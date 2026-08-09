"""Developer tool: score the answerability verifier against a question set.

Runs every question through retrieval and the verifier and prints a row per
question, so a misclassification can be read off rather than guessed at. The
question set is either the built-in publication-language regression set or a
JSON file describing another paper.

    python answerability_eval.py "/path/to/paper.pdf"
    python answerability_eval.py "/path/to/paper.pdf" --eval "/path/to/eval.json"

Requires Ollama with the embedding model pulled (`ollama pull nomic-embed-text`).
No language model is called: retrieval embeds, the verifier is deterministic.

EVALUATION FILE FORMAT

    {
      "paper": "paper name",
      "questions": [
        {"question": "...", "expected_answerable": true,  "expected_pages": [4]},
        {"question": "...", "expected_answerable": false, "expected_pages": []}
      ]
    }

``paper`` and ``expected_pages`` are optional. ``expected_pages`` is reported
separately and never affects answerability accuracy: whether the verifier
accepted a question and whether it cited the right page are two different
questions, and conflating them hides which one regressed.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from answerability import Answerability, AnswerabilityReport, verify_answerability
from embeddings import EmbeddedChunk, embed_chunks
from evidence_ranking import retrieve_evidence
from fixtures import LANGUAGE_EXPECTED_PAGES, language_answerability_cases
from ollama_client import OllamaError
from pdf import PdfError, extract_document
from retrieval import build_retrieval_chunks
from schemas import PageInput

DEFAULT_TOP_K = 5
RULE = "=" * 124
HEADER = "{:<50} {:<9} {:<20} {:>7} {:>7} {:<10} {:<6} {}".format(
    "question", "expected", "predicted", "top", "gap", "pages", "page?", "result"
)

# supported and partially_supported both mean "there is enough to answer".
ANSWERABLE_STATUSES = frozenset(
    {Answerability.SUPPORTED, Answerability.PARTIALLY_SUPPORTED}
)


class EvaluationFileError(Exception):
    """An evaluation file that could not be used, with the reason why."""


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


class EvaluationCase(BaseModel):
    question: str
    expected_answerable: bool
    expected_pages: List[int] = Field(default_factory=list)


class EvaluationSet(BaseModel):
    paper: str = ""
    questions: List[EvaluationCase] = Field(default_factory=list)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationFileError(message)


def parse_evaluation(payload: Any, source: str = "evaluation file") -> EvaluationSet:
    """Validate a decoded evaluation document, explaining any rejection."""
    _require(isinstance(payload, dict), "{}: top level must be an object.".format(source))

    paper = payload.get("paper", "")
    _require(
        isinstance(paper, str),
        "{}: 'paper' must be a string.".format(source),
    )

    questions = payload.get("questions")
    _require(questions is not None, "{}: missing 'questions'.".format(source))
    _require(
        isinstance(questions, list),
        "{}: 'questions' must be a list.".format(source),
    )
    _require(bool(questions), "{}: 'questions' is empty.".format(source))

    cases = []
    for index, entry in enumerate(questions):
        where = "{}: question {}".format(source, index + 1)
        _require(isinstance(entry, dict), "{} must be an object.".format(where))

        question = entry.get("question")
        _require(
            isinstance(question, str) and question.strip(),
            "{} needs a non-empty 'question'.".format(where),
        )

        answerable = entry.get("expected_answerable")
        _require(
            isinstance(answerable, bool),
            "{} needs 'expected_answerable' as true or false.".format(where),
        )

        pages = entry.get("expected_pages", [])
        _require(
            isinstance(pages, list),
            "{}: 'expected_pages' must be a list.".format(where),
        )
        for page in pages:
            _require(
                isinstance(page, int) and not isinstance(page, bool) and page >= 1,
                "{}: 'expected_pages' must contain page numbers from 1.".format(where),
            )

        cases.append(
            EvaluationCase(
                question=question.strip(),
                expected_answerable=answerable,
                expected_pages=list(pages),
            )
        )

    return EvaluationSet(paper=paper, questions=cases)


def load_evaluation(path: Path) -> EvaluationSet:
    """Read and validate an evaluation file."""
    if not path.is_file():
        raise EvaluationFileError("No such evaluation file: {}".format(path))

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise EvaluationFileError(
            "{} is not valid JSON: {}".format(path.name, error.msg)
        )
    except OSError as error:
        raise EvaluationFileError("Could not read {}: {}".format(path, error))

    evaluation = parse_evaluation(payload, path.name)
    if not evaluation.paper:
        evaluation.paper = path.stem
    return evaluation


def builtin_evaluation() -> EvaluationSet:
    """The publication-language regression set, kept as the default."""
    return EvaluationSet(
        paper="publication language (built-in regression set)",
        questions=[
            EvaluationCase(
                question=question,
                expected_answerable=answerable,
                expected_pages=LANGUAGE_EXPECTED_PAGES.get(question, []),
            )
            for question, answerable in language_answerability_cases()
        ],
    )


def _as_case(case: Union[EvaluationCase, Tuple[str, bool]]) -> EvaluationCase:
    """Accept either a case object or the older (question, answerable) tuple."""
    if isinstance(case, EvaluationCase):
        return case
    question, answerable = case
    return EvaluationCase(question=question, expected_answerable=answerable)


# --------------------------------------------------------------------------- #
# rows and metrics
# --------------------------------------------------------------------------- #


class EvaluationRow(object):
    """One evaluated question: what was expected, and what happened."""

    def __init__(self, case: EvaluationCase, report: AnswerabilityReport):
        self.case = case
        self.report = report

    @property
    def question(self) -> str:
        return self.case.question

    @property
    def answerable(self) -> bool:
        return self.case.expected_answerable

    @property
    def expected_pages(self) -> List[int]:
        return self.case.expected_pages

    @property
    def predicted_answerable(self) -> bool:
        return self.report.status in ANSWERABLE_STATUSES

    @property
    def correct(self) -> bool:
        return self.predicted_answerable == self.answerable

    @property
    def page_checked(self) -> bool:
        return bool(self.expected_pages)

    @property
    def page_hit(self) -> Optional[bool]:
        """True when at least one expected page backed the answer."""
        if not self.page_checked:
            return None
        return bool(set(self.expected_pages) & set(self.report.supporting_pages))


class EvaluationMetrics(BaseModel):
    """Answerability accuracy, plus the page check reported alongside it."""

    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    pages_checked: int = 0
    pages_hit: int = 0

    @property
    def page_accuracy(self) -> Optional[float]:
        if not self.pages_checked:
            return None
        return self.pages_hit / self.pages_checked


def compute_metrics(rows: Sequence[EvaluationRow]) -> EvaluationMetrics:
    """Confusion counts over the answerable/unanswerable decision."""
    metrics = EvaluationMetrics(total=len(rows))
    for row in rows:
        if row.answerable and row.predicted_answerable:
            metrics.true_positives += 1
        elif not row.answerable and not row.predicted_answerable:
            metrics.true_negatives += 1
        elif not row.answerable and row.predicted_answerable:
            metrics.false_positives += 1
        else:
            metrics.false_negatives += 1

        if row.page_checked:
            metrics.pages_checked += 1
            if row.page_hit:
                metrics.pages_hit += 1

    metrics.correct = metrics.true_positives + metrics.true_negatives
    metrics.accuracy = metrics.correct / metrics.total if metrics.total else 0.0
    return metrics


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def format_row(row: EvaluationRow) -> str:
    def number(value):
        return "  n/a" if value is None else "{:.3f}".format(value)

    pages = ",".join(str(page) for page in row.report.supporting_pages) or "-"
    page_hit = {None: "-", True: "yes", False: "no"}[row.page_hit]
    return "{:<50} {:<9} {:<20} {:>7} {:>7} {:<10} {:<6} {}".format(
        row.question[:50],
        "yes" if row.answerable else "no",
        row.report.status.value,
        number(row.report.retrieval_top_score),
        number(row.report.retrieval_score_gap),
        pages[:10],
        page_hit,
        "PASS" if row.correct else "MISS",
    )


def format_metrics(metrics: EvaluationMetrics) -> str:
    lines = [
        "total questions {}   {}/{} correct (accuracy {:.1%})".format(
            metrics.total, metrics.correct, metrics.total, metrics.accuracy
        ),
        "true positives {}   true negatives {}   "
        "false positives {}   false negatives {}".format(
            metrics.true_positives,
            metrics.true_negatives,
            metrics.false_positives,
            metrics.false_negatives,
        ),
    ]
    if metrics.pages_checked:
        lines.append(
            "expected-page check: {}/{} questions cited at least one expected "
            "page ({:.1%})".format(
                metrics.pages_hit,
                metrics.pages_checked,
                metrics.page_accuracy or 0.0,
            )
        )
        lines.append(
            "  (reported separately; page correctness is not part of "
            "answerability accuracy)"
        )
    return "\n".join(lines)


def format_table(rows: Sequence[EvaluationRow], paper: str = "") -> str:
    lines = [RULE]
    if paper:
        lines.append("Evaluation set: {}".format(paper))
    lines.extend([HEADER, "-" * 124])
    lines.extend(format_row(row) for row in rows)
    lines.append("-" * 124)
    lines.append(format_metrics(compute_metrics(rows)))
    lines.append(RULE)

    for row in rows:
        if not row.correct:
            lines.append("MISS  {}\n      {}".format(row.question, row.report.reason))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #


async def evaluate(
    cases: Sequence[Union[EvaluationCase, Tuple[str, bool]]],
    embedded: Sequence[EmbeddedChunk],
    top_k: int = DEFAULT_TOP_K,
) -> List[EvaluationRow]:
    """Run every case through retrieval and the verifier."""
    rows = []
    for raw in cases:
        case = _as_case(raw)
        ranking = await retrieve_evidence(case.question, embedded, top_k=top_k)
        rows.append(EvaluationRow(case, verify_answerability(case.question, ranking)))
    return rows


def load_pages(path: Path) -> List[PageInput]:
    if not path.is_file():
        raise SystemExit("No such file: {}".format(path))
    try:
        document = extract_document(path.read_bytes(), path.name)
    except PdfError as error:
        raise SystemExit("Could not read {}: {}".format(path.name, error.message))
    return [PageInput(**page) for page in document["pages"]]


async def run(
    path: Path, evaluation: EvaluationSet, top_k: int = DEFAULT_TOP_K
) -> int:
    pages = load_pages(path)
    chunks = build_retrieval_chunks(pages, path.name)
    if not chunks:
        print("The PDF produced no usable text.", file=sys.stderr)
        return 1

    try:
        embedded = await embed_chunks(chunks)
    except OllamaError as error:
        print("Embedding failed: {}".format(error.message), file=sys.stderr)
        return 1

    rows = await evaluate(evaluation.questions, embedded, top_k)
    print(format_table(rows, evaluation.paper))
    print(
        "\nStatuses are {}. supported and partially_supported both count as "
        "answerable. Scores are raw cosine statistics, not confidence "
        "percentages.".format(", ".join(status.value for status in Answerability))
    )
    return 0 if all(row.correct for row in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score the answerability verifier against a question set.",
        epilog=(
            'examples:\n'
            '  python answerability_eval.py "/path/to/paper.pdf"\n'
            '  python answerability_eval.py "/path/to/paper.pdf" '
            '--eval "/path/to/evaluation.json"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", help="path to a local PDF file")
    parser.add_argument(
        "--eval",
        dest="evaluation",
        metavar="JSON",
        help="evaluation file; omit to use the built-in regression set",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


def resolve_evaluation(argument: Optional[str]) -> EvaluationSet:
    if not argument:
        return builtin_evaluation()
    try:
        return load_evaluation(Path(argument))
    except EvaluationFileError as error:
        raise SystemExit(str(error))


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation = resolve_evaluation(args.evaluation)
    return asyncio.run(run(Path(args.pdf), evaluation, args.top_k))


def to_json(evaluation: EvaluationSet) -> Dict[str, Any]:
    """The on-disk form of an evaluation set."""
    return evaluation.model_dump()


if __name__ == "__main__":
    sys.exit(main())
