"""Developer tool: check semantic retrieval against a real PDF.

Extracts a paper, chunks it, embeds the chunks with the local embedding model,
and prints the top matches for a handful of hard questions. It is a diagnostic
for retrieval quality only: no answer is generated and the generation model is
never called. Everything stays in memory.

    python retrieval_demo.py "/path/to/paper.pdf"

Requires Ollama to be running with the embedding model pulled:

    ollama pull nomic-embed-text
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Sequence

from config import OLLAMA_EMBED_MODEL
from embeddings import (
    EmbeddedChunk,
    RankedChunk,
    RetrievalDiagnostics,
    embed_chunks,
)
from evidence_ranking import retrieve_evidence
from ollama_client import OllamaError
from pdf import PdfError, extract_document
from retrieval import RetrievalChunk, build_retrieval_chunks, exclusion_summary
from schemas import PageInput

DEFAULT_QUESTIONS = (
    "What applications did students use?",
    "How many participants were in the study?",
    "How were the data collected and analysed?",
    "What limitations did the authors identify?",
    "What role did teachers play in mobile learning?",
    "Did the paper report improved test scores?",
)

DEFAULT_TOP_K = 5
SNIPPET_CHARS = 300
RULE = "=" * 78


def load_pages(path: Path) -> List[PageInput]:
    """Read a PDF from disk using the same extraction as POST /upload."""
    if not path.exists():
        raise SystemExit("No such file: {}".format(path))
    if not path.is_file():
        raise SystemExit("Not a file: {}".format(path))

    try:
        document = extract_document(path.read_bytes(), path.name)
    except PdfError as error:
        raise SystemExit("Could not read {}: {}".format(path.name, error.message))

    return [PageInput(**page) for page in document["pages"]]


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """The opening of a chunk, collapsed onto one line."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def format_result(rank: int, result: RankedChunk) -> str:
    """One ranked hit, as printed."""
    return (
        "  {rank}. score {score:.4f}   page {page}   "
        "section {section}   {chunk_id}\n"
        "     {snippet}".format(
            rank=rank,
            score=result.score,
            page=result.page_number,
            section=result.section or "(unlabelled)",
            chunk_id=result.chunk_id,
            snippet=snippet(result.text),
        )
    )


def format_diagnostics(diagnostics: RetrievalDiagnostics) -> str:
    """Score shape for one ranking. Not confidence, not a probability."""

    def number(value):
        return "n/a" if value is None else "{:.4f}".format(value)

    return (
        "     [top {top}  second {second}  gap {gap}  mean@{k} {mean}]".format(
            top=number(diagnostics.top_score),
            second=number(diagnostics.second_score),
            gap=number(diagnostics.score_gap),
            k=diagnostics.returned,
            mean=number(diagnostics.mean_top_k),
        )
    )


def format_question(
    question: str,
    results: Sequence[RankedChunk],
    diagnostics: RetrievalDiagnostics = None,
) -> str:
    """A whole question block, as printed."""
    lines = ["", "Q: {}".format(question), "-" * 78]
    if not results:
        lines.append("  (no chunks matched)")
    else:
        lines.extend(
            format_result(rank, result) for rank, result in enumerate(results, start=1)
        )
    if diagnostics is not None:
        lines.append(format_diagnostics(diagnostics))
    return "\n".join(lines)


def describe_chunks(chunks: Sequence[RetrievalChunk]) -> str:
    """A one-line summary of what the chunker produced."""
    if not chunks:
        return "  no chunks"
    words = [len(chunk.text.split()) for chunk in chunks]
    sections = sorted({chunk.section for chunk in chunks if chunk.section})
    return (
        "  {count} chunks over {pages} pages, {low}-{high} words "
        "(mean {mean})\n  sections: {sections}".format(
            count=len(chunks),
            pages=len({chunk.page_number for chunk in chunks}),
            low=min(words),
            high=max(words),
            mean=sum(words) // len(words),
            sections=", ".join(sections) or "none detected",
        )
    )


async def run_demo(
    path: Path, questions: Sequence[str], top_k: int = DEFAULT_TOP_K
) -> int:
    """Print retrieval results for every question. Returns an exit code."""
    pages = load_pages(path)
    chunks = build_retrieval_chunks(pages, path.name)

    print(RULE)
    print("Retrieval demo: {}".format(path.name))
    print("Embedding model: {}".format(OLLAMA_EMBED_MODEL))
    print(RULE)
    print(describe_chunks(chunks))

    if not chunks:
        print("\nNothing to rank. The PDF produced no usable text.")
        return 1

    try:
        embedded: List[EmbeddedChunk] = await embed_chunks(chunks)
    except OllamaError as error:
        print("\nEmbedding failed: {}".format(error.message), file=sys.stderr)
        return 1

    print("  embedded {} chunks, {} dimensions".format(
        len(embedded), len(embedded[0].embedding) if embedded else 0
    ))

    excluded = exclusion_summary(embedded)
    if excluded:
        print("  excluded from retrieval: {}".format(
            ", ".join("{} {}".format(count, reason)
                      for reason, count in sorted(excluded.items()))
        ))

    for question in questions:
        try:
            ranking = await retrieve_evidence(question, embedded, top_k=top_k)
        except OllamaError as error:
            print("\nRanking failed: {}".format(error.message), file=sys.stderr)
            return 1
        print(format_question(question, ranking.results, ranking.diagnostics))

    print("")
    print(RULE)
    print("Scores are cosine similarity in [-1, 1]. Compare them within a")
    print("question, not across questions: only the ordering is meaningful.")
    print("A wide gap means the top chunk stood out; a gap near zero means")
    print("several chunks matched equally and the ordering is close to a")
    print("coin toss. These are not confidence percentages.")
    print(RULE)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect semantic retrieval quality for one PDF.",
        epilog='example: python retrieval_demo.py "/path/to/paper.pdf"',
    )
    parser.add_argument("pdf", help="path to a local PDF file")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="results per question (default: %(default)s)",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        metavar="TEXT",
        help="ask this instead of the built-in questions; repeatable",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    questions = tuple(args.questions) if args.questions else DEFAULT_QUESTIONS
    return asyncio.run(run_demo(Path(args.pdf), questions, args.top_k))


if __name__ == "__main__":
    sys.exit(main())
