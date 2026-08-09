"""Grounded answer generation for Ask the Paper.

The deterministic pipeline decides *what the evidence is*. The model decides
only *how to phrase it*. Everything a reader would rely on - which passages
count, which pages they came from, whether the paper answers the question at
all - is produced before the model is called and is never taken back from it.

Concretely:

* Phase 3A owns retrieval, ranking, answerability and page provenance.
* select_evidence() narrows the ranked passages to the smallest set that still
  supports an answer.
* The model receives only those passages, and returns only prose plus the ids
  of the passages it used.
* Every id is checked against what was sent. Unknown ids are dropped, pages are
  looked up from the parser's own metadata, and an answer that cannot be
  attributed to a specific passage is discarded rather than published.

A not_supported verdict never reaches the model at all.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from answerability import (
    Answerability,
    AnswerabilityReport,
    asks_for_specific_attribute,
    attribute_present,
    content_terms,
    detect_question_types,
    focus_terms,
    term_coverage,
    verify_answerability,
)
from config import (
    MAX_EVIDENCE_CHUNKS,
    OLLAMA_NUM_PREDICT_ANSWER,
    SELECTION_POOL_SIZE,
)
from diagnostics import debug
from embedding_cache import EMBEDDING_CACHE, document_key
from embeddings import RankedChunk, embed_chunks
from evidence_ranking import carries_question_evidence, retrieve_evidence
from ollama_client import OllamaError, generate_json
from retrieval import build_retrieval_chunks
from schemas import (
    AnswerSource,
    AskDiagnostics,
    AskResponse,
    PageInput,
)
from sections import split_sentences

logger = logging.getLogger(__name__)

# Two passages are treated as the same evidence above this token overlap.
# Chunks share a 60-word overlap by construction, so neighbours on a page are
# routinely near-duplicates.
DUPLICATE_OVERLAP = 0.6
# Upper bound on a quoted excerpt in the API response.
MAX_SNIPPET_CHARS = 320
# How much of a passage the model is shown. Long enough to answer from, short
# enough that four passages fit comfortably in context.
MAX_EVIDENCE_CHARS = 1200

ABSTAIN_ANSWER = (
    "The paper does not provide enough evidence to answer this question."
)
ABSTAIN_NOTE = "No sufficiently supported answer was found in the paper."
UNATTRIBUTABLE_ANSWER = (
    "An answer could not be traced to specific passages in the paper, so none "
    "is given."
)
SUPPORTED_NOTE = "Answer based only on retrieved evidence from the paper."
PARTIAL_NOTE = (
    "The paper only partially addresses this question; the answer below is "
    "based on limited evidence and should be checked against the paper."
)

SYSTEM_PROMPT = (
    "You are a careful research assistant. You are given numbered passages "
    "taken verbatim from one academic paper, and a question about that paper. "
    "You answer only from those passages. You never use outside knowledge, "
    "never infer facts the passages do not state, and never invent names, "
    "numbers, methods, outcomes, citations or page numbers. You reply with "
    "JSON only."
)


class GeneratedAnswer(BaseModel):
    """What the model is allowed to produce.

    Note what is absent: page numbers. Pages are parser-owned metadata and are
    attached afterwards, so the model has no way to invent one.
    """

    answer: str = ""
    used_chunk_ids: List[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = " ".join(str(item) for item in value)
        return str(value or "").strip()

    @field_validator("used_chunk_ids", mode="before")
    @classmethod
    def _coerce_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


# --------------------------------------------------------------------------- #
# evidence selection
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> Set[str]:
    return {word.strip(".,;:()[]\"'").lower() for word in text.split() if len(word) > 3}


def _overlap(first: str, second: str) -> float:
    """Jaccard-style containment of the smaller passage in the larger."""
    left, right = _tokens(first), _tokens(second)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


# Weights for the evidence-contribution score. These order candidates; none is
# a threshold, so a change shifts which passage is cited first, never whether a
# question is answerable. They are listed in descending importance: what the
# passage says about the question outranks where retrieval happened to put it.
W_TERM_COVERAGE = 3.0
W_NAMED_ITEMS = 2.5
W_TYPE_CUE = 2.0
W_ATTRIBUTE = 2.0
W_SUPPORTING = 1.0
W_RANK = 1.0

# Distinct proper nouns at which the named-item signal saturates.
NAMED_ITEM_SATURATION = 4

# A passage is cited only if it contributes at least this fraction of what the
# best passage contributes. Ordering constant, not a threshold on answerability:
# it decides how many passages are quoted, never whether a question is answered.
RELATIVE_KEEP = 0.75

# Capitalised words that are not names: sentence openers, headings, and the
# months and weekdays that litter academic prose.
_NOT_A_NAME = frozenset(
    """
    The This That These Those There Their They It Its As At In On For With From
    Table Figure Section Appendix Page Chapter Note Notes Study Results
    Findings Discussion Conclusion Introduction Method Methods Participants
    However Moreover Furthermore Although Because Since While When Where What
    Which Who How Why All Some Most Many Both Each One Two Three Four Five
    January February March April May June July August September October
    November December Monday Tuesday Wednesday Thursday Friday English
    """.split()
)
_PARTICIPANT_CODE = re.compile(r"^[A-Z]{1,2}\d{1,3}$")


def named_items(text: str) -> Set[str]:
    """Distinct proper nouns in a passage.

    A question like "which apps did students use" is answered by *names*, so a
    passage that contains several is a better citation than one that discusses
    the same topic abstractly. Sentence-initial words are skipped because
    English capitalises them regardless, and participant codes such as "S10"
    are skipped because every interview study is full of them.
    """
    found = set()
    for sentence in split_sentences(text):
        words = sentence.split()
        for position, word in enumerate(words):
            token = word.strip(".,;:()[]\"'?!")
            if len(token) < 3 or not token[0].isupper():
                continue
            if position == 0 or token in _NOT_A_NAME or _PARTICIPANT_CODE.match(token):
                continue
            if token.isupper() and len(token) > 5:
                continue
            found.add(token)
    return found


def asks_for_named_items(question: str) -> bool:
    """True for "which/what <thing>" questions our type taxonomy does not cover.

    Questions about limitations, methods, participants and so on already have a
    precise evidence family driving selection. What is left - "which apps",
    "what tools", "which databases" - is answered by naming things, and needs
    no list of noun classes to recognise: it is simply the residue.
    """
    if not re.match(r"^\s*(?:which|what)\b", question.strip(), re.IGNORECASE):
        return False
    return not any(
        question_type.precise for question_type, _ in detect_question_types(question)
    )


def content_contribution(
    question: str,
    chunk: RankedChunk,
    supporting: Set[str],
    wants_names: bool,
) -> float:
    """The part of the score that comes from what the passage says.

    Separated from the rank prior so a passage can be required to contribute
    *something* before it takes a citation slot. Retrieval rank alone is not a
    reason to quote a paragraph at the reader.
    """
    coverage, _ = term_coverage(content_terms(question), chunk.text)
    score = W_TERM_COVERAGE * coverage

    if carries_question_evidence(question, chunk):
        score += W_TYPE_CUE

    if asks_for_specific_attribute(question):
        wanted = focus_terms(question)
        chunk_terms = content_terms(chunk.text)
        if wanted and any(
            attribute_present(term, chunk_terms, chunk.text) for term in wanted
        ):
            score += W_ATTRIBUTE

    if wants_names:
        count = len(named_items(chunk.text))
        score += W_NAMED_ITEMS * min(1.0, count / float(NAMED_ITEM_SATURATION))

    if chunk.chunk_id in supporting:
        score += W_SUPPORTING

    return score


def contribution_score(
    question: str,
    chunk: RankedChunk,
    rank: int,
    supporting: Set[str],
    wants_names: bool,
) -> float:
    """How much a passage contributes to answering this specific question.

    Deterministic and independent of the model. Retrieval's own scores are read
    only through ``rank`` and are never modified.
    """
    content = content_contribution(question, chunk, supporting, wants_names)
    return content + W_RANK / float(1 + rank)


def select_evidence(
    report: AnswerabilityReport,
    ranked: Sequence[RankedChunk],
    question: str = "",
    limit: int = MAX_EVIDENCE_CHUNKS,
) -> List[RankedChunk]:
    """Choose the smallest evidence set that still answers the question.

    Earlier this took the first few answerability-supporting passages in rank
    order, which cited whatever cosine happened to like rather than whatever
    actually answered the question. Candidates are now scored on what they
    contain - question terms, the question type's own evidence language, a
    requested attribute, named items - with retrieval rank as a mild prior and
    a deterministic tie-break.

    Near-duplicates are dropped, keeping the higher-scoring copy, because
    adjacent chunks overlap by design. The result is capped but never emptied.
    """
    if not ranked:
        return []

    supporting = set(report.supporting_chunk_ids)
    wants_names = asks_for_named_items(question)

    scored = []
    for rank, chunk in enumerate(ranked):
        content = content_contribution(question, chunk, supporting, wants_names)
        total = content + W_RANK / float(1 + rank)
        scored.append((total, content, rank, chunk))

    # A passage must say something about the question to be worth citing;
    # a good retrieval rank on its own is not enough.
    contributing = [item for item in scored if item[1] > 0.0]
    if not contributing:
        # Nothing scored: fall back to what answerability endorsed, or failing
        # that the best-ranked passage. Selection is capped, never emptied.
        contributing = [item for item in scored if item[3].chunk_id in supporting]
    if not contributing:
        contributing = scored[:1]

    # Highest contribution first; ties fall back to retrieval order.
    contributing.sort(key=lambda item: (-item[0], item[2]))

    # Cite the smallest sufficient set: keep only passages that contribute
    # nearly as much as the best one, rather than padding to the cap. This is
    # what stops a limitations question from quoting two paragraphs that state
    # the limitations plus two more that merely sit near them. Relative, so it
    # adapts to a question whose evidence is spread thin.
    best_content = max(item[1] for item in contributing)
    if best_content > 0:
        floor = RELATIVE_KEEP * best_content
        contributing = [item for item in contributing if item[1] >= floor]

    selected: List[RankedChunk] = []
    for _total, _content, _rank, candidate in contributing:
        if any(
            _overlap(candidate.text, chosen.text) >= DUPLICATE_OVERLAP
            for chosen in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max(1, limit):
            break

    return selected


def evidence_snippet(
    text: str, question: str, limit: int = MAX_SNIPPET_CHARS
) -> str:
    """A short excerpt of a passage, taken verbatim from the passage.

    The window of consecutive sentences with the most words in common with the
    question is preferred, so the excerpt shows why the passage was cited. The
    model is never asked to produce or rewrite this text.
    """
    sentences = split_sentences(text)
    if not sentences:
        return " ".join(text.split())[:limit]

    wanted = _tokens(question)
    best_start, best_score = 0, -1
    for start in range(len(sentences)):
        window: List[str] = []
        length = 0
        for sentence in sentences[start : start + 3]:
            if length and length + len(sentence) + 1 > limit:
                break
            window.append(sentence)
            length += len(sentence) + 1
        if not window:
            continue
        score = len(wanted & _tokens(" ".join(window)))
        if score > best_score:
            best_score, best_start = score, start

    window = []
    length = 0
    for sentence in sentences[best_start : best_start + 3]:
        if length and length + len(sentence) + 1 > limit:
            break
        window.append(sentence)
        length += len(sentence) + 1

    excerpt = " ".join(window).strip() or sentences[0]
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rsplit(" ", 1)[0] + "..."
    return excerpt


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #


def render_evidence(chunks: Sequence[RankedChunk]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            "[Evidence {index}]\n"
            "Page: {page}\n"
            "Chunk ID: {chunk_id}\n"
            "Text:\n{text}".format(
                index=index,
                page=chunk.page_number,
                chunk_id=chunk.chunk_id,
                text=" ".join(chunk.text.split())[:MAX_EVIDENCE_CHARS],
            )
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str, chunks: Sequence[RankedChunk], partial: bool
) -> str:
    caution = ""
    if partial:
        caution = (
            "\nIMPORTANT: the evidence below only partially addresses this "
            "question. Say so plainly in your answer, use cautious wording "
            "such as 'the paper suggests' or 'based on the available "
            "evidence', and do not present a partial finding as a settled "
            "conclusion.\n"
        )

    return (
        "Answer the question using only the evidence passages below.\n"
        "{caution}\n"
        "RULES\n"
        "- Use only the passages provided. Do not use outside knowledge.\n"
        "- Do not infer facts the passages do not state.\n"
        "- If the passages do not fully answer the question, say what is "
        "missing rather than filling the gap.\n"
        "- Never invent names, numbers, methods, outcomes, citations or page "
        "numbers.\n"
        "- Preserve the source's own uncertainty: if the paper hedges, hedge.\n"
        "- Do not claim causation where the passages describe an association "
        "or a description.\n"
        "- Do not claim statistical significance unless a passage states it.\n"
        "- Page numbers are supplied metadata. Do not repeat or invent them in "
        "your answer.\n"
        "- Keep the answer to a few sentences.\n\n"
        "OUTPUT\n"
        'Return JSON with "answer" (your prose) and "used_chunk_ids" (the '
        "Chunk ID of every passage you actually used, copied exactly).\n\n"
        "{evidence}\n\n"
        "QUESTION:\n{question}"
    ).format(caution=caution, evidence=render_evidence(chunks), question=question)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def _abstention(
    report: AnswerabilityReport,
    diagnostics: AskDiagnostics,
    answer: str = ABSTAIN_ANSWER,
    note: str = ABSTAIN_NOTE,
) -> AskResponse:
    return AskResponse(
        status=Answerability.NOT_SUPPORTED.value,
        answer=answer,
        sources=[],
        confidence_note=note,
        diagnostics=diagnostics,
    )


def _sources_for(
    chunks: Sequence[RankedChunk], used: Sequence[str], question: str
) -> List[AnswerSource]:
    """Build citations from parser metadata for the passages actually used."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    sources = []
    for chunk_id in used:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            continue
        sources.append(
            AnswerSource(
                page=chunk.page_number,
                chunk_id=chunk.chunk_id,
                evidence=evidence_snippet(chunk.text, question),
                section=chunk.section,
            )
        )
    return sources


def validate_used_ids(
    produced: Sequence[str], selected: Sequence[RankedChunk]
) -> Tuple[List[str], List[str]]:
    """Split model-produced ids into those we sent and those we did not."""
    allowed = [chunk.chunk_id for chunk in selected]
    known, unknown = [], []
    for chunk_id in produced:
        if chunk_id in allowed and chunk_id not in known:
            known.append(chunk_id)
        elif chunk_id not in allowed:
            unknown.append(chunk_id)
    # Report in the order the evidence was supplied, not the order the model
    # happened to mention them.
    known.sort(key=allowed.index)
    return known, unknown


async def generate_grounded_answer(
    question: str,
    answerability_result: AnswerabilityReport,
    ranked_evidence: Sequence[RankedChunk],
    diagnostics: Optional[AskDiagnostics] = None,
) -> AskResponse:
    """Turn a verdict plus evidence into an answer, or abstain.

    Never retrieves anything: everything it can cite is in ``ranked_evidence``.
    """
    diagnostics = diagnostics or AskDiagnostics()
    diagnostics.answerability_reason = answerability_result.reason
    diagnostics.retrieved_chunk_ids = [c.chunk_id for c in ranked_evidence]

    # ---- not_supported: the model is never called ----
    if answerability_result.status is Answerability.NOT_SUPPORTED:
        logger.info("Abstaining without generation: %s", answerability_result.reason)
        return _abstention(answerability_result, diagnostics)

    selected = select_evidence(answerability_result, ranked_evidence, question)
    diagnostics.selected_chunk_ids = [c.chunk_id for c in selected]
    if not selected:
        logger.warning("Answerable verdict but no evidence survived selection.")
        return _abstention(answerability_result, diagnostics)

    partial = answerability_result.status is Answerability.PARTIALLY_SUPPORTED

    diagnostics.generation_called = True
    raw = await generate_json(
        build_prompt(question, selected, partial),
        system=SYSTEM_PROMPT,
        schema=GeneratedAnswer.model_json_schema(),
        num_ctx=8192,
        num_predict=OLLAMA_NUM_PREDICT_ANSWER,
        label="ASK",
    )

    try:
        produced = GeneratedAnswer.model_validate(raw)
    except Exception as exc:
        raise OllamaError(
            "The model returned an answer in an unexpected shape ({}). "
            "Try asking again.".format(type(exc).__name__),
            502,
        )

    if not produced.answer:
        raise OllamaError("The model returned an empty answer.", 502)

    known, unknown = validate_used_ids(produced.used_chunk_ids, selected)
    if unknown:
        logger.warning(
            "Dropped %d chunk id(s) the model invented or altered.", len(unknown)
        )

    # ---- fail closed on unattributable answers ----
    if not known:
        if len(selected) == 1:
            # Only one passage was supplied, so attribution is unambiguous.
            # This is a deterministic fallback, not a guess.
            known = [selected[0].chunk_id]
            debug(logger, "ASK: single-evidence attribution fallback")
        else:
            logger.warning(
                "Discarding an answer that cited none of the %d supplied "
                "passages.",
                len(selected),
            )
            diagnostics.model_used_chunk_ids = []
            return _abstention(
                answerability_result,
                diagnostics,
                answer=UNATTRIBUTABLE_ANSWER,
                note=ABSTAIN_NOTE,
            )

    sources = _sources_for(selected, known, question)
    diagnostics.model_used_chunk_ids = known
    diagnostics.cited_pages = sorted({source.page for source in sources})

    return AskResponse(
        status=answerability_result.status.value,
        answer=produced.answer,
        sources=sources,
        confidence_note=PARTIAL_NOTE if partial else SUPPORTED_NOTE,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


async def answer_question(
    question: str,
    filename: str,
    pages: Sequence[PageInput],
    top_k: int = 5,
) -> AskResponse:
    """Run the frozen Phase 3A pipeline, then generate or abstain."""
    key = document_key(filename, pages)
    embedded = EMBEDDING_CACHE.get(key)
    cache_hit = embedded is not None

    if embedded is None:
        chunks = build_retrieval_chunks(pages, filename)
        embedded = await embed_chunks(chunks)
        EMBEDDING_CACHE.put(key, embedded)

    diagnostics = AskDiagnostics(embedding_cache_hit=cache_hit)

    if not embedded:
        report = AnswerabilityReport(
            status=Answerability.NOT_SUPPORTED,
            reason="The paper produced no passages that could be searched.",
        )
        diagnostics.answerability_reason = report.reason
        return _abstention(report, diagnostics)

    # The answerability verdict is taken from exactly the window Phase 3A was
    # accepted on. Nothing about that call changes.
    ranking = await retrieve_evidence(question, embedded, top_k=top_k)
    report = verify_answerability(question, ranking)
    answerable = report.status is not Answerability.NOT_SUPPORTED

    diagnostics.retrieval_top_score = ranking.diagnostics.top_score
    diagnostics.retrieval_score_gap = ranking.diagnostics.score_gap
    diagnostics.retrieval_mean_top_k = ranking.diagnostics.mean_top_k

    # Evidence selection gets a wider view. A passage that names exactly what a
    # question asks for can sit just outside the answerability window: the
    # verdict does not need it, but the citation does. This is a second ranking
    # call rather than a slice of a wider one, so the verdict above is bit for
    # bit what it was before. The cost is one extra question embedding.
    candidates = ranking.results
    if answerable and len(embedded) > top_k:
        wider = await retrieve_evidence(
            question, embedded, top_k=max(top_k, SELECTION_POOL_SIZE)
        )
        candidates = wider.results

    return await generate_grounded_answer(question, report, candidates, diagnostics)


def cache_stats() -> Dict[str, int]:
    return EMBEDDING_CACHE.stats()
