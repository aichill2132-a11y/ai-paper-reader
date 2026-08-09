"""Evidence-aware reranking between retrieval and verification.

Cosine ranking scores a whole chunk. When a paper states its limitations in the
last forty words of a four-hundred-word Discussion chunk, that statement is a
tenth of the vector and the chunk ranks below unrelated prose that happens to be
about the same subject. The verifier then refuses a question whose evidence the
paper plainly contains, because retrieval never handed it over.

The fix is to widen the candidate pool and reorder it deterministically before
truncation:

    1. rank a pool CANDIDATE_MULTIPLIER times larger than the caller wants;
    2. stable-partition it so chunks carrying the question type's own evidence
       language come first, keeping cosine order inside each group;
    3. truncate to top_k.

No score is altered and no threshold is involved: this is a reordering, so the
best-matching chunk still leads its group. Only question types whose evidence
family is specific take part (see QuestionType.precise), because promoting on a
family that matches any digit or any past-tense verb would reorder everything
and mean nothing.
"""

from typing import List, Sequence

from answerability import detect_question_types, evidence_text
from embeddings import (
    EmbeddedChunk,
    RankedChunk,
    RetrievalDiagnostics,
    RetrievalResult,
    rank_with_diagnostics,
)

# How many candidates to consider before reordering. Four is enough to reach an
# evidence chunk that dense similarity placed just outside the top five, without
# scanning the whole document.
CANDIDATE_MULTIPLIER = 4
# Small documents should still get a usable pool.
MIN_CANDIDATE_POOL = 20


def carries_question_evidence(question: str, chunk: RankedChunk) -> bool:
    """True when a chunk speaks the language of the question's own type."""
    for question_type, match in detect_question_types(question):
        if not question_type.precise:
            continue
        if question_type.evidence_present(evidence_text(chunk), match):
            return True
    return False


def rerank_by_evidence_cues(
    question: str, ranked: Sequence[RankedChunk]
) -> List[RankedChunk]:
    """Stable-partition a ranking so cue-bearing chunks come first."""
    if not any(question_type.precise for question_type, _ in detect_question_types(question)):
        return list(ranked)

    with_cue = [chunk for chunk in ranked if carries_question_evidence(question, chunk)]
    if not with_cue or len(with_cue) == len(ranked):
        # Nothing to promote, or everything qualifies: leave the order alone.
        return list(ranked)

    without_cue = [
        chunk for chunk in ranked if not carries_question_evidence(question, chunk)
    ]
    return with_cue + without_cue


def _rebuild_diagnostics(
    pool: RetrievalDiagnostics,
    pool_results: Sequence[RankedChunk],
    results: Sequence[RankedChunk],
) -> RetrievalDiagnostics:
    """Diagnostics for the returned set.

    top_score, second_score and score_gap describe *retrieval strength*, so
    they are taken from the score-ordered pool rather than from the reordered
    evidence set. Reading them off a reordered list produced negative gaps,
    which is meaningless and fed straight into the verifier's noise-floor test.
    mean_top_k describes the evidence actually returned.
    """
    ordered = sorted((item.score for item in pool_results), reverse=True)
    top = ordered[0] if ordered else None
    second = ordered[1] if len(ordered) > 1 else None
    returned_scores = [item.score for item in results]
    return RetrievalDiagnostics(
        considered=pool.considered,
        excluded=pool.excluded,
        excluded_by_reason=dict(pool.excluded_by_reason),
        returned=len(results),
        top_score=top,
        second_score=second,
        score_gap=None if second is None else top - second,
        mean_top_k=(
            sum(returned_scores) / len(returned_scores) if returned_scores else None
        ),
    )


async def retrieve_evidence(
    question: str,
    embedded_chunks: Sequence[EmbeddedChunk],
    top_k: int = 5,
    evidence_only: bool = True,
) -> RetrievalResult:
    """Rank, rerank on evidence cues, and return the top_k passages."""
    if not embedded_chunks or top_k <= 0:
        return await rank_with_diagnostics(
            question, embedded_chunks, top_k=top_k, evidence_only=evidence_only
        )

    pool_size = max(MIN_CANDIDATE_POOL, top_k * CANDIDATE_MULTIPLIER)
    pool = await rank_with_diagnostics(
        question, embedded_chunks, top_k=pool_size, evidence_only=evidence_only
    )

    reordered = rerank_by_evidence_cues(question, pool.results)[:top_k]
    return RetrievalResult(
        results=reordered,
        diagnostics=_rebuild_diagnostics(
            pool.diagnostics, pool.results, reordered
        ),
    )
