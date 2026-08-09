"""Embedding and semantic ranking for Ask the Paper.

Chunks produced by ``retrieval.build_retrieval_chunks`` are embedded with a
dedicated local embedding model served by Ollama, then ranked against a
question by cosine similarity. Everything stays in memory and on this machine:
there is no vector database, no persistence, and no external service.

MODEL
    ``OLLAMA_EMBED_MODEL`` defaults to ``nomic-embed-text``. A purpose-built
    embedding model is preferred over reusing ``qwen3:8b``: it is roughly a
    twentieth of the size, an order of magnitude faster per chunk, and its
    vectors are actually trained so that cosine distance means something. The
    model is never silently substituted; if it is missing, the error says so
    and gives the ``ollama pull`` command.

TRANSPORT
    ``/api/embed`` is used because it embeds a whole batch per request. Older
    Ollama builds only expose the single-input ``/api/embeddings``; that is
    detected once on a 404 and remembered for the rest of the process.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from config import OLLAMA_EMBED_BATCH, OLLAMA_EMBED_MODEL
from diagnostics import debug
from ollama_client import OllamaError, post_json
from retrieval import RetrievalChunk, eligible_chunks, exclusion_summary

logger = logging.getLogger(__name__)

EMBED_TIMEOUT_HINT = (
    "Embedding a long paper can take a while on first run, while the model "
    "is loaded into memory."
)

# Ranking scores are compared at this precision before the tie-break runs, so
# that two vectors which are equal in every practical sense do not order
# themselves by floating-point noise.
SCORE_PRECISION = 12

# Set to False the first time /api/embed returns 404 (older Ollama builds).
_CAPABILITIES = {"batch_embed": True}


def reset_capabilities() -> None:
    """Re-enable batch embedding. Used by the tests."""
    _CAPABILITIES["batch_embed"] = True


class EmbeddingError(OllamaError):
    """The embedding response was unusable. Distinguishable from transport errors."""


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


class EmbeddedChunk(BaseModel):
    """A retrieval chunk plus its vector. Held in memory only."""

    chunk_id: str
    page_number: int = Field(ge=1)
    section: str = ""
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    embedding: List[float]

    @classmethod
    def from_chunk(
        cls, chunk: RetrievalChunk, embedding: Sequence[float]
    ) -> "EmbeddedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            section=chunk.section,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            embedding=list(embedding),
        )


class RankedChunk(BaseModel):
    """One search result: chunk metadata plus its similarity to the question.

    The vector is deliberately left out; callers want the text and provenance,
    and a 768-float array per result would dominate any response payload.
    """

    chunk_id: str
    page_number: int = Field(ge=1)
    section: str = ""
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    score: float

    @classmethod
    def from_embedded(cls, chunk: EmbeddedChunk, score: float) -> "RankedChunk":
        return cls(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            section=chunk.section,
            text=chunk.text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            score=score,
        )


# --------------------------------------------------------------------------- #
# response validation
# --------------------------------------------------------------------------- #


def _validate_vector(raw: Any, index: int, expected_dimension: Optional[int]) -> List[float]:
    """Turn one raw embedding from Ollama into a usable vector, or explain why not."""
    if not isinstance(raw, (list, tuple)):
        raise EmbeddingError(
            "The embedding model returned {} instead of a vector for input {}.".format(
                type(raw).__name__, index
            ),
            502,
        )

    vector: List[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EmbeddingError(
                "The embedding model returned a non-numeric value for input "
                "{}.".format(index),
                502,
            )
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            raise EmbeddingError(
                "The embedding model returned a non-finite value for input "
                "{}.".format(index),
                502,
            )
        vector.append(number)

    if not vector:
        raise EmbeddingError(
            "The embedding model returned an empty vector for input {}. "
            "Check that '{}' is an embedding model.".format(index, OLLAMA_EMBED_MODEL),
            502,
        )

    if not any(vector):
        raise EmbeddingError(
            "The embedding model returned an all-zero vector for input {}, "
            "which has no direction and cannot be compared.".format(index),
            502,
        )

    if expected_dimension is not None and len(vector) != expected_dimension:
        raise EmbeddingError(
            "The embedding model returned inconsistent dimensions "
            "({} then {}). All chunks must be embedded by the same model.".format(
                expected_dimension, len(vector)
            ),
            502,
        )

    return vector


def _model_not_installed() -> OllamaError:
    return OllamaError(
        "The embedding model '{0}' is not installed in Ollama. "
        "Install it with: ollama pull {0}".format(OLLAMA_EMBED_MODEL),
        503,
    )


def _raise_for_status(response: Any) -> None:
    if response.status_code == 404:
        raise _model_not_installed()
    if response.status_code < 400:
        return

    try:
        body = response.json()
        detail = str(body.get("error", "")).strip() if isinstance(body, dict) else ""
    except ValueError:
        detail = response.text.strip()
    detail = detail or "HTTP {}".format(response.status_code)

    if "not found" in detail.lower() or "try pulling" in detail.lower():
        raise _model_not_installed()
    raise OllamaError("Ollama returned an error: {}".format(detail), 502)


def _response_body(response: Any) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        raise EmbeddingError("Ollama returned a response that was not JSON.", 502)
    if not isinstance(body, dict):
        raise EmbeddingError("Ollama returned a response that was not an object.", 502)
    return body


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


async def _embed_batch(
    texts: Sequence[str], expected_dimension: Optional[int] = None
) -> List[List[float]]:
    """Embed a batch of texts in one request where the server supports it."""
    if not texts:
        return []

    if _CAPABILITIES["batch_embed"]:
        response = await post_json(
            "/api/embed",
            {"model": OLLAMA_EMBED_MODEL, "input": list(texts)},
            EMBED_TIMEOUT_HINT,
        )
        if response.status_code == 404:
            # Either the model is missing or this build predates /api/embed.
            # The legacy endpoint tells us which.
            _CAPABILITIES["batch_embed"] = False
            debug(logger, "/api/embed unavailable; falling back to /api/embeddings")
        else:
            _raise_for_status(response)
            raw = _response_body(response).get("embeddings")
            if not isinstance(raw, list) or len(raw) != len(texts):
                raise EmbeddingError(
                    "The embedding model returned {} vectors for {} inputs.".format(
                        len(raw) if isinstance(raw, list) else "no",
                        len(texts),
                    ),
                    502,
                )
            vectors: List[List[float]] = []
            dimension = expected_dimension
            for index, item in enumerate(raw):
                vector = _validate_vector(item, index, dimension)
                dimension = len(vector)
                vectors.append(vector)
            return vectors

    # Legacy single-input endpoint.
    vectors = []
    dimension = expected_dimension
    for index, text in enumerate(texts):
        response = await post_json(
            "/api/embeddings",
            {"model": OLLAMA_EMBED_MODEL, "prompt": text},
            EMBED_TIMEOUT_HINT,
        )
        _raise_for_status(response)
        vector = _validate_vector(
            _response_body(response).get("embedding"), index, dimension
        )
        dimension = len(vector)
        vectors.append(vector)
    return vectors


async def embed_text(text: str) -> List[float]:
    """Embed a single string and return its vector."""
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text.", 400)

    vectors = await _embed_batch([text])
    if not vectors:
        raise EmbeddingError("The embedding model returned no vector.", 502)
    return vectors[0]


async def embed_chunks(
    chunks: Sequence[RetrievalChunk], batch_size: int = OLLAMA_EMBED_BATCH
) -> List[EmbeddedChunk]:
    """Embed every chunk, in order, batching requests to Ollama.

    Chunks with no usable text are skipped rather than sent to the model.
    """
    usable = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]
    if not usable:
        return []

    size = max(1, batch_size)
    started = time.monotonic()

    embedded: List[EmbeddedChunk] = []
    dimension: Optional[int] = None
    for offset in range(0, len(usable), size):
        window = usable[offset : offset + size]
        vectors = await _embed_batch([chunk.text for chunk in window], dimension)
        dimension = len(vectors[0]) if vectors else dimension
        for chunk, vector in zip(window, vectors):
            embedded.append(EmbeddedChunk.from_chunk(chunk, vector))

    logger.info(
        "Embedded %d chunks with %s (%d dimensions) in %.1fs",
        len(embedded),
        OLLAMA_EMBED_MODEL,
        dimension or 0,
        time.monotonic() - started,
    )
    return embedded


# --------------------------------------------------------------------------- #
# similarity and ranking
# --------------------------------------------------------------------------- #


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors.

    Raises on empty or mismatched inputs, which are programming errors. A
    zero-magnitude vector has no direction, so its similarity is reported as
    0.0 rather than raising; ``_validate_vector`` already rejects such vectors
    at embedding time.
    """
    if not vector_a or not vector_b:
        raise EmbeddingError("Cannot compare an empty vector.", 500)
    if len(vector_a) != len(vector_b):
        raise EmbeddingError(
            "Vector dimensions do not match ({} and {}).".format(
                len(vector_a), len(vector_b)
            ),
            500,
        )

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vector_a, vector_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Guard against a hair over 1.0 from floating-point rounding.
    return max(-1.0, min(1.0, similarity))


class RetrievalDiagnostics(BaseModel):
    """Raw statistics about one ranking, for judging retrieval quality.

    These are cosine statistics, **not** confidence percentages and not
    calibrated probabilities. A top score of 0.62 does not mean 62% correct.
    What they are useful for is comparing results *within* one question:
    a large ``score_gap`` means the top chunk stood out, while a gap near zero
    means several chunks matched equally well and the ordering is close to
    arbitrary. Abstention is deliberately not implemented on top of these yet.
    """

    considered: int = 0
    excluded: int = 0
    excluded_by_reason: Dict[str, int] = Field(default_factory=dict)
    returned: int = 0
    top_score: Optional[float] = None
    second_score: Optional[float] = None
    score_gap: Optional[float] = None
    mean_top_k: Optional[float] = None


class RetrievalResult(BaseModel):
    """Ranked evidence plus the diagnostics for that ranking."""

    results: List[RankedChunk] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)


def _diagnostics(
    results: Sequence[RankedChunk], considered: int, excluded_by_reason: Dict[str, int]
) -> RetrievalDiagnostics:
    scores = [item.score for item in results]
    top = scores[0] if scores else None
    second = scores[1] if len(scores) > 1 else None
    return RetrievalDiagnostics(
        considered=considered,
        excluded=sum(excluded_by_reason.values()),
        excluded_by_reason=dict(excluded_by_reason),
        returned=len(results),
        top_score=top,
        second_score=second,
        score_gap=None if second is None else top - second,
        mean_top_k=sum(scores) / len(scores) if scores else None,
    )


def _rank_key(ranked: RankedChunk):
    """Highest score first, then document order, then id.

    Scores are rounded before comparison so that vectors which are identical
    in every practical sense tie-break on position rather than on the last
    bits of a float.
    """
    return (
        -round(ranked.score, SCORE_PRECISION),
        ranked.page_number,
        ranked.start_char,
        ranked.chunk_id,
    )


async def rank_with_diagnostics(
    question: str,
    embedded_chunks: Sequence[EmbeddedChunk],
    top_k: int = 5,
    evidence_only: bool = True,
) -> RetrievalResult:
    """Rank chunks against a question and report how the ranking looked.

    ``evidence_only`` filters out references, acknowledgements, keywords,
    front matter and stub chunks before anything is scored, so they cannot
    occupy a slot that real evidence needs. Nothing is removed from the corpus
    the caller holds; this is a view applied at ranking time.

    The question is embedded exactly once. An empty corpus or a non-positive
    ``top_k`` short-circuits without calling the model at all.
    """
    excluded_by_reason = exclusion_summary(embedded_chunks) if evidence_only else {}
    candidates = (
        eligible_chunks(embedded_chunks) if evidence_only else list(embedded_chunks)
    )

    if not candidates or top_k <= 0:
        return RetrievalResult(
            results=[],
            diagnostics=_diagnostics([], len(candidates), excluded_by_reason),
        )

    question_vector = await embed_text(question)

    dimension = len(candidates[0].embedding)
    if len(question_vector) != dimension:
        raise EmbeddingError(
            "The question was embedded to {} dimensions but the chunks have {}. "
            "They must be embedded by the same model.".format(
                len(question_vector), dimension
            ),
            502,
        )

    ranked = [
        RankedChunk.from_embedded(
            chunk, cosine_similarity(question_vector, chunk.embedding)
        )
        for chunk in candidates
    ]
    ranked.sort(key=_rank_key)
    results = ranked[:top_k]

    diagnostics = _diagnostics(results, len(candidates), excluded_by_reason)
    debug(
        logger,
        "ranked %d of %d chunks (%d excluded); top %.4f, gap %s",
        len(results),
        len(candidates),
        diagnostics.excluded,
        diagnostics.top_score or 0.0,
        "n/a" if diagnostics.score_gap is None else "{:.4f}".format(diagnostics.score_gap),
    )
    return RetrievalResult(results=results, diagnostics=diagnostics)


async def rank_chunks(
    question: str,
    embedded_chunks: Sequence[EmbeddedChunk],
    top_k: int = 5,
    evidence_only: bool = True,
) -> List[RankedChunk]:
    """Rank chunks against a question by cosine similarity, best first.

    Thin wrapper over ``rank_with_diagnostics`` for callers that only want the
    evidence. The ordering is identical.
    """
    ranking = await rank_with_diagnostics(
        question, embedded_chunks, top_k=top_k, evidence_only=evidence_only
    )
    return ranking.results
