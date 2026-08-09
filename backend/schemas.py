"""Request and response models for the summary endpoint.

Two "not stated" sentinels are used deliberately:

* ``NOT_STATED_IN_CHUNK`` is what the MAP stage says about a single excerpt.
  It means "this excerpt does not cover it", never "the paper does not cover it".
* ``NOT_STATED`` is the final answer for the whole paper, and is only correct
  when every chunk came back empty.

The validators recover malformed-but-usable model output instead of discarding
it. Falling back to a sentinel is a last resort and is logged (without any
paper text) so the loss is visible in development.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

NOT_STATED = "Not stated in the paper"
NOT_STATED_IN_CHUNK = "Not stated in this chunk"

# Anything a model might emit to mean "nothing here".
_EMPTY_MARKERS = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "not stated",
    "not applicable",
    "not mentioned",
    "not specified",
    "not available",
    "not stated in this chunk",
    "not stated in the paper",
    "not stated in the excerpt",
    "not stated in the text",
}

# Keys a model tends to wrap a real answer in.
_VALUE_KEYS = ("value", "text", "summary", "content", "description", "answer")

# Fields of PaperSummary that carry page provenance.
SOURCE_PAGE_FIELDS = (
    "research_question",
    "methods",
    "key_findings",
    "limitations",
)


def is_missing(text: Any) -> bool:
    """True when a value carries no information."""
    if text is None:
        return True
    return str(text).strip().lower().rstrip(".") in _EMPTY_MARKERS


def _unwrap(value: Any) -> Any:
    """Pull the real answer out of a wrapper object the model may have used."""
    if isinstance(value, dict):
        for key in _VALUE_KEYS:
            if key in value and value[key] is not None:
                return value[key]
    return value


def _as_text(value: Any, fallback: str = NOT_STATED) -> str:
    """Coerce a model-produced value into a non-empty string.

    Nested and list-shaped values are flattened rather than thrown away, so a
    structurally odd but informative answer survives.
    """
    value = _unwrap(value)

    if value is None:
        return fallback

    if isinstance(value, dict):
        parts = [_as_text(item, "") for item in value.values()]
        text = " ".join(part for part in parts if part and not is_missing(part))
    elif isinstance(value, (list, tuple, set)):
        parts = [_as_text(item, "") for item in value]
        text = " ".join(part for part in parts if part and not is_missing(part))
    else:
        text = str(value).strip()

    text = text.strip()
    if not text or is_missing(text):
        return fallback
    return text


def _as_text_list(value: Any, fallback: str = NOT_STATED) -> List[str]:
    """Coerce a model-produced value into a de-duplicated list of strings."""
    value = _unwrap(value)

    if value is None:
        return [fallback]
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    items: List[str] = []
    for item in value:
        text = _as_text(item, "")
        if text and not is_missing(text) and text not in items:
            items.append(text)
    return items or [fallback]


def _as_page_list(value: Any) -> List[int]:
    """Coerce a model-produced value into a sorted list of page numbers."""
    value = _unwrap(value)

    if value is None:
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    pages = set()
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            page = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if page > 0:
            pages.add(page)
    return sorted(pages)


# --------------------------------------------------------------------------- #
# request
# --------------------------------------------------------------------------- #


class PageInput(BaseModel):
    """One extracted page, as returned by POST /upload."""

    page_number: int = Field(ge=1)
    text: str = ""


class SummaryRequest(BaseModel):
    filename: str = "paper.pdf"
    pages: List[PageInput] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# MAP stage
# --------------------------------------------------------------------------- #


class Evidence(BaseModel):
    """One extracted field: the answer, the words it came from, and its pages."""

    value: str = NOT_STATED_IN_CHUNK
    evidence: str = ""
    pages: List[int] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def _coerce_value(cls, value: Any) -> str:
        return _as_text(value, NOT_STATED_IN_CHUNK)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, value: Any) -> str:
        text = _as_text(value, "")
        return "" if is_missing(text) else text

    @field_validator("pages", mode="before")
    @classmethod
    def _coerce_pages(cls, value: Any) -> List[int]:
        return _as_page_list(value)

    @property
    def has_value(self) -> bool:
        return not is_missing(self.value)


def _empty_evidence() -> Evidence:
    return Evidence()


class ChunkSummary(BaseModel):
    """Output of the MAP step: explicit extraction from one chunk of pages."""

    page_start: int = 0
    page_end: int = 0
    title_candidates: List[str] = Field(default_factory=list)
    author_candidates: List[str] = Field(default_factory=list)
    research_question: Evidence = Field(default_factory=_empty_evidence)
    background: Evidence = Field(default_factory=_empty_evidence)
    methods: Evidence = Field(default_factory=_empty_evidence)
    participants_or_data: Evidence = Field(default_factory=_empty_evidence)
    key_findings: List[Evidence] = Field(default_factory=list)
    limitations: List[Evidence] = Field(default_factory=list)

    @field_validator("page_start", "page_end", mode="before")
    @classmethod
    def _coerce_page(cls, value: Any) -> int:
        pages = _as_page_list(value)
        return pages[0] if pages else 0

    @field_validator("title_candidates", "author_candidates", mode="before")
    @classmethod
    def _coerce_candidates(cls, value: Any) -> List[str]:
        if value is None:
            return []
        items = _as_text_list(value, "")
        return [item for item in items if item and not is_missing(item)]

    @field_validator(
        "research_question",
        "background",
        "methods",
        "participants_or_data",
        mode="before",
    )
    @classmethod
    def _coerce_evidence_field(cls, value: Any) -> Any:
        # A bare string is a valid, recoverable answer: keep it as the value.
        if value is None:
            return {}
        if isinstance(value, (list, tuple, set)):
            items = [item for item in value if item is not None]
            if not items:
                return {}
            # A single-item list is just a wrapper; more than one gets joined
            # rather than silently dropped.
            value = items[0] if len(items) == 1 else {"value": items}
        if isinstance(value, dict):
            return value
        return {"value": value}

    @field_validator("key_findings", "limitations", mode="before")
    @classmethod
    def _coerce_evidence_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            # Either one evidence object, or a mapping of items.
            if any(key in value for key in ("value", "evidence", "pages")):
                return [value]
            value = list(value.values())
        if not isinstance(value, (list, tuple, set)):
            value = [value]
        return [
            {"value": item} if isinstance(item, (str, int, float)) else item
            for item in value
            if item is not None
        ]

    def populated_fields(self) -> List[str]:
        """Names of the fields this chunk actually filled in. Used for debug logs."""
        filled = []
        if self.title_candidates:
            filled.append("title")
        if self.author_candidates:
            filled.append("authors")
        for name in ("research_question", "background", "methods", "participants_or_data"):
            if getattr(self, name).has_value:
                filled.append(name)
        if any(item.has_value for item in self.key_findings):
            filled.append("key_findings")
        if any(item.has_value for item in self.limitations):
            filled.append("limitations")
        return filled


# --------------------------------------------------------------------------- #
# REDUCE stage / response
# --------------------------------------------------------------------------- #


class SourcePages(BaseModel):
    """Page numbers backing each section of the summary."""

    research_question: List[int] = Field(default_factory=list)
    methods: List[int] = Field(default_factory=list)
    key_findings: List[int] = Field(default_factory=list)
    limitations: List[int] = Field(default_factory=list)

    @field_validator(*SOURCE_PAGE_FIELDS, mode="before")
    @classmethod
    def _coerce_pages(cls, value: Any) -> List[int]:
        return _as_page_list(value)


class SummaryFields(BaseModel):
    """The fields shared by the condensed and the final summary.

    Every text field falls back to NOT_STATED and every list falls back to
    [NOT_STATED], so a partial model response still validates.
    """

    title: str = NOT_STATED
    authors: List[str] = Field(default_factory=lambda: [NOT_STATED])
    research_question: str = NOT_STATED
    background: str = NOT_STATED
    methods: str = NOT_STATED
    participants_or_data: str = NOT_STATED
    key_findings: List[str] = Field(default_factory=lambda: [NOT_STATED])
    limitations: List[str] = Field(default_factory=lambda: [NOT_STATED])

    @field_validator(
        "title",
        "research_question",
        "background",
        "methods",
        "participants_or_data",
        mode="before",
    )
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        return _as_text(value)

    @field_validator("authors", "key_findings", "limitations", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> List[str]:
        return _as_text_list(value)


class CondensedSummary(SummaryFields):
    """Output of the CONDENSE step: the model's restatement of section text.

    Source pages are deliberately absent. We already know which pages each
    evidence package came from, so they are never left to the model.
    """


class PaperSummary(SummaryFields):
    """Output of the REDUCE step: the final structured summary."""

    plain_english_summary: str = NOT_STATED
    confidence_notes: str = NOT_STATED
    source_pages: SourcePages = Field(default_factory=SourcePages)

    @field_validator("plain_english_summary", "confidence_notes", mode="before")
    @classmethod
    def _coerce_prose(cls, value: Any) -> str:
        return _as_text(value)

    @field_validator("source_pages", mode="before")
    @classmethod
    def _coerce_source_pages(cls, value: Any) -> Any:
        return value if isinstance(value, (dict, SourcePages)) else {}

    def missing_fields(self) -> List[str]:
        """Names of the fields that came back with no information."""
        missing = []
        for name in (
            "title",
            "research_question",
            "background",
            "methods",
            "participants_or_data",
            "plain_english_summary",
        ):
            if is_missing(getattr(self, name)):
                missing.append(name)
        for name in ("authors", "key_findings", "limitations"):
            if all(is_missing(item) for item in getattr(self, name)):
                missing.append(name)
        return missing

    def clamp_source_pages(self, max_page: int) -> None:
        """Drop hallucinated page numbers that fall outside the document."""
        for field in SOURCE_PAGE_FIELDS:
            pages = getattr(self.source_pages, field)
            setattr(
                self.source_pages,
                field,
                [page for page in pages if 1 <= page <= max_page],
            )


class SummaryResponse(BaseModel):
    filename: str
    model: str
    page_count: int
    chunk_count: int
    summary: PaperSummary


def validate_chunk(raw: Dict[str, Any], page_start: int, page_end: int) -> ChunkSummary:
    """Validate one MAP response, logging (never the paper text) if it degrades."""
    try:
        chunk = ChunkSummary.model_validate(raw)
    except Exception as exc:
        # Keep the page range and the field names, never the content.
        logger.warning(
            "MAP response for pages %d-%d failed validation (%s: %s); "
            "falling back to an empty chunk. Keys returned: %s",
            page_start,
            page_end,
            type(exc).__name__,
            str(exc).splitlines()[0][:200],
            sorted(raw.keys()) if isinstance(raw, dict) else type(raw).__name__,
        )
        return ChunkSummary(page_start=page_start, page_end=page_end)

    if not chunk.page_start:
        chunk.page_start = page_start
    if not chunk.page_end:
        chunk.page_end = page_end
    return chunk


# --------------------------------------------------------------------------- #
# Ask the Paper
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    """A question about one already-uploaded paper.

    Deliberately the same shape as SummaryRequest plus a question: the client
    holds the extracted pages returned by POST /upload and sends them back.
    No server-side paper storage, no session, no database.
    """

    question: str = ""
    filename: str = "paper.pdf"
    pages: List[PageInput] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerSource(BaseModel):
    """One cited passage. Page and chunk id come from the parser, never the model."""

    page: int = Field(ge=1)
    chunk_id: str
    evidence: str
    section: str = ""


class AskDiagnostics(BaseModel):
    """How the answer was assembled, for measuring citation precision.

    Retrieval scores are raw cosine statistics, not confidence percentages.
    """

    answerability_reason: str = ""
    generation_called: bool = False
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    selected_chunk_ids: List[str] = Field(default_factory=list)
    model_used_chunk_ids: List[str] = Field(default_factory=list)
    cited_pages: List[int] = Field(default_factory=list)
    retrieval_top_score: Optional[float] = None
    retrieval_score_gap: Optional[float] = None
    retrieval_mean_top_k: Optional[float] = None
    embedding_cache_hit: bool = False


class AskResponse(BaseModel):
    status: str
    answer: str
    sources: List[AnswerSource] = Field(default_factory=list)
    confidence_note: str = ""
    diagnostics: AskDiagnostics = Field(default_factory=AskDiagnostics)
