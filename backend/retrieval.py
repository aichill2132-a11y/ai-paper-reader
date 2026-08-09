"""Retrieval-ready document chunks.

This is the first layer of "Ask the Paper": it turns the page array returned by
POST /upload into overlapping, section-aware chunks that a retriever can later
embed and rank. Nothing here calls a model, and the summarisation pipeline is
untouched.

CHUNKING STRATEGY
    The document is first divided into *units*. A unit is one contiguous run of
    text belonging to a single page **and** a single section, so a chunk can
    never straddle a page break or mix two sections. Section boundaries come
    from the summariser's own parser (``sections.heading_on_line``), so the
    section a chunk is labelled with is the same one the summary pipeline would
    assign. A section that continues onto the next page keeps its name; only the
    page number changes.

    Each unit is then divided into as many evenly sized chunks as it needs:
    the chunk count is chosen from the unit's total length, and the per-chunk
    target is the unit length divided by that count, clamped to the 300-600
    band. Balancing this way avoids the common failure of filling every chunk
    to the ceiling and leaving a stub at the end of the page. Units shorter
    than the ceiling become a single chunk rather than being padded from a
    neighbouring page, and a single line longer than the ceiling becomes its
    own chunk, since lines are the smallest unit carrying an exact offset.

OVERLAP
    Each chunk after the first in a unit replays the trailing lines of its
    predecessor, up to OVERLAP_WORDS. Overlap is measured in whole lines so the
    character offsets stay contiguous and exact, and it never crosses a unit
    boundary, so no overlap leaks between pages or sections.

METADATA
    ``page_number`` is the page the chunk was taken from, and ``start_char`` /
    ``end_char`` are offsets into that page's original, unmodified text. Running
    heads, footers and standalone page numbers are dropped from ``text`` using
    the same rules as the summariser, so ``text`` is a whitespace-normalised
    rendering of the span rather than a byte-identical slice of it.

    ``chunk_id`` is a SHA-256 digest of the document name, page, section and
    span, so identical input always produces identical ids and two different
    papers cannot collide.
"""

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

from pydantic import BaseModel, Field

from metadata import repeated_lines
from schemas import PageInput
from sections import PAGE_NUMBER_LINE, heading_on_line, is_substantive_note

# Chunk sizing, in words. 500-600 word chunks proved too coarse for question
# answering: a single chunk covered several distinct claims, so the specific
# sentence that answered a question was diluted by everything around it.
TARGET_MIN_WORDS = 300
TARGET_MAX_WORDS = 400
TARGET_WORDS = (TARGET_MIN_WORDS + TARGET_MAX_WORDS) // 2
OVERLAP_WORDS = 60
# A trailing chunk smaller than this is folded back into its predecessor,
# provided the merge does not breach TARGET_MAX_WORDS.
MIN_TAIL_WORDS = 120

# Section name used for text that appears before any recognised heading.
UNLABELLED_SECTION = ""

# Author notes and endnotes. Evidence-bearing, unlike the bibliography they
# usually follow, so deliberately absent from NON_EVIDENCE_SECTIONS below.
NOTES_SECTION = "notes"

# Sections that are part of the document but are not evidence about the study.
# They share a paper's topical vocabulary, so an embedding model ranks them
# highly for almost any question about that paper: a reference list is largely
# made of the paper's own subject terms, and matches everything weakly.
NON_EVIDENCE_SECTIONS = frozenset({"references", "acknowledgements", "keywords"})

# Front matter (title, authors, affiliation, DOI) is the unlabelled text above
# the first heading on page 1. It names the topic without saying anything about
# it, which is exactly the profile that scores well and answers nothing.
FRONT_MATTER_PAGE = 1

# Below this a chunk is a heading remnant or a stray line, not evidence.
MIN_EVIDENCE_WORDS = 8


class RetrievalChunk(BaseModel):
    """One retrieval unit: a span of a single page within a single section."""

    chunk_id: str
    page_number: int = Field(ge=1)
    section: str = UNLABELLED_SECTION
    text: str
    # Offsets into the original text of `page_number`.
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class _Line(BaseModel):
    """A retained line of a page, with its span in the original page text."""

    text: str
    start: int
    end: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class _Unit(BaseModel):
    """Contiguous text from one page belonging to one section."""

    page_number: int
    section: str
    lines: List[_Line] = Field(default_factory=list)


def _line_spans(text: str) -> List[Tuple[str, int, int]]:
    """Every raw line of a page with its (start, end) character offsets."""
    spans = []
    cursor = 0
    for raw in text.split("\n"):
        spans.append((raw, cursor, cursor + len(raw)))
        cursor += len(raw) + 1  # the newline that split() consumed
    return spans


def _retained_line(
    raw: str, start: int, end: int, dropped: Sequence[str]
) -> Optional[_Line]:
    """Normalise a raw line, or return None if it is page furniture."""
    text = " ".join(raw.split())
    if not text:
        return None
    if text.lower() in dropped or PAGE_NUMBER_LINE.match(text):
        return None

    # Trim the offsets to the non-whitespace content of the line.
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    return _Line(text=text, start=start + leading, end=end - trailing)


def build_units(
    pages: Sequence[PageInput], drop_lines: Optional[Sequence[str]] = None
) -> List[_Unit]:
    """Split the document into (page, section) units, preserving offsets.

    Section detection reuses ``sections.heading_on_line``, so unit boundaries
    match the summariser's section boundaries by construction.
    """
    dropped = {" ".join(line.split()).lower() for line in (drop_lines or [])}

    units: List[_Unit] = []
    # A section that runs past a page break keeps its name on the next page.
    section = UNLABELLED_SECTION

    for page in pages:
        current = _Unit(page_number=page.page_number, section=section)

        for raw, start, end in _line_spans(page.text):
            line = _retained_line(raw, start, end, dropped)
            if line is None:
                continue

            # A numbered author note after the bibliography starts a new unit.
            # build_units walks lines itself (to keep character offsets), so it
            # applies the same rule sections.parse_sections does rather than
            # inheriting it.
            if section == "references" and is_substantive_note(line.text):
                if current.lines:
                    units.append(current)
                section = NOTES_SECTION
                current = _Unit(page_number=page.page_number, section=section)
                current.lines.append(line)
                continue

            found = heading_on_line(line.text)
            if found is None:
                current.lines.append(line)
                continue

            # A heading closes the current unit and opens the next one.
            name, remainder = found
            if current.lines:
                units.append(current)
            section = name
            current = _Unit(page_number=page.page_number, section=section)

            if remainder:
                # Run-in heading: the prose after it starts the new unit.
                offset = line.text.find(remainder)
                current.lines.append(
                    _Line(
                        text=remainder,
                        start=line.start + max(offset, 0),
                        end=line.end,
                    )
                )

        if current.lines:
            units.append(current)

    return units


def _chunk_size_for(total_words: int, count: int) -> int:
    """Size of each chunk when a unit of ``total_words`` is cut into ``count``.

    Overlap is replayed rather than consumed, so ``count`` chunks cover
    ``size + (count - 1) * (size - OVERLAP_WORDS)`` words. Solving for the size
    gives the expression below; ignoring the overlap term is what leaves a stub
    at the end of a long page.
    """
    return math.ceil((total_words + (count - 1) * OVERLAP_WORDS) / max(count, 1))


def _unit_target(total_words: int) -> int:
    """Per-chunk word target that divides a unit into even pieces.

    The chunk count is whichever value puts the resulting size closest to
    TARGET_WORDS, so a unit is never cut just because it edges over the ceiling:
    a 410-word section stays whole rather than becoming two 235-word fragments.
    Ties prefer fewer, larger chunks.
    """
    if total_words <= TARGET_MAX_WORDS:
        return total_words

    best_count = 1
    best_deviation = abs(total_words - TARGET_WORDS)
    count = 2
    while _chunk_size_for(total_words, count) >= OVERLAP_WORDS * 2:
        deviation = abs(_chunk_size_for(total_words, count) - TARGET_WORDS)
        if deviation < best_deviation:
            best_deviation = deviation
            best_count = count
        count += 1

    return _chunk_size_for(total_words, best_count)


def _chunk_ranges(lines: Sequence[_Line]) -> List[Tuple[int, int]]:
    """Half-open [start, end) line-index ranges for one unit's chunks."""
    if not lines:
        return []

    target = _unit_target(sum(line.word_count for line in lines))
    # When _unit_target decided this unit should stay whole, its target is the
    # ceiling. Using the global TARGET_MAX_WORDS here instead would split a
    # 410-word section into 390 + 86 despite the decision not to divide it.
    ceiling = max(target, TARGET_MAX_WORDS)

    ranges: List[Tuple[int, int]] = []
    start = 0

    while start < len(lines):
        end = start
        words = 0
        while end < len(lines):
            extended = words + lines[end].word_count
            # Stop before overflowing, but only once the chunk is usable.
            if words >= TARGET_MIN_WORDS and extended > ceiling:
                break
            words = extended
            end += 1
            if words >= target:
                break

        ranges.append((start, end))
        if end >= len(lines):
            break
        start = _overlap_start(lines, end)

    return _merge_small_tail(ranges, lines, ceiling)


def _overlap_start(lines: Sequence[_Line], end: int) -> int:
    """Index of the first line to replay in the next chunk."""
    words = 0
    start = end
    while start > 0 and words < OVERLAP_WORDS:
        start -= 1
        words += lines[start].word_count
    # Never replay the entire previous chunk.
    return min(start, end - 1) if end > 0 else 0


def _merge_small_tail(
    ranges: List[Tuple[int, int]],
    lines: Sequence[_Line],
    ceiling: int = TARGET_MAX_WORDS,
) -> List[Tuple[int, int]]:
    """Fold a very short trailing chunk back into its predecessor.

    Skipped when the merged chunk would breach the ceiling, so a small tail is
    preferred over an oversized chunk.
    """
    if len(ranges) < 2:
        return ranges

    start, end = ranges[-1]
    if sum(line.word_count for line in lines[start:end]) >= MIN_TAIL_WORDS:
        return ranges

    previous_start, _ = ranges[-2]
    merged = sum(line.word_count for line in lines[previous_start:end])
    if merged > ceiling:
        return ranges

    return ranges[:-2] + [(previous_start, end)]


def _make_chunk(
    unit: _Unit, lines: Sequence[_Line], document_id: str
) -> RetrievalChunk:
    text = " ".join(line.text for line in lines)
    start_char = lines[0].start
    end_char = lines[-1].end

    digest = hashlib.sha256(
        "|".join(
            [
                document_id,
                str(unit.page_number),
                unit.section,
                str(start_char),
                str(end_char),
                text,
            ]
        ).encode("utf-8")
    ).hexdigest()

    return RetrievalChunk(
        chunk_id="p{:04d}-{}".format(unit.page_number, digest[:12]),
        page_number=unit.page_number,
        section=unit.section,
        text=text,
        start_char=start_char,
        end_char=end_char,
    )


def build_retrieval_chunks(
    pages: Sequence[PageInput], document_id: str = ""
) -> List[RetrievalChunk]:
    """Convert uploaded page data into retrieval-ready chunks.

    ``pages`` is the array POST /upload returns. ``document_id`` (normally the
    filename) only salts the chunk ids so two papers cannot collide.

    The output is deterministic: the same input always produces the same chunks
    in the same order with the same ids.
    """
    ordered = sorted(pages, key=lambda page: page.page_number)
    running_heads = repeated_lines([(page.page_number, page.text) for page in ordered])

    chunks: List[RetrievalChunk] = []
    for unit in build_units(ordered, running_heads):
        for start, end in _chunk_ranges(unit.lines):
            window = unit.lines[start:end]
            if not window:
                continue
            chunk = _make_chunk(unit, window, document_id)
            if chunk.text.strip():
                chunks.append(chunk)

    return chunks


# --------------------------------------------------------------------------- #
# retrieval eligibility
# --------------------------------------------------------------------------- #

ChunkT = TypeVar("ChunkT")


def exclusion_reason(chunk: Any, document_has_sections: bool = True) -> Optional[str]:
    """Why this chunk is not evidence, or None if it is eligible.

    Deliberately returns the reason rather than a bool so the decision can be
    inspected, logged and tested one rule at a time. Accepts anything carrying
    ``section``, ``page_number`` and ``text``, so it works on both
    ``RetrievalChunk`` and ``embeddings.EmbeddedChunk``.

    ``document_has_sections`` guards the front-matter rule: in a paper with no
    recognised headings at all, every chunk is unlabelled and excluding page 1
    would throw away real content.
    """
    section = (getattr(chunk, "section", "") or "").strip().lower()

    if section in NON_EVIDENCE_SECTIONS:
        return section

    if len(getattr(chunk, "text", "").split()) < MIN_EVIDENCE_WORDS:
        return "too-short"

    if (
        document_has_sections
        and not section
        and getattr(chunk, "page_number", 0) == FRONT_MATTER_PAGE
    ):
        return "front-matter"

    return None


def is_evidence_chunk(chunk: Any, document_has_sections: bool = True) -> bool:
    """True when a chunk may be returned as evidence for a question."""
    return exclusion_reason(chunk, document_has_sections) is None


def has_labelled_sections(chunks: Sequence[Any]) -> bool:
    """True when the parser recognised at least one heading in the document."""
    return any((getattr(chunk, "section", "") or "").strip() for chunk in chunks)


def eligible_chunks(chunks: Sequence[ChunkT]) -> List[ChunkT]:
    """Filter a corpus down to the chunks that may be used as evidence.

    Nothing is deleted from the document model; this is a view over it, applied
    at retrieval time so the full corpus stays available for other uses.
    """
    document_has_sections = has_labelled_sections(chunks)
    return [
        chunk
        for chunk in chunks
        if exclusion_reason(chunk, document_has_sections) is None
    ]


def exclusion_summary(chunks: Sequence[Any]) -> Dict[str, int]:
    """How many chunks each rule removed. For diagnostics and tests."""
    document_has_sections = has_labelled_sections(chunks)
    counts: Dict[str, int] = {}
    for chunk in chunks:
        reason = exclusion_reason(chunk, document_has_sections)
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def chunks_by_page(chunks: Sequence[RetrievalChunk]) -> Dict[int, List[RetrievalChunk]]:
    """Group chunks by the page they came from, preserving order."""
    grouped: Dict[int, List[RetrievalChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.page_number, []).append(chunk)
    return grouped
