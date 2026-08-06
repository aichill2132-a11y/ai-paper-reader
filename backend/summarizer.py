"""Summarisation of an academic paper using a local Ollama model.

The pipeline is still map-reduce, but the model's job has been narrowed. Papers
are structured documents, so the structure is found deterministically first and
the model is asked to *condense supplied evidence* rather than to discover
fields inside unconstrained text.

    1. PARSE      sections.parse_sections finds headings and their page ranges.
    2. PACKAGE    sections.build_evidence_packages maps sections onto fields.
    3. CONDENSE   one call asking the model to restate the extracted evidence.
                  A field with non-empty evidence may not come back empty.
    4. MAP        chunk extraction, but only for fields no section covered.
    5. REDUCE     merge into the final structure and write the plain-English
                  summary.
    6. RECOVER    deterministic condensers fill anything still missing, and
                  package page numbers overwrite the model's guesses.

The full paper text is never logged. Debug output is limited to page ranges,
field names, timings, and short truncated values.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from condensers import LIST_CONDENSERS, TEXT_CONDENSERS, condense_findings
from config import (
    CHUNK_CHAR_LIMIT,
    CHUNK_PAGE_LIMIT,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT_MAP,
    OLLAMA_NUM_PREDICT_REDUCE,
)
from diagnostics import debug, short
from metadata import DocumentHints, extract_hints, render_hints
from ollama_client import OllamaError, generate_json
from schemas import (
    NOT_STATED,
    NOT_STATED_IN_CHUNK,
    SOURCE_PAGE_FIELDS,
    ChunkSummary,
    CondensedSummary,
    Evidence,
    PageInput,
    PaperSummary,
    is_missing,
    validate_chunk,
)
from sections import (
    EvidencePackage,
    Section,
    build_evidence_packages,
    filter_findings,
    parse_sections,
)

logger = logging.getLogger(__name__)

# Guard rail so a single very long page cannot blow past the context window.
MAX_PAGE_CHARS = 20000
# How much of one field's section text is shown to the model.
MAX_FIELD_EVIDENCE_CHARS = 4000
# Evidence excerpts quoted back in the reduce prompt.
MAX_EVIDENCE_CHARS = 300
# Context window for the two whole-document calls.
REDUCE_NUM_CTX = 16384

SCALAR_FIELDS = ("research_question", "background", "methods", "participants_or_data")
LIST_FIELDS = ("key_findings", "limitations")
CORE_FIELDS = SCALAR_FIELDS + LIST_FIELDS

CONDENSE_SYSTEM = (
    "You are a meticulous research assistant. You are given text copied "
    "verbatim from the labelled sections of one academic paper. Your only job "
    "is to condense that text faithfully. Every statement you write must be "
    "supported by the supplied text. You never add outside knowledge, and you "
    "never claim something is missing when text for it was supplied. You reply "
    "with JSON only."
)

MAP_SYSTEM = (
    "You are a meticulous research assistant extracting facts from an academic "
    "paper. You work in two steps: first you find the exact sentences in the "
    "text that answer each field, then you write a short faithful value based "
    "only on those sentences. You never invent facts, numbers, authors, or "
    "citations. You reply with JSON only."
)

REDUCE_SYSTEM = (
    "You are a meticulous research assistant assembling a final summary from "
    "verified extractions. Your job is to preserve information, not to discard "
    "it. Anything marked VERIFIED has already been confirmed against the paper "
    "and must be carried through. You never add facts of your own. You reply "
    "with JSON only."
)


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def chunk_pages(
    pages: List[PageInput],
    char_limit: int = CHUNK_CHAR_LIMIT,
    page_limit: int = CHUNK_PAGE_LIMIT,
) -> List[List[PageInput]]:
    """Group consecutive pages into chunks bounded by characters and page count."""
    chunks: List[List[PageInput]] = []
    current: List[PageInput] = []
    current_chars = 0

    for page in pages:
        if not page.text.strip():
            continue

        size = min(len(page.text), MAX_PAGE_CHARS)
        too_long = current and current_chars + size > char_limit
        too_many = current and len(current) >= max(1, page_limit)
        if too_long or too_many:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(page)
        current_chars += size

    if current:
        chunks.append(current)
    return chunks


def _page_range(chunk: List[PageInput]) -> Tuple[int, int]:
    numbers = [page.page_number for page in chunk]
    return min(numbers), max(numbers)


def _render_chunk(chunk: List[PageInput]) -> str:
    parts = []
    for page in chunk:
        parts.append(f"[page {page.page_number}]\n{page.text[:MAX_PAGE_CHARS].strip()}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# CONDENSE
# --------------------------------------------------------------------------- #

FIELD_INSTRUCTIONS = {
    "research_question": "State the question or aim the study set out to answer.",
    "background": "State the context and prior work that motivated the study.",
    "methods": "State how the data were collected and how they were analysed.",
    "participants_or_data": (
        "State who took part or what data were used: counts, characteristics, "
        "and how they were recruited or obtained."
    ),
    "key_findings": (
        "List the distinct empirical results, observed patterns, measured "
        "outcomes and reported themes. A conclusion is fine when it summarises "
        "what was observed. Do not include recommendations, implications, "
        "policy points or future work: anything phrased as what someone should, "
        "ought to or needs to do is not a finding."
    ),
    "limitations": (
        "List only the weaknesses the paper itself acknowledges. Do not turn a "
        "result or a recommendation for future work into a limitation."
    ),
}


def _condense_prompt(
    packages: Dict[str, EvidencePackage],
    filename: str,
    hints: DocumentHints,
) -> str:
    blocks = []
    for field in ("research_question", "background", "methods",
                  "participants_or_data", "key_findings", "limitations"):
        package = packages.get(field)
        if package is None or not package.has_text:
            continue
        pages = ", ".join(str(page) for page in package.pages) or "unknown"
        blocks.append(
            f"### {field}\n"
            f"Taken from: {'; '.join(package.headings) or 'unlabelled text'} "
            f"(pages {pages})\n"
            f"Instruction: {FIELD_INSTRUCTIONS[field]}\n"
            f"TEXT:\n{package.text[:MAX_FIELD_EVIDENCE_CHARS].strip()}"
        )

    supplied = sorted(packages.keys())
    return (
        f"File: {filename}\n\n"
        f"Front matter detected automatically:\n{render_hints(hints)}\n\n"
        "Below is text copied verbatim from this paper's own sections. "
        "Condense each block into the matching field.\n\n"
        "RULES\n"
        "- Use only the supplied text. Add nothing.\n"
        f"- Fields supplied here: {', '.join(supplied)}. You must return a real "
        f'value for every one of them. Returning "{NOT_STATED}" for a field '
        "whose text was supplied is wrong.\n"
        f'- Only use "{NOT_STATED}" for a field with no block below.\n'
        "- key_findings and limitations are lists of distinct points.\n"
        "- title and authors: choose from the candidates above, or repeat the "
        "title exactly as printed. Do not invent them.\n"
        "- Do not include page numbers in your values.\n\n"
        + "\n\n".join(blocks)
    )


def _apply_deterministic_condensers(
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
) -> List[str]:
    """A field with evidence may not come back empty. Fix it if it did."""
    repaired: List[str] = []

    for field, condenser in TEXT_CONDENSERS.items():
        package = packages.get(field)
        if package is None or not package.has_text:
            continue
        if not is_missing(getattr(condensed, field)):
            continue
        value = condenser(package.text)
        if value:
            setattr(condensed, field, value)
            repaired.append(field)

    for field, condenser in LIST_CONDENSERS.items():
        package = packages.get(field)
        if package is None or not package.has_text:
            continue
        current = getattr(condensed, field)
        if any(not is_missing(item) for item in current):
            continue
        values = condenser(package.text)
        if values:
            setattr(condensed, field, values)
            repaired.append(field)

    return repaired


async def condense_evidence(
    packages: Dict[str, EvidencePackage],
    filename: str,
    hints: DocumentHints,
) -> CondensedSummary:
    """Ask the model to restate the extracted sections, then check its work."""
    try:
        raw = await generate_json(
            _condense_prompt(packages, filename, hints),
            system=CONDENSE_SYSTEM,
            schema=CondensedSummary.model_json_schema(),
            num_ctx=REDUCE_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT_REDUCE,
            label="CONDENSE",
        )
        condensed = CondensedSummary.model_validate(raw)
    except OllamaError as error:
        # A failure here is recoverable: the evidence is already extracted.
        logger.warning(
            "CONDENSE failed (%s); falling back to deterministic condensers.",
            error.message,
        )
        condensed = CondensedSummary()
    except Exception as exc:
        logger.warning(
            "CONDENSE response failed validation (%s); falling back to "
            "deterministic condensers.",
            type(exc).__name__,
        )
        condensed = CondensedSummary()

    kept = filter_findings(
        [item for item in condensed.key_findings if not is_missing(item)]
    )
    dropped = len(
        [item for item in condensed.key_findings if not is_missing(item)]
    ) - len(kept)
    if dropped:
        debug(logger, "CONDENSE: dropped %d prescriptive key_finding(s)", dropped)
    # An empty list reads as "missing", so the deterministic condenser below
    # will refill it from the section text.
    condensed.key_findings = kept or [NOT_STATED]

    repaired = _apply_deterministic_condensers(condensed, packages)
    if repaired:
        logger.info(
            "CONDENSE returned nothing for fields that had evidence; "
            "used deterministic condensers for: %s",
            ", ".join(repaired),
        )
    debug(logger, 
        "CONDENSE values: %s",
        "; ".join(
            f"{field}={short(getattr(condensed, field))}"
            for field in SCALAR_FIELDS
        ),
    )
    return condensed


# --------------------------------------------------------------------------- #
# MAP (only for fields no section covered)
# --------------------------------------------------------------------------- #


def _map_prompt(
    chunk: List[PageInput],
    filename: str,
    index: int,
    total: int,
    hints: DocumentHints,
    wanted: List[str],
) -> str:
    first, last = _page_range(chunk)
    return (
        f"File: {filename}\n"
        f"This is excerpt {index} of {total}, covering pages {first} to {last} "
        "of ONE academic paper.\n\n"
        f"Structural hints found automatically:\n{render_hints(hints)}\n\n"
        f"The following fields were NOT found under any heading and must be "
        f"located here if the excerpt covers them: {', '.join(wanted)}.\n\n"
        "TASK\n"
        "Step 1 - Locate. Scan the excerpt for headings and their content, "
        "including: Title, Authors, Abstract, Introduction, Background, "
        "Research question, Aims, Objectives, Method, Methods, Study design, "
        "Participants, Sample, Data collection and analysis, Results, "
        "Findings, Discussion, Conclusions, Limitations. Facts also appear in "
        "the abstract even when no heading is present.\n"
        "Step 2 - Quote. For every field you can answer, copy the exact "
        "sentence from the excerpt that supports it into 'evidence'.\n"
        "Step 3 - Summarise. Only then write 'value' as a short faithful "
        "restatement, and record the page numbers in 'pages'.\n\n"
        "RULES\n"
        f'- If a field is absent from THIS excerpt, set its value to '
        f'"{NOT_STATED_IN_CHUNK}" and leave evidence empty.\n'
        "- Absence here does NOT mean the paper lacks it.\n"
        "- Never invent numbers, names, or citations.\n"
        f"- page_start is {first} and page_end is {last}.\n\n"
        "----- EXCERPT START -----\n"
        f"{_render_chunk(chunk)}\n"
        "----- EXCERPT END -----"
    )


# --------------------------------------------------------------------------- #
# REDUCE
# --------------------------------------------------------------------------- #


def _render_verified(
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
) -> str:
    lines = []
    for field in SCALAR_FIELDS:
        value = getattr(condensed, field)
        if is_missing(value):
            continue
        pages = packages[field].pages if field in packages else []
        suffix = f" [pages {', '.join(str(page) for page in pages)}]" if pages else ""
        lines.append(f"- {field}: {value}{suffix}")

    for field in LIST_FIELDS:
        values = [item for item in getattr(condensed, field) if not is_missing(item)]
        if not values:
            continue
        pages = packages[field].pages if field in packages else []
        suffix = f" [pages {', '.join(str(page) for page in pages)}]" if pages else ""
        lines.append(f"- {field}{suffix}:")
        lines.extend(f"    * {value}" for value in values)

    return "\n".join(lines) or "- (nothing was extracted from labelled sections)"


def _render_evidence(label: str, item: Evidence) -> str:
    if not item.has_value:
        return ""
    pages = ", ".join(str(page) for page in item.pages) or "unspecified"
    line = f"- {label}: {item.value} [pages {pages}]"
    if item.evidence:
        line += f'\n    quoted: "{item.evidence[:MAX_EVIDENCE_CHARS]}"'
    return line


def _render_notes(notes: List[ChunkSummary]) -> str:
    blocks = []
    for chunk in notes:
        lines = [f"### Notes from pages {chunk.page_start}-{chunk.page_end}"]
        if chunk.title_candidates:
            lines.append("- title candidates: " + " | ".join(chunk.title_candidates))
        if chunk.author_candidates:
            lines.append("- author candidates: " + " | ".join(chunk.author_candidates))
        for field in SCALAR_FIELDS:
            rendered = _render_evidence(field.replace("_", " "), getattr(chunk, field))
            if rendered:
                lines.append(rendered)
        for field in LIST_FIELDS:
            for item in getattr(chunk, field):
                rendered = _render_evidence(field.replace("_", " ")[:-1], item)
                if rendered:
                    lines.append(rendered)
        if len(lines) == 1:
            lines.append("- (this chunk yielded nothing)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _reduce_prompt(
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
    notes: List[ChunkSummary],
    filename: str,
    page_count: int,
    hints: DocumentHints,
) -> str:
    unlabelled = (
        "\n\nUNVERIFIED notes from parts of the paper with no headings:\n"
        + _render_notes(notes)
        if notes
        else ""
    )
    return (
        f"File: {filename} ({page_count} pages)\n\n"
        f"Front matter detected automatically:\n{render_hints(hints)}\n\n"
        "VERIFIED extractions, already confirmed against the paper's own "
        f"labelled sections:\n{_render_verified(condensed, packages)}"
        f"{unlabelled}\n\n"
        "ASSEMBLY RULES\n"
        "- Every VERIFIED value must appear in your output. Copy or lightly "
        "tighten it; never drop it.\n"
        f'- Never replace a VERIFIED value with "{NOT_STATED}".\n'
        f'- Only write "{NOT_STATED}" for a field that appears in neither the '
        "verified list nor the notes.\n"
        "- Where the notes add a distinct point not already covered, merge it "
        "in. Merge duplicates.\n"
        "- source_pages: reuse the page numbers shown in brackets above. Page "
        f"numbers must be between 1 and {page_count}.\n"
        "- title and authors: take them from the candidates above.\n"
        "- plain_english_summary: 3 to 5 sentences a non-specialist can "
        "follow, built only from the values above.\n"
        "- confidence_notes: say plainly which parts are thin or missing.\n"
    )


# --------------------------------------------------------------------------- #
# deterministic recovery
# --------------------------------------------------------------------------- #


def _evidence_pages(item: Evidence, chunk: ChunkSummary) -> List[int]:
    if item.pages:
        return item.pages
    if chunk.page_start and chunk.page_end:
        return sorted({chunk.page_start, chunk.page_end})
    return []


def _gather(notes: List[ChunkSummary], field: str) -> List[Tuple[Evidence, ChunkSummary]]:
    found: List[Tuple[Evidence, ChunkSummary]] = []
    for chunk in notes:
        value = getattr(chunk, field)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if item.has_value:
                found.append((item, chunk))
    return found


def _most_specific(
    items: List[Tuple[Evidence, ChunkSummary]],
) -> Optional[Tuple[Evidence, ChunkSummary]]:
    if not items:
        return None
    return max(items, key=lambda pair: (bool(pair[0].evidence), len(pair[0].value)))


def _deterministic_value(
    field: str,
    packages: Dict[str, EvidencePackage],
    condensers: Dict[str, Any],
) -> Any:
    """Run the deterministic condenser for a field, if it has section text."""
    package = packages.get(field)
    condenser = condensers.get(field)
    if package is None or not package.has_text or condenser is None:
        return None
    return condenser(package.text)


def _unique(values: List[str]) -> List[str]:
    seen: List[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if text and not is_missing(text) and text not in seen:
            seen.append(text)
    return seen


def tighten_key_findings(
    summary: PaperSummary,
    packages: Dict[str, EvidencePackage],
) -> int:
    """Remove recommendation-style entries from the final key_findings.

    If that empties the list, the deterministic condenser is re-run over the
    section text so a paper with real results never ends up with none.
    Returns how many entries were dropped.
    """
    original = [item for item in summary.key_findings if not is_missing(item)]
    kept = filter_findings(original)
    dropped = len(original) - len(kept)

    if not kept:
        package = packages.get("key_findings")
        if package is not None and package.has_text:
            kept = _unique(condense_findings(package.text))

    summary.key_findings = kept or [NOT_STATED]
    return dropped


def _recover_title_and_authors(
    summary: PaperSummary,
    condensed: CondensedSummary,
    notes: List[ChunkSummary],
    hints: DocumentHints,
) -> List[str]:
    """Fill in the title and authors from the next most trustworthy source."""
    recovered: List[str] = []
    confident = hints.title_candidates if hints.title_is_confident else []

    if is_missing(summary.title):
        candidates = _unique(
            ([condensed.title] if not is_missing(condensed.title) else [])
            + confident
            + [title for chunk in notes for title in chunk.title_candidates]
            + hints.title_candidates
        )
        if candidates:
            summary.title = candidates[0]
            recovered.append("title")

    if all(is_missing(author) for author in summary.authors):
        candidates = _unique(
            [item for item in condensed.authors if not is_missing(item)]
            + (hints.author_candidates if hints.title_is_confident else [])
            + [author for chunk in notes for author in chunk.author_candidates]
            + hints.author_candidates
        )
        if candidates:
            summary.authors = candidates[:3]
            recovered.append("authors")

    return recovered


def _recover_scalar_fields(
    summary: PaperSummary,
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
    notes: List[ChunkSummary],
) -> List[str]:
    """Condensed evidence, then chunk notes, then a deterministic condenser."""
    recovered: List[str] = []

    for field in SCALAR_FIELDS:
        if not is_missing(getattr(summary, field)):
            continue

        value = getattr(condensed, field)
        if not is_missing(value):
            setattr(summary, field, value)
            recovered.append(field)
            continue

        best = _most_specific(_gather(notes, field))
        if best:
            setattr(summary, field, best[0].value)
            recovered.append(field)
            continue

        derived = _deterministic_value(field, packages, TEXT_CONDENSERS)
        if derived:
            setattr(summary, field, derived)
            recovered.append(field + " (deterministic)")

    return recovered


def _recover_list_fields(
    summary: PaperSummary,
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
    notes: List[ChunkSummary],
) -> List[str]:
    recovered: List[str] = []

    for field in LIST_FIELDS:
        if any(not is_missing(item) for item in getattr(summary, field)):
            continue

        values = _unique(list(getattr(condensed, field)))
        suffix = ""
        if not values:
            values = _unique([item.value for item, _ in _gather(notes, field)])
        if not values:
            derived = _deterministic_value(field, packages, LIST_CONDENSERS)
            values = _unique(derived or [])
            suffix = " (deterministic)"

        if values:
            setattr(summary, field, values)
            recovered.append(field + suffix)

    return recovered


def _recover_source_pages(
    summary: PaperSummary,
    packages: Dict[str, EvidencePackage],
    notes: List[ChunkSummary],
) -> List[str]:
    """Package pages are ground truth; they come from the parser, not the model."""
    recovered: List[str] = []

    for field in SOURCE_PAGE_FIELDS:
        package = packages.get(field)
        if package is not None and package.pages:
            setattr(summary.source_pages, field, list(package.pages))
            continue

        if getattr(summary.source_pages, field):
            continue

        pages: List[int] = []
        for item, chunk in _gather(notes, field):
            for page in _evidence_pages(item, chunk):
                if page not in pages:
                    pages.append(page)
        if pages:
            setattr(summary.source_pages, field, sorted(pages))
            recovered.append("source_pages." + field)

    return recovered


def _recover_prose(summary: PaperSummary) -> List[str]:
    """Compose the two narrative fields from values that are already supported."""
    recovered: List[str] = []

    if is_missing(summary.plain_english_summary):
        sentences = [
            text
            for text in (
                summary.research_question,
                summary.methods,
                summary.participants_or_data,
            )
            if not is_missing(text)
        ]
        sentences.extend(
            [item for item in summary.key_findings if not is_missing(item)][:2]
        )
        if sentences:
            summary.plain_english_summary = " ".join(
                text if text.endswith(".") else text + "." for text in sentences
            )
            recovered.append("plain_english_summary")

    if is_missing(summary.confidence_notes):
        missing = summary.missing_fields()
        if missing:
            summary.confidence_notes = (
                "The paper text did not yield: " + ", ".join(missing) + "."
            )
        else:
            summary.confidence_notes = (
                "All fields were supported by extracted section text, but the "
                "wording was produced by a local model and should be checked "
                "against the paper."
            )
        recovered.append("confidence_notes")

    return recovered


def apply_recovery(
    summary: PaperSummary,
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
    notes: List[ChunkSummary],
    hints: DocumentHints,
) -> List[str]:
    """Restore anything REDUCE dropped, in order of how trustworthy it is.

    Precedence per field: REDUCE output, then the condensed section evidence,
    then chunk notes, then a deterministic condenser over the raw section text.
    Returns the names of the fields that had to be recovered.
    """
    return (
        _recover_title_and_authors(summary, condensed, notes, hints)
        + _recover_scalar_fields(summary, condensed, packages, notes)
        + _recover_list_fields(summary, condensed, packages, notes)
        + _recover_source_pages(summary, packages, notes)
        + _recover_prose(summary)
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


async def _run_map_stage(
    chunks: List[List[PageInput]],
    filename: str,
    hints: DocumentHints,
    wanted: List[str],
) -> List[ChunkSummary]:
    """Extract the fields no labelled section covered, chunk by chunk."""
    notes: List[ChunkSummary] = []
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        first, last = _page_range(chunk)
        raw = await generate_json(
            _map_prompt(chunk, filename, index, total, hints, wanted),
            system=MAP_SYSTEM,
            schema=ChunkSummary.model_json_schema(),
            num_predict=OLLAMA_NUM_PREDICT_MAP,
            label="MAP {}/{} (pages {}-{})".format(index, total, first, last),
        )
        note = validate_chunk(raw, first, last)
        notes.append(note)
        debug(
            logger,
            "MAP %d/%d pages %d-%d populated: %s",
            index,
            total,
            first,
            last,
            ", ".join(note.populated_fields()) or "nothing",
        )

    return notes


async def _run_reduce_stage(
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
    notes: List[ChunkSummary],
    filename: str,
    page_count: int,
    hints: DocumentHints,
) -> PaperSummary:
    """Assemble the final structure. Failure here is survivable when the
    deterministic extraction already produced evidence packages."""
    try:
        raw = await generate_json(
            _reduce_prompt(condensed, packages, notes, filename, page_count, hints),
            system=REDUCE_SYSTEM,
            schema=PaperSummary.model_json_schema(),
            num_ctx=REDUCE_NUM_CTX,
            num_predict=OLLAMA_NUM_PREDICT_REDUCE,
            label="REDUCE",
        )
        return PaperSummary.model_validate(raw)
    except OllamaError as error:
        if not packages:
            # Nothing was extracted deterministically, so there is nothing to
            # fall back on: this really is a failure.
            raise
        logger.warning(
            "REDUCE failed (%s); assembling the summary from verified "
            "extractions instead.",
            error.message,
        )
    except Exception as exc:
        logger.warning(
            "REDUCE response failed validation (%s); assembling the summary "
            "from verified extractions instead.",
            type(exc).__name__,
        )
    return PaperSummary()


def _log_document_shape(
    filename: str,
    pages: List[PageInput],
    sections: List[Section],
    packages: Dict[str, EvidencePackage],
    hints: DocumentHints,
) -> None:
    logger.info(
        "Summarising %s: %d pages, %d sections, packages=%s, model=%s",
        filename,
        len(pages),
        len(sections),
        sorted(packages.keys()),
        OLLAMA_MODEL,
    )
    debug(
        logger,
        "sections: %s",
        "; ".join(
            "{} p{}({} chars)".format(
                section.name,
                section.pages[0] if section.pages else "?",
                len(section.text),
            )
            for section in sections
        )
        or "none",
    )
    debug(
        logger,
        "title candidates=%d authors=%d confidence=%s repeated_lines=%d",
        len(hints.title_candidates),
        len(hints.author_candidates),
        hints.title_confidence,
        len(hints.repeated_lines),
    )


async def summarize_paper(
    filename: str, pages: List[PageInput]
) -> Tuple[PaperSummary, int]:
    """Run the pipeline and return the final summary and the chunk count."""
    if not any(page.text.strip() for page in pages):
        raise OllamaError("The paper contains no extractable text to summarise.", 400)

    page_count = max(page.page_number for page in pages)
    page_tuples = [(page.page_number, page.text) for page in pages]

    hints = extract_hints(page_tuples)
    sections = parse_sections(page_tuples, drop_lines=hints.repeated_lines)
    packages = build_evidence_packages(sections)
    _log_document_shape(filename, pages, sections, packages, hints)

    condensed = CondensedSummary()
    if packages:
        condensed = await condense_evidence(packages, filename, hints)

    chunks = chunk_pages(pages)
    uncovered = [field for field in CORE_FIELDS if field not in packages]
    notes: List[ChunkSummary] = []
    if uncovered and chunks:
        debug(logger, "running MAP for uncovered fields: %s", ", ".join(uncovered))
        notes = await _run_map_stage(chunks, filename, hints, uncovered)
    else:
        debug(logger, "every field was covered by a labelled section; skipping MAP")

    summary = await _run_reduce_stage(
        condensed, packages, notes, filename, page_count, hints
    )
    debug(
        logger,
        "REDUCE selected: %s",
        "; ".join(
            "{}={}".format(field, short(getattr(summary, field)))
            for field in ("title",) + SCALAR_FIELDS
        ),
    )
    debug(
        logger,
        "REDUCE missing before recovery: %s",
        ", ".join(summary.missing_fields()) or "nothing",
    )

    recovered = apply_recovery(summary, condensed, packages, notes, hints)
    if recovered:
        logger.info(
            "Recovered fields that REDUCE dropped or left empty: %s",
            ", ".join(recovered),
        )

    dropped = tighten_key_findings(summary, packages)
    if dropped:
        logger.info(
            "Removed %d recommendation-style entr%s from key_findings.",
            dropped,
            "y" if dropped == 1 else "ies",
        )

    summary.clamp_source_pages(page_count)

    still_missing = summary.missing_fields()
    if still_missing:
        debug(logger, "still missing after recovery: %s", ", ".join(still_missing))

    return summary, max(1, len(chunks))
