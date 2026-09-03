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
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from condensers import (
    LIST_CONDENSERS,
    NO_RESEARCH_QUESTION,
    TEXT_CONDENSERS,
    condense_findings,
)
from config import (
    CHUNK_CHAR_LIMIT,
    CHUNK_PAGE_LIMIT,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT_MAP,
    OLLAMA_NUM_PREDICT_REDUCE,
)
from diagnostics import debug, short
from metadata import (
    DocumentHints,
    extract_hints,
    render_hints,
    split_author_names,
)
from ollama_client import MalformedJSONError, OllamaError, generate_json
from schemas import (
    reduce_response_schema,
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
    split_sentences,
)
from sections import LIMITATION_CUES  # read-only: reused as a critique detector

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
LIST_FIELDS = ("key_findings", "author_stated_limitations")
# An empty author_stated_limitations list is a valid answer, so the field is
# excluded from CORE_FIELDS (it must not make a paper look "uncovered" and
# trigger MAP) and from deterministic recovery (nothing may be forced in).
OPTIONAL_LIST_FIELDS = ("author_stated_limitations",)
# Optional, model-inferred, and deliberately outside LIST_FIELDS so the
# recovery pass never invents one.
INFERRED_FIELDS = ("model_identified_considerations",)

# sections.build_evidence_packages is frozen and keys the limitation package
# "limitations". The summary now separates what the authors stated from what
# the model infers, so the new field name is mapped back to the package name
# here rather than by changing the parser.
PACKAGE_ALIASES = {"author_stated_limitations": "limitations"}


def package_for(packages, field):
    """The evidence package backing a summary field, if the parser found one."""
    return packages.get(PACKAGE_ALIASES.get(field, field))


def chunk_field(field: str) -> str:
    """The ChunkSummary attribute holding a summary field's raw extraction.

    The MAP stage extracts limitation evidence without judging whether the
    authors stated it; that split happens when the summary is assembled, so
    ChunkSummary keeps the original field name.
    """
    return PACKAGE_ALIASES.get(field, field)
CORE_FIELDS = SCALAR_FIELDS + tuple(
    field for field in LIST_FIELDS if field not in OPTIONAL_LIST_FIELDS
)

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
    "research_question": (
        "Work in two steps. First find the study's stated aim, objective, "
        "purpose or hypothesis. Then rewrite that aim as ONE concise research "
        "question, preserving its scientific meaning. Do not copy a sentence "
        "from the results or discussion, and do not quote the aim verbatim if "
        "it is not already a question. If no aim or objective is stated, "
        'answer exactly: "The research question was not explicitly '
        'identifiable in the paper."'
    ),
    "background": (
        "Synthesise, do not copy. In one or two short paragraphs explain what "
        "was already known, what gap in that knowledge motivated this study, "
        "and why the study was needed. Do not reproduce the literature review "
        "sentence by sentence."
    ),
    "methods": (
        "Summarise the STUDY DESIGN in about three to six sentences: the type "
        "of study, the sample or participants, any intervention or exposure, "
        "the main procedure, what was measured, and how it was analysed. "
        "Exclude equipment models, reagent concentrations, column dimensions, "
        "flow rates, temperatures, software versions and other protocol "
        "detail unless a reader could not understand the study without it. "
        "Tell a researcher what was done; do not reproduce the methods section."
    ),
    "participants_or_data": (
        "State who took part or what data were used: counts, characteristics, "
        "and how they were recruited or obtained."
    ),
    "key_findings": (
        "List the three to six most important empirical results, observed "
        "patterns, measured outcomes or reported themes - not every result. "
        "Keep quantitative values where they matter scientifically. A "
        "conclusion is fine when it summarises what was observed. Do not mix "
        "in discussion or interpretation, and do not include recommendations, "
        "implications, policy points or future work: anything phrased as what "
        "someone should, ought to or needs to do is not a finding."
    ),
    "author_stated_limitations": (
        "OPTIONAL - an empty list is a correct answer and is expected for many "
        "papers. Include an entry ONLY where the paper itself clearly states "
        "one of the following about its own work:\n"
        "  - a methodological limitation or acknowledged weakness;\n"
        "  - a measurement that was missing, unavailable or not taken;\n"
        "  - an unresolved methodological uncertainty;\n"
        "  - an explicit constraint on how the results may be interpreted;\n"
        "  - a need for further research that follows from such a limitation "
        "or uncertainty.\n"
        "Do NOT list any of the following:\n"
        "  - an ordinary finding, result or observed value;\n"
        "  - a clinical, practical or policy implication;\n"
        "  - a gap in prior literature that merely motivated this study;\n"
        "  - an explanation or interpretation of a result;\n"
        "  - a difference from previous studies;\n"
        "  - a speculative mechanism;\n"
        "  - a successful method-validation statement, including phrasing such "
        'as "no interference was observed";\n'
        "  - ordinary descriptive variation in the results.\n"
        "The authors must acknowledge it themselves, in their own words. A "
        "methodological choice is not a limitation merely because it can be "
        "criticised: excluding animals or cases under a stated protocol, a "
        "small sample, and an absence of clinical data are limitations ONLY "
        "where the authors themselves frame them that way. Never write an "
        "entry of the form \"the study did not ...\" or \"which may "
        "introduce bias\" unless the paper says so. Every entry must restate "
        "a sentence that is present in the supplied text.\n"
        "If the supplied text contains none of the qualifying kinds, return an "
        "empty list. An empty list is preferred over an inferred limitation. "
        f'Never write "{NOT_STATED}" here. Put your own methodological '
        "concerns in model_identified_considerations instead."
    ),
    "model_identified_considerations": (
        "OPTIONAL. Reasonable study-design concerns that the authors did NOT "
        "state, such as a small sample, a single site, or limited "
        "generalisability. These are your own inference, not claims by the "
        "authors. Return an empty list if the supplied text does not support "
        "a useful consideration. Never move an author-stated limitation here, "
        "and never present a consideration as something the paper said. This "
        "is the only field for your own methodological concerns; they must "
        "not appear in confidence_notes."
    ),
}


def _field_rules(fields: Sequence[str]) -> str:
    """Render FIELD_INSTRUCTIONS for the given fields.

    CONDENSE places each instruction next to its own block of section text.
    MAP and REDUCE have no such blocks, so they get the same wording as one
    list. FIELD_INSTRUCTIONS stays the single source of truth for all three,
    which is what stops the stages disagreeing about a field.
    """
    return "\n".join(
        "- {}: {}".format(field, FIELD_INSTRUCTIONS[field])
        for field in fields
        if field in FIELD_INSTRUCTIONS
    )


def _condense_prompt(
    packages: Dict[str, EvidencePackage],
    filename: str,
    hints: DocumentHints,
) -> str:
    blocks = []
    supplied: List[str] = []
    for field in ("research_question", "background", "methods",
                  "participants_or_data", "key_findings",
                  "author_stated_limitations"):
        package = package_for(packages, field)
        if package is None or not package.has_text:
            continue
        supplied.append(field)
        pages = ", ".join(str(page) for page in package.pages) or "unknown"
        blocks.append(
            f"### {field}\n"
            f"Taken from: {'; '.join(package.headings) or 'unlabelled text'} "
            f"(pages {pages})\n"
            f"Instruction: {FIELD_INSTRUCTIONS[field]}\n"
            f"TEXT:\n{package.text[:MAX_FIELD_EVIDENCE_CHARS].strip()}"
        )

    # Field names, not package names: the parser keys the limitation package
    # "limitations", which is not a field the model can return.
    required = [field for field in supplied if field not in OPTIONAL_LIST_FIELDS]
    return (
        f"File: {filename}\n\n"
        f"Front matter detected automatically:\n{render_hints(hints)}\n\n"
        "Below is text copied verbatim from this paper's own sections. "
        "Condense each block into the matching field.\n\n"
        "RULES\n"
        "- Use only the supplied text. Add nothing from general knowledge.\n"
        "- Never silently fill a gap. If the text does not say, write exactly "
        '"Not clearly stated in the paper".\n'
        "- Keep author-stated facts separate from your own inference. Only "
        "model_identified_considerations may contain inference.\n"
        f"- You must return a real value for: {', '.join(required) or 'none'}. "
        f'Returning "{NOT_STATED}" for one of those is wrong.\n'
        "- author_stated_limitations is OPTIONAL even though text is supplied "
        "for it. If none of that text meets its rule, return an empty list. "
        "Supplying evidence is not a reason to fill the field.\n"
        f'- Only use "{NOT_STATED}" for a field with no block below.\n'
        "- key_findings and the two limitation fields are lists of distinct "
        "points.\n"
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
        package = package_for(packages, field)
        if package is None or not package.has_text:
            continue
        if not is_missing(getattr(condensed, field)):
            continue
        value = condenser(package.text)
        if value:
            setattr(condensed, field, value)
            repaired.append(field)

    for field, condenser in LIST_CONDENSERS.items():
        package = package_for(packages, field)
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
    rules = _field_rules(wanted)
    # The absence rule above is deliberately restated: a field instruction may
    # name its own "not found" wording, which is correct for the whole paper
    # but wrong for a single excerpt.
    rules_block = (
        "FIELD RULES - follow these when you write a value. They do not "
        "override the absence rule: a field missing from THIS excerpt is "
        f'still "{NOT_STATED_IN_CHUNK}", whatever its rule says.\n'
        f"{rules}\n\n"
        if rules
        else ""
    )
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
        f"{rules_block}"
        "----- EXCERPT START -----\n"
        f"{_render_chunk(chunk)}\n"
        "----- EXCERPT END -----"
    )


# --------------------------------------------------------------------------- #
# REDUCE
# --------------------------------------------------------------------------- #


def _has_evidence(packages: Dict[str, EvidencePackage], field: str) -> bool:
    """True when the parser supplied section text for this field.

    The dividing line between a parser-backed extraction and a model-authored
    guess. A CONDENSE value for a field with no package is the latter: it must
    not be labelled VERIFIED to REDUCE, and it must not outrank MAP evidence
    during recovery.
    """
    package = package_for(packages, field)
    return package is not None and package.has_text


def _render_verified(
    condensed: CondensedSummary,
    packages: Dict[str, EvidencePackage],
) -> str:
    lines = []
    for field in SCALAR_FIELDS:
        value = getattr(condensed, field)
        if is_missing(value) or not _has_evidence(packages, field):
            continue
        pages = package_for(packages, field).pages
        suffix = f" [pages {', '.join(str(page) for page in pages)}]" if pages else ""
        lines.append(f"- {field}: {value}{suffix}")

    for field in LIST_FIELDS:
        values = [item for item in getattr(condensed, field) if not is_missing(item)]
        if not values or not _has_evidence(packages, field):
            continue
        pages = package_for(packages, field).pages
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
            for item in getattr(chunk, chunk_field(field)):
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
    rules = _field_rules(SCALAR_FIELDS + LIST_FIELDS + INFERRED_FIELDS)
    return (
        f"File: {filename} ({page_count} pages)\n\n"
        f"Front matter detected automatically:\n{render_hints(hints)}\n\n"
        "VERIFIED extractions, already confirmed against the paper's own "
        f"labelled sections:\n{_render_verified(condensed, packages)}"
        f"{unlabelled}\n\n"
        "ASSEMBLY RULES\n"
        "- Every VERIFIED value must appear in your output. Copy or lightly "
        "tighten it; never drop it. Rewriting a value so it obeys its field "
        "rule below is tightening, not dropping - research_question in "
        "particular must end up phrased as a question.\n"
        f'- Never replace a VERIFIED value with "{NOT_STATED}".\n'
        f'- Only write "{NOT_STATED}" for a field that appears in neither the '
        "verified list nor the notes.\n"
        "- Where the notes add a distinct point not already covered, merge it "
        "in. Merge duplicates.\n"
        "- key_findings is the exception to that merge rule: SELECT, do not "
        "collect. Deduplicate the candidates, then keep only the most "
        "important empirical results, at most six. Prefer findings that "
        "answer the research question and carry the paper's central "
        "conclusions. Do not list every distinct supporting, control or "
        "procedural result.\n"
        "- source_pages: reuse the page numbers shown in brackets above. Page "
        f"numbers must be between 1 and {page_count}.\n"
        "- title and authors: take them from the candidates above.\n"
        "- plain_english_summary: 3 to 5 sentences a non-specialist can "
        "follow, built only from the values above.\n"
        "- confidence_notes: describe the EVIDENCE, not the study. Say which "
        "fields came from labelled sections, which rest on thin or partial "
        "text, which could not be identified at all, and that the wording is "
        "model-generated. Never put a methodological critique here: sample "
        "size, generalisability, design weaknesses, inferred limitations and "
        "suggestions for improving the study all belong in "
        "model_identified_considerations.\n"
        "- author_stated_limitations is optional. If nothing above is clearly "
        "a limitation the authors stated about their own work, return an "
        "empty list rather than promoting a finding or an implication.\n\n"
        f"FIELD RULES - the wording each field must follow:\n{rules}\n"
    )


# --------------------------------------------------------------------------- #
# limitation attribution
# --------------------------------------------------------------------------- #

# Share of an entry's content words that must appear in one collected sentence
# before the entry counts as a restatement of that sentence rather than a new
# claim about it. Model critique ("did not account for", "may introduce bias")
# adds words the source sentence does not contain, which is what fails here.
ATTRIBUTION_OVERLAP = 0.6

ATTRIBUTION_NOTE = (
    "Author-stated limitations are based only on explicit paper text. Any "
    "additional study-design considerations are model-generated and labelled "
    "separately."
)

# Wording that claims a limitation was worked out rather than read off the page.
_INFERRED_CLAIM = re.compile(
    r"\b(inferred|inference|implicit(ly)?|not explicitly stated|"
    r"derived from|assumed|surmised|model[- ]generated|our own reading)\b",
    re.IGNORECASE,
)
_MENTIONS_LIMITATION = re.compile(r"\blimitation", re.IGNORECASE)


def _is_attributable(entry: str, sources: Sequence[str]) -> bool:
    """True when the entry restates one sentence the parser collected."""
    words = _content_words(entry)
    if not words:
        return False
    needed = ATTRIBUTION_OVERLAP * len(words)
    return any(len(words & _content_words(source)) >= needed for source in sources)


def enforce_limitation_attribution(
    summary: PaperSummary,
    packages: Dict[str, EvidencePackage],
) -> int:
    """Keep only limitations the paper itself states; relabel the rest.

    The model reliably produces plausible methodological critique - excluded
    animals, absent clinical data, unmeasured metabolites - and files it under
    author_stated_limitations, which is the one thing that field must never
    hold. Prompting alone did not stop it, so attribution is checked here
    against the sentences the parser actually collected from the paper. An
    entry that is not a restatement of one of them is the model's own
    inference: it is moved, not deleted, to model_identified_considerations,
    where it is labelled as such. Returns how many entries were moved.
    """
    entries = [item for item in summary.author_stated_limitations
               if not is_missing(item)]
    if not entries:
        summary.author_stated_limitations = []
        return 0

    package = package_for(packages, "author_stated_limitations")
    # collect_limitations joins one collected sentence per line, so the source
    # sentences are recovered exactly rather than re-split.
    sources = [line.strip() for line in package.text.splitlines() if line.strip()] \
        if package is not None and package.has_text else []

    kept = [item for item in entries if _is_attributable(item, sources)]
    moved = [item for item in entries if item not in kept]

    summary.author_stated_limitations = kept
    if moved:
        existing = list(summary.model_identified_considerations)
        summary.model_identified_considerations = _unique(existing + moved)
    return len(moved)


# Deliberately free of implementation vocabulary: a reader should learn that
# some of the paper was left out, not how the pipeline is built.
PARTIAL_EVIDENCE_NOTE = (
    "Some parts of the paper could not be processed, so this summary was "
    "generated from the remaining available evidence."
)


def disclose_partial_evidence(summary: PaperSummary, skipped: List[str]) -> bool:
    """Say plainly that the summary is missing part of the paper.

    A skipped chunk must never be invisible: the fields still look complete,
    and only this note distinguishes "the paper did not say" from "we could not
    read that part". Nothing is fabricated for the skipped pages.
    """
    if not skipped:
        return False

    notes = (summary.confidence_notes or "").strip()
    if PARTIAL_EVIDENCE_NOTE in notes:
        return False
    if is_missing(notes):
        summary.confidence_notes = PARTIAL_EVIDENCE_NOTE
    else:
        summary.confidence_notes = (notes + " " + PARTIAL_EVIDENCE_NOTE).strip()
    return True


# Share of a consideration's content words that must already appear in the
# authors' own limitation text before it counts as their point rather than the
# model's. Measured against a discrimination set: paraphrases of author
# limitations score 0.36-0.70, genuine model-only concerns score 0.00-0.20.
CONSIDERATION_COVERAGE = 0.3

# A consideration describes the STUDY. These two together describe the summary
# itself - "the author-stated limitations were directly quoted from the
# limitations section" is provenance commentary, not a methodological concern.
# Both halves are required so that an ordinary concern which merely contains a
# generic word ("no power analysis was reported") is not discarded.
_META_SUBJECT = re.compile(
    r"\b(summar(y|ies)|limitations section|evidence|quotations?|quotes?|"
    r"citations?|extraction|provenance|confidence notes?|page numbers?|"
    r"source text|the model|this (?:field|section|list))\b",
    re.IGNORECASE,
)
_META_PROCESS = re.compile(
    r"\b(quoted|extracted|derived|copied|listed|taken|generated|paraphrased|"
    r"verbatim|reproduced)\b",
    re.IGNORECASE,
)


def _is_meta_commentary(text: str) -> bool:
    """True for a statement about the summary rather than about the study."""
    return bool(_META_SUBJECT.search(text) and _META_PROCESS.search(text))


def _covered_by(text: str, source: str) -> float:
    """Share of the text's content words that already appear in the source."""
    words = _finding_words(text)
    if not words:
        return 0.0
    return len(words & _finding_words(source)) / len(words)


def prune_model_considerations(
    summary: PaperSummary,
    packages: Dict[str, EvidencePackage],
) -> int:
    """Drop anything in model_identified_considerations that is not one.

    Two things end up here that should not. Provenance commentary, because
    confidence_notes and the attribution split both feed this list. And the
    authors' own points, because the attribution gate compares a model
    paraphrase against single collected sentences: a loose restatement of a
    real limitation fails that check and gets relabelled as inference, which
    misattributes the authors' work to the model. Comparing against the whole
    limitation package catches the paraphrase without lowering the bar for
    entering author_stated_limitations. Returns how many entries were dropped.
    """
    entries = [item for item in summary.model_identified_considerations
               if not is_missing(item)]
    if not entries:
        summary.model_identified_considerations = []
        return 0

    package = package_for(packages, "author_stated_limitations")
    authored = package.text if package is not None and package.has_text else ""
    stated = " ".join(summary.author_stated_limitations)

    kept = []
    for item in entries:
        if _is_meta_commentary(item):
            continue
        if authored and _covered_by(item, authored) >= CONSIDERATION_COVERAGE:
            continue
        if stated and _covered_by(item, stated) >= CONSIDERATION_COVERAGE:
            continue
        kept.append(item)

    summary.model_identified_considerations = _unique(kept)
    return len(entries) - len(summary.model_identified_considerations)


def align_confidence_notes(summary: PaperSummary) -> bool:
    """Keep the notes about evidence quality, and rehome anything else.

    confidence_notes is free text, and the model uses it as an overflow bin
    for methodological criticism - the very content the summary works to keep
    separate from what the authors said. Rather than add a critique classifier,
    this reuses the paper's own limitation vocabulary: a note that trips
    LIMITATION_CUES is talking about the study, not about the evidence, so it
    is moved into model_identified_considerations where it is labelled as the
    model's inference. Notes that merely claim limitations were inferred are
    dropped outright - they are commentary on attribution, not a consideration.
    """
    notes = (summary.confidence_notes or "").strip()
    if is_missing(notes):
        return False

    kept: List[str] = []
    moved: List[str] = []
    for sentence in split_sentences(notes):
        # Checked first: this wording trips LIMITATION_CUES too, but it is
        # meta-commentary and must not resurface as a consideration.
        if _MENTIONS_LIMITATION.search(sentence) and _INFERRED_CLAIM.search(sentence):
            continue
        if LIMITATION_CUES.search(sentence):
            moved.append(sentence.strip())
            continue
        kept.append(sentence)

    if moved:
        # Never say the same thing twice: an entry the authors are already
        # credited with is dropped rather than repeated as model inference.
        stated = {" ".join(item.split()) for item in summary.author_stated_limitations}
        summary.model_identified_considerations = _unique(
            list(summary.model_identified_considerations)
            + [item for item in moved if " ".join(item.split()) not in stated]
        )

    rewritten = " ".join(kept).strip()
    if summary.model_identified_considerations:
        rewritten = (rewritten + " " + ATTRIBUTION_NOTE).strip()

    rewritten = rewritten or ATTRIBUTION_NOTE
    changed = rewritten != notes
    summary.confidence_notes = rewritten
    return changed


# --------------------------------------------------------------------------- #
# research question repair
# --------------------------------------------------------------------------- #

REWRITE_SYSTEM = (
    "You rephrase one sentence as one research question. You never add "
    "information that is not already present in the sentence you are given, "
    "and you never answer the question."
)

_REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"research_question": {"type": "string"}},
    "required": ["research_question"],
}

# Words a question may legitimately introduce that the declarative aim did not
# contain. Kept deliberately tiny; this is a guard, not an ontology.
_INTERROGATIVE_WORDS = frozenset(
    {"what", "when", "where", "which", "does", "did", "were", "have",
     "extent", "ways", "there"}
)


def _content_words(text: str) -> set:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3}


def _preserves_meaning(original: str, rewrite: str) -> bool:
    """True when the rewrite says the same thing in question form.

    Reordering and dropping words is expected. Introducing them is not: a
    rewrite that names a population, variable or claim absent from the aim has
    invented a research question rather than rephrased one.
    """
    words = _content_words(rewrite)
    if not words:
        return False
    added = words - _content_words(original) - _INTERROGATIVE_WORDS
    return len(added) <= max(2, len(words) // 4)


def is_interrogative(value: str) -> bool:
    """A research question has to actually be a question."""
    return value.strip().endswith("?")


async def ensure_research_question(value: str) -> str:
    """Make research_question interrogative, with exactly one corrective call.

    The model still sometimes returns the declarative aim sentence it found,
    which is the extractive behaviour the field instruction exists to prevent.
    One rewrite is allowed and there is no loop: a failed call, a still
    declarative answer, or a rewrite that introduced facts all fall back to
    NO_RESEARCH_QUESTION, because admitting the question was not identified is
    better than presenting a copied sentence as one.
    """
    text = " ".join((value or "").split())
    if not text or is_missing(text) or text == NO_RESEARCH_QUESTION:
        return value
    if is_interrogative(text):
        return value

    debug(logger, "research_question is not a question; attempting one rewrite")
    try:
        raw = await generate_json(
            "Rewrite the statement below as ONE concise research question.\n"
            "- Preserve its scientific meaning exactly.\n"
            "- Use only information already in the statement. Add no new "
            "variables, populations, numbers, outcomes or claims.\n"
            "- Return a single question ending in a question mark.\n\n"
            f"STATEMENT:\n{text}",
            system=REWRITE_SYSTEM,
            schema=_REWRITE_SCHEMA,
            label="REWRITE_QUESTION",
        )
    except OllamaError as error:
        logger.warning("research_question rewrite failed (%s).", error)
        return NO_RESEARCH_QUESTION

    rewritten = " ".join(str(raw.get("research_question", "") or "").split())
    if is_missing(rewritten) or not is_interrogative(rewritten):
        debug(logger, "rewrite was still not a question; using the fallback")
        return NO_RESEARCH_QUESTION
    if not _preserves_meaning(text, rewritten):
        debug(logger, "rewrite introduced new content; using the fallback")
        return NO_RESEARCH_QUESTION
    return rewritten


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
        value = getattr(chunk, chunk_field(field))
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
    package = package_for(packages, field)
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


# Two findings count as restatements when this share of the SHORTER one's
# content words also appear in the other. Set high on purpose: measured against
# a discrimination set, restatements score 1.00 while genuinely distinct
# findings that merely share a topic score 0.50 or less, so 0.8 sits in open
# space rather than on the boundary. It deliberately does NOT catch paraphrases
# with little shared vocabulary ("LTD inactivated the memory" vs "optical LTD
# removed the response", 0.40) - those are indistinguishable from distinct
# findings without a semantic model, and a threshold low enough to catch them
# would collapse unrelated results.
NEAR_DUPLICATE_OVERLAP = 0.8

_ACRONYM = re.compile(r"^[A-Z]{2,}$")


def _finding_words(text: str) -> set:
    """Content words for comparing findings, keeping short acronyms.

    _content_words drops tokens of three characters or fewer, which is right
    for prose but discards exactly the tokens findings turn on: LTD, LTP, CR,
    IQ. It is left untouched because the research-question and limitation
    checks depend on its current behaviour.
    """
    words = set()
    for token in re.findall(r"[A-Za-z0-9]+", text):
        if len(token) > 3 or _ACRONYM.match(token):
            words.add(token.lower())
    return words


def collapse_near_duplicates(findings: List[str]) -> List[str]:
    """Drop restatements of a finding already kept, preferring the fuller one.

    Exact duplicates are already removed by the schema validator; this catches
    the case where the model restates one result with more or less detail and
    spends two of six slots on it.
    """
    kept: List[str] = []

    for item in findings:
        words = _finding_words(item)
        duplicate = False

        for index, existing in enumerate(kept):
            other = _finding_words(existing)
            if not words or not other:
                continue
            shared = len(words & other)
            if shared < NEAR_DUPLICATE_OVERLAP * min(len(words), len(other)):
                continue
            duplicate = True
            if len(words) > len(other):
                # The restatement carries more detail, so it replaces the
                # shorter one rather than being dropped.
                kept[index] = item
            break

        if not duplicate:
            kept.append(item)

    return kept


def tighten_key_findings(
    summary: PaperSummary,
    packages: Dict[str, EvidencePackage],
) -> int:
    """Remove recommendation-style entries and restatements from key_findings.

    If filtering empties the list, the deterministic condenser is re-run over
    the section text so a paper with real results never ends up with none.
    Returns how many entries the prescriptive filter dropped; near-duplicate
    collapsing is reported separately in the debug log.
    """
    original = [item for item in summary.key_findings if not is_missing(item)]
    kept = filter_findings(original)
    dropped = len(original) - len(kept)

    before_collapse = len(kept)
    kept = collapse_near_duplicates(kept)
    if len(kept) < before_collapse:
        debug(
            logger,
            "collapsed %d near-duplicate key_finding(s)",
            before_collapse - len(kept),
        )

    if not kept:
        package = package_for(packages, "key_findings")
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
    elif hints.title_is_confident:
        # The model owns this field whenever it returns anything at all, and it
        # drops a name from time to time - the same paper produced a four-name
        # and a three-name byline on consecutive runs. The parser read the
        # byline off the page, so when it found MORE names than the model
        # returned, the model silently shortened the list and the parser's
        # reading is preferred. Never fewer: a confident parse that found less
        # must not truncate a good answer.
        parsed = _unique(
            [name for line in hints.author_candidates
             for name in split_author_names(line)]
        )
        current = [item for item in summary.authors if not is_missing(item)]
        if len(parsed) > len(current):
            summary.authors = parsed
            recovered.append("authors (parser byline)")

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

        from_condense = _unique(list(getattr(condensed, field)))
        from_notes = _unique([item.value for item, _ in _gather(notes, field)])

        # CONDENSE only earns first place when the parser actually gave it
        # text for this field. Otherwise its value is a guess made from other
        # sections' evidence, and MAP - which read the pages themselves - is
        # the better source.
        if _has_evidence(packages, field):
            ordered = (("", from_condense), (" (map notes)", from_notes))
        else:
            ordered = ((" (map notes)", from_notes), (" (unsupported condense)",
                                                      from_condense))

        values: List[str] = []
        suffix = ""
        for label, candidate in ordered:
            if candidate:
                values, suffix = candidate, label
                break

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
        package = package_for(packages, field)
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
    """Extract the fields no labelled section covered, chunk by chunk.

    Chunks are independent passes over disjoint pages, so one unparseable
    response costs that chunk's evidence and nothing else. It used to abort the
    whole request. Returns the notes plus the page ranges that were skipped, so
    the caller can disclose the gap instead of hiding it.
    """
    notes: List[ChunkSummary] = []
    skipped: List[str] = []
    last_error: Optional[MalformedJSONError] = None
    total = len(chunks)

    for index, chunk in enumerate(chunks, start=1):
        first, last = _page_range(chunk)
        try:
            raw = await generate_json(
                _map_prompt(chunk, filename, index, total, hints, wanted),
                system=MAP_SYSTEM,
                schema=ChunkSummary.model_json_schema(),
                num_predict=OLLAMA_NUM_PREDICT_MAP,
                label="MAP {}/{} (pages {}-{})".format(index, total, first, last),
            )
        except MalformedJSONError as error:
            # Only unparseable model output is skipped. A timeout, a missing
            # model or an unreachable Ollama is a plain OllamaError and still
            # propagates, because the next chunk would fail the same way.
            last_error = error
            skipped.append("{}-{}".format(first, last))
            logger.warning(
                "MAP %d/%d malformed JSON; skipping pages %d-%d",
                index,
                total,
                first,
                last,
            )
            continue

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

    if skipped and not notes and last_error is not None:
        # Every chunk failed. There is no partial evidence to fall back on, so
        # producing a summary here would mean inventing one.
        raise last_error

    return notes, skipped


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
            schema=reduce_response_schema(),
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
    # Uses the alias so a field whose package the parser names differently is
    # not treated as uncovered, which would run the MAP stage on every paper.
    uncovered = [
        field for field in CORE_FIELDS if package_for(packages, field) is None
    ]
    notes: List[ChunkSummary] = []
    skipped: List[str] = []
    if uncovered and chunks:
        debug(logger, "running MAP for uncovered fields: %s", ", ".join(uncovered))
        notes, skipped = await _run_map_stage(chunks, filename, hints, uncovered)
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

    relabelled = enforce_limitation_attribution(summary, packages)
    if relabelled:
        logger.info(
            "Moved %d unattributable limitation(s) to "
            "model_identified_considerations.",
            relabelled,
        )
    if align_confidence_notes(summary):
        debug(logger, "confidence_notes realigned with the attribution split")

    pruned = prune_model_considerations(summary, packages)
    if pruned:
        logger.info(
            "Dropped %d model consideration(s) that were provenance "
            "commentary or already stated by the authors.",
            pruned,
        )

    # After realignment, so the disclosure is not treated as a stray sentence.
    if disclose_partial_evidence(summary, skipped):
        logger.info(
            "Summary generated without pages %s: their evidence could not be "
            "parsed.",
            "; ".join(skipped),
        )

    before = summary.research_question
    summary.research_question = await ensure_research_question(before)
    if summary.research_question != before:
        logger.info("research_question was not a question; applied one rewrite.")

    summary.clamp_source_pages(page_count)

    still_missing = summary.missing_fields()
    if still_missing:
        debug(logger, "still missing after recovery: %s", ", ".join(still_missing))

    return summary, max(1, len(chunks))
