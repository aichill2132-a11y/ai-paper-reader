"""Deterministic section parsing for academic papers.

The model used to be asked to *discover* fields inside unconstrained chunks,
which is where title, research question, methods and participants were getting
lost. Papers are structured documents, so we find the structure ourselves and
hand the model the relevant text to condense.

HEADING DETECTION
    A line is a heading when, after stripping list numbering ("4.", "4.1",
    "IV."), leading bullets and a trailing colon, the remaining text matches one
    of HEADING_PATTERNS in full. Heading lines must be short (<= MAX_HEADING_CHARS)
    and must not end in sentence punctuation. A run-in heading ("Limitations. The
    study ...") is also recognised: the heading is matched against the text
    before the first ':' or '.', and the remainder of the line becomes the first
    line of that section's body.

SECTION BOUNDARIES
    A section starts on the line after its heading (or mid-line for a run-in
    heading) and ends at the line before the next recognised heading, or at the
    end of the document. Pages are tracked line by line, so a section that
    straddles a page break records every page it covers.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

MAX_HEADING_CHARS = 70

# Canonical section name -> pattern that must match the whole heading text.
HEADING_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("abstract", r"abstract|summary"),
    ("keywords", r"key ?words?"),
    ("introduction", r"introduction"),
    ("background", r"background|literature review|related work|theoretical framework"),
    (
        "research_question",
        r"research question(s)?|research aims?|aims?( of the study)?|"
        r"objectives?|study objectives?|hypothes[ei]s|hypotheses|purpose( of the study)?",
    ),
    (
        "data_collection_and_analysis",
        r"data collection( and analysis)?|data analysis|data gathering|"
        r"analysis|analytic(al)? (approach|strategy)|measures|instruments?",
    ),
    (
        "participants",
        r"participants?|subjects?|sample|study (sample|population)|population|"
        r"respondents|recruitment|setting and participants",
    ),
    (
        "methods",
        r"methods?|methodology|materials and methods|method(s)? and materials|"
        r"study design|design|research design|procedures?|"
        r"(methods?|methodology) and (materials|design)",
    ),
    ("findings", r"findings|results|results and findings|findings and results"),
    (
        "discussion",
        r"discussion|discussion and conclusions?|findings and discussion|"
        r"discussion of findings",
    ),
    ("limitations", r"limitations|study limitations|limitations of the study|"
                    r"strengths and limitations|threats to validity"),
    ("conclusion", r"conclusions?|concluding remarks|implications( for practice)?"),
    ("references", r"references|bibliography|works cited|reference list"),
    ("acknowledgements", r"acknowledge?ments?|funding|conflicts? of interest"),
)

# Sections that mark the end of the body text.
TERMINAL_SECTIONS = {"references", "acknowledgements"}

# List numbering / bullets that may precede a heading.
_NUMBERING = re.compile(r"^\s*(?:[-*•]\s*)?(?:\d+(?:\.\d+)*\.?|[IVXivx]+\.)\s*")
_SENTENCE_END = re.compile(r"[.!?]\s*$")

# --------------------------------------------------------------------------- #
# limitation vocabulary
# --------------------------------------------------------------------------- #

LIMITATION_CUES = re.compile(
    r"\b(limitation|limitations|limited by|weakness(es)?|caveat|shortcoming|"
    r"generali[sz]ability|generali[sz]ed?|generali[sz]e|small sample|"
    r"sample size|homogeneous sample|homogen(e|ei)ty|single interview|"
    r"single site|single centre|single center|convenience sample|"
    r"self-report(ed|ing)? bias|recall bias|selection bias|response bias|"
    r"not representative|cannot be assumed|may not (apply|transfer)|"
    r"one time point|cross-sectional design limits|no (long-term )?follow[- ]up)\b",
    re.IGNORECASE,
)

# Result-like sentences that sometimes sit inside a limitations paragraph.
FINDING_CUES = re.compile(
    r"\b(we found|our (findings|results)|results? (show|showed|indicate|"
    r"indicated|suggest|suggested|demonstrate)|"
    r"the (first|second|third|fourth|final) theme|"
    r"themes? (was|were|emerged|developed|identified)|"
    r"reported (markedly |significantly |substantially |much )?"
    r"(higher|lower|greater|more|less|better|worse)|"
    r"(increased|decreased|improved|declined|rose|fell) (by|significantly)|"
    r"was (significantly|markedly) (higher|lower|greater)|"
    r"\d+(\.\d+)?\s*(%|per cent|percent)|p\s*[<=]\s*0?\.\d+)\b",
    re.IGNORECASE,
)

# Prescriptive language. A sentence telling someone what to do is a
# recommendation, not a finding, however well evidenced it is.
PRESCRIPTIVE_CUES = re.compile(
    r"\b(should|shouldn'?t|ought to|need(s)? to|must (be|now|therefore)?|"
    r"recommend\w*|advocat\w+|call(s)? for|"
    r"we suggest|suggest(s|ed)? that (practitioners|teachers|educators|"
    r"clinicians|nurses|managers|leaders|schools|services|providers|"
    r"organi[sz]ations|policy\s?makers|policymakers|trusts)|"
    r"implications? for|implications? of these findings|"
    r"future (work|research|studies|directions)|further (research|work|studies)|"
    r"polic(y|ies|ymakers)|training should|guidance should|"
    r"it is important that|it would be (useful|helpful|advisable))\b",
    re.IGNORECASE,
)

# Language that marks a sentence as reporting evidence rather than opinion.
EMPIRICAL_CUES = re.compile(
    r"\b(found|finding(s)?|show(s|ed|n)?|demonstrat\w+|indicat\w+|reveal\w+|"
    r"report(s|ed|ing)?|describ(e|es|ed)|captur(e|es|ed)|identif\w+|"
    r"emerg\w+|observ\w+|record\w+|measur\w+|"
    r"theme(s)?|pattern(s)?|categor(y|ies)|"
    r"associat\w+|correlat\w+|predict\w+|relationship|"
    r"increas\w+|decreas\w+|reduc\w+|improv\w+|declin\w+|ros(e|se)|fell|"
    r"higher|lower|greater|fewer|more likely|less likely|"
    r"no (significant |clear )?(difference|effect|change|association)|"
    r"significant\w*|appear(s|ed)? to|seem(s|ed)? to|tend(s|ed)? to|"
    r"participants?|respondents?|interviewees?|informants?|"
    r"nurses?|students?|teachers?|patients?|clinicians?|"
    r"we (found|observed|identified|measured|saw)|"
    r"\d+(\.\d+)?\s*(%|per cent|percent)|p\s*[<=]\s*0?\.\d+|"
    r"mean|median|odds ratio|confidence interval|effect size)\b",
    re.IGNORECASE,
)

RECOMMENDATION_CUES = re.compile(
    r"\b(future (research|studies|work)|further (research|work|studies)|"
    r"we recommend|recommendations?|should be (explored|investigated|considered)|"
    r"we suggest that (researchers|practitioners)|next steps?)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")

# A line that is nothing but a page number.
PAGE_NUMBER_LINE = re.compile(r"^(page\s*)?\d{1,4}$", re.IGNORECASE)


def clean_lines(text: str) -> List[str]:
    """Split into lines, collapse internal whitespace, drop blank lines."""
    return [" ".join(raw.split()) for raw in text.splitlines() if raw.strip()]


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences. Deliberately simple and predictable."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT.split(cleaned) if part.strip()]


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


class Section(BaseModel):
    """One recognised section of the paper."""

    name: str
    heading: str
    text: str = ""
    pages: List[int] = Field(default_factory=list)

    @property
    def start_page(self) -> int:
        return self.pages[0] if self.pages else 0


class EvidencePackage(BaseModel):
    """The text handed to the model for one summary field."""

    field: str
    text: str = ""
    pages: List[int] = Field(default_factory=list)
    headings: List[str] = Field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


# --------------------------------------------------------------------------- #
# heading recognition
# --------------------------------------------------------------------------- #


def normalise_heading(line: str) -> str:
    """Strip numbering, bullets, trailing colon and surrounding whitespace."""
    text = _NUMBERING.sub("", line).strip()
    text = text.strip("–—-").strip()
    text = text.rstrip(":.").strip()
    return " ".join(text.split())


def match_heading(text: str) -> Optional[str]:
    """Return the canonical section name for a heading, or None."""
    candidate = normalise_heading(text)
    if not candidate or len(candidate) > MAX_HEADING_CHARS:
        return None
    if not candidate[0].isalpha():
        return None
    for name, pattern in HEADING_PATTERNS:
        if re.fullmatch(pattern, candidate, flags=re.IGNORECASE):
            return name
    return None


def _heading_on_line(line: str) -> Optional[Tuple[str, str]]:
    """Detect a heading at the start of a line.

    Returns ``(canonical_name, remainder_of_line)``. A standalone heading has an
    empty remainder; a run-in heading ("Limitations. The study ...") returns the
    prose that followed it.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Standalone heading line.
    if len(stripped) <= MAX_HEADING_CHARS and not _SENTENCE_END.search(stripped):
        name = match_heading(stripped)
        if name:
            return name, ""

    # Run-in heading: "Limitations: ..." or "Limitations. ..."
    for separator in (":", "."):
        head, found, tail = stripped.partition(separator)
        if not found or len(head) > MAX_HEADING_CHARS:
            continue
        name = match_heading(head)
        if name and tail.strip():
            return name, tail.strip()

    return None


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #


def parse_sections(
    pages: Sequence[Tuple[int, str]],
    drop_lines: Optional[Sequence[str]] = None,
) -> List[Section]:
    """Split a document into sections, tracking the pages each one covers.

    ``pages`` is ``(page_number, text)`` in reading order. ``drop_lines`` is the
    set of repeated header/footer lines to ignore (see metadata.repeated_lines).
    """
    dropped = {" ".join(line.split()).lower() for line in (drop_lines or [])}

    sections: List[Section] = []
    current: Optional[Section] = None
    buffer: List[str] = []

    def flush() -> None:
        if current is not None:
            current.text = "\n".join(buffer).strip()
            sections.append(current)

    for page_number, raw_text in pages:
        for line in clean_lines(raw_text):
            if line.lower() in dropped or PAGE_NUMBER_LINE.match(line):
                continue

            found = _heading_on_line(line)
            if found:
                name, remainder = found
                flush()
                current = Section(name=name, heading=line, pages=[page_number])
                buffer = [remainder] if remainder else []
                continue

            if current is not None:
                buffer.append(line)
                if page_number not in current.pages:
                    current.pages.append(page_number)

    flush()
    return sections


def sections_by_name(sections: Sequence[Section]) -> Dict[str, List[Section]]:
    grouped: Dict[str, List[Section]] = {}
    for section in sections:
        grouped.setdefault(section.name, []).append(section)
    return grouped


def body_sections(sections: Sequence[Section]) -> List[Section]:
    """Everything up to the references, which we never want to mine."""
    body = []
    for section in sections:
        if section.name in TERMINAL_SECTIONS:
            break
        body.append(section)
    return body


# --------------------------------------------------------------------------- #
# limitation filtering
# --------------------------------------------------------------------------- #


def is_limitation(sentence: str, in_limitations_section: bool) -> bool:
    """Only accept a sentence that the paper itself frames as a limitation.

    Inside an explicit limitations section the bar is low, but recommendations
    ("future research should ...") are still rejected. Anywhere else the
    sentence must carry an explicit limitation cue.
    """
    text = sentence.strip()
    if len(text) < 20:
        return False

    has_limitation_cue = bool(LIMITATION_CUES.search(text))

    # A recommendation for future work is not a limitation.
    if RECOMMENDATION_CUES.search(text) and not has_limitation_cue:
        return False

    # Neither is a result that happens to sit in the limitations paragraph.
    if FINDING_CUES.search(text) and not has_limitation_cue:
        return False

    if in_limitations_section:
        return True

    return has_limitation_cue


def is_prescriptive(sentence: str) -> bool:
    """True when a sentence tells the reader what someone ought to do."""
    return bool(PRESCRIPTIVE_CUES.search(sentence))


def is_finding(sentence: str) -> bool:
    """True for empirical results, observed patterns, reported themes and
    conclusions drawn from the study's own evidence.

    Prescriptive sentences are rejected outright. What remains must carry at
    least one marker of observation or measurement, so "implications" prose
    that slipped past the cue list is still kept out.
    """
    text = sentence.strip()
    if len(text) < 40:
        return False
    if is_prescriptive(text):
        return False
    return bool(EMPIRICAL_CUES.search(text))


def filter_findings(sentences: Sequence[str]) -> List[str]:
    """Drop prescriptive items from a list of candidate findings."""
    return [item for item in sentences if item and not is_prescriptive(item)]


def collect_limitations(
    sections: Sequence[Section],
) -> Tuple[List[str], List[int]]:
    """Gather limitation sentences plus the pages they came from."""
    accepted: List[str] = []
    pages: List[int] = []

    for section in body_sections(sections):
        explicit = section.name == "limitations"
        # Outside an explicit section only discussion-like prose is worth scanning.
        if not explicit and section.name not in {"discussion", "conclusion"}:
            continue

        for sentence in split_sentences(section.text):
            if not is_limitation(sentence, explicit):
                continue
            if sentence not in accepted:
                accepted.append(sentence)
                for page in section.pages:
                    if page not in pages:
                        pages.append(page)

    return accepted, sorted(pages)


# --------------------------------------------------------------------------- #
# field mapping
# --------------------------------------------------------------------------- #

# field -> section names, in priority order.
FIELD_SECTIONS: Dict[str, Tuple[str, ...]] = {
    "research_question": ("research_question",),
    "participants_or_data": ("participants",),
    "methods": ("data_collection_and_analysis", "methods"),
    "key_findings": ("findings", "discussion"),
    "background": ("background", "introduction", "abstract"),
}

# Sentences that state the research question, used only when the paper has no
# "Research question" heading. Deliberately narrower than
# condensers.QUESTION_CUES: this one decides whether to *invent* a research
# question package out of the abstract, so it errs on the side of not firing.
_QUESTION_CUES = re.compile(
    r"\b(research question|this (study|paper|article) (aims?|sought|set out|"
    r"examines?|explores?|investigates?)|the aim of this|we (aimed|sought|asked|"
    r"examine[d]?|explore[d]?|investigate[d]?)|our objective|the purpose of "
    r"(this|the) (study|paper))\b",
    re.IGNORECASE,
)


def _package(field: str, parts: Sequence[Section]) -> EvidencePackage:
    texts = []
    pages: List[int] = []
    headings: List[str] = []
    for section in parts:
        if not section.text.strip():
            continue
        texts.append(section.text.strip())
        headings.append(section.heading)
        for page in section.pages:
            if page not in pages:
                pages.append(page)
    return EvidencePackage(
        field=field,
        text="\n\n".join(texts),
        pages=sorted(pages),
        headings=headings,
    )


def build_evidence_packages(
    sections: Sequence[Section],
) -> Dict[str, EvidencePackage]:
    """Map recognised sections onto the summary fields."""
    body = body_sections(sections)
    grouped = sections_by_name(body)
    packages: Dict[str, EvidencePackage] = {}

    for field, names in FIELD_SECTIONS.items():
        parts: List[Section] = []
        for name in names:
            parts.extend(grouped.get(name, []))
        package = _package(field, parts)
        if package.has_text:
            packages[field] = package

    # Research question with no heading: mine the abstract and introduction.
    if "research_question" not in packages:
        sentences: List[str] = []
        pages: List[int] = []
        for name in ("abstract", "introduction", "background"):
            for section in grouped.get(name, []):
                for sentence in split_sentences(section.text):
                    if _QUESTION_CUES.search(sentence) and sentence not in sentences:
                        sentences.append(sentence)
                        for page in section.pages:
                            if page not in pages:
                                pages.append(page)
        if sentences:
            packages["research_question"] = EvidencePackage(
                field="research_question",
                text=" ".join(sentences),
                pages=sorted(pages),
                headings=["(inferred from abstract/introduction)"],
            )

    # Limitations go through the filter rather than straight section capture.
    limitation_sentences, limitation_pages = collect_limitations(body)
    if limitation_sentences:
        packages["limitations"] = EvidencePackage(
            field="limitations",
            text="\n".join(limitation_sentences),
            pages=limitation_pages,
            headings=["limitations"],
        )

    return packages
