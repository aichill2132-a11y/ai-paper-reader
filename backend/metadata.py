"""Deterministic front-matter extraction: title, authors, structural hints.

TITLE DETECTION
    1. Lines repeated across two or more pages are journal headers/footers and
       are removed everywhere.
    2. On page 1, standalone page numbers and any line matching a *furniture*
       pattern are dropped: journal/volume/issue/month/year, DOI, URL, email,
       ISSN/ISBN, received/accepted/published dates, article-type labels
       ("Research paper", "Original article"), copyright, and affiliation lines
       (university, department, school, faculty, hospital, institute, centre).
    3. Scanning stops at the first Abstract / Keywords / Introduction heading.
    4. The remaining lines are grouped into *blocks* of consecutive lines. A
       block breaks at a line that looks like an author list, an affiliation, or
       page furniture, so a title wrapped over three lines stays one block.
    5. The title is the longest block (by character count) that appears before
       the first author-looking line. Longest, not first, is what stops a short
       journal header from winning.
    6. Authors are the lines between the end of the title block and the first
       affiliation/email line that parse as a name list.
    7. Confidence is "high" only when a title block was found before an author
       or affiliation line and it is a plausible title. Low confidence means the
       candidates are passed to the model to choose from rather than being
       applied blindly.
"""

import math
import re
from typing import Dict, List, Sequence, Tuple

from pydantic import BaseModel, Field

from sections import PAGE_NUMBER_LINE, clean_lines, match_heading, parse_sections

# --------------------------------------------------------------------------- #
# line classifiers
# --------------------------------------------------------------------------- #

_PAGE_RANGE = re.compile(r"^\d{1,4}\s*[-–]\s*\d{1,4}$")

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL = re.compile(r"(https?://|www\.|doi\.org|\bdoi\b\s*:?\s*10\.)", re.IGNORECASE)

# Journal / bibliographic furniture.
_BIBLIOGRAPHIC = re.compile(
    r"(^journal of\b|^the journal\b|\bjournal\b.*\b(19|20)\d{2}\b|"
    r"\bvol(ume)?\.?\s*\d+|\bno\.\s*\d+|\bissue\s*\d+|\bpp?\.\s*\d+|"
    r"\bissn\b|\bisbn\b|\barxiv\b|\bpreprint\b|\bdoi\b|"
    r"^\(?(19|20)\d{2}\)?$|\b\d{1,3}\s*\(\s*\d{1,3}\s*\)\s*[:,]|"
    r"^(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\b|"
    r"\b(received|revised|accepted|published|first published|available online|"
    r"downloaded from|copyright|all rights reserved|licen[cs]e)\b|©|\(c\))",
    re.IGNORECASE,
)

# Article-type labels that sit above the title.
_ARTICLE_LABEL = re.compile(
    r"^(research paper|research article|original (research|article|paper)|"
    r"review( article| paper)?|short (report|communication)|case (report|study)|"
    r"editorial|commentary|brief report|systematic review|"
    r"empirical (research|paper|article)|article)$",
    re.IGNORECASE,
)

_AFFILIATION = re.compile(
    r"\b(universit|department|dept\.|school of|college|faculty|institute|"
    r"academy|hospital|clinic|laborator|research (centre|center|group|unit)|"
    r"centre for|center for|nhs|trust|foundation|corresponding author|"
    r"correspondence|address for correspondence|\binc\.|\bltd\b)\b",
    re.IGNORECASE,
)

_KEYWORDS = re.compile(r"^(key ?words?|index terms|jel classification)\b", re.IGNORECASE)

# Headings that mean the front matter is over.
_FRONT_MATTER_END = {"abstract", "keywords", "introduction"}

# Name-list shapes: "J. Smith", "Jane Smith", separated by commas / "and".
_NAME = re.compile(
    r"^[A-Z][A-Za-z'’\-]*\.?"          # first name or initial
    r"(\s+[A-Z]\.?)*"                   # middle initials
    r"(\s+(van|von|de|del|della|di|da|dos|le|la|el|bin|ibn|al))*"
    r"(\s+[A-Z][A-Za-z'’\-]+)+$"        # surname
)
_AUTHOR_SEPARATOR = re.compile(r"\s*(?:,|;|\band\b|&)\s*", re.IGNORECASE)
# Superscript affiliation markers attached to names.
_AUTHOR_MARKERS = re.compile(r"[\d\*†‡§¶\^]+")


def is_page_furniture(line: str) -> bool:
    """True for anything that is not part of the paper's own front matter."""
    if PAGE_NUMBER_LINE.match(line) or _PAGE_RANGE.match(line):
        return True
    if _EMAIL.search(line) or _URL.search(line):
        return True
    if _BIBLIOGRAPHIC.search(line):
        return True
    if _ARTICLE_LABEL.match(line):
        return True
    if _KEYWORDS.match(line):
        return True
    if _AFFILIATION.search(line):
        return True
    return False


def looks_like_authors(line: str) -> bool:
    """True when a line parses as a list of personal names."""
    if is_page_furniture(line):
        return False
    stripped = _AUTHOR_MARKERS.sub("", line).strip().rstrip(",;")
    if not stripped or len(stripped) > 200:
        return False

    parts = [part.strip() for part in _AUTHOR_SEPARATOR.split(stripped) if part.strip()]
    if not parts or len(parts) > 20:
        return False

    matched = sum(1 for part in parts if _NAME.match(part))
    if len(parts) == 1:
        # A lone name has to be short. Without this, a title in title case
        # ("Attention Is All You Need") parses as a person's name.
        words = stripped.split()
        return bool(matched) and 2 <= len(words) <= 4
    return matched >= max(2, (len(parts) + 1) // 2)


def looks_like_title_text(line: str) -> bool:
    if is_page_furniture(line) or looks_like_authors(line):
        return False
    words = line.split()
    return len(words) >= 2 and 6 <= len(line) <= 300


# --------------------------------------------------------------------------- #
# repeated headers and footers
# --------------------------------------------------------------------------- #


def repeated_lines(pages: Sequence[Tuple[int, str]]) -> List[str]:
    """Lines that appear on several pages: running heads and footers.

    The threshold scales with the document so a two-page paper does not have
    half its body discarded. Section headings are never dropped, and long lines
    are left alone because running heads are short.
    """
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for _number, text in pages:
        seen = set()
        for line in clean_lines(text):
            key = line.lower()
            if len(key) < 4 or key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, line)

    threshold = max(2, math.ceil(0.4 * len(pages)))
    return [
        originals[key]
        for key, count in counts.items()
        if count >= threshold
        and len(key) <= 100
        and match_heading(originals[key]) is None
    ]


# --------------------------------------------------------------------------- #
# title and authors
# --------------------------------------------------------------------------- #


class TitleAuthors(BaseModel):
    title_candidates: List[str] = Field(default_factory=list)
    author_candidates: List[str] = Field(default_factory=list)
    # "high" means safe to apply without the model's agreement.
    confidence: str = "low"

    @property
    def is_confident(self) -> bool:
        return self.confidence == "high"


def _front_matter_lines(page_one: str, drop: Sequence[str]) -> List[str]:
    """Page 1 lines above the abstract, with furniture removed."""
    dropped = {line.lower() for line in drop}
    lines: List[str] = []

    for line in clean_lines(page_one)[:60]:
        if line.lower() in dropped:
            continue
        heading = match_heading(line)
        if heading in _FRONT_MATTER_END:
            break
        if is_page_furniture(line):
            continue
        lines.append(line)

    return lines


def _title_blocks(lines: Sequence[str]) -> Tuple[List[List[str]], int]:
    """Group front-matter lines into runs of consecutive title-like lines.

    An author line breaks a run, which is what keeps a wrapped title together
    while separating it from the byline. Also returns how many blocks appeared
    before the first author line.
    """
    blocks: List[List[str]] = []
    current: List[str] = []
    blocks_before_authors = len(lines)

    for line in lines:
        if looks_like_authors(line):
            blocks_before_authors = min(blocks_before_authors, len(blocks))
            if current:
                blocks.append(current)
                current = []
        elif looks_like_title_text(line):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []

    if current:
        blocks.append(current)
    return blocks, blocks_before_authors


def _pick_title(
    blocks: List[List[str]], blocks_before_authors: int
) -> Tuple[List[str], List[str]]:
    """Choose the title block and turn it into candidates.

    The longest block before the byline wins: a journal header is short, a real
    title is not. Returns the candidates and the block they came from, so the
    caller knows where the byline can start.
    """
    considered = blocks[:blocks_before_authors] or blocks
    best = max(considered, key=lambda block: sum(len(line) for line in block))

    candidates = [" ".join(best)]
    if len(best) > 1:
        # Offer the opening line alone, in case the block swallowed a subtitle.
        candidates.append(best[0])
    return candidates, best


def _pick_authors(lines: Sequence[str], after_index: int) -> Tuple[List[str], bool]:
    """Name-list lines following the title, up to the first affiliation.

    Returns the candidates and whether they were found in that position; a
    name-like line *above* the title is a running head, not a byline.
    """
    candidates: List[str] = []
    for line in lines[after_index + 1 :]:
        if _AFFILIATION.search(line) or _EMAIL.search(line):
            break
        if looks_like_authors(line) and line not in candidates:
            candidates.append(line)

    follow_title = bool(candidates)
    if not candidates:
        candidates = [line for line in lines if looks_like_authors(line)]

    # A real byline usually lists several names, so prefer those: it stops a
    # two-word organisation name ("Google Brain") masquerading as an author.
    multi = [line for line in candidates if len(_AUTHOR_SEPARATOR.split(line)) > 1]
    return (multi or candidates)[:3], follow_title


def extract_title_and_authors(
    page_one: str, drop: Sequence[str] = ()
) -> TitleAuthors:
    """Find the title and authors on page 1. See the module docstring."""
    lines = _front_matter_lines(page_one, drop)
    if not lines:
        return TitleAuthors()

    blocks, blocks_before_authors = _title_blocks(lines)
    if not blocks:
        return TitleAuthors(
            author_candidates=[line for line in lines if looks_like_authors(line)][:3]
        )

    title_candidates, title_block = _pick_title(blocks, blocks_before_authors)
    title = title_candidates[0]

    # The byline can only start after the title block's last line.
    title_end = lines.index(title_block[-1])
    authors, authors_follow_title = _pick_authors(lines, title_end)

    confident = (
        bool(authors)
        and authors_follow_title
        and len(title) >= 20
        and len(title.split()) >= 4
    )

    return TitleAuthors(
        title_candidates=title_candidates,
        author_candidates=authors,
        confidence="high" if confident else "low",
    )


# --------------------------------------------------------------------------- #
# document-level hints
# --------------------------------------------------------------------------- #


class DocumentHints(BaseModel):
    """Everything we can establish about a paper without the model."""

    title_candidates: List[str] = Field(default_factory=list)
    author_candidates: List[str] = Field(default_factory=list)
    title_confidence: str = "low"
    headings: Dict[str, List[int]] = Field(default_factory=dict)
    repeated_lines: List[str] = Field(default_factory=list)

    @property
    def title_is_confident(self) -> bool:
        return self.title_confidence == "high"


def extract_hints(pages: Sequence[Tuple[int, str]]) -> DocumentHints:
    """Build the deterministic candidate set for a whole document."""
    if not pages:
        return DocumentHints()

    running = repeated_lines(pages)
    found = extract_title_and_authors(pages[0][1], running)

    headings: Dict[str, List[int]] = {}
    for section in parse_sections(pages, drop_lines=running):
        pages_for_name = headings.setdefault(section.name, [])
        if section.start_page not in pages_for_name:
            pages_for_name.append(section.start_page)

    return DocumentHints(
        title_candidates=found.title_candidates,
        author_candidates=found.author_candidates,
        title_confidence=found.confidence,
        headings=headings,
        repeated_lines=running,
    )


def render_hints(hints: DocumentHints) -> str:
    """A compact, prompt-friendly rendering of the hints."""
    lines = []
    if hints.title_candidates:
        lines.append(
            f"Title candidates from page 1 (confidence: {hints.title_confidence}): "
            + " | ".join(hints.title_candidates)
        )
    if hints.author_candidates:
        lines.append(
            "Author candidates from page 1: " + " | ".join(hints.author_candidates)
        )
    if hints.headings:
        parts = [
            f"{name} (p{', p'.join(str(page) for page in pages)})"
            for name, pages in sorted(hints.headings.items())
        ]
        lines.append("Section headings detected: " + "; ".join(parts))

    if not lines:
        return "No structural hints were detected."
    return "\n".join(lines)
