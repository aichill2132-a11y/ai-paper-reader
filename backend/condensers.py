"""Deterministic condensers.

These run when the model fails, times out, or returns "Not stated in the paper"
for a field we *know* has evidence. They never invent anything: every word they
return is copied from the supplied section text.
"""

import re
from typing import List

from sections import is_finding, split_sentences

MAX_CHARS = 600

_PEOPLE = (
    r"participants?|subjects?|respondents?|interviewees?|informants?|"
    r"volunteers?|patients?|nurses?|midwives|doctors?|physicians?|clinicians?|"
    r"students?|teachers?|staff|employees?|managers?|adults?|children|"
    r"women|men|families|carers?|households?|firms?|companies|documents?|"
    r"records?|articles?|papers?|transcripts?|interviews?|cases?|observations?"
)

_NUMBER_WORD = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred"
)

_COUNT = re.compile(
    r"\b(\d{1,5}|" + _NUMBER_WORD + r")\s+"
    r"((?:[a-z\-]+\s+){0,3}?(?:" + _PEOPLE + r"))\b",
    re.IGNORECASE,
)

_AGE = re.compile(
    r"\b(aged?\s+(?:between\s+)?\d{1,3}\s*(?:to|and|-|–)\s*\d{1,3}"
    r"|mean age (?:of|was)\s+[\d.]+"
    r"|age range(?: of| was)?\s+\d{1,3}\s*(?:to|and|-|–)\s*\d{1,3})",
    re.IGNORECASE,
)

_RECRUITMENT = re.compile(
    r"\b(recruit|sampl|enrol|selected from|drawn from|purposive|snowball|"
    r"convenience|volunteer|approached|invited)\w*\b",
    re.IGNORECASE,
)

_COLLECTION = re.compile(
    r"\b(semi[- ]structured interview|in[- ]depth interview|interview|"
    r"focus group|survey|questionnaire|observation|ethnograph\w*|diary|"
    r"diaries|field ?notes|workshop|document analysis|secondary data|"
    r"randomi[sz]ed|controlled trial|experiment|crossover|longitudinal|"
    r"cross[- ]sectional|case study)\w*",
    re.IGNORECASE,
)

_ANALYSIS = re.compile(
    r"\b(thematic analysis|framework analysis|content analysis|"
    r"grounded theory|constant comparison|discourse analysis|"
    r"narrative analysis|coded|coding|codebook|nvivo|atlas\.ti|"
    r"descriptive statistics|regression|anova|t-test|chi[- ]squared?|"
    r"mixed[- ]effects|statistical analys\w*|inductive|deductive)\w*",
    re.IGNORECASE,
)

# Cues for picking the research question out of text already known to be about
# it. Broader than sections._QUESTION_CUES, which gates a riskier decision.
_QUESTION_CUES = re.compile(
    r"\b(research question|this (study|paper|article|review) (aims?|sought|"
    r"set out|examines?|explores?|investigates?|asks?)|the aims? of (this|the)|"
    r"we (aimed|sought|asked|examined?|explored?|investigated?)|"
    r"our (aim|objective|purpose)|the (purpose|objective) of (this|the))\b",
    re.IGNORECASE,
)


def _trim(text: str, limit: int = MAX_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit].rsplit(" ", 1)[0]
    return cut + "..."


def clean_extract(text: str, max_sentences: int = 2) -> str:
    """The first few sentences of a section, tidied up."""
    sentences = split_sentences(text)
    return _trim(" ".join(sentences[:max_sentences]))


def condense_research_question(text: str) -> str:
    """Prefer an actual question, then an aim statement, then the opening."""
    sentences = split_sentences(text)
    if not sentences:
        return ""

    questions = [item for item in sentences if item.rstrip().endswith("?")]
    if questions:
        return _trim(" ".join(questions[:2]))

    cued = [item for item in sentences if _QUESTION_CUES.search(item)]
    if cued:
        return _trim(" ".join(cued[:2]))

    return clean_extract(text, 2)


def condense_participants(text: str) -> str:
    """Report who took part: counts, characteristics, and how they were found."""
    sentences = split_sentences(text)
    if not sentences:
        return ""

    parts: List[str] = []

    recruitment = next((item for item in sentences if _RECRUITMENT.search(item)), "")

    counts = []
    for match in _COUNT.finditer(text):
        phrase = " ".join(match.group(0).split())
        # Skip a count that the recruitment sentence already states.
        if phrase.lower() in recruitment.lower():
            continue
        if phrase.lower() not in [item.lower() for item in counts]:
            counts.append(phrase)
    if counts:
        parts.append("; ".join(counts[:3]))

    ages = [" ".join(match.group(0).split()) for match in _AGE.finditer(text)]
    if ages and not any(age.lower() in recruitment.lower() for age in ages):
        parts.append(ages[0])

    if recruitment:
        parts.append(recruitment)

    if not parts:
        return clean_extract(text, 2)

    summary = ". ".join(part.rstrip(".") for part in parts) + "."
    return _trim(summary)


def condense_methods(text: str) -> str:
    """Combine how the data were collected with how they were analysed."""
    sentences = split_sentences(text)
    if not sentences:
        return ""

    collection = next((item for item in sentences if _COLLECTION.search(item)), "")
    analysis = next(
        (item for item in sentences if _ANALYSIS.search(item) and item != collection),
        "",
    )

    parts = [part for part in (collection, analysis) if part]
    if not parts:
        return clean_extract(text, 2)
    return _trim(" ".join(parts))


def condense_findings(text: str, limit: int = 5) -> List[str]:
    """Keep only empirical results, observed patterns and reported themes.

    Recommendations, implications and future-work statements are excluded; see
    sections.is_finding for the rule.
    """
    findings: List[str] = []
    for sentence in split_sentences(text):
        if not is_finding(sentence):
            continue
        trimmed = _trim(sentence, 300)
        if trimmed not in findings:
            findings.append(trimmed)
        if len(findings) >= limit:
            break
    return findings


def condense_limitations(text: str, limit: int = 5) -> List[str]:
    """The limitation sentences are already filtered; just tidy and cap them."""
    items: List[str] = []
    for sentence in split_sentences(text):
        trimmed = _trim(sentence, 300)
        if trimmed and trimmed not in items:
            items.append(trimmed)
        if len(items) >= limit:
            break
    return items


# field name -> deterministic condenser producing a string
TEXT_CONDENSERS = {
    "research_question": condense_research_question,
    "participants_or_data": condense_participants,
    "methods": condense_methods,
    "background": lambda text: clean_extract(text, 3),
}

# field name -> deterministic condenser producing a list
LIST_CONDENSERS = {
    "key_findings": condense_findings,
    "limitations": condense_limitations,
}
