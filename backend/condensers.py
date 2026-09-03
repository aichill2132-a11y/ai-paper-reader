"""Deterministic condensers.

These run when the model fails, times out, or returns "Not stated in the paper"
for a field we *know* has evidence. They never invent anything: every word they
return is copied from the supplied section text.
"""

import re
from typing import List

from sections import is_finding, split_sentences

MAX_CHARS = 600

# Sentence-level protocol detail. A methods summary should say what kind of
# study was run, not restate the bench protocol, so sentences that are mostly
# instrument settings are skipped when a design sentence is available.
_PROTOCOL_DETAIL = re.compile(
    r"\b(?:\d+(?:\.\d+)?\s*(?:mm|cm|nm|\u00b5m|um|ml|\u00b5l|ul|mg|\u00b5g|g/l|mol|mM|nM|"
    r"rpm|psi|kV|mA|Hz|\u00b0C|min/ml|ml/min)\b"
    r"|flow rate|column (?:dimension|temperature|oven)|particle size"
    r"|wavelength|gradient elution|mobile phase|injection volume"
    r"|centrifug\w+ at|catalogue number|cat\.? no|lot number"
    r"|model\s+[A-Z0-9-]{3,}|version\s+\d)",
    re.IGNORECASE,
)

# The paper's own statement of purpose, used to build a research question when
# the model is unavailable.
_AIM_STATEMENT = re.compile(
    r"\b(?:aims?|aimed|objectives?|purpose|research question)\b"
    r"|\b(?:this|the present|the current|our)\s+(?:\w+\s+){0,2}"
    r"(?:stud(?:y|ies)|paper|article|analysis|research|work)\b[^.]{0,120}?"
    r"\bto\s+(?:determin|examin|investigat|explor|assess|establish|identif|"
    r"compar|evaluat|test|measur|describ)\w*"
    r"|\bwe\b[^.]{0,80}?\bto\s+(?:determin|examin|investigat|explor|assess|"
    r"establish|identif|compar|evaluat|test|measur|describ)\w*",
    re.IGNORECASE,
)

NO_RESEARCH_QUESTION = (
    "The research question was not explicitly identifiable in the paper."
)

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
    """Return the paper's aim, or say it could not be identified.

    Only a fallback: the model is asked to rewrite the aim as a question. This
    path runs when the model is unavailable, so it returns the aim verbatim
    rather than attempting a rewrite it cannot do reliably. It never falls back
    to the opening sentences, because a Results or Discussion sentence is worse
    than admitting the aim was not found.
    """
    sentences = split_sentences(text)
    if not sentences:
        return NO_RESEARCH_QUESTION

    questions = [item for item in sentences if item.rstrip().endswith("?")]
    if questions:
        return _trim(" ".join(questions[:2]))

    cued = [item for item in sentences if _QUESTION_CUES.search(item)]
    if cued:
        return _trim(" ".join(cued[:2]))

    aims = [item for item in sentences if _AIM_STATEMENT.search(item)]
    if aims:
        return _trim(aims[0])

    return NO_RESEARCH_QUESTION


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

    # Prefer design sentences over bench protocol. Protocol sentences are only
    # used if nothing else is available, so a chemistry paper still gets an
    # answer rather than an empty methods field.
    design = [item for item in sentences if not _PROTOCOL_DETAIL.search(item)]
    pool = design or sentences

    collection = next((item for item in pool if _COLLECTION.search(item)), "")
    analysis = next(
        (item for item in pool if _ANALYSIS.search(item) and item != collection),
        "",
    )

    parts = [part for part in (collection, analysis) if part]
    if not parts:
        return _trim(" ".join(pool[:2]))
    return _trim(" ".join(parts))


def condense_findings(text: str, limit: int = 6) -> List[str]:
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
# condense_limitations is deliberately NOT registered here. Recovery runs the
# condensers in this table whenever a field came back empty, which would force
# a limitation into author_stated_limitations every time the parser found
# limitation-shaped text - exactly the over-classification the field is meant
# to avoid. An empty list is a valid answer, so the field has no condenser.
# The function is kept for callers that want the deterministic extraction.
LIST_CONDENSERS = {
    "key_findings": condense_findings,
}
