"""Phase 3A: decide whether retrieved passages support answering a question.

The point of this layer is to separate *topical similarity* from *evidence*.
An embedding model happily returns a paper's methods section for a question
about participant demographics, because both are about the same study. Cosine
similarity ranks; it does not verify.

So no single cosine value decides anything here. The verifier combines three
independent families of signal:

A. RETRIEVAL STRENGTH   how the ranking looked: top score, mean over the
                        returned set, and the gap between first and second.
                        Used only to separate "supported" from "partially
                        supported"; it can never on its own promote a result
                        to supported.

B. EVIDENCE COVERAGE    how many distinct retrieved passages actually share
                        the question's content words, and which pages and
                        sections they came from.

C. QUESTION-TYPE CUES   what kind of question this is (participants, methods,
                        quantity, findings, limitations, location, named
                        instrument, intervention, aims), and whether the
                        retrieved text contains the kind of language that an
                        answer to that type would have to contain.

C is the veto. If a question asks how many people were recruited and no
retrieved passage contains participant or recruitment language, the result is
not_supported no matter how high the cosine scores were. This is deliberately
conservative: a wrong "not_supported" costs a missed answer, while a wrong
"supported" costs a fabricated one.

No language model is called from this module.
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field

from embeddings import RankedChunk, RetrievalResult
from sections import split_sentences
# --------------------------------------------------------------------------- #
# tunable constants
#
# All of these are deliberately loose. The type gate in section C does the
# discriminating work; these only shade the answer between supported and
# partially supported, so mis-setting one degrades a label rather than
# inverting a decision.
# --------------------------------------------------------------------------- #

# A passage counts as relevant when it shares at least this fraction of the
# question's content words. One word in three is low on purpose: the type gate,
# not this ratio, is what rejects off-target evidence.
TERM_COVERAGE_MIN = 0.34

# Below this cosine the ranking is closer to noise than to retrieval. It is a
# floor, not a decision threshold, and it is model dependent: nomic-embed-text
# puts unrelated text around 0.2-0.4, so this sits below anything meaningful.
MIN_USEFUL_TOP_SCORE = 0.15

# How many independently relevant passages are wanted before a question is
# called fully supported. One passage can be a coincidence.
MIN_SUPPORTING_CHUNKS = 2

# Gap between first and second, as a fraction of the top score. A wide relative
# gap means one passage stood out; a narrow one means the ordering among the
# top results is close to arbitrary.
CLEAR_RELATIVE_GAP = 0.15

# Questions shorter than this carry too few content words for coverage to mean
# anything, so coverage is skipped and the type gate decides alone.
MIN_QUESTION_TERMS = 2

STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for with without from by as is are was
    were be been being do does did doing done have has had having how what which
    who whom whose when where why that this these those there here it its it's
    they them their we our you your i me my he she his her not no nor so than
    then too very can could should would may might must will shall about into
    over under between among during before after above below again further once
    all any both each few more most other some such only own same s t just don
    now paper study article authors author report reported main
    given give gives said say says tell tells told describe describes described
    mention mentions mentioned state states stated call called
    """.split()
)

_PUNCTUATION = str.maketrans({char: " " for char in "?!.,;:()[]{}'\"/\\-–—"})


def _RAW_PATTERN(source):
    return re.compile(source, re.IGNORECASE)

# Words that describe *how* an attribute is reported rather than *which*
# attribute it is. "mean", "score" and "participants" are scaffolding; "IQ",
# "income" and "age" are the thing actually being asked about. Removing the
# scaffolding is what stops generic participant language from licensing a
# question about a specific measured attribute.
GENERIC_ATTRIBUTE_TERMS = frozenset(
    """
    mean average median modal typical overall total combined
    score scores level levels rate rates value values index indices
    number amount figure figures measurement measurements reading readings
    result results outcome outcomes statistic statistics
    percentage proportion share size count time times year years
    participant participants subject subjects respondent respondents
    sample samples cohort group groups people person persons individual
    individuals student students
    """.split()
)

# Standard ways papers express a measured attribute without naming it.
#
# A paper writes "22.22 years old", never "the age was 22.22". Prefix matching
# cannot bridge that, so the attribute gate needs to know the handful of
# conventional measurement constructions. This is a short table of *units and
# constructions*, not a synonym ontology: each entry describes how a quantity of
# that kind is conventionally written, so it transfers to any paper reporting
# the same kind of quantity.
#
# Each pattern is deliberately anchored to a measurement context. "years" alone
# must not satisfy an age question, or "studied English for 11 years" would.
ATTRIBUTE_EQUIVALENTS = {
    "age": _RAW_PATTERN(
        r"\b(?:\d+(?:\.\d+)?\s*[-–]?\s*years?[\s-]?old"
        r"|aged?\s+(?:between\s+)?\d"
        r"|(?:mean|average|median|modal)\s+age"
        r"|age\s+(?:range|group|of\s+\d)"
        r"|\d+(?:\.\d+)?\s+years?\s+of\s+age"
        r"|years?\s+of\s+age)\b"
    ),
    "bmi": _RAW_PATTERN(r"\b(?:bmi|body\s+mass\s+index)\b"),
    "income": _RAW_PATTERN(r"\b(?:income|salar\w+|earnings|wage\w*)\b"),
    "educat": _RAW_PATTERN(
        r"\b(?:years?\s+of\s+(?:education|schooling)|educational\s+level"
        r"|highest\s+(?:degree|qualification))\b"
    ),
    "gender": _RAW_PATTERN(r"\b(?:male|female|men|women|gender|sex)\b"),
    "sex": _RAW_PATTERN(r"\b(?:male|female|men|women|gender|sex)\b"),
}

# A term shorter than this is only matched exactly: "IQ" must be "IQ", never a
# prefix of something longer.
MIN_PREFIX_MATCH_LENGTH = 3
# How much longer a matched word may be than the term it matches. "age" may
# match "aged" or "ages"; it may not match "agency".
MAX_VARIANT_SUFFIX = 2


class Answerability(str, Enum):
    """How well the retrieved passages support answering the question."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class AnswerabilityReport(BaseModel):
    """The verdict, plus everything needed to inspect how it was reached.

    The retrieval scores are carried through for transparency. They are raw
    cosine statistics: not probabilities, not confidence percentages, and not
    comparable between questions.
    """

    status: Answerability
    reason: str
    supporting_chunk_ids: List[str] = Field(default_factory=list)
    supporting_pages: List[int] = Field(default_factory=list)
    retrieval_top_score: Optional[float] = None
    retrieval_score_gap: Optional[float] = None
    retrieval_mean_top_k: Optional[float] = None
    evidence_count: int = 0
    # Inspection aids.
    question_types: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    matched_terms: List[str] = Field(default_factory=list)

    @property
    def is_answerable(self) -> bool:
        return self.status is not Answerability.NOT_SUPPORTED


# --------------------------------------------------------------------------- #
# term handling
# --------------------------------------------------------------------------- #


def stem(word: str) -> str:
    """Crude suffix stripping, enough to match plurals and simple inflections.

    The ``-ion`` rule matters more than it looks: a question says "how were the
    data collected" while the paper's own heading says "data collection", and
    without it those never meet.
    """
    for suffix in ("ies", "ions", "ion", "ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            if suffix == "ies":
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


def content_terms(text: str) -> Set[str]:
    """Distinctive words of a question or passage, stemmed and de-duplicated."""
    terms = set()
    for token in text.lower().translate(_PUNCTUATION).split():
        if len(token) < 3 or token in STOPWORDS:
            continue
        terms.add(stem(token))
    return terms


def term_present(term: str, text_terms: Set[str]) -> bool:
    """True when a term appears in a passage, allowing close variants.

    Papers write "aged 18 to 24" where a question says "age", so an exact match
    is too strict. A prefix match bounded to two extra characters is the
    smallest rule that covers inflection without inventing synonyms.
    """
    if term in text_terms:
        return True
    if len(term) < MIN_PREFIX_MATCH_LENGTH:
        return False
    for candidate in text_terms:
        longer, shorter = (
            (candidate, term) if len(candidate) >= len(term) else (term, candidate)
        )
        if longer.startswith(shorter) and len(longer) - len(shorter) <= MAX_VARIANT_SUFFIX:
            return True
    return False


def attribute_present(term: str, text_terms: Set[str], raw_text: str) -> bool:
    """True when a passage reports the attribute a question asked about.

    Either the word itself appears (allowing close variants), or the passage
    uses the conventional construction for that kind of quantity, such as
    "22 years old" for age.
    """
    if term_present(term, text_terms):
        return True
    equivalent = ATTRIBUTE_EQUIVALENTS.get(term)
    return bool(equivalent and equivalent.search(raw_text))


def focus_terms(question: str) -> Set[str]:
    """The specific things a question asks about, minus its scaffolding.

    Tokenised down to two characters rather than three, because attribute names
    are often two-letter acronyms: IQ would otherwise be discarded before the
    attribute gate ever saw it. Two-letter English words are almost all
    function words, and STOPWORDS already covers them.
    """
    terms = set()
    for token in question.lower().translate(_PUNCTUATION).split():
        if len(token) < 2 or token in STOPWORDS:
            continue
        term = stem(token)
        if term not in GENERIC_ATTRIBUTE_TERMS:
            terms.add(term)
    return terms


def evidence_text(chunk: Any) -> str:
    """A passage's text plus its section name.

    The section a passage came from is evidence in its own right: a question
    about how data were collected is answered in part by the fact that the
    passage sits under "Data collection and analysis".
    """
    section = (getattr(chunk, "section", "") or "").replace("_", " ")
    return "{} {}".format(getattr(chunk, "text", ""), section)


def term_coverage(question_terms: Set[str], text: str) -> Tuple[float, Set[str]]:
    """Fraction of the question's content words that appear in a passage."""
    if not question_terms:
        return 0.0, set()
    matched = question_terms & content_terms(text)
    return len(matched) / len(question_terms), matched


# --------------------------------------------------------------------------- #
# question types
# --------------------------------------------------------------------------- #


class QuestionType(BaseModel):
    """A kind of research question and the language an answer must contain."""

    name: str
    label: str
    cue: Any
    evidence: Any
    # When set, the noun captured from the cue must itself appear in the
    # evidence. "Which questionnaire..." must find the word questionnaire.
    require_captured_noun: bool = False
    # True when the evidence family is specific enough that its presence is a
    # real signal. "findings" and "quantity" match almost any prose or any
    # digit, so promoting on them would reorder the whole corpus for nothing.
    precise: bool = False

    def matches(self, question: str) -> Optional[re.Match]:
        return self.cue.search(question)

    def evidence_present(self, text: str, match: Optional[re.Match]) -> bool:
        if self.require_captured_noun and match is not None:
            noun = (match.group("noun") or "").strip().lower()
            if noun and stem(noun) not in content_terms(text):
                return False
        # A family is normally a pattern, but may be a predicate when the
        # decision needs sentence-level context rather than a single match.
        if hasattr(self.evidence, "search"):
            return bool(self.evidence.search(text))
        return bool(self.evidence(text))


def _pattern(source: str) -> Any:
    return re.compile(source, re.IGNORECASE)


# Papers rarely write "this is a limitation". They write "the sample was drawn
# from a single institution" or "this may partly reflect coverage of the
# index". sections.LIMITATION_CUES is deliberately narrow because it decides
# what goes in a summary's limitations list; here the question has already
# announced it is about limitations, so recognising the hedged constructions
# costs nothing and is what these questions were failing on.
#
# Note what is absent: a bare "may". "Scores may increase with practice" is a
# finding, not a caveat, so only hedges that qualify an interpretation count.
IMPLICIT_LIMITATION_CUES = _pattern(
    r"\b("
    # explicit
    r"limitation\w*|caveat\w*|weakness\w*|shortcoming\w*|drawback\w*"
    r"|threats? to validity"
    # sample and setting
    r"|small(?:er)? sample|sample size|limited sample|homogen(?:e|ei)ous"
    r"|convenience sampl\w*|self[- ]?select\w*|single (?:institution|site|centre"
    r"|center|university|school|department|class|cohort|country)"
    r"|one (?:institution|site|centre|center|university|school)"
    # design and measurement
    r"|only once|single (?:measurement|time ?point|occasion|interview)"
    r"|one time ?point|cross[- ]?sectional|no control group|not randomi[sz]ed"
    r"|self[- ]?report\w*|did not (?:measure|collect|assess|control|record)"
    # inference limits
    r"|cannot (?:be )?(?:determine|establish|rule out|infer|generali[sz]e"
    r"|be inferred|be established|be generali[sz]ed)"
    r"|unable to (?:determine|establish|distinguish|assess|verify)"
    r"|no causal|causal(?:ity)? cannot|not possible to (?:determine|establish)"
    # hedged interpretation
    r"|may (?:partly |partially |in part )?(?:reflect|result|be explained"
    r"|be due|arise|stem|indicate|understate|overstate|not generali[sz]e"
    r"|not (?:be )?representative|not apply|not transfer)"
    r"|might (?:partly |partially )?(?:reflect|explain|be due)"
    r"|possible explanation|one explanation|alternative explanation"
    r"|should be interpreted (?:cautiously|with caution)"
    r"|interpret\w* with caution|treated with caution"
    # coverage and bias
    r"|generali[sz]ability|generali[sz]ed?|generali[sz]e"
    r"|potential(?:ly)? bias\w*|possible bias\w*|selection bias|response bias"
    r"|recall bias|confound\w*|under[- ]?represent\w*|over[- ]?represent\w*"
    r"|understate\w*|overstate\w*|not indexed|coverage of|restricted to"
    r"|limited to|confined to|only covers?"
    r")\b"
)


# A paper states its aims in one of two ways: with an explicit aim noun, or
# with a purpose construction attached to the current study. The second is by
# far the more common and is what the old closed verb list missed.
_PURPOSE_VERB = (
    r"(?:determin\w*|examin\w*|investigat\w*|explor\w*|assess\w*|establish\w*"
    r"|identif\w*|compar\w*|evaluat\w*|test\w*|analys\w*|analyz\w*|measur\w*"
    r"|answer\w*|ascertain\w*|quantif\w*|describ\w*|understand|find out"
    r"|report on|address\w*|shed light on)"
)

# "this study", "the present analysis", "our review" - the *current* work.
_CURRENT_STUDY = (
    r"(?:this|the present|the current|the reported|our)\s+(?:\w+\s+){0,2}?"
    r"(?:stud(?:y|ies)|paper|article|analysis|review|research|work"
    r"|investigation|report|survey)"
)

AIM_CONSTRUCTIONS: Tuple[Any, ...] = (
    # An explicit aim noun. "The aim of this study was to ..."
    _pattern(r"\b(?:aims?|aimed|objectives?|goals?|purpose|research questions?"
             r"|hypothes[ie]s)\b"),
    # The current study doing the verb: "this study examines", "our analysis
    # sets out to", "the present paper seeks to".
    _pattern(_CURRENT_STUDY + r"\s+(?:\w+\s+){0,2}?"
             r"(?:" + _PURPOSE_VERB + r"|sets? out to|seeks? to|aims? to"
             r"|intends? to|attempts? to|sought to)"),
    # The current study with a purpose infinitive later in the same sentence:
    # "This article reports on a short study using Scopus data to determine ..."
    _pattern(_CURRENT_STUDY + r"\b[^.]{0,140}?\bto\s+" + _PURPOSE_VERB),
    # First person with a purpose infinitive: "we sought to determine",
    # "we decided to replicate this analysis, to determine ...".
    _pattern(r"\bwe\b[^.]{0,80}?\bto\s+" + _PURPOSE_VERB),
    _pattern(r"\bwe\s+(?:\w+\s+){0,1}?" + _PURPOSE_VERB),
    # Passive: "the study was conducted to determine", "a search was carried
    # out to determine".
    _pattern(r"\b(?:this|the|our|an?)\s+(?:\w+\s+){0,2}?"
             r"(?:stud(?:y|ies)|analysis|research|investigation|survey|search"
             r"|experiment|review)\s+(?:was|were)\s+"
             r"(?:conducted|carried out|performed|undertaken|designed|run)"
             r"\b[^.]{0,60}?\bto\s+" + _PURPOSE_VERB),
)

# Sentences describing somebody else's work. An aim construction inside one of
# these is a report of prior research, not a statement of this paper's purpose,
# so "Previous studies examined X" and "Smith (2019) investigated Y" are
# skipped before any aim pattern is tried.
PRIOR_WORK_MARKER = _pattern(
    r"\b(?:previous|earlier|prior|past|existing|other|another|recent)\s+"
    r"(?:stud(?:y|ies)|work|research|articles?|papers?|literature|authors?"
    r"|researchers?|investigations?|reviews?)"
    r"|\bet al\b|\(\s*\d{4}[a-z]?\s*\)"
    r"|\b[A-Z][a-z]+\s+(?:and\s+[A-Z][a-z]+\s+)?\(\d{4}\)"
    r"|\bresearch\s+(?:suggests?|shows?|has shown|indicates?|demonstrates?)"
    r"|\bhas\s+been\s+(?:shown|reported|found|argued|suggested)"
    r"|\bothers?\s+(?:have|has|who)\b"
    r"|\bbuilds?\s+on\b|\baccording to\b"
)


def aims_evidence_present(text: str) -> bool:
    """True when the text states what *this* paper set out to do.

    Checked sentence by sentence so that an aim construction can be attributed
    to the current study rather than to the literature it cites.
    """
    for sentence in split_sentences(text):
        if PRIOR_WORK_MARKER.search(sentence):
            continue
        if any(construction.search(sentence) for construction in AIM_CONSTRUCTIONS):
            return True
    return False


QUESTION_TYPES: Tuple[QuestionType, ...] = (
    QuestionType(
        name="participants",
        precise=True,
        label="participant or recruitment",
        cue=_pattern(
            r"\b(participants?|subjects?|respondents?|interviewees?|volunteers?"
            r"|sample size|who took part|took part|recruit\w*|enrol\w*"
            r"|demographics?|average age|mean age|age range)\b"
        ),
        evidence=_pattern(
            r"\b(participants?|respondents?|interviewees?|informants?"
            r"|volunteers?|recruit\w*|enrol\w*|took part|cohort"
            # "subject" is only a person when a person predicate or a count
            # follows it. Bare "subject" is a subject *area* in half the
            # literature, which is how a bibliometric paper came to look as
            # though it had human participants.
            r"|human subjects?|subjects?\s+(?:were|who|aged|completed|reported)"
            r"|\d+\s+subjects?"
            # How a sample was obtained answers a recruitment question even when
            # the word "recruited" never appears. Each phrase ties the strategy
            # word to the sample, so "convenience" alone is not enough.
            r"|(?:convenience|purposive|snowball|opportunity|quota)\s+sampl\w*"
            r"|(?:choos\w+|chose|chosen|select\w+|drawn|obtained)"
            r"\s+(?:the\s+|this\s+|a\s+)?sampl\w*"
            r"|for convenience|sampled"
            r"|accessib\w*\s+to\s+the\s+(?:researcher|author|investigator)"
            r"|sample of \d+"
            r"|\d+\s+(?:\w+\s+){0,2}(?:people|adults|students|patients|nurses"
            r"|teachers|children|men|women))\b"
        ),
    ),
    QuestionType(
        name="methods",
        precise=True,
        label="data collection or analysis",
        cue=_pattern(
            r"\b(how (?:were|was|did|do)\b.{0,60}?\b(?:collect|gather|obtain"
            r"|assembl|analys|analyz|measur|conduct|determin|compil)\w*"
            r"|method\w*|methodolog\w*|procedure|study design|data collection"
            r"|how .{0,30}\bdata\b)"
        ),
        evidence=_pattern(
            r"\b(collect\w*|gather\w*|assembl\w*|extract\w*|analys\w*|analyz\w*"
            r"|coded|coding|survey\w*|interview\w*|questionnaire|corpus"
            r"|index\w*|dataset|data ?set|sampl\w*|aggregat\w*|estimat\w*"
            r"|model\w*|metadata|records?)\b"
        ),
    ),
    QuestionType(
        name="quantity",
        label="numeric",
        cue=_pattern(
            r"\b(how many|how much|how often|what (?:percentage|proportion"
            r"|share|number|fraction|total)|by how much)\b"
        ),
        evidence=_pattern(
            r"(\d|\bper cent\b|\bpercent\b|%|\b(one|two|three|four|five|six"
            r"|seven|eight|nine|ten|dozen|hundred|thousand|million)\b)"
        ),
    ),
    QuestionType(
        name="findings",
        label="result or association",
        cue=_pattern(
            r"\b(find(?:ing)?s?\b|what did .{0,30}\b(?:find|show|report)"
            r"|results?\b|outcomes?\b|effects?\b|associat\w*|correlat\w*"
            r"|increase\w*|decrease\w*|declin\w*|rose|fell|grew|change\w*"
            r"|difference|which (?:field|area|discipline|subject|domain)s?)\b"
        ),
        evidence=_pattern(
            r"\b(found|show\w*|report\w*|result\w*|rose|fell|grew|increas\w*"
            r"|decreas\w*|declin\w*|associat\w*|correlat\w*|strong\w*|weak\w*"
            r"|highest|lowest|steepest|slowest|dominan\w*|retained"
            r"|per cent|percent)\b|%|\d"
        ),
    ),
    QuestionType(
        name="limitations",
        precise=True,
        label="limitation or caveat",
        cue=_pattern(
            r"\b(limitation\w*|caveat\w*|weakness\w*|shortcoming\w*|drawback\w*"
            r"|generalis\w*|generaliz\w*|bias\w*|threats? to validity"
            r"|under-?represent\w*|understate\w*|not representative)\b"
        ),
        evidence=IMPLICIT_LIMITATION_CUES,
    ),
    QuestionType(
        name="location",
        label="country or region",
        cue=_pattern(
            r"\b(?:which|what|in which)\s+(?:\w+\s+){0,2}"
            r"(?:countr\w+|nation\w*|region\w*|continent\w*|state\w*|cit\w+)\b"
            r"|\bwhere\b|\bgeograph\w*"
        ),
        evidence=_pattern(
            r"\b(countr\w+|nation\w*|region\w*|continent\w*|worldwide|global"
            r"|domestic|Europe\w*|America\w*|Asia\w*|Africa\w*|Oceania"
            r"|Germany|Italy|Poland|France|Japan|Brazil|China|Spain|Portugal"
            r"|Netherlands|India|Russia|Mexico)\b"
        ),
    ),
    QuestionType(
        name="instrument",
        precise=True,
        label="named instrument",
        cue=_pattern(
            r"\b(?:which|what)\s+(?P<noun>questionnaire|survey|instrument|scale"
            r"|inventory|checklist|app|application|software|platform|tool"
            r"|database|index|measure|test|assessment)\b"
        ),
        evidence=_pattern(
            r"\b(questionnaire|survey|instrument|scale|inventory|checklist"
            r"|app|application|software|platform|tool|database|index\w*"
            r"|measure\w*|assessment)\b"
        ),
        require_captured_noun=True,
    ),
    QuestionType(
        name="intervention",
        precise=True,
        label="intervention or treatment",
        cue=_pattern(
            r"\b(intervention\w*|treatment\w*|programme|program\b|trial\w*"
            r"|control group|randomis\w*|randomiz\w*|placebo"
            r"|improved .{0,20}scores?|what improved)\b"
        ),
        evidence=_pattern(
            # Every phrase names an experimental design. Bare "programme",
            # "condition", "trial" and "arm" are ordinary academic words: a
            # philology programme is not an intervention, and matching on them
            # let an interview study answer questions about control groups.
            r"\b(intervention\w*|placebo"
            r"|(?:control|treatment|experimental|comparison|intervention)\s+"
            r"(?:group|condition|arm|participants?)"
            r"|randomi[sz]ed|randomly (?:assigned|allocated)|random allocation"
            r"|controlled trial|clinical trial|treated with"
            r"|pre-?test|post-?test|test scores?|baseline measure)\b"
        ),
    ),
    QuestionType(
        name="aims",
        precise=True,
        label="aim or research question",
        cue=_pattern(
            r"\b(research questions?|aims?\b|objectives?\b|purpose\b"
            r"|set out to|what (?:were|was) .{0,30}(?:studying|investigating"
            r"|examining|asking))\b"
        ),
        evidence=aims_evidence_present,
    ),
)


# Syntactic constructions that mean "give me the value of a specific
# attribute". These are shapes, not a list of attributes: no attribute name is
# hard-coded anywhere, so BMI, GPA, reaction time and blood pressure are all
# caught by the same four patterns.
ATTRIBUTE_QUESTION_CUES: Tuple[Any, ...] = (
    # "mean IQ", "average age", "median household income"
    _pattern(r"\b(?:mean|average|median|modal|typical)\s+(?:\w+\s+){0,2}\w{2,}\b"),
    # "depression score", "reaction time", "blood pressure", "response rate"
    _pattern(
        r"\b\w{3,}\s+(?:scores?|levels?|rates?|index|indices|quotients?|ratios?"
        r"|readings?|measurements?|pressure|duration|latency|accuracy)\b"
    ),
    # "the participants' BMI", "their weight", "the sample's education".
    # The possessive is required: without it "how were participants recruited"
    # reads as an attribute question and gets vetoed for not reporting a
    # "recruit" attribute.
    _pattern(
        r"\b(?:participants?|subjects?|respondents?|students?|sample|group)"
        r"(?:'s|s'|\u2019s|s\u2019)\s+(?:\w+\s+){0,2}\w{2,}\b"
        r"|\btheir\s+(?:mean|average|median|typical)?\s*\w{2,}\b"
    ),
    # "years of education", "hours of study"
    _pattern(r"\b(?:years?|hours?|months?|days?|weeks?)\s+of\s+\w{3,}\b"),
)


def asks_for_specific_attribute(question: str) -> bool:
    """True when the question requests the value of a named attribute."""
    return any(cue.search(question) for cue in ATTRIBUTE_QUESTION_CUES)


def detect_question_types(question: str) -> List[Tuple[QuestionType, Any]]:
    """Every type whose cue fires for this question, with the matching object."""
    detected = []
    for question_type in QUESTION_TYPES:
        match = question_type.matches(question)
        if match:
            detected.append((question_type, match))
    return detected


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #


def _pages_phrase(pages: Sequence[int]) -> str:
    if not pages:
        return "no pages"
    if len(pages) == 1:
        return "page {}".format(pages[0])
    if pages == list(range(pages[0], pages[-1] + 1)):
        return "pages {}-{}".format(pages[0], pages[-1])
    return "pages {}".format(", ".join(str(page) for page in pages))


def _count_phrase(count: int) -> str:
    return "one passage" if count == 1 else "{} passages".format(count)


def _relative_gap(diagnostics: Any) -> Optional[float]:
    top = diagnostics.top_score
    gap = diagnostics.score_gap
    if top is None or gap is None or top <= 0:
        return None
    return gap / top


def _report(
    status: Answerability,
    reason: str,
    diagnostics: Any,
    supporting: Sequence[RankedChunk] = (),
    question_types: Sequence[str] = (),
    missing: Sequence[str] = (),
    matched: Sequence[str] = (),
) -> AnswerabilityReport:
    pages = sorted({chunk.page_number for chunk in supporting})
    return AnswerabilityReport(
        status=status,
        reason=reason,
        supporting_chunk_ids=[chunk.chunk_id for chunk in supporting],
        supporting_pages=pages,
        retrieval_top_score=diagnostics.top_score,
        retrieval_score_gap=diagnostics.score_gap,
        retrieval_mean_top_k=diagnostics.mean_top_k,
        evidence_count=len(supporting),
        question_types=list(question_types),
        missing_evidence=list(missing),
        matched_terms=sorted(matched),
    )


def verify_answerability(
    question: str, retrieval: RetrievalResult
) -> AnswerabilityReport:
    """Decide whether the retrieved passages can support an answer.

    Deterministic: the same question and the same ranking always produce the
    same report. No model is called.
    """
    results = retrieval.results
    diagnostics = retrieval.diagnostics

    if not results:
        return _report(
            Answerability.NOT_SUPPORTED,
            "No eligible passages were retrieved for this question.",
            diagnostics,
        )

    combined_text = "\n".join(evidence_text(chunk) for chunk in results)
    detected = detect_question_types(question)
    type_names = [question_type.name for question_type, _ in detected]

    # ---- C. question-type gate (the veto) ----
    missing = [
        question_type.label
        for question_type, match in detected
        if not question_type.evidence_present(combined_text, match)
    ]
    if missing:
        return _report(
            Answerability.NOT_SUPPORTED,
            "Retrieved passages are topically related but contain no {} "
            "evidence.".format(" or ".join(missing)),
            diagnostics,
            question_types=type_names,
            missing=missing,
        )

    # ---- C2. specific-attribute gate ----
    # A question naming a measured attribute needs that attribute in the
    # evidence. Participant language plus an unrelated number is not an answer
    # to "what was their mean IQ".
    evidence_terms = content_terms(combined_text)
    if asks_for_specific_attribute(question):
        wanted = focus_terms(question)
        if wanted and not any(
            attribute_present(term, evidence_terms, combined_text) for term in wanted
        ):
            return _report(
                Answerability.NOT_SUPPORTED,
                "Retrieved passages describe the study but never report {}.".format(
                    " or ".join(sorted(wanted))
                ),
                diagnostics,
                question_types=type_names,
                missing=sorted(wanted),
            )

    # ---- B. evidence coverage ----
    question_terms = content_terms(question)
    scored = [
        (chunk,) + term_coverage(question_terms, evidence_text(chunk))
        for chunk in results
    ]
    matched_terms: Set[str] = set()
    for _chunk, _coverage, matched in scored:
        matched_terms |= matched

    if len(question_terms) < MIN_QUESTION_TERMS:
        # Too few content words for coverage to be meaningful; the type gate
        # above has already had its say, so fall back to the top passage.
        supporting = list(results[:MIN_SUPPORTING_CHUNKS])
        return _report(
            Answerability.PARTIALLY_SUPPORTED,
            "Question is too short to verify against the retrieved text; "
            "returning the closest {}.".format(_count_phrase(len(supporting))),
            diagnostics,
            supporting,
            type_names,
            matched=matched_terms,
        )

    # A passage counts when it shares enough of the question's words, or when
    # it carries the language of the question's own type. The second clause is
    # what finds an implicit caveat: a passage saying "drawn from a single
    # institution" answers "what were the limitations" while sharing none of
    # its words.
    supporting = [
        chunk
        for chunk, coverage, _ in scored
        if coverage >= TERM_COVERAGE_MIN
        or any(
            question_type.evidence_present(evidence_text(chunk), match)
            for question_type, match in detected
        )
    ]

    if not supporting:
        return _report(
            Answerability.NOT_SUPPORTED,
            "Only weak indirect evidence was found: no retrieved passage "
            "shares enough of the question's terms.",
            diagnostics,
            question_types=type_names,
            matched=matched_terms,
        )

    pages = sorted({chunk.page_number for chunk in supporting})
    sections = sorted({chunk.section for chunk in supporting if chunk.section})

    # ---- A. retrieval strength, used only to shade the verdict ----
    top_score = diagnostics.top_score or 0.0
    relative_gap = _relative_gap(diagnostics)

    if top_score < MIN_USEFUL_TOP_SCORE:
        return _report(
            Answerability.PARTIALLY_SUPPORTED,
            "Evidence found in {} on {}, but the retrieval scores are close to "
            "noise.".format(_count_phrase(len(supporting)), _pages_phrase(pages)),
            diagnostics,
            supporting,
            type_names,
            matched=matched_terms,
        )

    if len(supporting) >= MIN_SUPPORTING_CHUNKS:
        detail = ""
        if len(sections) > 1:
            detail = " across the {} sections".format(" and ".join(sections))
        return _report(
            Answerability.SUPPORTED,
            "Relevant evidence found in {} on {}{}.".format(
                _count_phrase(len(supporting)), _pages_phrase(pages), detail
            ),
            diagnostics,
            supporting,
            type_names,
            matched=matched_terms,
        )

    # Exactly one relevant passage. A clear lead makes that more convincing,
    # but one passage is never called fully supported.
    if relative_gap is not None and relative_gap >= CLEAR_RELATIVE_GAP:
        reason = (
            "A single passage on {} matches clearly, but no second passage "
            "corroborates it.".format(_pages_phrase(pages))
        )
    else:
        reason = (
            "A single passage on {} matches, and the next results score "
            "almost as highly.".format(_pages_phrase(pages))
        )
    return _report(
        Answerability.PARTIALLY_SUPPORTED,
        reason,
        diagnostics,
        supporting,
        type_names,
        matched=matched_terms,
    )


def summarise(report: AnswerabilityReport) -> Dict[str, Any]:
    """A flat dict of the report, for logging or an evaluation table."""
    return {
        "status": report.status.value,
        "reason": report.reason,
        "pages": report.supporting_pages,
        "evidence_count": report.evidence_count,
        "top_score": report.retrieval_top_score,
        "score_gap": report.retrieval_score_gap,
        "mean_top_k": report.retrieval_mean_top_k,
        "question_types": report.question_types,
        "missing_evidence": report.missing_evidence,
    }
