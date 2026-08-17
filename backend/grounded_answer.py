"""Grounded answer generation for Ask the Paper.

The deterministic pipeline decides *what the evidence is*. The model decides
only *how to phrase it*. Everything a reader would rely on - which passages
count, which pages they came from, whether the paper answers the question at
all - is produced before the model is called and is never taken back from it.

Concretely:

* Phase 3A owns retrieval, ranking, answerability and page provenance.
* select_evidence() narrows the ranked passages to the smallest set that still
  supports an answer.
* The model receives only those passages, and returns only prose plus the ids
  of the passages it used.
* Every id is checked against what was sent. Unknown ids are dropped, pages are
  looked up from the parser's own metadata, and an answer that cannot be
  attributed to a specific passage is discarded rather than published.

A not_supported verdict never reaches the model at all.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from answerability import (
    Answerability,
    AnswerabilityReport,
    asks_for_specific_attribute,
    attribute_present,
    content_terms,
    detect_question_types,
    focus_terms,
    verify_answerability,
)
from config import (
    MAX_EVIDENCE_CHUNKS,
    OLLAMA_NUM_PREDICT_ANSWER,
    SELECTION_POOL_SIZE,
)
from diagnostics import debug
from embedding_cache import EMBEDDING_CACHE, document_key
from embeddings import RankedChunk, embed_chunks
from evidence_ranking import retrieve_evidence
from ollama_client import OllamaError, generate_json
from retrieval import build_retrieval_chunks
from schemas import (
    AnswerSource,
    AskDiagnostics,
    AskResponse,
    PageInput,
)
from sections import split_sentences

logger = logging.getLogger(__name__)

# Two passages are treated as the same evidence above this token overlap.
# Chunks share a 60-word overlap by construction, so neighbours on a page are
# routinely near-duplicates.
DUPLICATE_OVERLAP = 0.6
# Upper bound on a quoted excerpt in the API response.
MAX_SNIPPET_CHARS = 320
# How much of a passage the model is shown. Long enough to answer from, short
# enough that four passages fit comfortably in context.
MAX_EVIDENCE_CHARS = 1200

ABSTAIN_ANSWER = (
    "The paper does not provide enough evidence to answer this question."
)
ABSTAIN_NOTE = "No sufficiently supported answer was found in the paper."
UNATTRIBUTABLE_ANSWER = (
    "An answer could not be traced to specific passages in the paper, so none "
    "is given."
)
SUPPORTED_NOTE = "Answer based only on retrieved evidence from the paper."
PARTIAL_NOTE = (
    "The paper only partially addresses this question; the answer below is "
    "based on limited evidence and should be checked against the paper."
)

SYSTEM_PROMPT = (
    "You are a careful research assistant. You are given numbered passages "
    "taken verbatim from one academic paper, and a question about that paper. "
    "You answer only from those passages. You never use outside knowledge, "
    "never infer facts the passages do not state, and never invent names, "
    "numbers, methods, outcomes, citations or page numbers. You reply with "
    "JSON only."
)


class GeneratedAnswer(BaseModel):
    """What the model is allowed to produce.

    Note what is absent: page numbers. Pages are parser-owned metadata and are
    attached afterwards, so the model has no way to invent one.
    """

    answer: str = ""
    used_chunk_ids: List[str] = Field(default_factory=list)

    @field_validator("answer", mode="before")
    @classmethod
    def _coerce_answer(cls, value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = " ".join(str(item) for item in value)
        return str(value or "").strip()

    @field_validator("used_chunk_ids", mode="before")
    @classmethod
    def _coerce_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            return []
        return [str(item).strip() for item in value if str(item).strip()]


# --------------------------------------------------------------------------- #
# evidence selection
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> Set[str]:
    return {word.strip(".,;:()[]\"'").lower() for word in text.split() if len(word) > 3}


def _overlap(first: str, second: str) -> float:
    """Jaccard-style containment of the smaller passage in the larger."""
    left, right = _tokens(first), _tokens(second)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


# Weights for the evidence-contribution score. These order candidates; none is
# a threshold, so a change shifts which passage is cited first, never whether a
# question is answerable. They are listed in descending importance: what the
# passage says about the question outranks where retrieval happened to put it.
W_TERM_COVERAGE = 3.0
W_NAMED_ITEMS = 2.5
W_TYPE_CUE = 2.0
W_ATTRIBUTE = 2.0
W_SUPPORTING = 1.0
W_RANK = 1.0
# A number attached to the thing being counted. The strongest possible signal
# for "how many X", and absent from a passage that merely mentions X.
W_QUANTITY = 3.0
# A sentence that lists several requested items ("such as A, B and C").
W_ENUMERATION = 1.0

# Distinct proper nouns at which the named-item signal saturates.
NAMED_ITEM_SATURATION = 4

# Interrogative quantifiers. "many" in "how many" is part of the question, not
# a content word, and a passage does not answer a counting question by using
# the word "many". Applied only to selection; answerability is frozen.
_QUESTION_QUANTIFIERS = frozenset({"many", "much", "often", "long"})

# A quantity question: "how many participants", "what was the sample size".
_QUANTITY_QUESTION = re.compile(
    r"\bhow many\s+(?P<entity>[a-z]+)"
    r"|\bnumber of\s+(?P<entity2>[a-z]+)"
    r"|\b(?P<entity3>sample|cohort|group)\s+size",
    re.IGNORECASE,
)

_NUMBER = r"(?:\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|"     r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"     r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)"

# Words that make a number mean something other than a count of the entity.
_NON_COUNT_UNIT = re.compile(
    r"^(?:%|per ?cent|percent|years?|months?|weeks?|days?|hours?|minutes?"
    r"|seconds?|pages?|euros?|dollars?|pounds?)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Direct-answer features
#
# A passage can be *about* a question without answering it. These three
# discrete features mark the passages that actually carry an answer, and are
# what the selector looks for first. They are booleans, not weights: a passage
# either enumerates the requested items, explains the requested procedure, or
# states the requested count, or it does not.
# --------------------------------------------------------------------------- #

# Families of study-procedure verb. A "how were participants recruited"
# question is answered by sampling language, not by data-collection language,
# so the families are kept apart.
PROCESS_FAMILIES = (
    (
        "sampling",
        r"recruit|select|choos|chose|chosen|sampl|draw|drew|drawn|enrol|obtain"
        r"|identif|approach|invit|volunteer",
    ),
    (
        "collection",
        r"collect|gather|record|administer|conduct|elicit|interview|survey|measur",
    ),
    ("assignment", r"assign|allocat|randomi[sz]"),
)

# The clause that turns a statement into an explanation of *how* or *why*.
_EXPLANATORY = re.compile(
    r"\b(?:because|since|as they were|as these were|owing to|due to|in order to"
    r"|through|throughout|via|by means of|by using|using|on the basis of"
    r"|based on|for convenience|for reasons of|so as to|with the help of"
    # "identified by searching the records" - instrumental "by" plus a gerund,
    # the commonest way a method sentence says how something was done.
    r"|by\s+\w+ing)\b",
    re.IGNORECASE,
)

# A procedural question: "how were participants recruited", "how was the
# sample obtained", "how were cases identified".
_PROCEDURAL_QUESTION = re.compile(
    r"\bhow\s+(?:were|was|did|do|are|is)\b", re.IGNORECASE
)

# A comma- or conjunction-joined run of capitalised names: "Duolingo, Anki and
# Quizlet". This is what distinguishes a list from a title block.
_NAME_RUN = re.compile(
    r"\b[A-Z][\w'-]{2,}(?:\s+[A-Z][\w'-]{2,})?"
    r"(?:\s*,\s*|\s+and\s+|\s+or\s+)"
    r"[A-Z][\w'-]{2,}"
)

# Bibliographic prose: "(Cakir, 2015)", "Byrne & Diem, 2014", "Smith et al.".
# "&" and "et al" mark a citation. "(e.g." deliberately does not: in a findings
# section it introduces exactly the list of tools a named-item question wants.
_CITATION_DENSE = re.compile(r"\bet al\b|&", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CITATION_YEAR = re.compile(r"^,?\s*\(?\s*(?:19|20)\d{2}")

# Sentences that enumerate: "such as X, Y and Z", "including A and B".
_ENUMERATION = re.compile(
    r"\b(?:such as|including|namely|for example|e\.g\.|like)\b", re.IGNORECASE
)

# Weight of each additional relevant sentence in a span, relative to the
# strongest one. Keeps span scores comparable between a one-sentence answer and
# a four-sentence paragraph.
CORROBORATION = 0.25
# Cost of dragging a sentence that says nothing about the question into a span.
PADDING_PENALTY = 0.1

# How the two families of signal are balanced. Content says what a passage
# contains; rank carries what the embedding model understood semantically and
# lexical scoring cannot see - a note about "convenience" answering a question
# about recruitment without sharing a word with it.
CONTENT_SHARE = 0.65
RANK_SHARE = 0.35

# Deterministic evidence span limits.
MAX_SPAN_SENTENCES = 4
MAX_SPAN_CHARS = 700

# A passage must contribute at least this fraction of the best passage to be
# considered at all. Deliberately generous: the elbow below decides how many
# are actually cited. Ordering constant, never a threshold on answerability.
RELATIVE_FLOOR = 0.4

# Capitalised words that are not names: sentence openers, headings, and the
# months and weekdays that litter academic prose.
_NOT_A_NAME = frozenset(
    """
    The This That These Those There Their They It Its As At In On For With From
    Table Figure Section Appendix Page Chapter Note Notes Study Results
    Findings Discussion Conclusion Introduction Method Methods Participants
    However Moreover Furthermore Although Because Since While When Where What
    Which Who How Why All Some Most Many Both Each One Two Three Four Five
    January February March April May June July August September October
    November December Monday Tuesday Wednesday Thursday Friday English
    """.split()
)
_PARTICIPANT_CODE = re.compile(r"^[A-Z]{1,2}\d{1,3}$")


def named_items(text: str) -> Set[str]:
    """Distinct proper nouns in a passage.

    A question like "which apps did students use" is answered by *names*, so a
    passage that contains several is a better citation than one that discusses
    the same topic abstractly. Sentence-initial words are skipped because
    English capitalises them regardless, and participant codes such as "S10"
    are skipped because every interview study is full of them.
    """
    found = set()
    for sentence in split_sentences(text):
        # Citation-dense prose is full of capitalised surnames. They are names,
        # but never the names a "which tools were used" question asks for.
        if _CITATION_DENSE.search(sentence) or len(_YEAR.findall(sentence)) >= 2:
            continue
        words = sentence.split()
        for position, word in enumerate(words):
            token = word.strip(".,;:()[]\"'?!")
            if len(token) < 3 or not token[0].isupper():
                continue
            if position == 0 or token in _NOT_A_NAME or _PARTICIPANT_CODE.match(token):
                continue
            # "Cakir, 2015" is a citation even when an abbreviation such as
            # "e.g." has split the sentence in the middle of one.
            following = " ".join(words[position + 1 : position + 3])
            if _CITATION_YEAR.match(following):
                continue
            if token.isupper() and len(token) > 5:
                continue
            found.add(token)
    return found


def selection_terms(question: str) -> Set[str]:
    """Question content words, minus interrogative quantifiers."""
    return {
        term for term in content_terms(question)
        if term not in _QUESTION_QUANTIFIERS
    }


def quantity_entity(question: str) -> Optional[str]:
    """The thing a counting question wants counted, or None."""
    match = _QUANTITY_QUESTION.search(question)
    if not match:
        return None
    for group in ("entity", "entity2", "entity3"):
        value = match.groupdict().get(group)
        if value:
            return value.rstrip("s")
    return None


def states_quantity_of(text: str, entity: str) -> bool:
    """True when a passage attaches a plausible count to the requested entity.

    Number presence alone is not enough: a paper is full of years, percentages
    and page numbers. The number has to sit beside the thing being counted, and
    must not be carrying a unit that makes it something else.
    """
    if not entity:
        return False
    stem_entity = entity.rstrip("s")
    near = r"[^.]{0,40}?"
    patterns = (
        r"\b(?P<n>" + _NUMBER + r")\s+" + near + stem_entity + r"s?\b",
        stem_entity + r"s?\b" + near + r"\b(?:were|was|is|are|:|totall\w+|comprised|included)?\s*(?P<n2>" + _NUMBER + r")\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            tail = text[match.end():].lstrip()
            if _NON_COUNT_UNIT.match(tail):
                continue
            return True
    return False


def process_family(question: str) -> Optional[str]:
    """Which family of study procedure a "how was X done" question asks about."""
    if not _PROCEDURAL_QUESTION.search(question):
        return None
    for name, pattern in PROCESS_FAMILIES:
        if re.search(r"\b(?:" + pattern + r")\w*", question, re.IGNORECASE):
            return name
    return None


def explains_process(text: str, family: str) -> bool:
    """True when a sentence says *how* or *why* a procedure was carried out.

    "The sample was chosen for convenience since the participants were
    accessible" explains recruitment; "twenty participants took part" does not,
    however much participant vocabulary it contains.
    """
    if not family:
        return False
    pattern = dict(PROCESS_FAMILIES)[family]
    verb = re.compile(r"\b(?:" + pattern + r")\w*", re.IGNORECASE)
    for sentence in split_sentences(text):
        if verb.search(sentence) and _EXPLANATORY.search(sentence):
            return True
    return False


def enumerates_named_items(text: str) -> bool:
    """True when a sentence actually lists several of the things being asked about.

    A count of proper nouns is not enough: a title block on page one is full of
    them. The sentence has to *list* - either with an enumeration marker
    ("such as", "including", "e.g.") or as a comma- or "and"-joined run of
    names. Citation-dense sentences are skipped so a literature review is not
    mistaken for a list of tools.
    """
    for sentence in split_sentences(text):
        if _CITATION_DENSE.search(sentence) or len(_YEAR.findall(sentence)) >= 2:
            continue
        names = named_items(sentence)
        if len(names) < 2:
            continue
        if _ENUMERATION.search(sentence):
            return True
        if _NAME_RUN.search(sentence):
            return True
    return False


def direct_answer_features(question: str, span: str) -> List[str]:
    """Which discrete answer-bearing features a span has, if any."""
    features = []
    if asks_for_named_items(question) and enumerates_named_items(span):
        features.append("enumerates")
    family = process_family(question)
    if family and explains_process(span, family):
        features.append("explains-{}".format(family))
    entity = quantity_entity(question)
    if entity and states_quantity_of(span, entity):
        features.append("states-quantity")
    return features


def asks_for_named_items(question: str) -> bool:
    """True for "which/what <thing>" questions our type taxonomy does not cover.

    Questions about limitations, methods, participants and so on already have a
    precise evidence family driving selection. What is left - "which apps",
    "what tools", "which databases" - is answered by naming things, and needs
    no list of noun classes to recognise: it is simply the residue.
    """
    if not re.match(r"^\s*(?:which|what)\b", question.strip(), re.IGNORECASE):
        return False
    return not any(
        question_type.precise for question_type, _ in detect_question_types(question)
    )


class SelectedEvidence(BaseModel):
    """One passage chosen to answer a question.

    The chunk keeps its identity - id, page, section come from the parser and
    are never touched. What changes is how much of it the model sees: instead
    of a whole 300-word Discussion chunk, the model gets the verbatim span of
    sentences that actually bear on the question. The span is also what the API
    returns as the evidence excerpt, so the reader sees exactly what the model
    saw.
    """

    chunk_id: str
    page_number: int
    section: str = ""
    span: str
    score: float = 0.0

    @classmethod
    def build(cls, chunk: RankedChunk, span: str, score: float) -> "SelectedEvidence":
        return cls(
            chunk_id=chunk.chunk_id,
            page_number=chunk.page_number,
            section=chunk.section,
            span=span,
            score=score,
        )


def sentence_relevance(
    sentence: str,
    terms: Set[str],
    entity: Optional[str],
    wants_names: bool,
    cue: Optional[Any],
    attribute: Optional[Set[str]],
) -> float:
    """How much one sentence bears on the question. Deterministic."""
    score = 0.0
    if terms:
        matched = terms & content_terms(sentence)
        score += W_TERM_COVERAGE * (len(matched) / float(len(terms)))
    if cue is not None and cue(sentence):
        score += W_TYPE_CUE
    if entity and states_quantity_of(sentence, entity):
        score += W_QUANTITY
    if attribute:
        sentence_terms = content_terms(sentence)
        if any(
            attribute_present(term, sentence_terms, sentence) for term in attribute
        ):
            score += W_ATTRIBUTE
    if wants_names:
        count = len(named_items(sentence))
        score += W_NAMED_ITEMS * min(1.0, count / float(NAMED_ITEM_SATURATION))
        if count >= 2 and _ENUMERATION.search(sentence):
            score += W_ENUMERATION
    return score


def _cue_for(precise_types):
    """A predicate testing a sentence against the question's evidence families."""

    def matches(text: str) -> bool:
        return any(t.evidence_present(text, None) for t in precise_types)

    return matches


def _question_signals(question: str):
    """Everything about a question that sentence scoring needs, computed once."""
    terms = selection_terms(question)
    entity = quantity_entity(question)
    wants_names = asks_for_named_items(question)

    precise = [t for t, _ in detect_question_types(question) if t.precise]
    cue = _cue_for(precise) if precise else None

    attribute = focus_terms(question) if asks_for_specific_attribute(question) else None
    return terms, entity, wants_names, cue, attribute


def evidence_span(chunk: RankedChunk, question: str) -> Tuple[str, float]:
    """The verbatim window of sentences in a chunk that answers the question.

    Returns the span and its relevance. A Discussion chunk that ends with the
    paper's limitations yields the limitation sentences, not the four hundred
    words of discussion before them, which is what let the model blend findings
    into a limitations answer.
    """
    sentences = split_sentences(chunk.text)
    if not sentences:
        return " ".join(chunk.text.split())[:MAX_SPAN_CHARS], 0.0

    terms, entity, wants_names, cue, attribute = _question_signals(question)
    relevance = [
        sentence_relevance(sentence, terms, entity, wants_names, cue, attribute)
        for sentence in sentences
    ]

    best_score, best_window = -1.0, (0, 1)
    for begin in range(len(sentences)):
        length = 0
        for finish in range(begin, min(begin + MAX_SPAN_SENTENCES, len(sentences))):
            length += len(sentences[finish]) + 1
            if length > MAX_SPAN_CHARS and finish > begin:
                break
            scores = relevance[begin : finish + 1]
            # Strongest sentence dominates; corroboration adds a little; a
            # sentence that bears nothing on the question costs a little. A
            # plain sum would reward longer windows, and a window that carries
            # unrelated prose into the model's context is the whole problem.
            best = max(scores)
            window = (
                best
                + CORROBORATION * (sum(scores) - best)
                - PADDING_PENALTY * sum(1 for value in scores if value <= 0.0)
            )
            # Strictly greater keeps the tightest window of equal relevance.
            if window > best_score + 1e-9:
                best_score, best_window = window, (begin, finish + 1)

    begin, finish = best_window
    span = " ".join(sentences[begin:finish]).strip()
    if len(span) > MAX_SPAN_CHARS:
        span = span[:MAX_SPAN_CHARS].rsplit(" ", 1)[0] + "..."
    return span, best_score


def content_contribution(
    question: str, chunk: RankedChunk, supporting: Set[str]
) -> float:
    """How much a passage contributes, judged on its best span.

    Scoring the span rather than the whole chunk is the point: a long chunk no
    longer wins by containing a little of everything, and the passage that
    directly answers the question wins even when retrieval ranked it lower.
    """
    _span, relevance = evidence_span(chunk, question)
    return relevance + (W_SUPPORTING if chunk.chunk_id in supporting else 0.0)


def select_evidence(
    report: AnswerabilityReport,
    ranked: Sequence[RankedChunk],
    question: str = "",
    limit: int = MAX_EVIDENCE_CHUNKS,
) -> List[SelectedEvidence]:
    """Choose the smallest set of source text that answers this question.

    Earlier versions took the first few answerability-supporting passages in
    retrieval order, which cited whatever cosine liked rather than whatever
    answered the question. Candidates are now judged on their best span, so
    "the participants were 20 Polish university students" beats a paragraph
    that merely uses the word participants.
    """
    if not ranked:
        return []

    supporting = set(report.supporting_chunk_ids)

    raw = []
    for rank, chunk in enumerate(ranked):
        span, relevance = evidence_span(chunk, question)
        content = relevance + (W_SUPPORTING if chunk.chunk_id in supporting else 0.0)
        raw.append((content, rank, chunk, span))

    # A passage must say something about the question to be worth citing;
    # a good retrieval rank on its own is not enough.
    contributing = [item for item in raw if item[0] > 0.0]
    if not contributing:
        contributing = [item for item in raw if item[2].chunk_id in supporting]
    if not contributing:
        contributing = raw[:1]

    # Normalise before combining, so neither family can swamp the other simply
    # by being measured on a bigger scale.
    ceiling = max(item[0] for item in contributing) or 1.0
    scored = [
        (
            CONTENT_SHARE * (content / ceiling)
            + RANK_SHARE * (1.0 / float(1 + rank)),
            content / ceiling,
            rank,
            chunk,
            span,
        )
        for content, rank, chunk, span in contributing
    ]
    scored.sort(key=lambda item: (-item[0], item[2]))

    # Stage 1: passages that carry a direct answer - they enumerate the
    # requested items, explain the requested procedure, or state the requested
    # count. Retrieval rank cannot promote a merely topical passage above one
    # of these, which is what let a literature review outrank the paragraph
    # naming the tools.
    direct, topical = [], []
    for item in scored:
        if direct_answer_features(question, item[4]):
            direct.append(item)
        else:
            topical.append(item)

    if direct:
        # Additional evidence is only genuinely complementary when it is
        # itself answer-bearing. A passage that merely shares the question's
        # topic adds nothing a reader can use, and asking the model to
        # attribute an answer across one real passage and one fragment is how
        # attribution comes back empty.
        contributing = direct[: max(1, limit)]
    else:
        # No passage answers outright. Fall back to contribution order, cut at
        # the largest drop so a question whose evidence is thin cites one
        # passage and one whose evidence is spread cites several.
        contributing = [item for item in scored if item[1] >= RELATIVE_FLOOR]
        contributing = (contributing or scored[:1])[: max(1, limit)]
        if len(contributing) > 1:
            drops = [
                (contributing[index - 1][1] - contributing[index][1], index)
                for index in range(1, len(contributing))
            ]
            biggest_drop, cut = max(drops, key=lambda pair: (pair[0], -pair[1]))
            if biggest_drop > 0:
                contributing = contributing[:cut]

    selected: List[SelectedEvidence] = []
    for total, _content, _rank, candidate, span in contributing:
        if any(_overlap(span, chosen.span) >= DUPLICATE_OVERLAP for chosen in selected):
            continue
        selected.append(SelectedEvidence.build(candidate, span, total))
        if len(selected) >= max(1, limit):
            break

    return selected


def evidence_snippet(text: str, question: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    """A short verbatim excerpt of a passage. Never written by the model."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rsplit(" ", 1)[0] + "..."


def render_evidence(evidence: Sequence[SelectedEvidence]) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            "[Evidence {index}]\n"
            "Page: {page}\n"
            "Chunk ID: {chunk_id}\n"
            "Text:\n{text}".format(
                index=index,
                page=item.page_number,
                chunk_id=item.chunk_id,
                text=item.span,
            )
        )
    return "\n\n".join(blocks)


def build_prompt(
    question: str, chunks: Sequence["SelectedEvidence"], partial: bool
) -> str:
    caution = ""
    if partial:
        caution = (
            "\nIMPORTANT: the evidence below only partially addresses this "
            "question. Say so plainly in your answer, use cautious wording "
            "such as 'the paper suggests' or 'based on the available "
            "evidence', and do not present a partial finding as a settled "
            "conclusion.\n"
        )

    return (
        "Answer the question using only the evidence passages below.\n"
        "{caution}\n"
        "RULES\n"
        "- Use only the passages provided. Do not use outside knowledge.\n"
        "- Do not infer facts the passages do not state.\n"
        "- If the passages do not fully answer the question, say what is "
        "missing rather than filling the gap.\n"
        "- Never invent names, numbers, methods, outcomes, citations or page "
        "numbers.\n"
        "- Preserve the source's own uncertainty: if the paper hedges, hedge.\n"
        "- Do not claim causation where the passages describe an association "
        "or a description.\n"
        "- Do not claim statistical significance unless a passage states it.\n"
        "- Page numbers are supplied metadata. Do not repeat or invent them in "
        "your answer.\n"
        "- Keep the answer to a few sentences.\n\n"
        "OUTPUT\n"
        'Return JSON with "answer" (your prose) and "used_chunk_ids" (the '
        "Chunk ID of every passage you actually used, copied exactly).\n\n"
        "{evidence}\n\n"
        "QUESTION:\n{question}"
    ).format(caution=caution, evidence=render_evidence(chunks), question=question)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


def _abstention(
    report: AnswerabilityReport,
    diagnostics: AskDiagnostics,
    answer: str = ABSTAIN_ANSWER,
    note: str = ABSTAIN_NOTE,
) -> AskResponse:
    return AskResponse(
        status=Answerability.NOT_SUPPORTED.value,
        answer=answer,
        sources=[],
        confidence_note=note,
        diagnostics=diagnostics,
    )


def _sources_for(
    evidence: Sequence["SelectedEvidence"], used: Sequence[str], question: str
) -> List[AnswerSource]:
    """Build citations from parser metadata for the passages actually used.

    The excerpt is the same span the model was shown, so a reader can check the
    answer against exactly the text that produced it.
    """
    by_id = {item.chunk_id: item for item in evidence}
    sources = []
    for chunk_id in used:
        item = by_id.get(chunk_id)
        if item is None:
            continue
        sources.append(
            AnswerSource(
                page=item.page_number,
                chunk_id=item.chunk_id,
                evidence=evidence_snippet(item.span, question),
                section=item.section,
            )
        )
    return sources


def validate_used_ids(
    produced: Sequence[str], selected: Sequence["SelectedEvidence"]
) -> Tuple[List[str], List[str]]:
    """Split model-produced ids into those we sent and those we did not."""
    allowed = [item.chunk_id for item in selected]
    known, unknown = [], []
    for chunk_id in produced:
        if chunk_id in allowed and chunk_id not in known:
            known.append(chunk_id)
        elif chunk_id not in allowed:
            unknown.append(chunk_id)
    # Report in the order the evidence was supplied, not the order the model
    # happened to mention them.
    known.sort(key=allowed.index)
    return known, unknown


async def generate_grounded_answer(
    question: str,
    answerability_result: AnswerabilityReport,
    ranked_evidence: Sequence[RankedChunk],
    diagnostics: Optional[AskDiagnostics] = None,
) -> AskResponse:
    """Turn a verdict plus evidence into an answer, or abstain.

    Never retrieves anything: everything it can cite is in ``ranked_evidence``.
    """
    diagnostics = diagnostics or AskDiagnostics()
    diagnostics.answerability_reason = answerability_result.reason
    diagnostics.retrieved_chunk_ids = [c.chunk_id for c in ranked_evidence]

    # ---- not_supported: the model is never called ----
    if answerability_result.status is Answerability.NOT_SUPPORTED:
        logger.info("Abstaining without generation: %s", answerability_result.reason)
        return _abstention(answerability_result, diagnostics)

    selected = select_evidence(answerability_result, ranked_evidence, question)
    diagnostics.selected_chunk_ids = [c.chunk_id for c in selected]
    if not selected:
        logger.warning("Answerable verdict but no evidence survived selection.")
        return _abstention(answerability_result, diagnostics)

    partial = answerability_result.status is Answerability.PARTIALLY_SUPPORTED

    diagnostics.generation_called = True
    raw = await generate_json(
        build_prompt(question, selected, partial),
        system=SYSTEM_PROMPT,
        schema=GeneratedAnswer.model_json_schema(),
        num_ctx=8192,
        num_predict=OLLAMA_NUM_PREDICT_ANSWER,
        label="ASK",
    )

    try:
        produced = GeneratedAnswer.model_validate(raw)
    except Exception as exc:
        raise OllamaError(
            "The model returned an answer in an unexpected shape ({}). "
            "Try asking again.".format(type(exc).__name__),
            502,
        )

    if not produced.answer:
        raise OllamaError("The model returned an empty answer.", 502)

    known, unknown = validate_used_ids(produced.used_chunk_ids, selected)
    if unknown:
        logger.warning(
            "Dropped %d chunk id(s) the model invented or altered.", len(unknown)
        )

    # ---- fail closed on unattributable answers ----
    if not known:
        if len(selected) == 1:
            # Only one passage was supplied, so attribution is unambiguous.
            # This is a deterministic fallback, not a guess.
            known = [selected[0].chunk_id]
            debug(logger, "ASK: single-evidence attribution fallback")
        else:
            logger.warning(
                "Discarding an answer that cited none of the %d supplied "
                "passages.",
                len(selected),
            )
            diagnostics.model_used_chunk_ids = []
            return _abstention(
                answerability_result,
                diagnostics,
                answer=UNATTRIBUTABLE_ANSWER,
                note=ABSTAIN_NOTE,
            )

    sources = _sources_for(selected, known, question)
    diagnostics.model_used_chunk_ids = known
    diagnostics.cited_pages = sorted({source.page for source in sources})

    return AskResponse(
        status=answerability_result.status.value,
        answer=produced.answer,
        sources=sources,
        confidence_note=PARTIAL_NOTE if partial else SUPPORTED_NOTE,
        diagnostics=diagnostics,
    )


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


async def answer_question(
    question: str,
    filename: str,
    pages: Sequence[PageInput],
    top_k: int = 5,
) -> AskResponse:
    """Run the frozen Phase 3A pipeline, then generate or abstain."""
    key = document_key(filename, pages)
    embedded = EMBEDDING_CACHE.get(key)
    cache_hit = embedded is not None

    if embedded is None:
        chunks = build_retrieval_chunks(pages, filename)
        embedded = await embed_chunks(chunks)
        EMBEDDING_CACHE.put(key, embedded)

    diagnostics = AskDiagnostics(embedding_cache_hit=cache_hit)

    if not embedded:
        report = AnswerabilityReport(
            status=Answerability.NOT_SUPPORTED,
            reason="The paper produced no passages that could be searched.",
        )
        diagnostics.answerability_reason = report.reason
        return _abstention(report, diagnostics)

    # The answerability verdict is taken from exactly the window Phase 3A was
    # accepted on. Nothing about that call changes.
    ranking = await retrieve_evidence(question, embedded, top_k=top_k)
    report = verify_answerability(question, ranking)
    answerable = report.status is not Answerability.NOT_SUPPORTED

    diagnostics.retrieval_top_score = ranking.diagnostics.top_score
    diagnostics.retrieval_score_gap = ranking.diagnostics.score_gap
    diagnostics.retrieval_mean_top_k = ranking.diagnostics.mean_top_k

    # Evidence selection gets a wider view. A passage that names exactly what a
    # question asks for can sit just outside the answerability window: the
    # verdict does not need it, but the citation does. This is a second ranking
    # call rather than a slice of a wider one, so the verdict above is bit for
    # bit what it was before. The cost is one extra question embedding.
    candidates = ranking.results
    if answerable and len(embedded) > top_k:
        wider = await retrieve_evidence(
            question, embedded, top_k=max(top_k, SELECTION_POOL_SIZE)
        )
        candidates = wider.results

    return await generate_grounded_answer(question, report, candidates, diagnostics)


def cache_stats() -> Dict[str, int]:
    return EMBEDDING_CACHE.stats()
