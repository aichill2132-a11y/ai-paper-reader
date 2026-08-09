"""Regression tests for the three real-paper evidence failures.

Each failure sat at a different pipeline stage, so each is tested at that
stage: the age failure against the verifier, the endnote failure against
parsing and eligibility, the limitations failure against ranking.
"""

import pytest

from answerability import (
    Answerability,
    attribute_present,
    content_terms,
    verify_answerability,
)
from conftest import embedded_interview_corpus  # noqa: F401
from embeddings import RankedChunk, RetrievalDiagnostics, RetrievalResult
from evidence_ranking import (
    carries_question_evidence,
    rerank_by_evidence_cues,
    retrieve_evidence,
)
from retrieval import build_retrieval_chunks, eligible_chunks, exclusion_reason
from schemas import PageInput
from sections import is_substantive_note, parse_sections


def chunk(text, page=4, section="participants", score=0.5, chunk_id=None):
    return RankedChunk(
        chunk_id=chunk_id or "p{:04d}-{}".format(page, abs(hash(text)) % 10 ** 10),
        page_number=page,
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        score=score,
    )


def retrieval(chunks, top=0.6, second=0.3):
    return RetrievalResult(
        results=list(chunks),
        diagnostics=RetrievalDiagnostics(
            considered=len(chunks),
            returned=len(chunks),
            top_score=top,
            second_score=second,
            score_gap=top - second,
            mean_top_k=sum(c.score for c in chunks) / len(chunks) if chunks else None,
        ),
    )


# --------------------------------------------------------------------------- #
# Failure 1: age reported as "N years old"
# --------------------------------------------------------------------------- #

AGE_AS_YEARS_OLD = (
    "The participants were 20 Polish university students. The study "
    "participants were on average 22.22 years old and had been learning "
    "English for 11 years."
)


def test_mean_age_is_answerable_when_reported_as_years_old():
    """The reported false negative: the paper never writes the word "age"."""
    assert "age" not in AGE_AS_YEARS_OLD.lower().replace("average", "")

    report = verify_answerability(
        "What was the participants' mean age?",
        retrieval([chunk(AGE_AS_YEARS_OLD, page=4)]),
    )
    assert report.status is not Answerability.NOT_SUPPORTED
    assert 4 in report.supporting_pages


@pytest.mark.parametrize(
    "question,evidence",
    [
        ("What was the average age?", "The mean age was 24 years."),
        ("What was the mean age?", "Respondents were on average 24 years old."),
        ("What was the participants' age?", "Participants ranged from 18 to 24 years old."),
        ("What was the mean age of the sample?", "Interviewees were aged between 30 and 45."),
        ("What was the participants' median age?", "Median age was 31.4 years of age."),
    ],
)
def test_standard_age_constructions_are_recognised(question, evidence):
    report = verify_answerability(question, retrieval([chunk(evidence)]))
    assert report.status is not Answerability.NOT_SUPPORTED, evidence


UNRELATED_NUMBERS = (
    "Twenty participants took part. Nine were female and eleven were male. "
    "They had studied English for 11 years and were all year 2 students at "
    "the same institution."
)


def test_unrelated_numbers_do_not_satisfy_an_age_question():
    """Counts, durations and cohort labels are not ages."""
    report = verify_answerability(
        "What was the participants' mean age?",
        retrieval([chunk(UNRELATED_NUMBERS, page=4)]),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "never report" in report.reason


def test_the_age_construction_requires_a_measurement_context():
    terms = content_terms(UNRELATED_NUMBERS)
    assert not attribute_present("age", terms, UNRELATED_NUMBERS)
    assert attribute_present("age", content_terms(AGE_AS_YEARS_OLD), AGE_AS_YEARS_OLD)


def test_other_attributes_are_still_rejected_on_the_same_passage():
    """The equivalence table must not become a blanket pass."""
    for question in (
        "What was the participants' mean IQ?",
        "What was the average BMI of the sample?",
        "What was the mean depression score?",
    ):
        assert (
            verify_answerability(question, retrieval([chunk(AGE_AS_YEARS_OLD)])).status
            is Answerability.NOT_SUPPORTED
        ), question


# --------------------------------------------------------------------------- #
# Failure 2: substantive notes printed after the bibliography
# --------------------------------------------------------------------------- #

BIBLIOGRAPHY_WITH_NOTES = [
    PageInput(
        page_number=11,
        text=(
            "References\n"
            "Smith, J. (2019). Sampling participants for interview research. "
            "Journal of Method, 12, 1-14.\n"
            "Jones, A. (2020). Recruitment strategies in applied linguistics. "
            "Applied Review, 8, 44-60.\n"
            "Brown, K. (2021). A study of learner samples. System, 90, 5-19.\n"
            "[1] The sample was selected for convenience because the "
            "participants were accessible to the researcher.\n"
        ),
    )
]


def note_chunks():
    return build_retrieval_chunks(BIBLIOGRAPHY_WITH_NOTES, "paper.pdf")


def test_a_numbered_note_after_the_bibliography_becomes_its_own_section():
    sections = {s.name: s for s in parse_sections([(11, BIBLIOGRAPHY_WITH_NOTES[0].text)])}
    assert "references" in sections and "notes" in sections
    assert "Smith" in sections["references"].text
    assert "Smith" not in sections["notes"].text
    assert "convenience" in sections["notes"].text


def test_bibliography_entries_stay_excluded_even_when_they_mention_sampling():
    """These entries contain "sampling", "participants" and "recruitment"."""
    chunks = note_chunks()
    references = [c for c in chunks if c.section == "references"]
    assert references
    for reference in references:
        assert exclusion_reason(reference) == "references"

    eligible_text = " ".join(c.text for c in eligible_chunks(chunks))
    for surname in ("Smith", "Jones", "Brown"):
        assert surname not in eligible_text


def test_the_note_is_evidence_eligible():
    notes = [c for c in note_chunks() if c.section == "notes"]
    assert len(notes) == 1
    assert exclusion_reason(notes[0]) is None
    assert notes[0].page_number == 11


def test_a_recruitment_question_can_be_answered_from_the_note():
    notes = [c for c in note_chunks() if c.section == "notes"]
    report = verify_answerability(
        "How were participants recruited?",
        retrieval([chunk(notes[0].text, page=11, section="notes")]),
    )
    assert report.status is not Answerability.NOT_SUPPORTED
    assert 11 in report.supporting_pages


def test_a_reference_entry_cannot_become_supporting_evidence():
    entry = chunk(
        "Smith, J. (2019). Sampling participants for interview research. "
        "Journal of Method, 12, 1-14.",
        page=11,
        section="references",
    )
    assert exclusion_reason(entry) == "references"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("[1] The sample was chosen because participants were available.", True),
        ("[2] We asked learners to keep a diary since recall was unreliable.", True),
        ("Smith, J. (2019). Sampling participants. Journal, 12, 1-14.", False),
        ("1. Jones, A. (2020). Recruitment strategies. Review, 8, 44-60.", False),
        ("[3] Amano, T. (2016) Languages are a barrier. PLoS Biology 14, e2000933.", False),
        ("[4] See above.", False),
    ],
)
def test_notes_are_distinguished_from_citations_by_shape(line, expected):
    assert is_substantive_note(line) is expected


def test_a_paper_whose_references_run_to_the_end_is_unchanged():
    """No notes, no behaviour change."""
    pages = [
        PageInput(
            page_number=9,
            text=(
                "References\n"
                "Smith, J. (2019). A study of things. Journal, 12, 1-14.\n"
                "Jones, A. (2020). Another study. Review, 8, 44-60.\n"
            ),
        )
    ]
    chunks = build_retrieval_chunks(pages, "paper.pdf")
    assert {c.section for c in chunks} == {"references"}
    assert eligible_chunks(chunks) == []


def test_note_chunk_ids_stay_deterministic():
    first = note_chunks()
    second = note_chunks()
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


# --------------------------------------------------------------------------- #
# Failure 3: limitations diluted inside a long Discussion chunk
# --------------------------------------------------------------------------- #

LIMITATION_TAIL = (
    "This study has several limitations. The small number of participants "
    "reduces the generalizability of the findings, and the group was largely "
    "homogeneous."
)
GENERIC_BACKGROUND = (
    "Prior research on autonomy and technology has described broadly similar "
    "patterns of study behaviour across higher education settings."
)


def test_a_diluted_limitation_chunk_is_rescued_from_outside_the_top_k():
    """The core of the ranking failure.

    Dense similarity puts the limitation chunk at rank 8 because it is mostly
    unrelated discussion. Reranking must pull it into the top 5 without
    reordering anything else.
    """
    ranked = [
        chunk(GENERIC_BACKGROUND, page=p, section="background", score=0.54 - p * 0.001)
        for p in range(1, 8)
    ] + [chunk(LIMITATION_TAIL, page=9, section="discussion", score=0.48)]

    reordered = rerank_by_evidence_cues("What limitations did the authors identify?", ranked)

    assert reordered[0].page_number == 9
    assert [c.page_number for c in reordered[1:]] == [c.page_number for c in ranked[:-1]]


def test_reranking_leaves_an_already_correct_ranking_alone():
    ranked = [
        chunk(LIMITATION_TAIL, page=9, section="discussion", score=0.6),
        chunk(GENERIC_BACKGROUND, page=2, section="background", score=0.5),
    ]
    reordered = rerank_by_evidence_cues("What limitations did the authors identify?", ranked)
    assert [c.page_number for c in reordered] == [9, 2]


def test_reranking_is_inert_for_broad_question_types():
    """"findings" matches almost any prose, so promoting on it means nothing."""
    ranked = [
        chunk(GENERIC_BACKGROUND, page=2, section="background", score=0.6),
        chunk(LIMITATION_TAIL, page=9, section="discussion", score=0.5),
    ]
    assert rerank_by_evidence_cues("What did the study find?", ranked) == ranked


def test_cue_detection_ignores_imprecise_types():
    assert carries_question_evidence(
        "What limitations did the authors identify?",
        chunk(LIMITATION_TAIL, section="discussion"),
    )
    assert not carries_question_evidence(
        "What limitations did the authors identify?",
        chunk(GENERIC_BACKGROUND, section="background"),
    )


async def test_limitations_reach_the_verifier_end_to_end(hashing_ollama):
    """Chunking, eligibility, candidate pool, ranking, then verification."""
    embedded = await embedded_interview_corpus()
    question = "What limitations did the authors identify?"

    ranking = await retrieve_evidence(question, embedded, top_k=5)
    pages = [c.page_number for c in ranking.results]
    assert 9 in pages or 10 in pages, pages

    report = verify_answerability(question, ranking)
    assert report.status is not Answerability.NOT_SUPPORTED
    assert set(report.supporting_pages) & {9, 10}


async def test_diagnostics_never_report_a_negative_gap(hashing_ollama):
    """Reordering must not be read back as a score ordering."""
    embedded = await embedded_interview_corpus()
    for question in (
        "What limitations did the authors identify?",
        "How were participants recruited?",
        "What was the participants' mean age?",
    ):
        gap = (await retrieve_evidence(question, embedded, top_k=5)).diagnostics.score_gap
        assert gap is None or gap >= 0, (question, gap)


# --------------------------------------------------------------------------- #
# The property that matters most: no new false positives
# --------------------------------------------------------------------------- #


async def test_unsupported_questions_stay_unsupported_after_reranking(hashing_ollama):
    embedded = await embedded_interview_corpus()
    for question in (
        "What was the participants' mean IQ?",
        "What was the average BMI of the sample?",
        "What was the participants' mean household income?",
        "What was the mean depression score?",
        "What intervention improved test scores?",
        "Which questionnaire did participants complete?",
    ):
        ranking = await retrieve_evidence(question, embedded, top_k=5)
        assert (
            verify_answerability(question, ranking).status
            is Answerability.NOT_SUPPORTED
        ), question


def test_subject_area_is_not_mistaken_for_a_human_subject():
    """A bibliometric paper has "subject categories", not research subjects."""
    report = verify_answerability(
        "How many human participants were recruited?",
        retrieval(
            [
                chunk(
                    "Each article was assigned to one subject category, and "
                    "counts were aggregated by subject field and by year.",
                    page=2,
                    section="data_collection_and_analysis",
                )
            ]
        ),
    )
    assert report.status is Answerability.NOT_SUPPORTED


# --------------------------------------------------------------------------- #
# False-positive guards found while making the three fixes
# --------------------------------------------------------------------------- #

OBSERVATIONAL_STUDY = (
    "The participants were 20 university students enrolled on a philology "
    "programme. All of them were year two students at the time of the "
    "interviews, which were audio recorded and transcribed."
)


@pytest.mark.parametrize(
    "question",
    [
        "What was the control group size?",
        "Which randomized treatment condition performed best?",
        "What intervention improved participants' test scores?",
    ],
)
def test_experimental_questions_are_rejected_by_an_observational_paper(question):
    """A philology "programme" is not an intervention, and a "condition" is
    not an experimental arm. Bare design words made an interview study look
    like a trial."""
    report = verify_answerability(
        question, retrieval([chunk(OBSERVATIONAL_STUDY, page=4)])
    )
    assert report.status is Answerability.NOT_SUPPORTED, report.reason


def test_a_real_experimental_paper_still_answers_them():
    trial = (
        "Participants were randomly assigned to an intervention group or a "
        "control group of 30 each. Post-test scores rose in the treatment "
        "group relative to controls."
    )
    report = verify_answerability(
        "What was the control group size?", retrieval([chunk(trial, page=4)])
    )
    assert report.status is not Answerability.NOT_SUPPORTED


# --------------------------------------------------------------------------- #
# Aim / research-question evidence
#
# The failing real paper states its aims as a purpose construction attached to
# the current article, not as an aim noun and not with one of the five verbs
# the old pattern knew about.
# --------------------------------------------------------------------------- #

from answerability import aims_evidence_present  # noqa: E402

# Verbatim from the publication-language PDF, page 2.
REAL_AIM_SENTENCE = (
    "This article reports on a short study using Scopus data to determine "
    "(a) whether the use of languages other than English for scientific "
    "communication is increasing or decreasing, and (b) in which subject "
    "fields researchers publish most when publishing in their native "
    "languages instead of in English."
)


def test_the_real_papers_aim_wording_is_recognised():
    assert aims_evidence_present(REAL_AIM_SENTENCE)


def test_the_real_aim_makes_the_question_answerable():
    report = verify_answerability(
        "What were the main research questions?",
        retrieval([chunk(REAL_AIM_SENTENCE, page=2, section="")]),
    )
    assert report.status is not Answerability.NOT_SUPPORTED
    assert 2 in report.supporting_pages


@pytest.mark.parametrize(
    "sentence",
    [
        # explicit research-question wording
        "The research questions were whether uptake differs by discipline.",
        "Our main research question concerned publication language.",
        # explicit aim / objective nouns
        "The aim of this study was to compare two rostering systems.",
        "The purpose of this study was to establish a baseline.",
        # current-study objective phrasing
        "This study examines how learners use handheld devices.",
        "This article investigates publication language across fields.",
        "The present paper seeks to identify barriers to uptake.",
        "This analysis sets out to compare two indexes.",
        # first person
        "We sought to determine whether mentorship reduces anxiety.",
        "We decided to replicate this analysis, to determine whether the "
        "trend has continued.",
        # passive current-study
        "The study was conducted to determine the prevalence of the condition.",
        "A survey was carried out to identify barriers to reporting.",
    ],
)
def test_legitimate_aim_constructions_are_recognised(sentence):
    assert aims_evidence_present(sentence), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        # previous-study / background phrasing
        "Previous studies examined whether the use of English is increasing.",
        "Earlier work has explored this question in other settings.",
        "Prior research investigated the same relationship.",
        "Other researchers have set out to determine the same thing.",
        "Research suggests that authors publish in their native language.",
        # cited work
        "Smith (2019) investigated language choice among researchers.",
        "Kruk et al. examined mobile device use among advanced learners.",
        "According to Jones, the effect is driven by indexing coverage.",
        # no aim at all
        "The data were collected in 2011 and are presented in Table 1.",
        "Figure 1 shows that the use of English has continued to rise.",
        "Participants were interviewed once during the spring semester.",
    ],
)
def test_background_and_cited_work_are_not_aims(sentence):
    assert not aims_evidence_present(sentence), sentence


def test_a_paper_with_no_stated_aim_stays_not_supported():
    body = (
        "Figure 1 shows that the use of English has continued to rise. "
        "Table 1 provides an overview of the percentage of articles published "
        "in each language between 1996 and 2011."
    )
    report = verify_answerability(
        "What were the main research questions?",
        retrieval([chunk(body, page=3, section="findings")]),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "aim or research question" in report.reason


def test_a_paper_that_only_cites_others_aims_stays_not_supported():
    """The false positive this rule exists to prevent."""
    body = (
        "Previous studies examined whether English dominance is increasing. "
        "Smith (2019) investigated language choice, and Kruk et al. explored "
        "device use among learners."
    )
    report = verify_answerability(
        "What were the main research questions?",
        retrieval([chunk(body, page=2, section="background")]),
    )
    assert report.status is Answerability.NOT_SUPPORTED


def test_an_aim_beside_cited_work_is_still_found():
    """Sentence-level checking: one background sentence must not mask an aim."""
    body = (
        "Previous studies examined this question. Smith (2019) investigated "
        "language choice. " + REAL_AIM_SENTENCE
    )
    assert aims_evidence_present(body)
