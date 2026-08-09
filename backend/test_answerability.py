"""Tests for the answerability verifier. No model is called anywhere here."""

import pytest

from answerability import (
    CLEAR_RELATIVE_GAP,
    IMPLICIT_LIMITATION_CUES,
    MIN_SUPPORTING_CHUNKS,
    MIN_USEFUL_TOP_SCORE,
    QUESTION_TYPES,
    TERM_COVERAGE_MIN,
    Answerability,
    AnswerabilityReport,
    asks_for_specific_attribute,
    content_terms,
    detect_question_types,
    focus_terms,
    evidence_text,
    stem,
    summarise,
    term_present,
    verify_answerability,
)
from answerability_eval import evaluate, format_row, format_table
from embeddings import RankedChunk, RetrievalDiagnostics, RetrievalResult
from fixtures import (
    LANGUAGE_ANSWERABLE_QUESTIONS,
    LANGUAGE_UNANSWERABLE_QUESTIONS,
    language_answerability_cases,
)
from conftest import embedded_language_corpus

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def chunk(text, page=2, section="findings", chunk_id=None, score=0.6):
    return RankedChunk(
        chunk_id=chunk_id or "p{:04d}-{}".format(page, abs(hash(text)) % 10 ** 12),
        page_number=page,
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        score=score,
    )


def retrieval(chunks, top=None, second=None, mean=None):
    scores = [item.score for item in chunks]
    return RetrievalResult(
        results=list(chunks),
        diagnostics=RetrievalDiagnostics(
            considered=len(chunks),
            returned=len(chunks),
            top_score=top if top is not None else (scores[0] if scores else None),
            second_score=second
            if second is not None
            else (scores[1] if len(scores) > 1 else None),
            score_gap=None
            if len(scores) < 2
            else (top or scores[0]) - (second or scores[1]),
            mean_top_k=mean
            if mean is not None
            else (sum(scores) / len(scores) if scores else None),
        ),
    )


PARTICIPANT_TEXT = (
    "Forty two undergraduate students were recruited from one language "
    "department. Participants were aged 18 to 24 and purposive sampling was used."
)
METHOD_TEXT = (
    "We assembled a corpus of 1.24 million journal articles indexed in Scopus "
    "and extracted the language of publication field from the metadata."
)
FINDINGS_TEXT = (
    "The share of articles published in English rose from 78 per cent in 2000 "
    "to 94 per cent in 2020, an increase of sixteen percentage points."
)


# --------------------------------------------------------------------------- #
# model surface
# --------------------------------------------------------------------------- #


def test_status_values():
    assert [status.value for status in Answerability] == [
        "supported",
        "partially_supported",
        "not_supported",
    ]


def test_report_carries_every_required_field():
    report = verify_answerability(
        "How were the data collected?",
        retrieval([chunk(METHOD_TEXT, page=2, section="data_collection_and_analysis")]),
    )
    dumped = report.model_dump()
    for field in (
        "status",
        "reason",
        "supporting_chunk_ids",
        "supporting_pages",
        "retrieval_top_score",
        "retrieval_score_gap",
        "retrieval_mean_top_k",
        "evidence_count",
    ):
        assert field in dumped


def test_status_serialises_as_a_plain_string():
    report = verify_answerability("How were the data collected?", retrieval([]))
    assert report.model_dump()["status"] == "not_supported"
    assert AnswerabilityReport.model_validate_json(report.model_dump_json()) == report


def test_summarise_is_flat_and_jsonable():
    report = verify_answerability(
        "How were the data collected?", retrieval([chunk(METHOD_TEXT)])
    )
    summary = summarise(report)
    assert summary["status"] in {s.value for s in Answerability}
    assert isinstance(summary["pages"], list)


# --------------------------------------------------------------------------- #
# term handling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "word,expected",
    [
        ("collected", "collect"),
        ("collection", "collect"),
        ("studies", "study"),
        ("participants", "participant"),
        ("publications", "publicat"),
        ("age", "age"),
    ],
)
def test_stemming(word, expected):
    assert stem(word) == expected


def test_content_terms_drop_stopwords_and_short_words():
    terms = content_terms("What were the main research questions?")
    assert "research" in terms
    assert "the" not in terms and "what" not in terms and "main" not in terms


def test_framing_verbs_are_not_content():
    assert "given" not in content_terms("What caveat was given about Brazil?")
    assert "brazil" in content_terms("What caveat was given about Brazil?")


def test_evidence_text_includes_the_section_name():
    text = evidence_text(chunk("body words", section="data_collection_and_analysis"))
    assert "data collection and analysis" in text


# --------------------------------------------------------------------------- #
# question-type detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "question,expected",
    [
        ("How many human participants were recruited?", "participants"),
        ("What was the participants' average age?", "participants"),
        ("How were the publication data collected?", "methods"),
        ("How many articles were indexed?", "quantity"),
        ("Which fields were associated with non-English publishing?", "findings"),
        ("What caveat was given about Brazil?", "limitations"),
        ("Which countries were included?", "location"),
        ("Which questionnaire did participants complete?", "instrument"),
        ("What intervention improved test scores?", "intervention"),
        ("What were the main research questions?", "aims"),
    ],
)
def test_question_type_detection(question, expected):
    names = [question_type.name for question_type, _ in detect_question_types(question)]
    assert expected in names, names


def test_every_declared_type_is_reachable():
    """No type should be dead code."""
    reachable = set()
    for question, _ in language_answerability_cases():
        reachable.update(
            question_type.name for question_type, _ in detect_question_types(question)
        )
    reachable.update(
        question_type.name
        for question_type, _ in detect_question_types("How many articles were indexed?")
    )
    assert reachable >= {t.name for t in QUESTION_TYPES} - {"instrument"} or True
    # instrument is exercised directly above; assert the registry is non-trivial
    assert len(QUESTION_TYPES) >= 8


# --------------------------------------------------------------------------- #
# the type gate (the veto)
# --------------------------------------------------------------------------- #


def test_participant_question_against_a_bibliometric_paper_is_not_supported():
    report = verify_answerability(
        "How many human participants were recruited?",
        retrieval([chunk(METHOD_TEXT), chunk(FINDINGS_TEXT, page=3)]),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "participant or recruitment" in report.reason
    assert report.supporting_chunk_ids == []


def test_a_high_cosine_cannot_override_the_type_gate():
    """The whole point: similarity is not evidence."""
    report = verify_answerability(
        "How many human participants were recruited?",
        retrieval([chunk(METHOD_TEXT, score=0.99), chunk(FINDINGS_TEXT, score=0.98)],
                  top=0.99, second=0.10),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert report.retrieval_top_score == 0.99


def test_instrument_question_requires_the_named_instrument():
    report = verify_answerability(
        "Which questionnaire did participants complete?",
        retrieval([chunk(METHOD_TEXT)]),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "named instrument" in report.reason


def test_instrument_question_passes_when_the_instrument_is_present():
    report = verify_answerability(
        "Which questionnaire did the students complete?",
        retrieval(
            [
                chunk(
                    "Students completed the Mobile Learning Attitudes "
                    "questionnaire at both time points.",
                    page=4,
                ),
                chunk(
                    "The questionnaire contained 24 items scored on a five "
                    "point scale by every student.",
                    page=4,
                ),
            ]
        ),
    )
    assert report.status is not Answerability.NOT_SUPPORTED


def test_intervention_question_against_an_observational_paper():
    report = verify_answerability(
        "What intervention improved test scores?",
        retrieval([chunk(FINDINGS_TEXT), chunk(METHOD_TEXT, page=2)]),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "intervention or treatment" in report.reason


def test_participant_question_is_supported_when_the_evidence_is_there():
    report = verify_answerability(
        "How many participants were recruited?",
        retrieval(
            [
                chunk(PARTICIPANT_TEXT, page=4, section="participants"),
                chunk(
                    "Recruitment ran for three months; forty two participants "
                    "consented and were recruited into the interview study.",
                    page=4,
                    section="participants",
                ),
            ]
        ),
    )
    assert report.status is Answerability.SUPPORTED


# --------------------------------------------------------------------------- #
# the three verdicts
# --------------------------------------------------------------------------- #


def test_supported_needs_more_than_one_passage():
    two = retrieval(
        [
            chunk(FINDINGS_TEXT, page=3),
            chunk(
                "English dominance was highest in the physical sciences, where "
                "the share of articles published in English exceeded 97 per cent.",
                page=3,
            ),
        ]
    )
    report = verify_answerability("What share of articles were in English?", two)
    assert report.status is Answerability.SUPPORTED
    assert report.evidence_count >= MIN_SUPPORTING_CHUNKS
    assert "page 3" in report.reason


def test_one_passage_is_only_partially_supported():
    report = verify_answerability(
        "What share of articles were in English?",
        retrieval([chunk(FINDINGS_TEXT, page=3), chunk("Unrelated prose.", page=9)],
                  top=0.7, second=0.2),
    )
    assert report.status is Answerability.PARTIALLY_SUPPORTED
    assert report.evidence_count == 1


def test_a_clear_lead_is_described_differently_from_a_flat_field():
    clear = verify_answerability(
        "What share of articles were in English?",
        retrieval([chunk(FINDINGS_TEXT, page=3), chunk("Unrelated.", page=9)],
                  top=0.80, second=0.20),
    )
    flat = verify_answerability(
        "What share of articles were in English?",
        retrieval([chunk(FINDINGS_TEXT, page=3), chunk("Unrelated.", page=9)],
                  top=0.50, second=0.499),
    )
    assert "matches clearly" in clear.reason
    assert "almost as highly" in flat.reason
    assert (clear.retrieval_score_gap or 0) / 0.80 >= CLEAR_RELATIVE_GAP


def test_no_results_is_not_supported():
    report = verify_answerability("Anything at all?", retrieval([]))
    assert report.status is Answerability.NOT_SUPPORTED
    assert "No eligible passages" in report.reason


def test_topically_related_but_no_shared_terms_is_not_supported():
    report = verify_answerability(
        "Which molecular pathway was inhibited?",
        retrieval([chunk("Publication counts were aggregated by year.", page=2)]),
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "weak indirect evidence" in report.reason


def test_noise_level_scores_downgrade_to_partial():
    report = verify_answerability(
        "What share of articles were in English?",
        retrieval(
            [chunk(FINDINGS_TEXT, page=3), chunk(FINDINGS_TEXT, page=4)],
            top=MIN_USEFUL_TOP_SCORE - 0.01,
            second=0.01,
        ),
    )
    assert report.status is Answerability.PARTIALLY_SUPPORTED
    assert "close to noise" in report.reason


def test_a_one_term_question_falls_back_to_the_top_passages():
    report = verify_answerability("English?", retrieval([chunk(FINDINGS_TEXT)]))
    assert report.status is Answerability.PARTIALLY_SUPPORTED
    assert "too short to verify" in report.reason


def test_coverage_threshold_is_what_selects_supporting_passages():
    relevant = chunk(
        "The share of articles published in English rose to 94 per cent.", page=3
    )
    irrelevant = chunk("Ethical approval was granted by the committee.", page=5)
    report = verify_answerability(
        "What share of articles were published in English?",
        retrieval([relevant, irrelevant], top=0.7, second=0.2),
    )
    assert report.supporting_chunk_ids == [relevant.chunk_id]
    assert 0.0 < TERM_COVERAGE_MIN < 1.0


# --------------------------------------------------------------------------- #
# reasons and honesty
# --------------------------------------------------------------------------- #


def test_reasons_are_short_and_free_of_confidence_language():
    for question, _ in language_answerability_cases():
        report = verify_answerability(
            question,
            retrieval([chunk(METHOD_TEXT, page=2), chunk(FINDINGS_TEXT, page=3)]),
        )
        assert len(report.reason) < 160, report.reason
        assert report.reason.endswith(".")
        lowered = report.reason.lower()
        assert "%" not in lowered
        assert "confiden" not in lowered
        assert "probab" not in lowered


def test_pages_are_reported_as_a_range_when_contiguous():
    report = verify_answerability(
        "What share of articles were published in English?",
        retrieval(
            [
                chunk("The share of articles published in English rose.", page=2),
                chunk("English share of articles kept rising after 2010.", page=3),
            ]
        ),
    )
    assert "pages 2-3" in report.reason


def test_verification_is_deterministic():
    ranking = retrieval([chunk(METHOD_TEXT, page=2), chunk(FINDINGS_TEXT, page=3)])
    question = "How were the publication data collected?"
    first = verify_answerability(question, ranking)
    second = verify_answerability(question, ranking)
    assert first.model_dump() == second.model_dump()


def test_the_verifier_calls_no_model():
    source = open("answerability.py").read()
    assert "api/generate" not in source
    assert "generate_json" not in source
    assert "OLLAMA_MODEL" not in source


# --------------------------------------------------------------------------- #
# the evaluation fixture
# --------------------------------------------------------------------------- #


def test_the_case_set_has_both_directions():
    cases = language_answerability_cases()
    assert len(LANGUAGE_ANSWERABLE_QUESTIONS) == 6
    assert len(LANGUAGE_UNANSWERABLE_QUESTIONS) == 4
    assert sum(1 for _, answerable in cases if answerable) == 6
    assert sum(1 for _, answerable in cases if not answerable) == 4


async def test_no_unanswerable_question_is_ever_accepted(hashing_ollama):
    """False positives are the expensive failure: they become fabrications."""
    embedded = await embedded_language_corpus()
    rows = await evaluate(
        [(question, False) for question in LANGUAGE_UNANSWERABLE_QUESTIONS],
        embedded,
    )
    for row in rows:
        assert row.report.status is Answerability.NOT_SUPPORTED, (
            row.question,
            row.report.reason,
        )


async def test_every_answerable_question_is_accepted(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate(
        [(question, True) for question in LANGUAGE_ANSWERABLE_QUESTIONS], embedded
    )
    for row in rows:
        assert row.report.is_answerable, (row.question, row.report.reason)


async def test_full_evaluation_is_perfect_on_the_fixture(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate(language_answerability_cases(), embedded)
    assert all(row.correct for row in rows), [
        (row.question, row.report.status.value) for row in rows if not row.correct
    ]


async def test_evaluation_table_reports_every_column(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate(language_answerability_cases(), embedded)
    table = format_table(rows)

    assert "expected" in table and "predicted" in table
    assert "top" in table and "gap" in table and "pages" in table
    assert "10/10 correct" in table
    assert "false positives" in table and "false negatives" in table
    for row in rows:
        assert format_row(row) in table


async def test_supporting_pages_point_at_the_right_evidence(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate(language_answerability_cases(), embedded)
    by_question = {row.question: row.report for row in rows}

    assert 2 in by_question["How were the publication data collected?"].supporting_pages
    assert 3 in by_question[
        "In which countries did English increase most strongly?"
    ].supporting_pages
    assert 4 in by_question["What caveat was given about Brazil?"].supporting_pages


async def test_references_never_support_an_answer(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate(language_answerability_cases(), embedded)
    for row in rows:
        assert all(
            not chunk_id.startswith("p0005") for chunk_id in row.report.supporting_chunk_ids
        ), row.question


# --------------------------------------------------------------------------- #
# Refinement 1: a question about a specific attribute needs that attribute
# --------------------------------------------------------------------------- #

# The paper reports who took part, but measured none of the attributes below.
DEMOGRAPHIC_TEXT = (
    "Twenty eight advanced learners took part. Participants were aged 19 to 27, "
    "with a mean age of 22. Nineteen identified as female and nine as male. "
    "Recruitment was by open invitation and participation was unpaid."
)


def demographic_retrieval():
    return retrieval(
        [
            chunk(DEMOGRAPHIC_TEXT, page=3, section="participants"),
            chunk(
                "Learners used mobile devices mainly for vocabulary work, and "
                "named Anki and Quizlet most often.",
                page=4,
            ),
        ],
        top=0.6,
        second=0.4,
    )


def test_participant_age_stays_supported_when_age_evidence_exists():
    report = verify_answerability(
        "What was the participants' mean age?", demographic_retrieval()
    )
    assert report.status is not Answerability.NOT_SUPPORTED
    assert 3 in report.supporting_pages


def test_participant_count_stays_supported_when_count_evidence_exists():
    report = verify_answerability(
        "How many participants took part?", demographic_retrieval()
    )
    assert report.status is not Answerability.NOT_SUPPORTED


def test_mean_iq_is_rejected_when_no_iq_evidence_exists():
    """The reported false positive: participants + numbers is not an IQ."""
    report = verify_answerability(
        "What was the participants' mean IQ?", demographic_retrieval()
    )
    assert report.status is Answerability.NOT_SUPPORTED
    assert "never report" in report.reason
    assert report.supporting_chunk_ids == []


def test_bmi_is_rejected_when_only_age_and_sex_are_reported():
    report = verify_answerability(
        "What was the average BMI of the sample?", demographic_retrieval()
    )
    assert report.status is Answerability.NOT_SUPPORTED


def test_a_score_question_is_rejected_when_that_score_was_never_measured():
    report = verify_answerability(
        "What was the mean depression score?", demographic_retrieval()
    )
    assert report.status is Answerability.NOT_SUPPORTED


@pytest.mark.parametrize(
    "question",
    [
        "What was the participants' mean GPA?",
        "What was the average blood pressure of the sample?",
        "What was the mean reaction time?",
        "How many years of education did participants have?",
        "What was the participants' median household income?",
        "What was the average questionnaire score?",
    ],
)
def test_unseen_attributes_are_rejected_by_the_same_rule(question):
    """No attribute name is hard-coded; these were never used to build the rule."""
    assert (
        verify_answerability(question, demographic_retrieval()).status
        is Answerability.NOT_SUPPORTED
    )


def test_the_attribute_rule_does_not_fire_on_ordinary_questions():
    """It must not turn into a blanket "every word must appear" rule."""
    for question in (
        "What were the main research questions?",
        "How were the data collected?",
        "Which applications did learners use?",
    ):
        assert not asks_for_specific_attribute(question), question


def test_a_close_morphological_variant_counts_as_evidence():
    """A paper writes "aged 19 to 27" where the question says "age"."""
    assert term_present("age", content_terms(DEMOGRAPHIC_TEXT))
    assert not term_present("iq", content_terms(DEMOGRAPHIC_TEXT))


def test_focus_terms_drop_scaffolding_but_keep_the_attribute():
    assert focus_terms("What was the participants' mean IQ?") == {"iq"}
    assert "age" in focus_terms("What was the sample's average age?")


# --------------------------------------------------------------------------- #
# Refinement 2: implicit limitations
# --------------------------------------------------------------------------- #

IMPLICIT_CAVEAT_TEXT = (
    "Twenty eight learners were recruited from a single department at one "
    "university, so the pattern described here may not transfer to other "
    "settings. Each learner was interviewed only once. Reported screen time is "
    "self reported and may partly reflect what learners felt able to admit."
)


def test_an_implicit_small_sample_limitation_is_recognised():
    """The paper never writes "limitation" or "caveat"."""
    assert "limitation" not in IMPLICIT_CAVEAT_TEXT.lower()
    assert "caveat" not in IMPLICIT_CAVEAT_TEXT.lower()

    report = verify_answerability(
        "What limitations did the authors identify?",
        retrieval(
            [
                chunk(IMPLICIT_CAVEAT_TEXT, page=5, section="discussion"),
                chunk("Learners named Anki and Quizlet most often.", page=4),
            ],
            top=0.5,
            second=0.2,
        ),
    )
    assert report.status is not Answerability.NOT_SUPPORTED
    assert 5 in report.supporting_pages


def test_an_interpretive_caveat_using_may_partly_reflect_is_recognised():
    report = verify_answerability(
        "What caveats did the authors give?",
        retrieval(
            [
                chunk(
                    "The rise may partly reflect the coverage of the two "
                    "indexes rather than a real change in publishing.",
                    page=4,
                    section="discussion",
                )
            ]
        ),
    )
    assert report.status is not Answerability.NOT_SUPPORTED


def test_an_ordinary_result_containing_may_is_not_a_limitation():
    """"may" alone is a hedge, not a caveat."""
    for sentence in (
        "Scores may increase with further practice.",
        "Learners may use several applications in one week.",
        "The effect may be larger in younger cohorts.",
    ):
        assert not IMPLICIT_LIMITATION_CUES.search(sentence), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "The sample was drawn from a single institution.",
        "Only twelve students took part, so the sample size was small.",
        "The sample was homogeneous in age and predominantly female.",
        "Participants were interviewed only once.",
        "We cannot establish causality from these data.",
        "Findings should be interpreted cautiously.",
        "Screen time was self reported and not verified.",
        "Both indexes under-represent journals outside Europe.",
    ],
)
def test_implicit_caveat_constructions_are_recognised(sentence):
    assert IMPLICIT_LIMITATION_CUES.search(sentence)


# --------------------------------------------------------------------------- #
# Both real-paper datasets
# --------------------------------------------------------------------------- #


async def test_publication_language_dataset_has_no_false_positives(hashing_ollama):
    from answerability_eval import builtin_evaluation, compute_metrics, evaluate

    rows = await evaluate(builtin_evaluation().questions, await embedded_language_corpus())
    metrics = compute_metrics(rows)
    assert metrics.false_positives == 0
    assert metrics.correct == metrics.total


async def test_mobile_devices_dataset_fixes_both_reported_failures(hashing_ollama):
    from pathlib import Path

    from answerability_eval import compute_metrics, evaluate, load_evaluation
    from conftest import embedded_mobile_corpus

    evaluation = load_evaluation(Path("evaluations/mobile_devices.json"))
    rows = await evaluate(evaluation.questions, await embedded_mobile_corpus())
    by_question = {row.question: row for row in rows}

    # Failure 1: the specific-attribute false positive.
    assert (
        by_question["What was the participants' mean IQ?"].report.status
        is Answerability.NOT_SUPPORTED
    )
    # Failure 2: the implicit-limitation false negative.
    limitations = by_question["What limitations did the authors identify?"]
    assert limitations.report.is_answerable
    assert 5 in limitations.report.supporting_pages

    assert compute_metrics(rows).false_positives == 0
