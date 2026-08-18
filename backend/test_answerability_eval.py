"""Tests for the evaluation harness. The verifier itself is tested elsewhere."""

import json

import pytest

from answerability import Answerability, AnswerabilityReport
from answerability_eval import (
    EvaluationCase,
    EvaluationFileError,
    EvaluationRow,
    build_parser,
    builtin_evaluation,
    compute_metrics,
    evaluate,
    format_metrics,
    format_row,
    format_table,
    load_evaluation,
    parse_evaluation,
    resolve_evaluation,
)
from conftest import embedded_language_corpus  # noqa: F401

EVALUATIONS = "evaluations/real_papers/publication_language.json"


def report(status, pages=(), top=0.5, gap=0.1):
    return AnswerabilityReport(
        status=status,
        reason="reason.",
        supporting_pages=list(pages),
        retrieval_top_score=top,
        retrieval_score_gap=gap,
        evidence_count=len(pages),
    )


def row(answerable, status, pages=(), expected_pages=()):
    return EvaluationRow(
        EvaluationCase(
            question="Q?",
            expected_answerable=answerable,
            expected_pages=list(expected_pages),
        ),
        report(status, pages),
    )


def write(tmp_path, payload):
    path = tmp_path / "eval.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return path


VALID = {
    "paper": "example",
    "questions": [
        {"question": "Answerable?", "expected_answerable": True, "expected_pages": [4]},
        {"question": "Unanswerable?", "expected_answerable": False, "expected_pages": []},
    ],
}


# --------------------------------------------------------------------------- #
# loading and validation
# --------------------------------------------------------------------------- #


def test_a_valid_file_loads(tmp_path):
    evaluation = load_evaluation(write(tmp_path, VALID))
    assert evaluation.paper == "example"
    assert [case.question for case in evaluation.questions] == [
        "Answerable?",
        "Unanswerable?",
    ]
    assert evaluation.questions[0].expected_pages == [4]
    assert evaluation.questions[1].expected_pages == []


def test_optional_fields_default(tmp_path):
    evaluation = load_evaluation(
        write(tmp_path, {"questions": [{"question": "Q?", "expected_answerable": True}]})
    )
    assert evaluation.questions[0].expected_pages == []
    assert evaluation.paper == "eval"  # falls back to the file stem


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(EvaluationFileError) as error:
        load_evaluation(tmp_path / "absent.json")
    assert "No such evaluation file" in str(error.value)


def test_invalid_json_is_reported(tmp_path):
    with pytest.raises(EvaluationFileError) as error:
        load_evaluation(write(tmp_path, "{not json"))
    assert "not valid JSON" in str(error.value)


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ([], "top level must be an object"),
        ({"paper": 7, "questions": [{"question": "Q?", "expected_answerable": True}]},
         "'paper' must be a string"),
        ({}, "missing 'questions'"),
        ({"questions": "nope"}, "'questions' must be a list"),
        ({"questions": []}, "'questions' is empty"),
        ({"questions": ["nope"]}, "question 1 must be an object"),
        ({"questions": [{"expected_answerable": True}]}, "needs a non-empty 'question'"),
        ({"questions": [{"question": "  ", "expected_answerable": True}]},
         "needs a non-empty 'question'"),
        ({"questions": [{"question": "Q?"}]}, "needs 'expected_answerable'"),
        ({"questions": [{"question": "Q?", "expected_answerable": "yes"}]},
         "needs 'expected_answerable'"),
        ({"questions": [{"question": "Q?", "expected_answerable": True,
                         "expected_pages": 4}]}, "'expected_pages' must be a list"),
        ({"questions": [{"question": "Q?", "expected_answerable": True,
                         "expected_pages": [0]}]}, "page numbers from 1"),
        ({"questions": [{"question": "Q?", "expected_answerable": True,
                         "expected_pages": ["4"]}]}, "page numbers from 1"),
        ({"questions": [{"question": "Q?", "expected_answerable": True,
                         "expected_pages": [True]}]}, "page numbers from 1"),
    ],
)
def test_malformed_files_are_rejected_with_a_reason(payload, fragment):
    with pytest.raises(EvaluationFileError) as error:
        parse_evaluation(payload)
    assert fragment in str(error.value)


def test_the_offending_question_is_identified():
    with pytest.raises(EvaluationFileError) as error:
        parse_evaluation(
            {
                "questions": [
                    {"question": "fine", "expected_answerable": True},
                    {"question": "fine too", "expected_answerable": True},
                    {"question": "broken"},
                ]
            }
        )
    assert "question 3" in str(error.value)


# --------------------------------------------------------------------------- #
# built-in set is preserved
# --------------------------------------------------------------------------- #


def test_builtin_set_is_the_publication_language_regression_set():
    evaluation = builtin_evaluation()
    assert len(evaluation.questions) == 10
    assert sum(1 for case in evaluation.questions if case.expected_answerable) == 6
    assert sum(1 for case in evaluation.questions if not case.expected_answerable) == 4
    assert "How many human participants were recruited?" in [
        case.question for case in evaluation.questions
    ]


def test_no_eval_argument_uses_the_builtin_set():
    assert resolve_evaluation(None).questions == builtin_evaluation().questions


def test_shipped_evaluation_file_matches_the_builtin_set():
    """The example file doubles as documentation, so it must not drift."""
    from pathlib import Path

    loaded = load_evaluation(Path(EVALUATIONS))
    assert [case.model_dump() for case in loaded.questions] == [
        case.model_dump() for case in builtin_evaluation().questions
    ]


def test_a_bad_eval_path_exits_cleanly():
    with pytest.raises(SystemExit) as exit_info:
        resolve_evaluation("does/not/exist.json")
    assert "No such evaluation file" in str(exit_info.value)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_accepts_pdf_only():
    args = build_parser().parse_args(["/tmp/paper.pdf"])
    assert args.pdf == "/tmp/paper.pdf"
    assert args.evaluation is None


def test_cli_accepts_an_eval_file():
    args = build_parser().parse_args(["/tmp/paper.pdf", "--eval", "/tmp/eval.json"])
    assert args.evaluation == "/tmp/eval.json"


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_both_answerable_statuses_count_as_answerable():
    assert row(True, Answerability.SUPPORTED).predicted_answerable
    assert row(True, Answerability.PARTIALLY_SUPPORTED).predicted_answerable
    assert not row(True, Answerability.NOT_SUPPORTED).predicted_answerable


def test_confusion_counts():
    rows = [
        row(True, Answerability.SUPPORTED),            # TP
        row(True, Answerability.PARTIALLY_SUPPORTED),  # TP
        row(False, Answerability.NOT_SUPPORTED),       # TN
        row(False, Answerability.SUPPORTED),           # FP
        row(True, Answerability.NOT_SUPPORTED),        # FN
    ]
    metrics = compute_metrics(rows)

    assert metrics.total == 5
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.correct == 3
    assert metrics.accuracy == pytest.approx(0.6)


def test_metrics_on_an_empty_set():
    metrics = compute_metrics([])
    assert metrics.total == 0 and metrics.accuracy == 0.0


# --------------------------------------------------------------------------- #
# page reporting stays separate from accuracy
# --------------------------------------------------------------------------- #


def test_page_hit_needs_one_expected_page_in_the_evidence():
    assert row(True, Answerability.SUPPORTED, pages=[3], expected_pages=[3]).page_hit
    assert row(True, Answerability.SUPPORTED, pages=[2, 3], expected_pages=[3]).page_hit
    assert not row(True, Answerability.SUPPORTED, pages=[9], expected_pages=[3]).page_hit


def test_page_hit_is_none_when_no_pages_were_expected():
    assert row(True, Answerability.SUPPORTED, pages=[3]).page_hit is None


def test_a_wrong_page_does_not_reduce_answerability_accuracy():
    rows = [row(True, Answerability.SUPPORTED, pages=[9], expected_pages=[3])]
    metrics = compute_metrics(rows)

    assert metrics.accuracy == 1.0, "answerability was correct"
    assert metrics.pages_checked == 1
    assert metrics.pages_hit == 0
    assert metrics.page_accuracy == 0.0


def test_only_questions_with_expected_pages_are_page_checked():
    rows = [
        row(True, Answerability.SUPPORTED, pages=[3], expected_pages=[3]),
        row(True, Answerability.SUPPORTED, pages=[4]),
        row(False, Answerability.NOT_SUPPORTED),
    ]
    metrics = compute_metrics(rows)
    assert metrics.pages_checked == 1 and metrics.pages_hit == 1


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def test_row_shows_every_required_column():
    line = format_row(row(True, Answerability.SUPPORTED, pages=[2, 3], expected_pages=[3]))
    assert "Q?" in line
    assert "yes" in line and "supported" in line
    assert "0.500" in line and "0.100" in line
    assert "2,3" in line
    assert "PASS" in line


def test_row_marks_a_miss():
    assert "MISS" in format_row(row(True, Answerability.NOT_SUPPORTED))


def test_metrics_block_reports_every_figure():
    text = format_metrics(
        compute_metrics(
            [
                row(True, Answerability.SUPPORTED, pages=[3], expected_pages=[3]),
                row(False, Answerability.NOT_SUPPORTED),
            ]
        )
    )
    assert "total questions 2" in text
    assert "2/2 correct" in text and "accuracy 100.0%" in text
    assert "true positives 1" in text and "true negatives 1" in text
    assert "false positives 0" in text and "false negatives 0" in text
    assert "expected-page check: 1/1" in text
    assert "not part of answerability accuracy" in text


def test_page_line_is_omitted_when_nothing_was_page_checked():
    text = format_metrics(compute_metrics([row(True, Answerability.SUPPORTED)]))
    assert "expected-page check" not in text


def test_table_includes_the_paper_name_and_every_row():
    rows = [row(True, Answerability.SUPPORTED), row(False, Answerability.NOT_SUPPORTED)]
    table = format_table(rows, paper="example paper")
    assert "Evaluation set: example paper" in table
    for item in rows:
        assert format_row(item) in table


def test_table_explains_each_miss():
    table = format_table([row(True, Answerability.NOT_SUPPORTED)])
    assert "MISS  Q?" in table and "reason." in table


# --------------------------------------------------------------------------- #
# end to end over the fixture corpus
# --------------------------------------------------------------------------- #


async def test_evaluate_accepts_case_objects(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate(builtin_evaluation().questions, embedded)

    assert len(rows) == 10
    assert all(row.correct for row in rows)
    assert compute_metrics(rows).accuracy == 1.0


async def test_evaluate_still_accepts_the_older_tuple_form(hashing_ollama):
    embedded = await embedded_language_corpus()
    rows = await evaluate([("Which countries were included?", True)], embedded)
    assert rows[0].correct
    assert rows[0].expected_pages == []


async def test_a_json_file_and_the_builtin_set_agree(hashing_ollama):
    from pathlib import Path

    embedded = await embedded_language_corpus()
    from_file = await evaluate(load_evaluation(Path(EVALUATIONS)).questions, embedded)
    from_builtin = await evaluate(builtin_evaluation().questions, embedded)

    assert [r.report.status for r in from_file] == [
        r.report.status for r in from_builtin
    ]


async def test_page_accuracy_is_reported_for_the_builtin_set(hashing_ollama):
    embedded = await embedded_language_corpus()
    metrics = compute_metrics(await evaluate(builtin_evaluation().questions, embedded))

    assert metrics.accuracy == 1.0
    assert metrics.pages_checked == 6, "the six answerable questions carry pages"
    assert metrics.page_accuracy is not None


async def test_custom_evaluation_file_runs_against_the_same_corpus(
    hashing_ollama, tmp_path
):
    path = write(
        tmp_path,
        {
            "paper": "custom",
            "questions": [
                {
                    "question": "How were the publication data collected?",
                    "expected_answerable": True,
                    "expected_pages": [2],
                },
                {
                    "question": "How many human participants were recruited?",
                    "expected_answerable": False,
                    "expected_pages": [],
                },
            ],
        },
    )
    embedded = await embedded_language_corpus()
    rows = await evaluate(load_evaluation(path).questions, embedded)

    assert all(row.correct for row in rows)
    assert rows[0].page_hit is True
    assert rows[1].page_hit is None
