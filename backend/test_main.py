"""Backend tests. Ollama is never contacted: httpx is stubbed out."""

import json

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

import diagnostics
import ollama_client
from condensers import (
    condense_findings,
    condense_methods,
    condense_participants,
    condense_research_question,
)
from fixtures import PAGES, RUNNING_HEAD, page_tuples, pages_payload
from main import app
from metadata import extract_hints, extract_title_and_authors, repeated_lines
from ollama_client import OllamaError, _extract_json, inline_schema_refs
from schemas import (
    NOT_STATED,
    NOT_STATED_IN_CHUNK,
    ChunkSummary,
    PageInput,
    PaperSummary,
    is_missing,
    validate_chunk,
)
from sections import (
    build_evidence_packages,
    filter_findings,
    is_finding,
    is_limitation,
    is_prescriptive,
    match_heading,
    parse_sections,
)
from summarizer import chunk_pages, tighten_key_findings

client = TestClient(app)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

EMPTY_CONDENSE = {
    "title": NOT_STATED,
    "authors": [NOT_STATED],
    "research_question": NOT_STATED,
    "background": NOT_STATED,
    "methods": NOT_STATED,
    "participants_or_data": NOT_STATED,
    "key_findings": [NOT_STATED],
    "limitations": [NOT_STATED],
}

EMPTY_REDUCE = dict(
    EMPTY_CONDENSE,
    plain_english_summary=NOT_STATED,
    confidence_notes=NOT_STATED,
    source_pages={
        "research_question": [],
        "methods": [],
        "key_findings": [],
        "limitations": [],
    },
)

GOOD_CONDENSE = {
    "title": "Supporting newly qualified nurses through the transition to "
    "clinical practice: a qualitative interview study",
    "authors": ["Jane A. Fielding", "Marcus O. Reyes", "Priya N. Shah"],
    "research_question": "How newly qualified nurses experience structured "
    "mentorship in their first year on acute wards.",
    "background": "Earlier preceptorship studies were single-site surveys.",
    "methods": "Semi-structured interviews analysed with reflexive thematic "
    "analysis in NVivo 12.",
    "participants_or_data": "Eighteen newly qualified nurses from three acute "
    "NHS trusts, aged 21 to 34.",
    "key_findings": [
        "Visible availability of a mentor reduced anxiety.",
        "Nurses calibrated questions against ward busyness.",
        "Supervised practice became independent judgement.",
    ],
    "limitations": [
        "Sample drawn from three trusts in one region.",
        "Each nurse was interviewed only once.",
    ],
}

GOOD_REDUCE = dict(
    GOOD_CONDENSE,
    plain_english_summary="Newly qualified nurses were interviewed about "
    "mentorship. Having a mentor nearby mattered more than formal scheduling.",
    confidence_notes="Based on a single round of interviews.",
    source_pages={
        "research_question": [4],
        "methods": [4],
        "key_findings": [6, 7, 8],
        "limitations": [9, 10],
    },
)


def make_pdf(page_texts):
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 100), text)
    data = document.tobytes()
    document.close()
    return data


def pages(count, text="Some paper text about methods and findings. "):
    return [PageInput(page_number=n, text=text * 5) for n in range(1, count + 1)]


class FakeClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        return self._handler(url, json)


@pytest.fixture
def fake_ollama(monkeypatch):
    """Patch httpx.AsyncClient with a scripted handler and record the calls."""
    ollama_client.reset_capabilities()

    def install(handler):
        calls = []

        def wrapped(url, payload):
            calls.append(payload)
            return handler(url, payload)

        monkeypatch.setattr(
            ollama_client.httpx, "AsyncClient", lambda **kwargs: FakeClient(wrapped)
        )
        return calls

    return install


def ok(body):
    return httpx.Response(200, json={"response": json.dumps(body)})


def stage_of(payload):
    prompt = payload["prompt"]
    if "EXCERPT START" in prompt:
        return "map"
    if "ASSEMBLY RULES" in prompt:
        return "reduce"
    return "condense"


def make_handler(condense=None, reduce=None):
    def handler(url, payload):
        stage = stage_of(payload)
        if stage == "condense":
            return ok(condense if condense is not None else GOOD_CONDENSE)
        if stage == "reduce":
            return ok(reduce if reduce is not None else GOOD_REDUCE)
        return ok({"page_start": 0, "page_end": 0})

    return handler


def post_summary(payload=None):
    return client.post(
        "/summary",
        json=payload or {"filename": "nurses.pdf", "pages": pages_payload()},
    )


# --------------------------------------------------------------------------- #
# existing endpoints
# --------------------------------------------------------------------------- #


def test_root_and_health():
    assert client.get("/").json() == {"message": "AI Paper Reader API is running"}
    assert client.get("/health").json() == {"status": "healthy"}


def test_upload_extracts_text():
    pdf = make_pdf(["First page text.", "Second page text."])
    response = client.post(
        "/upload", files={"file": ("paper.pdf", pdf, "application/pdf")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 2
    assert "First page text." in body["text_preview"]


@pytest.mark.parametrize(
    "name,data,content_type,expected",
    [
        ("notes.txt", b"hello", "text/plain", 400),
        ("empty.pdf", b"", "application/pdf", 400),
        ("broken.pdf", b"definitely not a pdf", "application/pdf", 400),
    ],
)
def test_upload_rejects_bad_files(name, data, content_type, expected):
    response = client.post("/upload", files={"file": (name, data, content_type)})
    assert response.status_code == expected


def test_upload_rejects_scanned_pdf():
    document = fitz.open()
    document.new_page()
    blank = document.tobytes()
    document.close()
    response = client.post(
        "/upload", files={"file": ("scan.pdf", blank, "application/pdf")}
    )
    assert response.status_code == 422


def test_response_contract_is_unchanged():
    assert set(PaperSummary.model_fields) == {
        "title",
        "authors",
        "research_question",
        "background",
        "methods",
        "participants_or_data",
        "key_findings",
        "limitations",
        "plain_english_summary",
        "confidence_notes",
        "source_pages",
    }


# --------------------------------------------------------------------------- #
# THE REGRESSION TEST
# --------------------------------------------------------------------------- #

METHODOLOGICAL_CUES = (
    "region",
    "interviewed once",
    "homogeneous",
    "limitations",
    "generalise",
    "sample",
)


@pytest.mark.parametrize(
    "label,handler",
    [
        ("model returns nothing", make_handler(EMPTY_CONDENSE, EMPTY_REDUCE)),
        ("model behaves", make_handler()),
    ],
)
def test_real_paper_fields_are_extracted(fake_ollama, label, handler):
    """Page 1 has a journal header, page number, label, wrapped title, author,
    affiliation, email and abstract. Pages 4 and 9-10 have explicit headings.
    None of the core fields may come back empty, even when the model is useless.
    """
    fake_ollama(handler)

    response = post_summary()
    assert response.status_code == 200
    summary = response.json()["summary"]

    # -- title is the paper's, not the journal's --
    assert "newly qualified nurses" in summary["title"].lower()
    assert "Journal of Advanced Nursing" not in summary["title"]
    assert RUNNING_HEAD not in summary["title"]
    assert "1234" not in summary["title"]
    assert "Research paper" != summary["title"]

    # -- author is a person, not an affiliation or an email --
    authors = " ".join(summary["authors"])
    assert "Fielding" in authors
    assert "University of Manchester" not in authors
    assert "School of Nursing" not in authors
    assert "@" not in authors

    # -- the fields that kept disappearing --
    for field in (
        "title",
        "research_question",
        "methods",
        "participants_or_data",
        "background",
    ):
        assert summary[field] != NOT_STATED, field
        assert not is_missing(summary[field]), field

    assert "mentorship" in summary["research_question"].lower()
    assert (
        "interview" in summary["methods"].lower()
        or "thematic" in summary["methods"].lower()
    )
    assert (
        "eighteen" in summary["participants_or_data"].lower()
        or "18" in summary["participants_or_data"]
        or "21 to 34" in summary["participants_or_data"]
    )

    findings = summary["key_findings"]
    assert findings and NOT_STATED not in findings
    joined_findings = " ".join(findings).lower()
    assert "should protect mentor time" not in joined_findings
    assert "ought to receive" not in joined_findings
    assert "future work should" not in joined_findings

    # -- limitations are methodological only --
    limitations = summary["limitations"]
    assert limitations and NOT_STATED not in limitations
    joined = " ".join(limitations).lower()
    assert any(cue in joined for cue in METHODOLOGICAL_CUES)
    assert "future research should" not in joined
    assert "markedly higher confidence" not in joined
    assert "services should protect" not in joined

    # -- provenance comes from the parser --
    source = summary["source_pages"]
    assert source["research_question"] == [4]
    assert 4 in source["methods"]
    assert source["limitations"] == [9, 10]


def test_structured_paper_skips_the_map_stage(fake_ollama):
    """Every field is under a heading, so no chunk discovery calls are needed."""
    calls = fake_ollama(make_handler())
    post_summary()
    stages = [stage_of(call) for call in calls]
    assert stages == ["condense", "reduce"]


def test_condense_prompt_supplies_section_text(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()

    prompt = next(c["prompt"] for c in calls if stage_of(c) == "condense")
    assert "### research_question" in prompt
    assert "### participants_or_data" in prompt
    assert "Eighteen newly qualified nurses" in prompt
    assert "reflexive thematic analysis" in prompt
    # The model is told it may not claim a supplied field is missing.
    assert f'Returning "{NOT_STATED}" for a field whose text was supplied' in prompt


def test_reduce_prompt_marks_extractions_verified(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()
    prompt = next(c["prompt"] for c in calls if stage_of(c) == "reduce")
    assert "VERIFIED extractions" in prompt
    assert "Every VERIFIED value must appear in your output" in prompt


def test_unstructured_paper_still_uses_chunk_discovery(fake_ollama):
    """A paper with no headings falls back to the old MAP behaviour."""
    calls = fake_ollama(make_handler())
    client.post(
        "/summary",
        json={
            "filename": "notes.pdf",
            "pages": [
                {"page_number": n, "text": "Prose with no headings at all. " * 20}
                for n in range(1, 5)
            ],
        },
    )
    assert "map" in [stage_of(call) for call in calls]


def test_condense_failure_falls_back_to_deterministic_condensers(fake_ollama):
    def handler(url, payload):
        if stage_of(payload) == "condense":
            raise httpx.ReadTimeout("too slow")
        return ok(EMPTY_REDUCE)

    fake_ollama(handler)
    summary = post_summary().json()["summary"]
    assert "mentorship" in summary["research_question"].lower()
    assert "thematic analysis" in summary["methods"].lower()
    assert "21 to 34" in summary["participants_or_data"]


def test_reduce_failure_still_returns_verified_content(fake_ollama):
    def handler(url, payload):
        if stage_of(payload) == "reduce":
            raise httpx.ReadTimeout("too slow")
        return ok(GOOD_CONDENSE)

    fake_ollama(handler)
    response = post_summary()
    assert response.status_code == 200
    summary = response.json()["summary"]
    assert summary["research_question"] == GOOD_CONDENSE["research_question"]
    assert summary["source_pages"]["research_question"] == [4]


# --------------------------------------------------------------------------- #
# title and author detection
# --------------------------------------------------------------------------- #


def test_journal_header_and_page_number_are_not_the_title():
    hints = extract_hints(page_tuples())
    title = hints.title_candidates[0]
    assert title.startswith("Supporting newly qualified nurses")
    assert "Journal" not in title and "1234" not in title
    assert hints.title_confidence == "high"


def test_title_spanning_three_lines_is_joined():
    hints = extract_hints(page_tuples())
    assert hints.title_candidates[0].endswith("interview study")


def test_affiliation_and_email_are_not_authors():
    hints = extract_hints(page_tuples())
    authors = " ".join(hints.author_candidates)
    assert "Fielding" in authors
    assert "University" not in authors and "@" not in authors


def test_running_head_is_detected_and_removed():
    running = repeated_lines(page_tuples())
    assert RUNNING_HEAD in running
    # Section headings are never treated as running heads.
    assert not any(match_heading(line) for line in running)


def test_arxiv_front_matter():
    found = extract_title_and_authors(
        "arXiv:1706.03762v5 [cs.CL] 2 Aug 2023\n"
        "Attention Is All You Need\n"
        "Ashish Vaswani, Noam Shazeer, Niki Parmar\n"
        "Google Brain\n"
        "noam@google.com\n"
        "Abstract\nThe dominant sequence transduction models..."
    )
    assert found.title_candidates[0] == "Attention Is All You Need"
    assert found.author_candidates == ["Ashish Vaswani, Noam Shazeer, Niki Parmar"]


def test_low_confidence_front_matter_is_not_applied_blindly():
    found = extract_title_and_authors("Some Heading\nMore text here to read\n")
    assert not found.is_confident


# --------------------------------------------------------------------------- #
# section parsing
# --------------------------------------------------------------------------- #


def test_sections_are_found_with_their_pages():
    hints = extract_hints(page_tuples())
    found = {
        section.name: section.pages
        for section in parse_sections(page_tuples(), drop_lines=hints.repeated_lines)
    }
    assert found["research_question"] == [4]
    assert found["participants"] == [4]
    assert found["data_collection_and_analysis"] == [4, 5]
    assert found["findings"] == [6, 7, 8]
    assert found["limitations"] == [9, 10]
    assert found["references"] == [10]


def test_section_body_stops_at_the_next_heading():
    sections = {
        section.name: section
        for section in parse_sections(page_tuples(), drop_lines=[RUNNING_HEAD])
    }
    text = sections["research_question"].text
    assert "acute wards" in text
    assert "Eighteen newly qualified nurses" not in text


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Abstract", "abstract"),
        ("4. Data collection and analysis", "data_collection_and_analysis"),
        ("METHODS:", "methods"),
        ("2.1 Participants", "participants"),
        ("Discussion and conclusions", "discussion"),
        ("IV. Findings", "findings"),
        ("Limitations of the study", "limitations"),
        ("The methods used here were varied and long.", None),
        ("", None),
    ],
)
def test_heading_matching(line, expected):
    assert match_heading(line) == expected


def test_run_in_heading_keeps_its_body():
    sections = parse_sections([(1, "Limitations. The sample was small and local.")])
    assert sections[0].name == "limitations"
    assert sections[0].text.startswith("The sample was small")


def test_references_are_not_mined():
    packages = build_evidence_packages(
        parse_sections(page_tuples(), drop_lines=[RUNNING_HEAD])
    )
    for package in packages.values():
        assert "Nursing Review 12" not in package.text


def test_research_question_is_inferred_when_there_is_no_heading():
    packages = build_evidence_packages(
        parse_sections(
            [
                (1, "Abstract\nThis study aims to measure ward handover delays."),
                (2, "Methods\nWe timed 40 handovers."),
            ]
        )
    )
    assert "research_question" in packages
    assert "handover delays" in packages["research_question"].text


# --------------------------------------------------------------------------- #
# key-findings filtering
# --------------------------------------------------------------------------- #

EMPIRICAL_RESULT = (
    "Nurses with a mentor rostered on the same shift reported lower anxiety "
    "scores across the whole year."
)
PRACTITIONER_RECOMMENDATION = (
    "Ward educators ought to receive dedicated preparation before taking on "
    "mentees, and training should be extended to every acute ward."
)
FUTURE_RESEARCH_RECOMMENDATION = (
    "Future work should test whether these patterns hold in community settings."
)


def test_empirical_result_is_a_finding():
    assert is_finding(EMPIRICAL_RESULT)
    assert not is_prescriptive(EMPIRICAL_RESULT)


def test_practitioner_recommendation_is_not_a_finding():
    assert is_prescriptive(PRACTITIONER_RECOMMENDATION)
    assert not is_finding(PRACTITIONER_RECOMMENDATION)


def test_future_research_recommendation_is_not_a_finding():
    assert is_prescriptive(FUTURE_RESEARCH_RECOMMENDATION)
    assert not is_finding(FUTURE_RESEARCH_RECOMMENDATION)


def test_conclusion_summarising_evidence_is_kept():
    """Requirement 3: conclusions survive, prescriptions do not."""
    conclusion = (
        "Structured mentorship appears to work through availability rather "
        "than formal scheduling."
    )
    assert is_finding(conclusion)


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Recall accuracy improved by 22 per cent in the extended condition.", True),
        ("Three themes were identified across the eighteen interviews.", True),
        ("Participants described handover as the most stressful moment.", True),
        ("No significant difference was observed between the two wards.", True),
        ("Services should protect mentor time on the roster.", False),
        ("Teachers ought to model questioning behaviour in the classroom.", False),
        ("Trusts need to invest in preceptorship programmes.", False),
        ("We recommend that mentors be given protected time.", False),
        ("We suggest that practitioners revisit their induction materials.", False),
        ("The implications for workforce policy are considerable.", False),
        ("Future research should follow a larger cohort across regions.", False),
        ("Policy makers must fund additional mentor posts.", False),
        ("Training should be extended to every acute ward.", False),
        ("Short result.", False),
    ],
)
def test_finding_filter(sentence, expected):
    assert is_finding(sentence) is expected


def test_condense_findings_keeps_results_and_drops_recommendations():
    packages = build_evidence_packages(
        parse_sections(page_tuples(), drop_lines=[RUNNING_HEAD])
    )
    findings = condense_findings(packages["key_findings"].text, limit=10)
    joined = " ".join(findings)

    assert "reported lower anxiety scores" in joined
    assert "appears to work through availability" in joined
    assert "should protect mentor time" not in joined
    assert "ought to receive dedicated preparation" not in joined
    assert "Future work should" not in joined
    assert "implications for workforce policy" not in joined


def test_filter_findings_removes_prescriptive_entries():
    assert filter_findings(
        [EMPIRICAL_RESULT, PRACTITIONER_RECOMMENDATION, FUTURE_RESEARCH_RECOMMENDATION]
    ) == [EMPIRICAL_RESULT]


def test_tighten_key_findings_never_empties_a_paper_with_results():
    packages = build_evidence_packages(
        parse_sections(page_tuples(), drop_lines=[RUNNING_HEAD])
    )
    summary = PaperSummary.model_validate(
        {"key_findings": [PRACTITIONER_RECOMMENDATION, FUTURE_RESEARCH_RECOMMENDATION]}
    )
    dropped = tighten_key_findings(summary, packages)

    assert dropped == 2
    assert summary.key_findings
    assert NOT_STATED not in summary.key_findings
    assert all(not is_prescriptive(item) for item in summary.key_findings)


def test_model_recommendations_are_stripped_end_to_end(fake_ollama):
    """The model returns a mix; only the empirical entries reach the response."""
    polluted = dict(
        GOOD_CONDENSE,
        key_findings=[
            EMPIRICAL_RESULT,
            PRACTITIONER_RECOMMENDATION,
            FUTURE_RESEARCH_RECOMMENDATION,
        ],
    )
    fake_ollama(make_handler(polluted, dict(GOOD_REDUCE, key_findings=[
        EMPIRICAL_RESULT,
        PRACTITIONER_RECOMMENDATION,
        FUTURE_RESEARCH_RECOMMENDATION,
    ])))

    findings = post_summary().json()["summary"]["key_findings"]
    assert EMPIRICAL_RESULT in findings
    assert PRACTITIONER_RECOMMENDATION not in findings
    assert FUTURE_RESEARCH_RECOMMENDATION not in findings


def test_key_findings_stay_a_list_of_strings(fake_ollama):
    """Requirement 5: the response contract does not change."""
    fake_ollama(make_handler())
    findings = post_summary().json()["summary"]["key_findings"]
    assert isinstance(findings, list)
    assert all(isinstance(item, str) for item in findings)


# --------------------------------------------------------------------------- #
# limitation filtering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sentence,in_section,expected",
    [
        ("The sample was drawn from a single region and is small.", True, True),
        ("The sample was homogeneous in age and predominantly female.", True, True),
        ("Future research should follow a larger cohort of nurses.", True, False),
        ("We recommend that trusts protect mentor time on the roster.", True, False),
        ("Nurses reported markedly higher confidence by month nine.", True, False),
        ("The first theme was visible availability of the mentor.", True, False),
        ("Findings may not generalise beyond acute wards in one region.", False, True),
        ("Handover delays fell by 14 per cent across the year.", False, False),
        ("Short.", True, False),
    ],
)
def test_limitation_filter(sentence, in_section, expected):
    assert is_limitation(sentence, in_section) is expected


def test_limitations_package_excludes_findings_and_recommendations():
    packages = build_evidence_packages(
        parse_sections(page_tuples(), drop_lines=[RUNNING_HEAD])
    )
    text = packages["limitations"].text
    assert "homogeneous" in text
    assert "Future research should" not in text
    assert "markedly higher confidence" not in text
    assert packages["limitations"].pages == [9, 10]


# --------------------------------------------------------------------------- #
# deterministic condensers
# --------------------------------------------------------------------------- #


def test_condense_research_question_prefers_the_question():
    text = (
        "This section sets out the focus. How do nurses experience mentorship "
        "during their first year? We describe the rationale below."
    )
    assert condense_research_question(text).startswith("How do nurses")


def test_condense_participants_reports_counts_and_characteristics():
    result = condense_participants(PAGES[4].split("Participants\n")[1])
    assert "21 to 34" in result
    assert "Eighteen newly qualified nurses" in result


def test_condense_methods_combines_collection_and_analysis():
    result = condense_methods(PAGES[4].split("Data collection and analysis\n")[1])
    assert "Semi-structured interviews" in result
    assert "thematic analysis" in result


def test_condensers_return_empty_for_empty_text():
    assert condense_research_question("") == ""
    assert condense_participants("") == ""
    assert condense_methods("") == ""


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #


def test_chunking_respects_page_limit():
    chunks = chunk_pages(pages(10), char_limit=1_000_000, page_limit=4)
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]


def test_chunking_respects_char_limit():
    chunks = chunk_pages(pages(6, text="x" * 500), char_limit=1200, page_limit=100)
    assert len(chunks) > 1
    assert sum(len(chunk) for chunk in chunks) == 6


def test_chunking_skips_blank_pages():
    supplied = [
        PageInput(page_number=1, text="real text"),
        PageInput(page_number=2, text="   \n "),
        PageInput(page_number=3, text="more real text"),
    ]
    chunks = chunk_pages(supplied, char_limit=1_000_000, page_limit=100)
    assert [page.page_number for page in chunks[0]] == [1, 3]


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #


def test_summary_rejects_missing_pages():
    assert client.post("/summary", json={"pages": []}).status_code == 400


def test_summary_rejects_blank_text():
    response = client.post(
        "/summary", json={"pages": [{"page_number": 1, "text": "  "}]}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# logging safety
# --------------------------------------------------------------------------- #


def test_paper_text_is_not_logged(fake_ollama, caplog):
    fake_ollama(make_handler())
    with caplog.at_level("DEBUG"):
        post_summary()
    assert "Eighteen newly qualified nurses were recruited" not in caplog.text
    assert RUNNING_HEAD not in caplog.text


def test_debug_logging_reports_sections_and_recovery(fake_ollama, caplog, monkeypatch):
    monkeypatch.setattr(diagnostics, "SUMMARY_DEBUG", True)
    monkeypatch.setattr(ollama_client, "SUMMARY_DEBUG", True)
    fake_ollama(make_handler(EMPTY_CONDENSE, EMPTY_REDUCE))

    with caplog.at_level("INFO"):
        post_summary()

    text = caplog.text
    assert "sections:" in text
    assert "research_question p4" in text
    assert "CONDENSE" in text
    assert "REDUCE selected:" in text
    assert "Recovered fields" in text
    assert "skipping MAP" in text


# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #


def test_ollama_not_running_on_an_unstructured_paper(fake_ollama):
    def handler(url, payload):
        raise httpx.ConnectError("connection refused")

    fake_ollama(handler)
    response = client.post(
        "/summary",
        json={"pages": [{"page_number": 1, "text": "prose with no headings"}]},
    )
    assert response.status_code == 503


def test_model_not_installed(fake_ollama):
    def handler(url, payload):
        return httpx.Response(
            404, json={"error": "model 'qwen3:8b' not found, try pulling it first"}
        )

    fake_ollama(handler)
    response = client.post(
        "/summary",
        json={"pages": [{"page_number": 1, "text": "prose with no headings"}]},
    )
    assert response.status_code == 503
    assert "ollama pull" in response.json()["detail"]


def test_malformed_output_on_an_unstructured_paper(fake_ollama):
    fake_ollama(
        lambda url, payload: httpx.Response(200, json={"response": "sorry, no JSON"})
    )
    response = client.post(
        "/summary",
        json={"pages": [{"page_number": 1, "text": "prose with no headings"}]},
    )
    assert response.status_code == 502


def test_generation_options_are_set_for_extraction(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()
    for call in calls:
        options = call["options"]
        assert options["temperature"] == 0
        assert options["top_p"] == 0.9
        assert options["num_predict"] >= 2048
        assert call["think"] is False


def test_retries_once_when_think_option_is_rejected(fake_ollama):
    state = {"calls": 0}
    good = make_handler()

    def handler(url, payload):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(
                400, json={"error": "model does not support think option"}
            )
        assert "think" not in payload
        return good(url, payload)

    fake_ollama(handler)
    assert post_summary().status_code == 200


# --------------------------------------------------------------------------- #
# parsing, schema and coercion
# --------------------------------------------------------------------------- #


def test_extract_json_strips_think_block_and_fences():
    parsed, cleanups = _extract_json(
        '<think>hmm</think>\n```json\n{"a": 1}\n```'
    )
    assert parsed == {"a": 1}
    assert "stripped-think-block" in cleanups and "stripped-code-fence" in cleanups


def test_extract_json_handles_unclosed_think_block():
    parsed, cleanups = _extract_json('<think>never closed {"b": 0}')
    assert parsed == {"b": 0}
    assert "stripped-unclosed-think" in cleanups


def test_extract_json_finds_object_inside_prose():
    parsed, cleanups = _extract_json('Here you go: {"a": 2} hope that helps')
    assert parsed == {"a": 2}
    assert "extracted-object-from-prose" in cleanups


def test_extract_json_rejects_garbage():
    with pytest.raises(OllamaError):
        _extract_json("no json at all")


def test_schema_sent_to_ollama_has_no_refs():
    dumped = json.dumps(inline_schema_refs(ChunkSummary.model_json_schema()))
    assert "$ref" not in dumped and "$defs" not in dumped


def test_nested_values_are_recovered_not_discarded():
    chunk = ChunkSummary.model_validate(
        {
            "research_question": "Does sleep help?",
            "methods": {"value": {"text": "Crossover design"}},
            "participants_or_data": ["48 adults"],
            "key_findings": "Recall improved",
            "limitations": {"value": "Single site", "pages": ["9"]},
        }
    )
    assert chunk.research_question.value == "Does sleep help?"
    assert chunk.methods.value == "Crossover design"
    assert chunk.participants_or_data.value == "48 adults"
    assert chunk.key_findings[0].value == "Recall improved"
    assert chunk.limitations[0].pages == [9]


def test_empty_markers_are_treated_as_missing():
    chunk = ChunkSummary.model_validate(
        {"methods": "N/A", "research_question": NOT_STATED_IN_CHUNK}
    )
    assert not chunk.methods.has_value
    assert not chunk.research_question.has_value


def test_missing_fields_become_not_stated():
    summary = PaperSummary.model_validate({"title": "  "})
    assert summary.title == NOT_STATED
    assert summary.key_findings == [NOT_STATED]


def test_source_pages_tolerate_strings_and_nulls():
    summary = PaperSummary.model_validate(
        {"source_pages": {"methods": ["3", 4, None, "x"], "key_findings": None}}
    )
    assert summary.source_pages.methods == [3, 4]
    assert summary.source_pages.key_findings == []


def test_validation_failure_keeps_the_page_range(caplog):
    with caplog.at_level("WARNING"):
        chunk = validate_chunk({"page_start": ["oops", {"nested": 1}]}, 3, 5)
    assert chunk.page_start == 3 and chunk.page_end == 5
