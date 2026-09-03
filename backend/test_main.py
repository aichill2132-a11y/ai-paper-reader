"""Backend tests. Ollama is never contacted: httpx is stubbed out."""

import asyncio
import json
import logging

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

import condensers
import diagnostics
import ollama_client
import summarizer
from condensers import (
    condense_findings,
    condense_methods,
    condense_participants,
    condense_research_question,
)
from fixtures import PAGES, RUNNING_HEAD, page_tuples, pages_payload
from main import app
from metadata import (
    DocumentHints,
    _pick_title,
    extract_hints,
    extract_title_and_authors,
    is_page_furniture,
    looks_like_affiliation,
    looks_like_authors,
    looks_like_title_text,
    repeated_lines,
    split_author_names,
)
from ollama_client import (
    MalformedJSONError,
    OllamaError,
    _extract_json,
    inline_schema_refs,
)
from schemas import (
    NOT_STATED,
    NOT_STATED_IN_CHUNK,
    ChunkSummary,
    CondensedSummary,
    Evidence,
    reduce_response_schema,
    PageInput,
    PaperSummary,
    is_missing,
    validate_chunk,
)
from sections import (
    EvidencePackage,
    heading_on_line,
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
    "author_stated_limitations": [NOT_STATED],
}

EMPTY_REDUCE = dict(
    EMPTY_CONDENSE,
    plain_english_summary=NOT_STATED,
    confidence_notes=NOT_STATED,
    source_pages={
        "research_question": [],
        "methods": [],
        "key_findings": [],
        "author_stated_limitations": [],
    },
)

GOOD_CONDENSE = {
    "title": "Supporting newly qualified nurses through the transition to "
    "clinical practice: a qualitative interview study",
    "authors": ["Jane A. Fielding", "Marcus O. Reyes", "Priya N. Shah"],
    "research_question": "How do newly qualified nurses experience structured "
    "mentorship in their first year on acute wards?",
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
        "author_stated_limitations": [9, 10],
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
    if "STATEMENT:" in prompt:
        return "rewrite"
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
        "author_stated_limitations",
        "model_identified_considerations",
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

    # -- limitations are optional, and methodological only when present --
    # Recovery no longer forces a value, so a model that returns nothing
    # correctly leaves this empty rather than inventing a limitation.
    limitations = summary["author_stated_limitations"]
    assert NOT_STATED not in limitations
    joined = " ".join(limitations).lower()
    if limitations:
        assert any(cue in joined for cue in METHODOLOGICAL_CUES)
    assert "future research should" not in joined
    assert "markedly higher confidence" not in joined
    assert "services should protect" not in joined

    # -- provenance comes from the parser --
    source = summary["source_pages"]
    assert source["research_question"] == [4]
    assert 4 in source["methods"]
    assert source["author_stated_limitations"] == [9, 10]


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
    # The model is told it may not claim a supplied field is missing...
    assert f'Returning "{NOT_STATED}" for one of those is wrong' in prompt
    assert "You must return a real value for:" in prompt
    # ...except for the one field that is allowed to come back empty.
    assert "author_stated_limitations is OPTIONAL" in prompt
    assert "author_stated_limitations" not in prompt.split(
        "You must return a real value for:"
    )[1].split("\n")[0]


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
        # Under an explicit limitations heading the authors' own framing wins,
        # so future-work wording is retained (contract changed deliberately).
        ("Future research should follow a larger cohort of nurses.", True, True),
        ("We recommend that trusts protect mentor time on the roster.", True, True),
        # Outside such a section the same wording is still rejected.
        ("Future research should follow a larger cohort of nurses.", False, False),
        ("We recommend that trusts protect mentor time on the roster.", False, False),
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
    # Findings stay out wherever they sit.
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
    assert condense_participants("") == ""
    assert condense_methods("") == ""


def test_research_question_says_so_when_no_aim_is_stated():
    """Better to admit the aim was not found than to quote a Results sentence."""
    from condensers import NO_RESEARCH_QUESTION

    assert condense_research_question("") == NO_RESEARCH_QUESTION
    assert (
        condense_research_question(
            "Recall improved by 22 per cent. The effect was strongest after REM."
        )
        == NO_RESEARCH_QUESTION
    )


def test_research_question_uses_the_stated_aim():
    assert "determine whether sleep" in condense_research_question(
        "This study aimed to determine whether sleep improves recall. "
        "Results showed a 22 per cent gain."
    )


def test_methods_summary_skips_bench_protocol():
    """Design over instrument settings: issue 2."""
    summary = condense_methods(
        "Samples were injected onto a C18 column with 5 um particle size at a "
        "flow rate of 1.0 ml/min. A within-subjects crossover design was used "
        "with 48 adults. Data were analysed with mixed-effects models."
    )
    assert "crossover design" in summary
    assert "flow rate" not in summary
    assert "particle size" not in summary


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


# --------------------------------------------------------------------------- #
# research_question must be interrogative (one corrective rewrite, no loop)
# --------------------------------------------------------------------------- #


def _rewriter(monkeypatch, reply, calls):
    """Stand in for the single corrective generation call."""

    async def fake(prompt, **kwargs):
        calls.append(prompt)
        if isinstance(reply, Exception):
            raise reply
        return {"research_question": reply}

    monkeypatch.setattr(summarizer, "generate_json", fake)


def test_a_question_is_left_alone_and_costs_no_call(monkeypatch):
    calls = []
    _rewriter(monkeypatch, "unused", calls)
    value = "How does mentorship affect retention?"
    assert asyncio.run(summarizer.ensure_research_question(value)) == value
    assert calls == []


def test_the_declarative_aim_is_rewritten_once(monkeypatch):
    calls = []
    _rewriter(
        monkeypatch,
        "How does structured mentorship affect newly qualified nurses?",
        calls,
    )
    result = asyncio.run(
        summarizer.ensure_research_question(
            "The effect of structured mentorship upon newly qualified nurses "
            "was also determined."
        )
    )
    assert result.endswith("?")
    assert "mentorship" in result
    assert len(calls) == 1


def test_a_rewrite_that_is_still_not_a_question_falls_back(monkeypatch):
    calls = []
    _rewriter(monkeypatch, "Mentorship was examined in nurses.", calls)
    result = asyncio.run(
        summarizer.ensure_research_question("Mentorship was examined in nurses.")
    )
    assert result == condensers.NO_RESEARCH_QUESTION
    # Exactly one attempt: the fallback is used rather than a retry loop.
    assert len(calls) == 1


def test_a_rewrite_that_invents_content_falls_back(monkeypatch):
    calls = []
    _rewriter(
        monkeypatch,
        "How does randomised chemotherapy dosing affect paediatric survival "
        "rates across European hospitals?",
        calls,
    )
    result = asyncio.run(
        summarizer.ensure_research_question(
            "The effect of structured mentorship upon newly qualified nurses "
            "was also determined."
        )
    )
    assert result == condensers.NO_RESEARCH_QUESTION
    assert len(calls) == 1


def test_a_failed_rewrite_call_falls_back(monkeypatch):
    calls = []
    _rewriter(monkeypatch, OllamaError("model is unavailable", 503), calls)
    result = asyncio.run(
        summarizer.ensure_research_question("Mentorship was examined in nurses.")
    )
    assert result == condensers.NO_RESEARCH_QUESTION


def test_the_not_identifiable_sentence_is_never_rewritten(monkeypatch):
    calls = []
    _rewriter(monkeypatch, "unused", calls)
    value = condensers.NO_RESEARCH_QUESTION
    assert asyncio.run(summarizer.ensure_research_question(value)) == value
    assert calls == []


# --------------------------------------------------------------------------- #
# every stage applies the same field wording
# --------------------------------------------------------------------------- #


def test_map_and_reduce_carry_the_field_instructions(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()

    reduce_prompt = next(c["prompt"] for c in calls if stage_of(c) == "reduce")
    assert "FIELD RULES" in reduce_prompt
    # The exact wording, not a paraphrase: one source of truth.
    assert summarizer.FIELD_INSTRUCTIONS["research_question"] in reduce_prompt
    assert summarizer.FIELD_INSTRUCTIONS["methods"] in reduce_prompt


def test_the_map_prompt_carries_rules_for_the_fields_it_hunts():
    prompt = summarizer._map_prompt(
        [PageInput(page_number=1, text="Some text about a study.")],
        "paper.pdf",
        1,
        1,
        extract_hints([(1, "Some text about a study.")]),
        ["research_question", "methods"],
    )
    assert "FIELD RULES" in prompt
    assert summarizer.FIELD_INSTRUCTIONS["research_question"] in prompt
    # The per-excerpt absence rule still wins over any field-level wording.
    assert NOT_STATED_IN_CHUNK in prompt


# --------------------------------------------------------------------------- #
# author_stated_limitations is optional
# --------------------------------------------------------------------------- #


def test_limitations_are_not_forced_when_the_model_returns_none():
    """No deterministic condenser may fill this field."""
    assert "author_stated_limitations" not in condensers.LIST_CONDENSERS
    assert "author_stated_limitations" not in summarizer.CORE_FIELDS


def test_an_empty_limitations_list_is_valid_and_not_missing():
    summary = PaperSummary()
    assert summary.author_stated_limitations == []
    assert "author_stated_limitations" not in summary.missing_fields()


# --------------------------------------------------------------------------- #
# author-stated limitations must be attributable to the paper
# --------------------------------------------------------------------------- #

COLLECTED = EvidencePackage(
    field="limitations",
    text=(
        "These pretreatment studies are limited by the small number of animals "
        "available in each dosage group.\n"
        "The generalizability of these results to human tumours cannot be "
        "assumed from the L1210 model alone."
    ),
    pages=[5, 6],
    headings=["discussion"],
)


def _with_limitations(entries, notes="All fields were supported."):
    summary = PaperSummary()
    summary.author_stated_limitations = entries
    summary.confidence_notes = notes
    return summary


def test_a_restated_author_limitation_is_kept():
    summary = _with_limitations(
        [
            "These pretreatment studies are limited by the small number of "
            "animals in each dosage group."
        ]
    )
    moved = summarizer.enforce_limitation_attribution(
        summary, {"limitations": COLLECTED}
    )
    assert moved == 0
    assert len(summary.author_stated_limitations) == 1
    assert summary.model_identified_considerations == []


@pytest.mark.parametrize(
    "critique",
    [
        # A methodological choice a model can criticise is not a limitation.
        "The study excluded mice that died within 5 days, which may introduce "
        "bias.",
        # Absent clinical data the authors never framed as limiting.
        "The study is based on preliminary pretreatment studies and does not "
        "provide comprehensive clinical data.",
        # An unmeasured variable the paper never mentions.
        "The study did not account for potential residual procarbazine "
        "metabolites in pretreated animals.",
    ],
)
def test_model_critique_is_relabelled_not_presented_as_an_author_claim(critique):
    summary = _with_limitations([critique])
    moved = summarizer.enforce_limitation_attribution(
        summary, {"limitations": COLLECTED}
    )
    assert moved == 1
    assert summary.author_stated_limitations == []
    assert summary.model_identified_considerations == [critique]


def test_nothing_survives_when_the_paper_collected_no_limitations():
    summary = _with_limitations(
        ["The sample was small and possibly unrepresentative."]
    )
    summarizer.enforce_limitation_attribution(summary, {})
    assert summary.author_stated_limitations == []
    # Moved, not deleted: the observation is kept under its honest label.
    assert summary.model_identified_considerations == [
        "The sample was small and possibly unrepresentative."
    ]


def test_confidence_notes_stop_claiming_limitations_were_inferred():
    summary = _with_limitations(
        ["The study excluded mice that died within 5 days, which may bias it."],
        notes=(
            "The key findings are well supported. Some limitations were "
            "inferred rather than stated by the authors."
        ),
    )
    summarizer.enforce_limitation_attribution(summary, {"limitations": COLLECTED})
    summarizer.align_confidence_notes(summary)

    notes = summary.confidence_notes
    assert "were inferred rather than stated" not in notes
    assert "The key findings are well supported." in notes
    assert summarizer.ATTRIBUTION_NOTE in notes


def test_the_attribution_note_is_absent_when_nothing_was_inferred():
    summary = _with_limitations(
        [
            "These pretreatment studies are limited by the small number of "
            "animals in each dosage group."
        ]
    )
    summarizer.enforce_limitation_attribution(summary, {"limitations": COLLECTED})
    summarizer.align_confidence_notes(summary)
    assert summarizer.ATTRIBUTION_NOTE not in summary.confidence_notes


# --------------------------------------------------------------------------- #
# confidence_notes describes the evidence; critique is rehomed
# --------------------------------------------------------------------------- #

N8_CRITIQUE = (
    "The study's findings are based on a small sample size (n=8), which may "
    "limit the generalizability of the results."
)


def test_a_methodological_critique_leaves_the_confidence_notes():
    """Paper 2: the n=8 sentence is a consideration, not a confidence note."""
    summary = PaperSummary()
    summary.confidence_notes = N8_CRITIQUE
    summarizer.align_confidence_notes(summary)

    assert summary.model_identified_considerations == [N8_CRITIQUE]
    assert "n=8" not in summary.confidence_notes
    # Author-stated limitations are untouched by this move.
    assert summary.author_stated_limitations == []


def test_provenance_notes_survive_while_critique_is_moved():
    summary = PaperSummary()
    summary.confidence_notes = (
        "Methods and key findings were taken directly from labelled sections. "
        "The sample size was small, which may limit generalizability. "
        "All wording was produced by a local model."
    )
    summarizer.align_confidence_notes(summary)

    notes = summary.confidence_notes
    assert "taken directly from labelled sections" in notes
    assert "produced by a local model" in notes
    assert "sample size was small" not in notes
    assert summary.model_identified_considerations == [
        "The sample size was small, which may limit generalizability."
    ]


def test_notes_without_critique_are_left_alone_and_add_no_considerations():
    summary = PaperSummary()
    notes = "Every field was supported by labelled section text."
    summary.confidence_notes = notes
    summarizer.align_confidence_notes(summary)

    # Nothing is invented, so the frontend keeps hiding the section.
    assert summary.model_identified_considerations == []
    assert summary.confidence_notes == notes
    assert summarizer.ATTRIBUTION_NOTE not in summary.confidence_notes


def test_a_critique_the_authors_already_stated_is_not_duplicated():
    stated = "The sample size was small, which may limit generalizability."
    summary = PaperSummary()
    summary.author_stated_limitations = [stated]
    summary.confidence_notes = stated
    summarizer.align_confidence_notes(summary)

    assert summary.author_stated_limitations == [stated]
    assert summary.model_identified_considerations == []


def test_the_reduce_prompt_sends_critique_to_the_right_field(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()
    prompt = next(c["prompt"] for c in calls if stage_of(c) == "reduce")

    assert "confidence_notes: describe the EVIDENCE, not the study" in prompt
    assert "belong in model_identified_considerations" in prompt
    # The old wording invited exactly the critique this change rehomes.
    assert "say plainly which parts are thin or missing" not in prompt


# --------------------------------------------------------------------------- #
# 1-2. REDUCE output contract: the KEY is required, the content is not
# --------------------------------------------------------------------------- #


def test_the_reduce_schema_requires_the_structured_keys():
    schema = reduce_response_schema()
    required = set(schema["required"])
    for field in (
        "title", "authors", "research_question", "background", "methods",
        "participants_or_data", "key_findings", "author_stated_limitations",
        "model_identified_considerations", "plain_english_summary",
        "confidence_notes", "source_pages",
    ):
        assert field in required, field


def test_paper_summary_itself_stays_permissive():
    """Only the copy sent to the model is tightened."""
    assert "required" not in PaperSummary.model_json_schema()
    # A partial response must still validate everywhere else in the app.
    assert PaperSummary.model_validate({}).title == NOT_STATED


def test_the_reduce_call_sends_the_required_schema(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()
    payload = next(c for c in calls if stage_of(c) == "reduce")
    assert "research_question" in payload["format"]["required"]
    assert "key_findings" in payload["format"]["required"]
    # MAP is unchanged: its schema is a different model and stays permissive.
    map_calls = [c for c in calls if stage_of(c) == "map"]
    for call in map_calls:
        assert "required" not in call["format"]


def test_a_required_key_may_still_carry_a_missing_value():
    """Requiring the key must not force invented content."""
    summary = PaperSummary.model_validate(
        {
            "title": NOT_STATED,
            "research_question": NOT_STATED,
            "key_findings": [NOT_STATED],
            "author_stated_limitations": [],
            "model_identified_considerations": [],
        }
    )
    assert summary.research_question == NOT_STATED
    assert summary.author_stated_limitations == []
    assert "research_question" in summary.missing_fields()


# --------------------------------------------------------------------------- #
# 3-5. provenance: package-backed beats model-authored
# --------------------------------------------------------------------------- #

FINDINGS_PACKAGE = EvidencePackage(
    field="key_findings",
    text="Scores rose by 12 points after the intervention.",
    pages=[5],
    headings=["results"],
)


def _recover_with(package):
    summary = PaperSummary()
    summary.key_findings = []
    condensed = CondensedSummary()
    condensed.key_findings = ["a finding CONDENSE wrote"]
    note = ChunkSummary(page_start=1, page_end=2)
    note.key_findings = [
        Evidence(value="a finding MAP read off the page", evidence="quote", pages=[1])
    ]
    labels = summarizer._recover_list_fields(
        summary, condensed, {"key_findings": package} if package else {}, [note]
    )
    return summary.key_findings, labels


def test_recovery_prefers_map_when_condense_had_no_evidence_package():
    values, labels = _recover_with(None)
    assert values == ["a finding MAP read off the page"]
    assert labels == ["key_findings (map notes)"]


def test_recovery_keeps_condense_first_when_it_was_package_backed():
    values, labels = _recover_with(FINDINGS_PACKAGE)
    assert values == ["a finding CONDENSE wrote"]
    assert labels == ["key_findings"]


def test_package_backed_condense_values_are_rendered_as_verified():
    condensed = CondensedSummary()
    condensed.methods = "Interviews were analysed thematically."
    package = EvidencePackage(
        field="methods", text="Methods text.", pages=[4], headings=["methods"]
    )
    rendered = summarizer._render_verified(condensed, {"methods": package})
    assert "Interviews were analysed thematically." in rendered
    assert "[pages 4]" in rendered


def test_unsupported_condense_values_are_not_rendered_as_verified():
    """A model guess must not acquire parser-backed authority."""
    condensed = CondensedSummary()
    condensed.key_findings = ["a finding with no supporting section"]
    condensed.research_question = "a question with no supporting section"
    rendered = summarizer._render_verified(condensed, {})
    assert "no supporting section" not in rendered
    assert rendered == "- (nothing was extracted from labelled sections)"


# --------------------------------------------------------------------------- #
# 6-9. title extraction
# --------------------------------------------------------------------------- #


def test_a_scientific_acronym_is_not_a_company_affiliation():
    """LTD is long-term depression here, not a limited company."""
    title = "Engineering a memory with LTD and LTP"
    assert not is_page_furniture(title)
    assert looks_like_title_text(title)


@pytest.mark.parametrize(
    "line",
    [
        "Genentech Ltd, South San Francisco, California",
        "Acme Biosciences Inc., Cambridge",
        "Novartis Pharma GmbH, Basel",
    ],
)
def test_real_company_affiliations_are_still_furniture(line):
    assert is_page_furniture(line)


def test_a_nature_style_letter_keeps_its_real_title():
    found = extract_title_and_authors(
        "LETTER\n"
        "doi:10.1038/nature13294\n"
        "Engineering a memory with LTD and LTP\n"
        "Sadegh Nabavi1,2*, Rocky Fox1* & Roberto Malinow1,2\n"
        "It has been proposed that memories are encoded by modification\n"
        "a tone with optogenetic stimulation of neural inputs to the lateral amyg-\n"
        "dala originating from auditory nuclei, and subsequently examined the"
    )
    assert found.title_candidates[0] == "Engineering a memory with LTD and LTP"
    assert "a tone with optogenetic" not in " ".join(found.title_candidates)


def test_no_block_before_the_byline_yields_no_title_rather_than_body_text():
    """Fail safe instead of promoting the longest paragraph on page 1."""
    found = extract_title_and_authors(
        "Jane A. Fielding, Marcus O. Reyes\n"
        "Some long body paragraph that would previously have been promoted to "
        "the title because it was simply the longest block on the first page.\n"
        "A second body line that is also long and also not a title at all."
    )
    assert found.title_candidates == []
    assert found.confidence == "low"
    assert found.author_candidates


def test_pick_title_returns_nothing_when_no_block_precedes_the_byline():
    candidates, block = _pick_title([["body text"], ["more body text"]], 0)
    assert candidates == []
    assert block == []


# --------------------------------------------------------------------------- #
# key_findings cardinality: schema cap, prompt wording, near-duplicates
# --------------------------------------------------------------------------- #


def test_the_reduce_schema_caps_key_findings_at_six():
    findings = reduce_response_schema()["properties"]["key_findings"]
    assert findings["maxItems"] == 6


def test_the_reduce_schema_sets_no_lower_bound():
    """A paper with two genuine findings must be allowed to return two."""
    schema = reduce_response_schema()
    assert "minItems" not in schema["properties"]["key_findings"]
    assert "minItems" not in json.dumps(schema)


def test_the_cap_reaches_the_payload_sent_to_ollama(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()
    payload = next(c for c in calls if stage_of(c) == "reduce")
    assert payload["format"]["properties"]["key_findings"]["maxItems"] == 6


def test_map_key_findings_are_not_capped():
    """MAP must keep collecting candidates; selection happens in REDUCE."""
    findings = ChunkSummary.model_json_schema()["properties"]["key_findings"]
    assert "maxItems" not in findings
    assert "maxItems" not in json.dumps(ChunkSummary.model_json_schema())


def test_paper_summary_validation_is_not_globally_tightened():
    """The cap is REDUCE-only; stored/validated summaries stay permissive."""
    assert "maxItems" not in json.dumps(PaperSummary.model_json_schema())
    many = PaperSummary.model_validate({"key_findings": [f"finding {n}" for n in range(9)]})
    assert len(many.key_findings) == 9


def test_the_reduce_prompt_tells_the_model_to_select_not_collect(fake_ollama):
    calls = fake_ollama(make_handler())
    post_summary()
    prompt = next(c["prompt"] for c in calls if stage_of(c) == "reduce")
    assert "key_findings is the exception to that merge rule" in prompt
    assert "SELECT, do not collect" in prompt
    assert "at most six" in prompt
    # The general merge rule survives for the other fields.
    assert "merge it in. Merge duplicates." in prompt


# -- near-duplicate collapsing ---------------------------------------------


def test_a_restatement_with_more_detail_replaces_the_shorter_one():
    kept = summarizer.collapse_near_duplicates(
        ["LTD inactivated the memory.",
         "Optical LTD inactivated the established memory."]
    )
    assert kept == ["Optical LTD inactivated the established memory."]


def test_the_same_claim_reordered_collapses():
    kept = summarizer.collapse_near_duplicates(
        ["Optical LTD inactivated the conditioned response.",
         "The conditioned response was inactivated by optical LTD."]
    )
    assert len(kept) == 1


def test_exact_duplicates_collapse():
    item = "Scores rose by 12 points after the intervention."
    assert summarizer.collapse_near_duplicates([item, item]) == [item]


@pytest.mark.parametrize(
    "first,second",
    [
        # Opposite manipulations with opposite outcomes.
        ("Optical LTD reduced auditory fear conditioning.",
         "Optical LTP reactivated the conditioned response."),
        # Same subject, different measurement.
        ("The CR was sensitive to extinction.",
         "The CR was blocked by NMDA receptor inhibition."),
        # Shared topic words only.
        ("Optical stimulation produced a conditioned response.",
         "Optical stimulation was delivered at 900 pulses."),
    ],
)
def test_distinct_findings_are_not_collapsed(first, second):
    assert len(summarizer.collapse_near_duplicates([first, second])) == 2


def test_two_genuine_findings_stay_two():
    summary = PaperSummary()
    summary.key_findings = [
        "Scores rose by 12 points after the intervention.",
        "Dropout was higher in the control arm.",
    ]
    summarizer.tighten_key_findings(summary, {})
    assert len(summary.key_findings) == 2


def test_prescriptive_filtering_still_composes_with_collapsing():
    summary = PaperSummary()
    summary.key_findings = [
        "Scores rose by 12 points after the intervention.",
        "Scores rose by 12 points after the intervention was delivered.",
        "Services should protect mentor time.",
    ]
    dropped = summarizer.tighten_key_findings(summary, {})
    assert dropped == 1                       # the recommendation
    assert len(summary.key_findings) == 1     # plus one restatement collapsed
    assert "should protect" not in " ".join(summary.key_findings)


def test_empty_key_findings_still_fall_back_to_not_stated():
    summary = PaperSummary()
    summary.key_findings = [NOT_STATED]
    summarizer.tighten_key_findings(summary, {})
    assert summary.key_findings == [NOT_STATED]


# --------------------------------------------------------------------------- #
# one malformed MAP chunk must not abort the summary
# --------------------------------------------------------------------------- #

# 14 filler pages with no headings: nothing is covered by a labelled section,
# so MAP runs, and chunk_pages splits them into 1-6, 7-12, 13-14.
UNSTRUCTURED = {
    "filename": "unstructured.pdf",
    "pages": [page.model_dump() for page in pages(14)],
}

# Shaped like the real Paper 4 failure: a leading brace and a nested closing
# brace, so the "extract the object from prose" salvage is attempted and still
# fails. That is the second raise in _extract_json, not the first.
TRUNCATED = (
    '{"page_start": 1, "background": {"value": "some text", "pages": [1]}, '
    '"participants_or_data": {"value": "the model stopped mid-obj'
)


def map_handler(fail_on=(), body=None):
    """Return malformed output for the given 1-based MAP call numbers."""
    seen = {"map": 0}

    def handler(url, payload):
        stage = stage_of(payload)
        if stage == "map":
            seen["map"] += 1
            if seen["map"] in fail_on:
                return httpx.Response(
                    200, json={"response": body if body is not None else TRUNCATED}
                )
            return ok({"page_start": 0, "page_end": 0,
                       "background": {"value": "some background", "pages": [1]}})
        if stage == "condense":
            return ok(GOOD_CONDENSE)
        return ok(GOOD_REDUCE)

    return handler


def test_a_malformed_map_chunk_is_skipped_and_the_summary_succeeds(fake_ollama):
    calls = fake_ollama(map_handler(fail_on=(2,)))
    response = post_summary(UNSTRUCTURED)
    assert response.status_code == 200
    assert len([c for c in calls if stage_of(c) == "map"]) == 3


def test_map_chunks_after_the_failure_are_still_attempted(fake_ollama):
    calls = fake_ollama(map_handler(fail_on=(1,)))
    assert post_summary(UNSTRUCTURED).status_code == 200
    map_prompts = [c["prompt"] for c in calls if stage_of(c) == "map"]
    assert len(map_prompts) == 3
    assert any("pages 13 to 14" in p for p in map_prompts)


def test_notes_either_side_of_a_failed_chunk_are_retained(fake_ollama):
    calls = fake_ollama(map_handler(fail_on=(2,)))
    assert post_summary(UNSTRUCTURED).status_code == 200
    reduce_prompt = next(c["prompt"] for c in calls if stage_of(c) == "reduce")
    assert "Notes from pages 1-6" in reduce_prompt
    assert "Notes from pages 13-14" in reduce_prompt
    assert "Notes from pages 7-12" not in reduce_prompt


def test_the_skipped_chunk_is_logged_with_index_and_pages(fake_ollama, caplog):
    fake_ollama(map_handler(fail_on=(2,)))
    with caplog.at_level(logging.WARNING):
        post_summary(UNSTRUCTURED)
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    assert any("MAP 2/3 malformed JSON; skipping pages 7-12" in m for m in warnings)


def test_the_malformed_response_is_logged_with_size_and_tail(fake_ollama, caplog):
    fake_ollama(map_handler(fail_on=(2,)))
    with caplog.at_level(logging.WARNING):
        post_summary(UNSTRUCTURED)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "malformed structured output" in messages
    assert "response_chars=%d" % len(TRUNCATED) in messages
    assert "stopped mid-obj" in messages          # the tail, not the whole reply
    # The whole paper is never logged.
    assert "Some paper text about methods and findings." not in messages


def test_partial_map_failure_is_disclosed_in_the_confidence_notes(fake_ollama):
    fake_ollama(map_handler(fail_on=(2,)))
    summary = post_summary(UNSTRUCTURED).json()["summary"]
    assert summarizer.PARTIAL_EVIDENCE_NOTE in summary["confidence_notes"]
    # No implementation vocabulary reaches the user.
    for word in ("JSON", "MAP", "Ollama", "chunk"):
        assert word not in summarizer.PARTIAL_EVIDENCE_NOTE


def test_a_clean_run_gets_no_partial_evidence_disclosure(fake_ollama):
    fake_ollama(map_handler(fail_on=()))
    summary = post_summary(UNSTRUCTURED).json()["summary"]
    assert summarizer.PARTIAL_EVIDENCE_NOTE not in summary["confidence_notes"]


def test_the_summary_still_fails_when_every_map_chunk_is_malformed(fake_ollama):
    calls = fake_ollama(map_handler(fail_on=(1, 2, 3)))
    response = post_summary(UNSTRUCTURED)
    assert response.status_code == 502
    assert "malformed JSON" in response.json()["detail"]
    # No REDUCE on invented evidence.
    assert not [c for c in calls if stage_of(c) == "reduce"]


def test_infrastructure_failures_are_not_swallowed(fake_ollama):
    """A timeout must still abort: the next chunk would fail the same way."""
    def handler(url, payload):
        if stage_of(payload) == "map":
            raise httpx.ReadTimeout("too slow")
        return ok(GOOD_CONDENSE)

    calls = fake_ollama(handler)
    response = post_summary(UNSTRUCTURED)
    assert response.status_code == 504
    assert not [c for c in calls if stage_of(c) == "reduce"]


def test_a_missing_model_is_not_swallowed(fake_ollama):
    def handler(url, payload):
        if stage_of(payload) == "map":
            return httpx.Response(404, json={"error": "model not found"})
        return ok(GOOD_CONDENSE)

    fake_ollama(handler)
    assert post_summary(UNSTRUCTURED).status_code in (404, 502, 503)


def test_malformed_json_is_a_narrow_subtype_of_ollama_error():
    """Uncaught, it must still surface exactly as before."""
    assert issubclass(MalformedJSONError, OllamaError)
    error = MalformedJSONError("bad", 502)
    assert error.status_code == 502
    with pytest.raises(OllamaError):
        raise error


# --------------------------------------------------------------------------- #
# compound limitations headings (Paper 4 class of failure)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "heading",
    [
        "Limitations and future directions",
        "Limitations and Future Directions",
        "LIMITATIONS AND FUTURE DIRECTIONS",
        "limitations and future direction",
        "Limitations and future research",
        "Limitations and further research",
        "Limitations and implications",
        "Limitations and recommendations",
        "Study limitations",
        "Potential limitations",
        "Methodological limitations",
        "Limitations",
        "Limitations of the study",
        "Strengths and limitations",
        "Limitations and strengths",
        "Threats to validity",
    ],
)
def test_standard_limitations_headings_are_recognised(heading):
    assert match_heading(heading) == "limitations"


@pytest.mark.parametrize(
    "heading",
    [
        # A literature-review heading, not this paper's own limitations.
        "Limitations of previous research",
        "Limitations of existing measures",
        "Discussion",
        "Findings",
        "Conclusion",
        "References",
        "Future directions",
    ],
)
def test_unrelated_headings_are_not_classified_as_limitations(heading):
    assert match_heading(heading) != "limitations"


@pytest.mark.parametrize(
    "heading",
    [
        "4.2 Limitations",
        "4.2. Limitations",
        "4.2 Limitations and future directions",
        "4.2. Limitations and future directions",
        "4.2.1 Limitations",
        "4 Limitations",
        "IV. Limitations",
        "Limitations:",
    ],
)
def test_numbered_limitations_headings_normalise(heading):
    assert match_heading(heading) == "limitations"


# -- is_limitation: the heading is attribution evidence ---------------------

RECOMMENDATION_LIMITATION = (
    "Only women participated in the current study, and future studies should "
    "expand these findings to men."
)


def test_future_work_under_an_explicit_heading_is_an_author_limitation():
    assert is_limitation(RECOMMENDATION_LIMITATION, True)


def test_the_same_wording_in_plain_discussion_is_still_rejected():
    assert not is_limitation(RECOMMENDATION_LIMITATION, False)


@pytest.mark.parametrize(
    "sentence",
    [
        "We did not directly ask participants about their felt emotions, but "
        "measured the perceived valence of the stimuli instead.",
        "We used a median split to create low and high HRV groups, an approach "
        "that has known disadvantages.",
    ],
)
def test_substantive_limitations_need_no_cue_under_an_explicit_heading(sentence):
    assert is_limitation(sentence, True)
    # Outside the section they are still not recognised - the heading is what
    # supplies the attribution, not a widened cue list.
    assert not is_limitation(sentence, False)


def test_results_under_a_limitations_heading_are_still_rejected():
    assert not is_limitation(
        "Results showed that reappraisal reduced negative affect by 22%.", True
    )


# -- end to end: package, then attribution ---------------------------------

LIMITATIONS_PAPER = [
    (1, "A Study of Emotion Regulation\nJane A. Fielding\n\n"
        "Abstract\nThis study examined emotion regulation strategies."),
    (2, "Discussion\nReappraisal outperformed suppression across conditions.\n\n"
        "Limitations and future directions\n"
        "The current study had some limitations. "
        "We did not directly ask participants about their felt emotions, but "
        "measured the perceived valence of the stimuli instead. "
        "Only women participated in the current study, and future studies "
        "should expand these findings to men. "
        "We used a median split to create low and high HRV groups, an approach "
        "that has known disadvantages."),
]


def test_the_package_carries_the_substantive_limitations_not_just_the_preamble():
    packages = build_evidence_packages(parse_sections(LIMITATIONS_PAPER))
    package = packages["limitations"]
    assert "median split" in package.text
    assert "felt emotions" in package.text
    assert "Only women participated" in package.text
    # More than the contentless opening sentence.
    assert len(package.text.splitlines()) >= 3


def test_the_attribution_gate_keeps_supported_author_limitations():
    packages = build_evidence_packages(parse_sections(LIMITATIONS_PAPER))
    summary = PaperSummary()
    summary.author_stated_limitations = [
        "Only women participated in the study, so future studies should "
        "expand these findings to men.",
        "A median split was used to create low and high HRV groups, an "
        "approach with known disadvantages.",
    ]
    moved = summarizer.enforce_limitation_attribution(summary, packages)
    assert moved == 0
    assert len(summary.author_stated_limitations) == 2
    assert summary.model_identified_considerations == []


def test_a_paper_without_a_limitations_section_is_unchanged():
    plain = [
        (1, "A Study\nJane A. Fielding\n\nAbstract\nWe examined outcomes."),
        (2, "Discussion\nScores improved across both arms of the trial."),
    ]
    packages = build_evidence_packages(parse_sections(plain))
    assert "limitations" not in packages
    summary = PaperSummary()
    summary.author_stated_limitations = ["The sample was small."]
    summarizer.enforce_limitation_attribution(summary, packages)
    assert summary.author_stated_limitations == []
    assert summary.model_identified_considerations == ["The sample was small."]


# --------------------------------------------------------------------------- #
# heading_on_line: trailing punctuation must not lose a real heading
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "Limitations and future directions",          # no punctuation
        "Limitations and future directions.",         # trailing full stop
        "Limitations and future directions. ",       # + em space (U+2003)
        "Limitations and future directions.\u00a0",   # + non-breaking space
        "Limitations and future directions:",         # trailing colon
        "  Limitations and future directions.  ",     # surrounding whitespace
        "4.2 Limitations and future directions",      # numbered
        "4.2 Limitations and future directions.",     # numbered + full stop
        "4.2. Limitations and future directions.",    # numbered, dotted
    ],
)
def test_a_standalone_limitations_heading_survives_trailing_punctuation(raw):
    """The real Paper 4 heading arrived as '...directions.\u2003'."""
    assert heading_on_line(raw) == ("limitations", "")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Discussion.", "discussion"),
        ("Methods:", "methods"),
        ("References.", "references"),
        ("Results.", "findings"),
        ("Conclusion.", "conclusion"),
    ],
)
def test_other_heading_types_also_tolerate_trailing_punctuation(raw, expected):
    assert heading_on_line(raw) == (expected, "")


@pytest.mark.parametrize(
    "raw",
    [
        # Ordinary prose ending in a period must never become a heading.
        "The current study had some limitations.",
        "We discuss the limitations of previous research below.",
        "This paper is about methods.",
        "Findings may not generalise beyond acute wards in one region.",
        "In conclusion the intervention worked.",
        # An unrecognised phrase ending in a period.
        "Some unrecognised phrase ending in a period.",
        "Notes on the analytic approach.",
    ],
)
def test_prose_ending_in_a_period_is_not_a_heading(raw):
    assert heading_on_line(raw) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "Limitations. The study was small and single-site.",
            ("limitations", "The study was small and single-site."),
        ),
        (
            "Results: scores rose by 12 points.",
            ("findings", "scores rose by 12 points."),
        ),
        (
            "Discussion. We interpret these findings cautiously.",
            ("discussion", "We interpret these findings cautiously."),
        ),
    ],
)
def test_run_in_headings_are_unchanged(raw, expected):
    assert heading_on_line(raw) == expected


def test_the_punctuated_heading_produces_a_limitations_section():
    """End to end: the parser must now emit the section, not fold it in."""
    paper = [
        (1, "A Study\nJane A. Fielding\n\nAbstract\nWe examined outcomes."),
        (
            2,
            "Discussion\nReappraisal outperformed suppression.\n\n"
            "Limitations and future directions. \n"
            "The current study had some limitations. "
            "We used a median split to create low and high HRV groups, an "
            "approach that has known disadvantages.",
        ),
    ]
    names = {section.name for section in parse_sections(paper)}
    assert "limitations" in names

    packages = build_evidence_packages(parse_sections(paper))
    assert "median split" in packages["limitations"].text


# --------------------------------------------------------------------------- #
# model_identified_considerations must hold only model-inferred study concerns
# --------------------------------------------------------------------------- #

AUTHOR_LIMITATIONS_PACKAGE = EvidencePackage(
    field="limitations",
    pages=[9],
    headings=["limitations"],
    text=(
        "We did not directly ask participants about the level of felt "
        "emotions, but rather asked for the perceived valence of the "
        "stimuli.\n"
        "We used a median split to create low and high HRV groups, and while "
        "we are aware that this approach has its disadvantages, it has also "
        "been suggested that dichotomization in factorial designs is "
        "acceptable.\n"
        "Moreover, only women participated in this study - future studies "
        "should expand our findings to men."
    ),
)


def _prune(considerations, stated=(), package=AUTHOR_LIMITATIONS_PACKAGE):
    summary = PaperSummary()
    summary.author_stated_limitations = list(stated)
    summary.model_identified_considerations = list(considerations)
    dropped = summarizer.prune_model_considerations(
        summary, {"limitations": package} if package else {}
    )
    return summary, dropped


@pytest.mark.parametrize(
    "paraphrase",
    [
        # The authors said this; a model paraphrase must not be credited to
        # the model just because the wording drifted.
        "The study involved only female participants, which may limit the "
        "generalizability of findings to other demographics.",
        "Participants were all women, so the results may not extend to men.",
        "A median split was used for the HRV grouping, which has known "
        "drawbacks.",
    ],
)
def test_a_paraphrased_author_limitation_is_not_a_model_consideration(paraphrase):
    summary, dropped = _prune([paraphrase])
    assert dropped == 1
    assert summary.model_identified_considerations == []


@pytest.mark.parametrize(
    "concern",
    [
        "The sample was recruited from a single university, which may limit "
        "generalisability.",
        "No power analysis was reported, so the study may be underpowered.",
        "The absence of a control condition makes causal claims difficult.",
        "Self-report measures may be subject to social desirability bias.",
        "Stimuli were presented in a fixed order, which could introduce "
        "sequence effects.",
    ],
)
def test_genuine_model_only_concerns_survive(concern):
    """Generic words alone must not get a real consideration dropped."""
    summary, dropped = _prune([concern])
    assert dropped == 0
    assert summary.model_identified_considerations == [concern]


@pytest.mark.parametrize(
    "meta",
    [
        "The author-stated limitations were directly quoted from the "
        "limitations section.",
        "The evidence was extracted verbatim from the source text.",
        "Page numbers were derived from the parser rather than the model.",
        "This summary was generated from the remaining available evidence.",
    ],
)
def test_provenance_commentary_is_not_a_methodological_consideration(meta):
    summary, dropped = _prune([meta], package=None)
    assert dropped == 1
    assert summary.model_identified_considerations == []


def test_a_consideration_repeating_an_author_entry_is_dropped():
    # The secondary net: comparison against the final author list catches a
    # close restatement. A loose paraphrase is caught by the package check
    # above, which sees all of the authors' wording rather than one entry.
    stated = ["Only women participated, so findings may not extend to men."]
    summary, dropped = _prune(
        ["Only women participated in the study, which limits generalisability "
         "to men."],
        stated=stated,
        package=None,
    )
    assert dropped == 1
    assert summary.author_stated_limitations == stated


def test_pruning_never_touches_author_stated_limitations():
    stated = ["We used a median split to create low and high HRV groups."]
    summary, _ = _prune(["No power analysis was reported."], stated=stated)
    assert summary.author_stated_limitations == stated


def test_pruning_a_paper_with_no_limitations_package_keeps_real_concerns():
    concern = "The absence of a control condition makes causal claims difficult."
    summary, dropped = _prune([concern], package=None)
    assert dropped == 0
    assert summary.model_identified_considerations == [concern]


# --------------------------------------------------------------------------- #
# authors: a confident parser byline must not be silently shortened
# --------------------------------------------------------------------------- #

BYLINE = "Ada N. Fielding1, Bruno O. Reyes1, Chandra P. Shah2, Dmitri Q. Ilves2"


def test_split_author_names_drops_affiliation_markers():
    assert split_author_names(BYLINE) == [
        "Ada N. Fielding", "Bruno O. Reyes", "Chandra P. Shah", "Dmitri Q. Ilves",
    ]


def _recover_authors(model_authors, confident=True):
    summary = PaperSummary()
    summary.authors = list(model_authors)
    hints = DocumentHints(
        title_candidates=["A Study of Something"],
        author_candidates=[BYLINE],
        title_confidence="high" if confident else "low",
    )
    summarizer._recover_title_and_authors(
        summary, CondensedSummary(), [], hints
    )
    return summary.authors


def test_a_shortened_model_byline_is_restored_from_the_parser():
    """The model dropped the first author on one run of the same paper."""
    authors = _recover_authors(
        ["Bruno O. Reyes", "Chandra P. Shah", "Dmitri Q. Ilves"]
    )
    assert len(authors) == 4
    assert "Ada N. Fielding" in authors


def test_a_complete_model_byline_is_left_alone():
    complete = [
        "Ada N. Fielding", "Bruno O. Reyes", "Chandra P. Shah", "Dmitri Q. Ilves",
    ]
    assert _recover_authors(complete) == complete


def test_a_longer_model_byline_is_not_truncated_by_the_parser():
    longer = [
        "Ada N. Fielding", "Bruno O. Reyes", "Chandra P. Shah",
        "Dmitri Q. Ilves", "Elena R. Vasquez",
    ]
    assert _recover_authors(longer) == longer


def test_low_confidence_hints_do_not_override_the_model():
    model = ["Bruno O. Reyes", "Chandra P. Shah"]
    assert _recover_authors(model, confident=False) == model


def test_the_empty_author_fallback_still_works():
    summary = PaperSummary()
    hints = DocumentHints(
        title_candidates=["A Study"], author_candidates=[BYLINE],
        title_confidence="high",
    )
    summarizer._recover_title_and_authors(
        summary, CondensedSummary(), [], hints
    )
    assert summary.authors and not all(is_missing(a) for a in summary.authors)


# --------------------------------------------------------------------------- #
# front matter: a wrapped byline must not be absorbed into the title
# --------------------------------------------------------------------------- #

# Invented names carrying diacritics that a PDF byline commonly wraps at the
# superscript: the first author ends a line, its markers head the next.
WRAPPED_BYLINE_PAGE = (
    "Perceived workload and recovery in shift nursing across three hospitals\n"
    "and the role of scheduling autonomy\n"
    "Zofia Wroblewska\n"
    "1,6*, Karol Nowacki\n"
    "3, Henri Delacroix\n"
    "4, Bjorn Ostergaard\n"
    "Department of Nursing, University of Someplace\n"
    "Abstract\nWe examined workload."
)
WRAPPED_TITLE = (
    "Perceived workload and recovery in shift nursing across three hospitals "
    "and the role of scheduling autonomy"
)


def test_a_wrapped_byline_is_not_absorbed_into_the_title():
    found = extract_title_and_authors(WRAPPED_BYLINE_PAGE)
    assert found.title_candidates[0] == WRAPPED_TITLE
    assert "Wroblewska" not in found.title_candidates[0]


def test_the_first_author_of_a_wrapped_byline_is_retained():
    found = extract_title_and_authors(WRAPPED_BYLINE_PAGE)
    names = [n for line in found.author_candidates
             for n in split_author_names(line)]
    assert "Zofia Wroblewska" == names[0]


def test_every_author_of_a_wrapped_byline_is_retained():
    found = extract_title_and_authors(WRAPPED_BYLINE_PAGE)
    names = [n for line in found.author_candidates
             for n in split_author_names(line)]
    assert names == [
        "Zofia Wroblewska", "Karol Nowacki", "Henri Delacroix",
        "Bjorn Ostergaard",
    ]


def test_superscript_markers_never_reach_the_clean_names():
    found = extract_title_and_authors(WRAPPED_BYLINE_PAGE)
    names = [n for line in found.author_candidates
             for n in split_author_names(line)]
    joined = " ".join(names)
    for marker in "0123456789*":
        assert marker not in joined


@pytest.mark.parametrize(
    "name",
    [
        "Zofia Wr\u00f3blewska",       # o-acute
        "Hans M\u00fcller",            # u-umlaut
        "Jos\u00e9 Jim\u00e9nez",      # e-acute
        "\u0141ukasz Nowak",           # L-stroke (atomic, no combining form)
        "Bj\u00f8rn \u00d8st",         # o-slash (atomic)
        "Ivan \u0160imi\u0107",        # caron / acute
        "Ana Fern\u00e1ndez-Lopez",    # hyphenated
    ],
)
def test_names_with_diacritics_parse_as_names(name):
    """The ASCII-only pattern was why an accented first author was lost."""
    assert looks_like_authors(name)


def test_an_accented_byline_is_not_mistaken_for_title_text():
    page = (
        "A study of clinical handover practices in three regional units\n"
        "Zofia Wr\u00f3blewska, Hans M\u00fcller, Jos\u00e9 Jim\u00e9nez\n"
        "Department of Nursing, University of Someplace\n"
        "Abstract\nWe examined handover."
    )
    found = extract_title_and_authors(page)
    assert found.title_candidates[0] == (
        "A study of clinical handover practices in three regional units"
    )
    assert "Wr\u00f3blewska" not in found.title_candidates[0]


def test_a_title_containing_a_name_like_phrase_is_not_split():
    """A capitalised phrase inside a real title must stay in the title."""
    page = (
        "The Bayesian Approach To Signal Detection In Clinical Screening\n"
        "Zofia Wroblewska, Karol Nowacki\n"
        "Department of Nursing, University of Someplace\n"
        "Abstract\nWe examined screening."
    )
    found = extract_title_and_authors(page)
    assert found.title_candidates[0] == (
        "The Bayesian Approach To Signal Detection In Clinical Screening"
    )
    names = [n for line in found.author_candidates
             for n in split_author_names(line)]
    assert names == ["Zofia Wroblewska", "Karol Nowacki"]


def test_a_trailing_organisation_line_is_still_not_an_author():
    """The multi-name preference must survive: only wrapped lines re-join."""
    found = extract_title_and_authors(
        "Attention Is All You Need\n"
        "Ashish Vaswani, Noam Shazeer, Niki Parmar\n"
        "Google Brain\n"
        "noam@google.com\n"
        "Abstract\nThe dominant sequence transduction models..."
    )
    assert found.author_candidates == ["Ashish Vaswani, Noam Shazeer, Niki Parmar"]


def test_split_author_names_keeps_the_original_spelling():
    """Folding is for matching only; accents survive into the output."""
    assert split_author_names("Zofia Wr\u00f3blewska1, Hans M\u00fcller2") == [
        "Zofia Wr\u00f3blewska", "Hans M\u00fcller",
    ]


# --------------------------------------------------------------------------- #
# an affiliation/address line must not become an author candidate
# --------------------------------------------------------------------------- #

# Invented institutions and places, with the byline wrapped at superscripts and
# an affiliation list following it, as PDF extraction commonly emits.
AFFILIATION_AFTER_BYLINE = (
    "Perceived workload and recovery in shift nursing across three hospitals\n"
    "Zofia Wr\u00f3blewska\n"
    "1,6*, Karol Nowacki\n"
    "2, Magdalena Wis\u0142a\n"
    "Henri Delacroix\n"
    "3, Bj\u00f6rn Ostergaard\n"
    "4 & Ursula Haas\n"
    "of Northmoor, Eastbourne, Eastbourne, USA. 4Institut de Psychologie, "
    "Universit\u00e9 de Lakeside, Riverton-Sur-Mer\n"
    "Abstract\nWe examined workload."
)
EXPECTED_NAMES = [
    "Zofia Wr\u00f3blewska", "Karol Nowacki", "Magdalena Wis\u0142a",
    "Henri Delacroix", "Bj\u00f6rn Ostergaard", "Ursula Haas",
]


def _names(page):
    found = extract_title_and_authors(page)
    return found, [n for line in found.author_candidates
                   for n in split_author_names(line)]


def test_an_affiliation_line_after_a_byline_is_not_an_author_candidate():
    found, _ = _names(AFFILIATION_AFTER_BYLINE)
    joined = " ".join(found.author_candidates)
    assert "Institut" not in joined
    assert "Eastbourne" not in joined
    assert "Lakeside" not in joined


def test_institution_and_location_fragments_are_never_returned_as_names():
    _, names = _names(AFFILIATION_AFTER_BYLINE)
    for fragment in ("Eastbourne", "Universit\u00e9 de Lakeside", "Institut de Psychologie",
                     "Riverton-Sur-Mer", "Northmoor"):
        assert fragment not in names


def test_the_byline_before_the_affiliation_line_survives_intact():
    _, names = _names(AFFILIATION_AFTER_BYLINE)
    assert names == EXPECTED_NAMES


def test_accented_spellings_survive_the_affiliation_filter():
    _, names = _names(AFFILIATION_AFTER_BYLINE)
    assert "Zofia Wr\u00f3blewska" in names      # o-acute preserved
    assert "Magdalena Wis\u0142a" in names        # l-stroke preserved
    assert "Bj\u00f6rn Ostergaard" in names       # o-umlaut preserved


@pytest.mark.parametrize(
    "line",
    [
        "of Northmoor, Eastbourne, Eastbourne, USA. 4Institut de Psychologie",
        "Universit\u00e9 de Lakeside, Riverton-Sur-Mer",
        "1Department of Nursing, University of Someplace",
        "2Laboratoire de Physique, Riverton",
        "Klinik fur Innere Medizin, Someplace",
    ],
)
def test_affiliation_lines_are_recognised(line):
    assert looks_like_affiliation(line)


@pytest.mark.parametrize(
    "line",
    [
        "Zofia Wr\u00f3blewska, Karol Nowacki, Magdalena Wis\u0142a",
        "Ursula Haas & Till Kastendorf",
        "Ana Fern\u00e1ndez-Lopez, Jos\u00e9 Jimenez",
        "Zofia Wr\u00f3blewska 1,6*, Karol Nowacki 1,2,6*",
    ],
)
def test_real_bylines_are_not_mistaken_for_affiliations(line):
    """An accented surname must never be rejected as an institution."""
    assert not looks_like_affiliation(line)
    assert looks_like_authors(line)


@pytest.mark.parametrize(
    "title",
    [
        "University students' wellbeing and academic performance",
        "Laboratory automation in clinical chemistry",
    ],
)
def test_titles_containing_institution_words_are_still_titles(title):
    """The byline filter must not leak into page-furniture detection."""
    assert not is_page_furniture(title)
    assert looks_like_title_text(title)
