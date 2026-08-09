"""Tests for Phase 3B grounded answering. Ollama is always stubbed."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

import ollama_client
from answerability import Answerability, AnswerabilityReport
from conftest import FakeClient, hashed_vector
from embedding_cache import EMBEDDING_CACHE, document_key
from embeddings import RankedChunk
from fixtures import interview_pages_payload
from grounded_answer import (
    ABSTAIN_ANSWER,
    PARTIAL_NOTE,
    SUPPORTED_NOTE,
    UNATTRIBUTABLE_ANSWER,
    build_prompt,
    evidence_snippet,
    generate_grounded_answer,
    select_evidence,
    validate_used_ids,
)
from main import app
from schemas import PageInput

client = TestClient(app)

PARTICIPANTS_TEXT = (
    "The participants were 20 Polish university students enrolled on a "
    "philology programme. The study participants were on average 22.22 years "
    "old. Nine of them were female and eleven were male."
)
METHODS_TEXT = (
    "Semi-structured interviews were audio recorded, transcribed verbatim and "
    "examined using thematic analysis by two independent coders."
)


def chunk(text, page=4, section="participants", chunk_id=None, score=0.6):
    return RankedChunk(
        chunk_id=chunk_id or "p{:04d}-{}".format(page, abs(hash(text)) % 10 ** 10),
        page_number=page,
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        score=score,
    )


def report(status, supporting=(), reason="reason."):
    return AnswerabilityReport(
        status=status,
        reason=reason,
        supporting_chunk_ids=[c.chunk_id for c in supporting],
        supporting_pages=sorted({c.page_number for c in supporting}),
        evidence_count=len(supporting),
    )


class Recorder:
    """Counts generation calls and lets a test script the reply."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.handler(url, payload)

    @property
    def generate_calls(self):
        return [p for url, p in self.calls if url.endswith("/api/generate")]

    @property
    def prompts(self):
        return [p["prompt"] for p in self.generate_calls]


def answers(body):
    """A well-behaved generation server."""

    def handler(url, payload):
        if url.endswith("/api/embed"):
            return httpx.Response(
                200,
                json={"embeddings": [hashed_vector(t) for t in payload["input"]]},
            )
        return httpx.Response(200, json={"response": json.dumps(body)})

    return handler


@pytest.fixture
def ollama(monkeypatch):
    import embeddings

    embeddings.reset_capabilities()
    ollama_client.reset_capabilities()
    EMBEDDING_CACHE.clear()

    def install(handler):
        recorder = Recorder(handler)
        monkeypatch.setattr(
            ollama_client.httpx, "AsyncClient", lambda **kw: FakeClient(recorder)
        )
        return recorder

    return install


# --------------------------------------------------------------------------- #
# the three policies
# --------------------------------------------------------------------------- #


async def test_supported_question_calls_generation(ollama):
    evidence = [chunk(PARTICIPANTS_TEXT)]
    recorder = ollama(
        answers({"answer": "Twenty students took part.",
                 "used_chunk_ids": [evidence[0].chunk_id]})
    )

    result = await generate_grounded_answer(
        "How many participants were there?",
        report(Answerability.SUPPORTED, evidence),
        evidence,
    )

    assert len(recorder.generate_calls) == 1
    assert result.status == "supported"
    assert result.answer == "Twenty students took part."
    assert result.confidence_note == SUPPORTED_NOTE
    assert [s.page for s in result.sources] == [4]


async def test_not_supported_never_calls_generation(ollama):
    """The abstention path must not reach qwen3:8b at all."""
    recorder = ollama(answers({"answer": "made up", "used_chunk_ids": []}))

    result = await generate_grounded_answer(
        "What was the participants' mean IQ?",
        report(Answerability.NOT_SUPPORTED),
        [chunk(PARTICIPANTS_TEXT)],
    )

    assert recorder.calls == [], "the generation client was called"
    assert result.status == "not_supported"
    assert result.answer == ABSTAIN_ANSWER
    assert result.sources == []
    assert result.diagnostics.generation_called is False


async def test_partially_supported_generates_with_a_partial_instruction(ollama):
    evidence = [chunk(PARTICIPANTS_TEXT)]
    recorder = ollama(
        answers({"answer": "The paper suggests twenty took part.",
                 "used_chunk_ids": [evidence[0].chunk_id]})
    )

    result = await generate_grounded_answer(
        "How many participants were there?",
        report(Answerability.PARTIALLY_SUPPORTED, evidence),
        evidence,
    )

    assert result.status == "partially_supported"
    assert result.confidence_note == PARTIAL_NOTE
    prompt = recorder.prompts[0]
    assert "only partially addresses" in prompt
    assert "cautious wording" in prompt


def test_the_partial_instruction_is_absent_for_supported():
    evidence = [chunk(PARTICIPANTS_TEXT)]
    assert "only partially addresses" not in build_prompt("q?", evidence, partial=False)


# --------------------------------------------------------------------------- #
# the model sees only what it is given
# --------------------------------------------------------------------------- #


async def test_the_model_sees_only_selected_evidence(ollama):
    kept = chunk(PARTICIPANTS_TEXT, page=4, chunk_id="keep-1")
    dropped = chunk("Unrelated background prose about prior work.", page=2,
                    section="background", chunk_id="drop-1")
    recorder = ollama(answers({"answer": "a", "used_chunk_ids": ["keep-1"]}))

    await generate_grounded_answer(
        "How many participants were there?",
        report(Answerability.SUPPORTED, [kept]),
        [kept, dropped],
    )

    prompt = recorder.prompts[0]
    assert "keep-1" in prompt
    assert "drop-1" not in prompt
    assert "Unrelated background prose" not in prompt


def test_evidence_is_presented_with_real_provenance():
    evidence = [chunk(PARTICIPANTS_TEXT, page=4, chunk_id="p0004-abc")]
    prompt = build_prompt("How many participants?", evidence, partial=False)

    assert "[Evidence 1]" in prompt
    assert "Page: 4" in prompt
    assert "Chunk ID: p0004-abc" in prompt
    assert "20 Polish university students" in prompt


def test_the_prompt_forbids_outside_knowledge_and_invented_pages():
    prompt = build_prompt("q?", [chunk(PARTICIPANTS_TEXT)], partial=False)
    for rule in (
        "only the passages provided",
        "Do not use outside knowledge",
        "Never invent names, numbers",
        "Do not claim causation",
        "statistical significance",
        "Page numbers are supplied metadata",
    ):
        assert rule in prompt, rule


# --------------------------------------------------------------------------- #
# source validation
# --------------------------------------------------------------------------- #


def test_unknown_chunk_ids_are_rejected():
    selected = [chunk(PARTICIPANTS_TEXT, chunk_id="real-1")]
    known, unknown = validate_used_ids(["real-1", "invented-9", "p9999-xyz"], selected)
    assert known == ["real-1"]
    assert unknown == ["invented-9", "p9999-xyz"]


async def test_valid_ids_map_to_parser_owned_pages(ollama):
    first = chunk(PARTICIPANTS_TEXT, page=4, chunk_id="a")
    second = chunk(METHODS_TEXT, page=7, section="methods", chunk_id="b")
    ollama(answers({"answer": "x", "used_chunk_ids": ["b"]}))

    result = await generate_grounded_answer(
        "How were the data analysed?",
        report(Answerability.SUPPORTED, [first, second]),
        [first, second],
    )

    assert [(s.chunk_id, s.page) for s in result.sources] == [("b", 7)]
    assert result.diagnostics.cited_pages == [7]


async def test_a_page_the_model_invents_cannot_reach_the_response(ollama):
    """The model has no page field at all; pages come from the parser."""
    evidence = [chunk(PARTICIPANTS_TEXT, page=4, chunk_id="a")]
    ollama(answers({"answer": "See page 99.", "used_chunk_ids": ["a"], "page": 99}))

    result = await generate_grounded_answer(
        "q?", report(Answerability.SUPPORTED, evidence), evidence
    )
    assert [s.page for s in result.sources] == [4]
    assert 99 not in result.diagnostics.cited_pages


async def test_an_unattributable_answer_is_discarded(ollama):
    """Fail closed: several passages sent, none cited back."""
    evidence = [
        chunk(PARTICIPANTS_TEXT, page=4, chunk_id="a"),
        chunk(METHODS_TEXT, page=7, section="methods", chunk_id="b"),
    ]
    ollama(answers({"answer": "Something confident.", "used_chunk_ids": ["ghost"]}))

    result = await generate_grounded_answer(
        "q?", report(Answerability.SUPPORTED, evidence), evidence
    )

    assert result.status == "not_supported"
    assert result.answer == UNATTRIBUTABLE_ANSWER
    assert result.sources == []


async def test_single_evidence_attribution_is_unambiguous(ollama):
    """One passage supplied: attribution needs no guessing."""
    evidence = [chunk(PARTICIPANTS_TEXT, page=4, chunk_id="only")]
    ollama(answers({"answer": "Twenty took part.", "used_chunk_ids": []}))

    result = await generate_grounded_answer(
        "q?", report(Answerability.SUPPORTED, evidence), evidence
    )
    assert result.status == "supported"
    assert [s.chunk_id for s in result.sources] == ["only"]


# --------------------------------------------------------------------------- #
# evidence selection and snippets
# --------------------------------------------------------------------------- #


def test_selection_prefers_the_supporting_set():
    supporting = chunk(PARTICIPANTS_TEXT, page=4, chunk_id="s")
    other = chunk("High cosine but not supporting.", page=2, chunk_id="o")
    selected = select_evidence(
        report(Answerability.SUPPORTED, [supporting]), [other, supporting]
    )
    assert [c.chunk_id for c in selected] == ["s"]


def test_overlapping_evidence_is_reduced():
    """Adjacent chunks share a 60-word overlap by construction."""
    shared = PARTICIPANTS_TEXT + " " + METHODS_TEXT
    first = chunk(shared, page=4, chunk_id="a")
    second = chunk(shared + " One extra sentence.", page=4, chunk_id="b")
    third = chunk("Entirely different prose about limitations.", page=9,
                  section="discussion", chunk_id="c")

    selected = select_evidence(
        report(Answerability.SUPPORTED, [first, second, third]),
        [first, second, third],
    )
    assert [c.chunk_id for c in selected] == ["a", "c"]


def test_selection_is_capped_but_never_empty():
    chunks = [chunk("Distinct passage number {}.".format(i) * 6, page=i, chunk_id=str(i))
              for i in range(1, 9)]
    selected = select_evidence(report(Answerability.SUPPORTED, chunks), chunks, limit=4)
    assert 1 <= len(selected) <= 4

    assert select_evidence(report(Answerability.SUPPORTED), []) == []
    fallback = select_evidence(report(Answerability.SUPPORTED), [chunks[0]])
    assert len(fallback) == 1, "no supporting ids must not mean no evidence"


def test_a_snippet_is_taken_verbatim_from_the_chunk():
    snippet = evidence_snippet(PARTICIPANTS_TEXT, "What was the average age?")
    assert snippet in " ".join(PARTICIPANTS_TEXT.split())
    assert "22.22" in snippet
    assert len(snippet) <= 330


def test_a_snippet_is_bounded_for_a_long_chunk():
    long_text = " ".join("Sentence number {} of the passage.".format(i) for i in range(200))
    assert len(evidence_snippet(long_text, "question")) <= 330


# --------------------------------------------------------------------------- #
# failure modes
# --------------------------------------------------------------------------- #


def post_ask(question="How many participants were in the study?", **kwargs):
    body = {
        "question": question,
        "filename": "interview.pdf",
        "pages": interview_pages_payload(),
    }
    body.update(kwargs)
    return client.post("/ask", json=body)


def test_ollama_unavailable(ollama):
    def handler(url, payload):
        raise httpx.ConnectError("connection refused")

    ollama(handler)
    response = post_ask()
    assert response.status_code == 503
    assert "ollama serve" in response.json()["detail"]


def test_generation_model_not_installed(ollama):
    def handler(url, payload):
        if url.endswith("/api/embed"):
            return httpx.Response(
                200, json={"embeddings": [hashed_vector(t) for t in payload["input"]]}
            )
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    ollama(handler)
    response = post_ask()
    assert response.status_code == 503
    assert "ollama pull" in response.json()["detail"]


def test_generation_timeout(ollama):
    def handler(url, payload):
        if url.endswith("/api/embed"):
            return httpx.Response(
                200, json={"embeddings": [hashed_vector(t) for t in payload["input"]]}
            )
        raise httpx.ReadTimeout("too slow")

    ollama(handler)
    assert post_ask().status_code == 504


def test_malformed_generation_fails_safely(ollama):
    def handler(url, payload):
        if url.endswith("/api/embed"):
            return httpx.Response(
                200, json={"embeddings": [hashed_vector(t) for t in payload["input"]]}
            )
        return httpx.Response(200, json={"response": "not json at all"})

    ollama(handler)
    response = post_ask()
    assert response.status_code == 502
    assert "detail" in response.json()
    assert "Traceback" not in response.text


def test_empty_generation_fails_safely(ollama):
    def handler(url, payload):
        if url.endswith("/api/embed"):
            return httpx.Response(
                200, json={"embeddings": [hashed_vector(t) for t in payload["input"]]}
            )
        return httpx.Response(200, json={"response": json.dumps({"answer": "  "})})

    ollama(handler)
    assert post_ask().status_code == 502


# --------------------------------------------------------------------------- #
# the endpoint
# --------------------------------------------------------------------------- #


def test_ask_rejects_an_empty_question():
    assert post_ask(question="   ").status_code == 400
    assert client.post("/ask", json={"pages": []}).status_code == 400


def test_ask_rejects_a_paper_with_no_pages():
    response = client.post("/ask", json={"question": "q?", "pages": []})
    assert response.status_code == 400


def test_ask_rejects_a_paper_with_no_text():
    response = client.post(
        "/ask",
        json={"question": "q?", "pages": [{"page_number": 1, "text": "   "}]},
    )
    assert response.status_code == 422


def test_ask_end_to_end_returns_the_response_contract(ollama):
    ollama(answers({"answer": "Twenty students took part.", "used_chunk_ids": []}))
    body = post_ask().json()

    assert set(body) == {"status", "answer", "sources", "confidence_note", "diagnostics"}
    assert body["status"] in {"supported", "partially_supported", "not_supported"}
    if body["sources"]:
        source = body["sources"][0]
        assert set(source) >= {"page", "chunk_id", "evidence"}
        assert isinstance(source["page"], int)


def test_ask_abstains_without_generation_end_to_end(ollama):
    recorder = ollama(answers({"answer": "fabricated", "used_chunk_ids": []}))
    body = post_ask(question="What was the participants' mean IQ?").json()

    assert body["status"] == "not_supported"
    assert body["sources"] == []
    assert recorder.generate_calls == [], "generation was called for an abstention"


def test_a_reference_chunk_can_never_be_cited(ollama):
    """Phase 3A excludes the bibliography; generation cannot bring it back."""
    ollama(answers({"answer": "x", "used_chunk_ids": ["p0011-references"]}))
    body = post_ask(question="Which sources did the authors cite?").json()
    for source in body["sources"]:
        assert "Godwin-Jones" not in source["evidence"]
        assert source["section"] != "references"


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


def test_the_paper_is_embedded_once_across_questions(ollama):
    """The corpus is embedded once. Each question then costs only its own
    embedding - twice, because answerability and evidence selection rank over
    different windows and each embeds the question. That second call is the
    documented price of keeping the accepted answerability window untouched."""
    recorder = ollama(answers({"answer": "a", "used_chunk_ids": []}))

    def embed_calls():
        return len([u for u, _ in recorder.calls if u.endswith("/api/embed")])

    post_ask(question="How many participants were in the study?")
    after_first = embed_calls()
    post_ask(question="How were the data analysed?")
    after_second = embed_calls()

    # No re-embedding of the corpus: the delta is question embeddings only.
    assert after_second - after_first <= 2


def test_two_different_papers_do_not_share_a_cache_entry():
    a = [PageInput(page_number=1, text="First paper about sleep.")]
    b = [PageInput(page_number=1, text="Second paper about language.")]
    assert document_key("a.pdf", a) != document_key("b.pdf", b)
    assert document_key("same.pdf", a) == document_key("same.pdf", list(a))


def test_the_cache_is_bounded():
    EMBEDDING_CACHE.clear()
    for index in range(EMBEDDING_CACHE.capacity + 3):
        EMBEDDING_CACHE.put("key-{}".format(index), [])
    assert EMBEDDING_CACHE.stats()["papers"] == EMBEDDING_CACHE.capacity
    EMBEDDING_CACHE.clear()


# --------------------------------------------------------------------------- #
# Evidence selection: contribution scoring
#
# Selection used to take the first few answerability-supporting passages in
# rank order, so a passage naming exactly what was asked for could lose to one
# that merely discussed the topic.
# --------------------------------------------------------------------------- #

from grounded_answer import (  # noqa: E402
    asks_for_named_items,
    content_contribution,
    named_items,
)

NAMED_TOOLS_CHUNK = (
    "Students reported using Duolingo and Fiszkoteka for vocabulary work, "
    "Google Translate and PONS for quick lookups, and Voscreen for listening "
    "practice. Several also used WhatsApp to talk to native speakers."
)
TOPICAL_CHUNK = (
    "Students used their devices frequently for language study and described "
    "them as convenient. Device use has grown steadily among learners in "
    "higher education and is now part of everyday study."
)
EXPLICIT_LIMITATIONS = (
    "As with all studies, the study reported here has some limitations. The "
    "small number of participants reduces the generalizability of the "
    "findings, and the group was homogeneous and drawn from one institution."
)
NEARBY_FINDING = (
    "The interviewees also had relatively limited experience of using their "
    "devices for study, which shaped how they described their routines."
)


def test_a_named_item_question_selects_the_chunk_with_the_names():
    """The reported failure: the passage naming the tools must be cited."""
    named = chunk(NAMED_TOOLS_CHUNK, page=6, section="findings", chunk_id="named")
    topical = chunk(TOPICAL_CHUNK, page=5, section="findings", chunk_id="topical")

    # Retrieval put the topical passage first.
    selected = select_evidence(
        report(Answerability.SUPPORTED, [topical, named]),
        [topical, named],
        "What mobile apps did students use?",
    )
    assert "named" in [c.chunk_id for c in selected]
    assert selected[0].chunk_id == "named", "the naming passage should lead"


def test_a_topical_chunk_does_not_displace_a_direct_answer():
    named = chunk(NAMED_TOOLS_CHUNK, page=6, chunk_id="named")
    topical = chunk(TOPICAL_CHUNK, page=5, chunk_id="topical")
    question = "What tools did students use?"

    assert content_contribution(
        question, named, set(), asks_for_named_items(question)
    ) > content_contribution(
        question, topical, set(), asks_for_named_items(question)
    )


def test_named_item_detection_is_generic():
    assert asks_for_named_items("What mobile apps did students use?")
    assert asks_for_named_items("Which databases were searched?")
    # Questions our type taxonomy already covers are handled by their family.
    assert not asks_for_named_items("What limitations did the authors identify?")
    assert not asks_for_named_items("How were participants recruited?")

    found = named_items(NAMED_TOOLS_CHUNK)
    assert {"Duolingo", "Fiszkoteka", "Voscreen", "WhatsApp"} <= found
    assert "Students" not in found, "sentence-initial words are not names"


def test_participant_codes_are_not_treated_as_names():
    assert named_items("S10 said this. S8 disagreed. S9 was unsure.") == set()


def test_a_limitations_question_prefers_explicit_limitation_evidence():
    explicit = chunk(EXPLICIT_LIMITATIONS, page=9, section="discussion", chunk_id="lim")
    nearby = chunk(NEARBY_FINDING, page=9, section="discussion", chunk_id="near")

    selected = select_evidence(
        report(Answerability.SUPPORTED, [nearby, explicit]),
        [nearby, explicit],
        "What limitations did the authors identify?",
    )
    assert [c.chunk_id for c in selected] == ["lim"]


def test_nearby_discussion_is_not_cited_just_for_being_adjacent():
    """Selection stops once the marginal passage adds little."""
    explicit = chunk(EXPLICIT_LIMITATIONS, page=9, chunk_id="lim")
    nearby = chunk(NEARBY_FINDING, page=9, chunk_id="near")
    unrelated = chunk(TOPICAL_CHUNK, page=5, chunk_id="topic")

    selected = select_evidence(
        report(Answerability.SUPPORTED, [explicit, nearby, unrelated]),
        [explicit, nearby, unrelated],
        "What limitations did the authors identify?",
    )
    assert [c.chunk_id for c in selected] == ["lim"]
    assert len(selected) < 3, "the cap must not be padded out"


def test_retrieval_rank_alone_does_not_earn_a_citation():
    answering = chunk(NAMED_TOOLS_CHUNK, page=6, chunk_id="answer")
    irrelevant = chunk(
        "The journal was founded in 1998 and appears twice a year.",
        page=1, section="abstract", chunk_id="noise",
    )
    selected = select_evidence(
        report(Answerability.SUPPORTED, [irrelevant, answering]),
        [irrelevant, answering],
        "What mobile apps did students use?",
    )
    assert "noise" not in [c.chunk_id for c in selected]


def test_selection_never_empties_when_nothing_scores():
    supporting = chunk("Opaque prose with no shared vocabulary.", page=3, chunk_id="s")
    selected = select_evidence(
        report(Answerability.SUPPORTED, [supporting]), [supporting], "zzz qqq?"
    )
    assert [c.chunk_id for c in selected] == ["s"]


def test_raw_retrieval_scores_are_never_modified():
    original = chunk(NAMED_TOOLS_CHUNK, page=6, chunk_id="a", score=0.4321)
    selected = select_evidence(
        report(Answerability.SUPPORTED, [original]), [original],
        "What apps did students use?",
    )
    assert selected[0].score == 0.4321
