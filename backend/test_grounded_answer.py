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


def sel(c, span=None):
    """Wrap a chunk as SelectedEvidence, as select_evidence now returns."""
    from grounded_answer import SelectedEvidence

    return SelectedEvidence.build(c, span if span is not None else c.text, 1.0)


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
    assert "only partially addresses" not in build_prompt(
        "q?", [sel(evidence[0])], partial=False
    )


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
    evidence = [sel(chunk(PARTICIPANTS_TEXT, page=4, chunk_id="p0004-abc"))]
    prompt = build_prompt("How many participants?", evidence, partial=False)

    assert "[Evidence 1]" in prompt
    assert "Page: 4" in prompt
    assert "Chunk ID: p0004-abc" in prompt
    assert "20 Polish university students" in prompt


def test_the_prompt_forbids_outside_knowledge_and_invented_pages():
    prompt = build_prompt("q?", [sel(chunk(PARTICIPANTS_TEXT))], partial=False)
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

    assert content_contribution(question, named, set()) > content_contribution(
        question, topical, set()
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
    """Selection reorders and excerpts; it never rewrites a cosine score."""
    original = chunk(NAMED_TOOLS_CHUNK, page=6, chunk_id="a", score=0.4321)
    select_evidence(
        report(Answerability.SUPPORTED, [original]), [original],
        "What apps did students use?",
    )
    assert original.score == 0.4321


# --------------------------------------------------------------------------- #
# Quantity questions: a number attached to the requested entity
# --------------------------------------------------------------------------- #

from grounded_answer import (  # noqa: E402
    evidence_span,
    quantity_entity,
    states_quantity_of,
)

COUNT_CHUNK = (
    "The participants were 20 Polish university students of English philology. "
    "Nine of them were female and eleven were male."
)
AGE_ONLY_CHUNK = (
    "The students had been learning English for 11.38 years on average and "
    "were 22.22 years old."
)
PERCENT_CHUNK = (
    "More than half of the students (55%) regarded themselves as experienced "
    "users of their devices in everyday life."
)


@pytest.mark.parametrize(
    "question,entity",
    [
        ("How many participants were in the study?", "participant"),
        ("How many papers were analyzed?", "paper"),
        ("How many trials were included?", "trial"),
        ("What was the sample size?", "sample"),
        ("What was the number of subjects?", "subject"),
        ("What limitations did the authors identify?", None),
    ],
)
def test_quantity_entity_extraction(question, entity):
    assert quantity_entity(question) == entity


def test_a_count_attached_to_the_entity_is_recognised():
    assert states_quantity_of(COUNT_CHUNK, "participant")
    assert states_quantity_of("Twenty participants took part.", "participant")
    assert states_quantity_of("A total of 42 subjects were enrolled.", "subject")


@pytest.mark.parametrize(
    "text",
    [
        AGE_ONLY_CHUNK,          # years, not a count of people
        PERCENT_CHUNK,           # a percentage
        "See page 20 for details of the coding frame.",
        "Data were collected over 20 weeks during 2016.",
    ],
)
def test_numbers_that_are_not_counts_do_not_satisfy_a_count_question(text):
    assert not states_quantity_of(text, "participant")


def test_a_quantity_of_another_entity_does_not_satisfy_the_question():
    text = "The team analysed 40 interview transcripts."
    assert states_quantity_of(text, "transcript")
    assert not states_quantity_of(text, "participant")


def test_the_count_passage_beats_passages_that_merely_mention_the_entity():
    """The reported regression: the passage stating the count must win."""
    counting = chunk(COUNT_CHUNK, page=4, section="participants", chunk_id="count")
    mentions = chunk(
        "The participants described their routines and the participants also "
        "reported using devices between classes.",
        page=8, section="findings", chunk_id="mentions",
    )
    question = "How many participants were in the study?"

    assert content_contribution(question, counting, set()) > content_contribution(
        question, mentions, set()
    )
    selected = select_evidence(
        report(Answerability.SUPPORTED, [mentions, counting]),
        [mentions, counting],
        question,
    )
    assert selected[0].chunk_id == "count"


def test_the_question_word_many_is_not_content():
    """"many" belongs to the question, not to a passage that happens to use it."""
    counting = chunk(COUNT_CHUNK, page=4, chunk_id="count")
    uses_many = chunk(
        "Many students reported that many of their classmates did the same.",
        page=8, chunk_id="many",
    )
    question = "How many participants were in the study?"
    assert content_contribution(question, counting, set()) > content_contribution(
        question, uses_many, set()
    )


# --------------------------------------------------------------------------- #
# Evidence spans
# --------------------------------------------------------------------------- #

BROAD_DISCUSSION = (
    "The learners used their devices intuitively and spontaneously throughout "
    "the day. Many of them felt inexperienced with the technology at first. "
    "Device use has grown steadily among students in higher education. "
    "As with all studies, the study reported in this paper has some "
    "limitations. The small number of participants reduces the "
    "generalizability of the findings, and the group was homogeneous."
)


def test_a_span_narrows_a_broad_chunk_to_the_limitation_sentences():
    """The remaining blending failure: qwen received the whole chunk."""
    broad = chunk(BROAD_DISCUSSION, page=9, section="discussion", chunk_id="disc")
    span, _score = evidence_span(broad, "What limitations did the authors identify?")

    assert "has some limitations" in span
    assert "generalizability" in span
    assert "intuitively and spontaneously" not in span
    assert "felt inexperienced" not in span
    assert len(span) < len(BROAD_DISCUSSION)


def test_a_span_is_verbatim_source_text():
    broad = chunk(BROAD_DISCUSSION, page=9, chunk_id="disc")
    span, _ = evidence_span(broad, "What limitations did the authors identify?")
    collapsed = " ".join(BROAD_DISCUSSION.split())
    for sentence in span.split(". "):
        assert sentence.strip(". ") in collapsed


async def test_the_model_receives_spans_not_whole_chunks(ollama):
    broad = chunk(BROAD_DISCUSSION, page=9, section="discussion", chunk_id="disc")
    recorder = ollama(answers({"answer": "a", "used_chunk_ids": ["disc"]}))

    await generate_grounded_answer(
        "What limitations did the authors identify?",
        report(Answerability.SUPPORTED, [broad]),
        [broad],
    )
    prompt = recorder.prompts[0]
    assert "has some limitations" in prompt
    assert "intuitively and spontaneously" not in prompt


async def test_the_citation_still_points_at_the_original_chunk_and_page(ollama):
    broad = chunk(BROAD_DISCUSSION, page=9, section="discussion", chunk_id="disc")
    ollama(answers({"answer": "a", "used_chunk_ids": ["disc"]}))

    result = await generate_grounded_answer(
        "What limitations did the authors identify?",
        report(Answerability.SUPPORTED, [broad]),
        [broad],
    )
    assert [(s.chunk_id, s.page) for s in result.sources] == [("disc", 9)]
    assert "has some limitations" in result.sources[0].evidence


def test_citation_dense_prose_does_not_supply_named_items():
    """Author surnames in a literature review are not the tools being asked about."""
    citations = (
        "Prior work has looked at this (Cakir, 2015), profiling mobile learners "
        "(e.g. Byrne & Diem, 2014) and their effect on learning (Kim, 2016)."
    )
    assert named_items(citations) == set()


# --------------------------------------------------------------------------- #
# Direct-answer selection: topical evidence must not beat an actual answer
# --------------------------------------------------------------------------- #

from grounded_answer import (  # noqa: E402
    direct_answer_features,
    enumerates_named_items,
    explains_process,
    process_family,
)

ENUMERATION_CHUNK = (
    "Resources and tools. The most frequently used language tools were online "
    "dictionaries (e.g. diki, ColorDict) and a variety of mobile apps, such as "
    "Google Translate, Duolingo and Fiszkoteka."
)
TOPICAL_APPS_CHUNK = (
    "Students used their mobile devices frequently for language study and "
    "described them as convenient. Device use has grown among learners."
)
PROCEDURAL_CHUNK = (
    "[1] It should be noted that the reason for choosing this sample was for "
    "convenience since the participants were accessible to the researcher."
)
GENERIC_METHODS_CHUNK = (
    "The participants were 20 Polish university students of English philology "
    "and they took part in semi-structured interviews about their devices."
)


def test_enumeration_beats_topical_discussion():
    """The reported failure: page 6 enumerates the tools, page 7 discusses them."""
    enumerating = chunk(ENUMERATION_CHUNK, page=6, section="findings", chunk_id="enum")
    topical = chunk(TOPICAL_APPS_CHUNK, page=7, section="findings", chunk_id="topic")
    question = "What mobile apps did students use?"

    # Retrieval ranked the topical passage first.
    selected = select_evidence(
        report(Answerability.SUPPORTED, [topical, enumerating]),
        [topical, enumerating],
        question,
    )
    assert selected[0].chunk_id == "enum"


def test_a_direct_answer_wins_from_a_worse_retrieval_rank():
    """Page 6 was last of twelve candidates and must still lead."""
    enumerating = chunk(ENUMERATION_CHUNK, page=6, chunk_id="enum")
    fillers = [
        chunk(TOPICAL_APPS_CHUNK, page=p, chunk_id="f{}".format(p))
        for p in range(1, 6)
    ]
    ranked = fillers + [enumerating]

    selected = select_evidence(
        report(Answerability.SUPPORTED, ranked), ranked,
        "What mobile apps did students use?",
    )
    assert selected[0].chunk_id == "enum"


@pytest.mark.parametrize(
    "question,family",
    [
        ("How were participants recruited?", "sampling"),
        ("How were participants selected?", "sampling"),
        ("How was the sample obtained?", "sampling"),
        ("How were cases identified?", "sampling"),
        ("How were the data collected?", "collection"),
        ("How were subjects assigned?", "assignment"),
        ("What limitations did the authors identify?", None),
    ],
)
def test_procedural_question_family(question, family):
    assert process_family(question) == family


def test_a_procedural_explanation_beats_generic_methods_language():
    """The reported failure: page 11 explains the sampling, page 4 describes it."""
    explaining = chunk(PROCEDURAL_CHUNK, page=11, section="notes", chunk_id="note")
    generic = chunk(GENERIC_METHODS_CHUNK, page=4, section="participants", chunk_id="gen")
    question = "How were participants recruited?"

    selected = select_evidence(
        report(Answerability.SUPPORTED, [generic, explaining]),
        [generic, explaining],
        question,
    )
    assert selected[0].chunk_id == "note"


@pytest.mark.parametrize(
    "sentence",
    [
        "The sample was selected because the students were available locally.",
        "Volunteers were recruited through an advertisement in the department.",
        "Cases were identified by searching the hospital records.",
        "Participants were obtained using snowball referral from earlier interviewees.",
    ],
)
def test_procedural_detection_needs_no_fixed_vocabulary(sentence):
    """No sampling strategy is hard-coded; the pattern is verb plus explanation."""
    assert explains_process(sentence, "sampling"), sentence


def test_describing_a_sample_is_not_explaining_recruitment():
    assert not explains_process(GENERIC_METHODS_CHUNK, "sampling")


def test_a_title_block_is_not_an_enumeration():
    """Front matter is dense with proper nouns but lists nothing."""
    title = (
        "Research paper A look at advanced learners use of mobile devices for "
        "English language study Insights from interview data Mariusz Kruk "
        "Uniwersytet Zielonogorski Poland"
    )
    assert not enumerates_named_items(title)


def test_count_question_still_resolves_to_the_counting_passage():
    counting = chunk(COUNT_CHUNK, page=4, section="participants", chunk_id="count")
    other = chunk(TOPICAL_APPS_CHUNK, page=7, chunk_id="other")
    selected = select_evidence(
        report(Answerability.SUPPORTED, [other, counting]), [other, counting],
        "How many participants were in the study?",
    )
    assert selected[0].chunk_id == "count"
    assert selected[0].page_number == 4


def test_mean_age_still_resolves_to_the_participants_passage():
    ages = chunk(
        "The study participants were on average 22.22 years old and had been "
        "learning English for 11.38 years.",
        page=4, section="participants", chunk_id="age",
    )
    other = chunk(TOPICAL_APPS_CHUNK, page=7, chunk_id="other")
    selected = select_evidence(
        report(Answerability.SUPPORTED, [other, ages]), [other, ages],
        "What was the average age of the participants?",
    )
    assert selected[0].chunk_id == "age"
    assert selected[0].page_number == 4


def test_limitation_selection_is_untouched_by_the_direct_answer_stage():
    """Limitations have no direct-answer feature and must keep their behaviour."""
    broad = chunk(BROAD_DISCUSSION, page=9, section="discussion", chunk_id="disc")
    unrelated = chunk(TOPICAL_APPS_CHUNK, page=7, chunk_id="other")
    question = "What limitations did the authors identify?"

    assert direct_answer_features(question, BROAD_DISCUSSION) == []
    selected = select_evidence(
        report(Answerability.SUPPORTED, [broad, unrelated]), [broad, unrelated],
        question,
    )
    assert selected[0].chunk_id == "disc"
    assert "has some limitations" in selected[0].span
    assert "intuitively and spontaneously" not in selected[0].span


# --------------------------------------------------------------------------- #
# Attribution: corroborating evidence must itself be answer-bearing
#
# Recruitment failed attribution because the answer-bearing note was sent
# alongside an 82-character sentence fragment that answered nothing. The model
# returned no usable source id, and the fail-closed policy discarded a correct
# answer.
# --------------------------------------------------------------------------- #

RECRUITMENT_NOTE = (
    "[1] It should be noted that the reason for choosing this sample was for "
    "convenience since they were accessible to the researcher."
)
TOPICAL_FRAGMENT = (
    "namely a research question, description of participants, data collection "
    "tools and"
)


def test_a_lone_direct_answer_is_not_padded_with_topical_noise():
    """The fix: a fragment that answers nothing must not be sent as evidence."""
    note = chunk(RECRUITMENT_NOTE, page=11, section="notes", chunk_id="note")
    fragment = chunk(TOPICAL_FRAGMENT, page=2, section="introduction", chunk_id="frag")

    selected = select_evidence(
        report(Answerability.SUPPORTED, [fragment, note]),
        [fragment, note],
        "How were participants recruited?",
    )
    assert [item.chunk_id for item in selected] == ["note"]
    assert direct_answer_features("How were participants recruited?", TOPICAL_FRAGMENT) == []


def test_genuinely_complementary_evidence_is_still_sent():
    """Two answer-bearing passages must both survive: this is not a cap of one."""
    first = chunk(
        "The participants were 20 Polish university students of English philology.",
        page=4, section="participants", chunk_id="a",
    )
    second = chunk(
        "Table 1 shows the 20 study participants and their devices.",
        page=5, section="findings", chunk_id="b",
    )
    selected = select_evidence(
        report(Answerability.SUPPORTED, [first, second]), [first, second],
        "How many participants were in the study?",
    )
    assert [item.chunk_id for item in selected] == ["a", "b"]


async def test_recruitment_attributes_to_page_11_when_the_model_omits_ids(ollama):
    """The end-to-end failure: qwen answered but cited nothing."""
    note = chunk(RECRUITMENT_NOTE, page=11, section="notes", chunk_id="note")
    fragment = chunk(TOPICAL_FRAGMENT, page=2, section="introduction", chunk_id="frag")
    ollama(answers({
        "answer": "The sample was chosen for convenience because the "
                  "participants were accessible to the researcher.",
        "used_chunk_ids": [],
    }))

    result = await generate_grounded_answer(
        "How were participants recruited?",
        report(Answerability.SUPPORTED, [fragment, note]),
        [fragment, note],
    )

    assert result.status == "supported"
    assert [(s.chunk_id, s.page) for s in result.sources] == [("note", 11)]
    assert "convenience" in result.sources[0].evidence


async def test_multiple_complementary_passages_still_fail_closed(ollama):
    """Safety unchanged: with two real passages, an uncited answer is discarded."""
    first = chunk(
        "The participants were 20 Polish university students.",
        page=4, section="participants", chunk_id="a",
    )
    second = chunk(
        "Table 1 shows the 20 study participants and their devices.",
        page=5, section="findings", chunk_id="b",
    )
    ollama(answers({"answer": "Twenty took part.", "used_chunk_ids": []}))

    result = await generate_grounded_answer(
        "How many participants were in the study?",
        report(Answerability.SUPPORTED, [first, second]),
        [first, second],
    )
    assert result.status == "not_supported"
    assert result.sources == []


def test_the_limitations_path_is_untouched_by_the_change():
    """Limitations carry no direct-answer feature and take the other branch."""
    broad = chunk(BROAD_DISCUSSION, page=9, section="discussion", chunk_id="disc")
    other = chunk(TOPICAL_APPS_CHUNK, page=7, chunk_id="other")
    question = "What limitations did the authors identify?"

    assert direct_answer_features(question, BROAD_DISCUSSION) == []
    selected = select_evidence(
        report(Answerability.SUPPORTED, [broad, other]), [broad, other], question
    )
    assert selected[0].chunk_id == "disc"
    assert "has some limitations" in selected[0].span
