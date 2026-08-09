"""Tests for the embedding and ranking layer. Ollama is always stubbed."""

import math

import httpx
import pytest

import embeddings as emb
import ollama_client
from conftest import FakeClient
from embeddings import (
    EmbeddedChunk,
    EmbeddingError,
    RankedChunk,
    cosine_similarity,
    embed_chunks,
    embed_text,
    rank_chunks,
)
from ollama_client import OllamaError
from retrieval import RetrievalChunk

# --------------------------------------------------------------------------- #
# a stub embedder with real (if crude) semantics
# --------------------------------------------------------------------------- #

# A bag-of-words "embedding": each dimension is one vocabulary term. Cosine
# over these vectors behaves like lexical overlap, which is enough to show that
# ranking works without pretending to be a real model.
VOCAB = (
    "participant student undergraduate age female recruited "
    "interview structured transcribed thematic analysis "
    "duolingo memrise anki mobile app used "
    "limitation sample generalise single site "
    "vocabulary language learning daily"
).split()

VOCAB_INDEX = {word: index for index, word in enumerate(VOCAB)}

# Without this, "applications" in the question and "application" in the chunk
# would occupy different dimensions and never match.
STEMS = {
    "apps": "app",
    "application": "app",
    "applications": "app",
    "use": "used",
    "uses": "used",
    "using": "used",
    "analysed": "analysis",
    "analyses": "analysis",
    "analyzed": "analysis",
}

PUNCTUATION = str.maketrans({char: " " for char in "?.,;:()[]\"'"})


def stem(token):
    if token in STEMS:
        return STEMS[token]
    if token in VOCAB_INDEX:
        return token
    return token[:-1] if token.endswith("s") else token


def lexical_vector(text):
    vector = [0.0] * len(VOCAB)
    for token in text.lower().translate(PUNCTUATION).split():
        position = VOCAB_INDEX.get(stem(token))
        if position is not None:
            vector[position] += 1.0
    if not any(vector):
        # Keep every vector non-zero; validation rejects all-zero vectors.
        vector[0] = 0.001
    return vector


class Recorder:
    """Counts requests and lets a test script the response."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, url, payload):
        self.calls.append((url, payload))
        return self.handler(url, payload)

    @property
    def inputs(self):
        """Every text sent for embedding, across both endpoint styles."""
        sent = []
        for _url, payload in self.calls:
            if "input" in payload:
                sent.extend(payload["input"])
            elif "prompt" in payload:
                sent.append(payload["prompt"])
        return sent


@pytest.fixture
def fake_ollama(monkeypatch):
    emb.reset_capabilities()
    ollama_client.reset_capabilities()

    def install(handler):
        recorder = Recorder(handler)
        monkeypatch.setattr(
            ollama_client.httpx, "AsyncClient", lambda **kw: FakeClient(recorder)
        )
        return recorder

    return install


def lexical_handler(url, payload):
    """A well-behaved /api/embed server backed by the lexical stub."""
    assert url.endswith("/api/embed")
    return httpx.Response(
        200,
        json={"embeddings": [lexical_vector(text) for text in payload["input"]]},
    )


@pytest.fixture
def lexical(fake_ollama):
    return fake_ollama(lexical_handler)


# --------------------------------------------------------------------------- #
# the paper used for the semantic-retrieval test
# --------------------------------------------------------------------------- #

PAPER = {
    "demographics": (
        "Participants were 42 undergraduate students, mean age 20, of whom "
        "60 per cent were female and recruited from one language department."
    ),
    "methods": (
        "Data were gathered through semi structured interviews, transcribed "
        "verbatim and examined with thematic analysis by two researchers."
    ),
    "apps": (
        "Students used the Duolingo and Memrise mobile applications daily, and "
        "a smaller group used the Anki application for vocabulary revision."
    ),
    "limitations": (
        "A limitation is the single site sample, so the findings may not "
        "generalise beyond this cohort of language learning students."
    ),
}


def paper_chunks():
    chunks = []
    for index, (name, text) in enumerate(sorted(PAPER.items()), start=1):
        chunks.append(
            RetrievalChunk(
                chunk_id="chunk-{}".format(name),
                page_number=index,
                section=name,
                text=text,
                start_char=0,
                end_char=len(text),
            )
        )
    return chunks


def make_embedded(chunk_id, vector, page_number=1, start_char=0, section="body"):
    text = (
        "Body text of {} describing what the study did, long enough to count "
        "as evidence rather than a heading remnant.".format(chunk_id)
    )
    return EmbeddedChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        section=section,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
        embedding=list(vector),
    )


# --------------------------------------------------------------------------- #
# cosine similarity
# --------------------------------------------------------------------------- #


def test_identical_vectors_score_one():
    vector = [0.2, -0.5, 0.9, 0.1]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0, 0.0], [0.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one():
    assert cosine_similarity([1.0, 2.0], [-1.0, -2.0]) == pytest.approx(-1.0)


def test_similarity_ignores_magnitude():
    assert cosine_similarity([1.0, 1.0], [7.0, 7.0]) == pytest.approx(1.0)


def test_similarity_is_symmetric():
    a, b = [0.3, 0.1, -0.7], [0.9, -0.2, 0.4]
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))


def test_similarity_is_clamped_to_the_valid_range():
    for _ in range(50):
        vector = [1e-8, 1e-8, 1e-8]
        assert -1.0 <= cosine_similarity(vector, vector) <= 1.0


def test_empty_vectors_are_rejected():
    with pytest.raises(EmbeddingError):
        cosine_similarity([], [1.0])
    with pytest.raises(EmbeddingError):
        cosine_similarity([1.0], [])


def test_mismatched_dimensions_are_rejected():
    with pytest.raises(EmbeddingError) as error:
        cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])
    assert "dimensions do not match" in str(error.value)


def test_zero_magnitude_vector_scores_zero():
    """No direction, so no similarity. Never reached in practice."""
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


async def test_embed_text_returns_a_vector(lexical):
    vector = await embed_text("students used mobile apps")
    assert len(vector) == len(VOCAB)
    assert any(vector)
    assert len(lexical.calls) == 1


async def test_embed_text_rejects_blank_input(lexical):
    for blank in ("", "   ", "\n\t"):
        with pytest.raises(EmbeddingError):
            await embed_text(blank)
    assert lexical.calls == [], "no request should reach Ollama"


async def test_embed_chunks_preserves_order_and_metadata(lexical):
    chunks = paper_chunks()
    embedded = await embed_chunks(chunks)

    assert len(embedded) == len(chunks)
    for original, result in zip(chunks, embedded):
        assert result.chunk_id == original.chunk_id
        assert result.page_number == original.page_number
        assert result.section == original.section
        assert result.text == original.text
        assert result.start_char == original.start_char
        assert result.end_char == original.end_char
        assert len(result.embedding) == len(VOCAB)


async def test_embed_chunks_batches_requests(lexical):
    chunks = paper_chunks() * 3  # 12 chunks
    await embed_chunks(chunks, batch_size=5)
    assert [len(payload["input"]) for _url, payload in lexical.calls] == [5, 5, 2]


async def test_embed_chunks_handles_empty_input(lexical):
    assert await embed_chunks([]) == []
    assert lexical.calls == []


async def test_blank_chunks_are_skipped(lexical):
    chunks = paper_chunks()
    chunks.append(
        RetrievalChunk(
            chunk_id="blank", page_number=9, section="", text="   ",
            start_char=0, end_char=3,
        )
    )
    embedded = await embed_chunks(chunks)
    assert "blank" not in {chunk.chunk_id for chunk in embedded}


async def test_falls_back_to_the_legacy_endpoint(fake_ollama):
    def handler(url, payload):
        if url.endswith("/api/embed"):
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json={"embedding": lexical_vector(payload["prompt"])})

    recorder = fake_ollama(handler)
    embedded = await embed_chunks(paper_chunks()[:2])

    assert len(embedded) == 2
    assert any(url.endswith("/api/embeddings") for url, _ in recorder.calls)


# --------------------------------------------------------------------------- #
# embedding response validation
# --------------------------------------------------------------------------- #


def responds(body):
    return lambda url, payload: httpx.Response(200, json=body)


@pytest.mark.parametrize(
    "body,fragment",
    [
        ({"embeddings": [[]]}, "empty vector"),
        ({"embeddings": [[0.0, 0.0, 0.0]]}, "all-zero"),
        ({"embeddings": [["a", "b"]]}, "non-numeric"),
        ({"embeddings": ["not a vector"]}, "instead of a vector"),
        ({"embeddings": []}, "0 vectors for 1 inputs"),
        ({"embeddings": [[1.0], [2.0]]}, "2 vectors for 1 inputs"),
        ({"nothing": True}, "no vectors for 1 inputs"),
    ],
)
async def test_malformed_embedding_responses_are_rejected(fake_ollama, body, fragment):
    fake_ollama(responds(body))
    with pytest.raises(EmbeddingError) as error:
        await embed_text("hello")
    assert fragment in str(error.value)
    assert error.value.status_code == 502


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
async def test_non_finite_values_are_rejected(fake_ollama, literal):
    """httpx refuses to encode these, so the body is sent as raw JSON text."""
    fake_ollama(
        lambda url, payload: httpx.Response(
            200,
            content='{{"embeddings": [[1.0, {}]]}}'.format(literal),
            headers={"content-type": "application/json"},
        )
    )
    with pytest.raises(EmbeddingError) as error:
        await embed_text("hello")
    assert "non-finite" in str(error.value)


async def test_inconsistent_dimensions_are_rejected(fake_ollama):
    def handler(url, payload):
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 2.0, 3.0] if i == 0 else [1.0, 2.0]
                                 for i, _ in enumerate(payload["input"])]},
        )

    fake_ollama(handler)
    with pytest.raises(EmbeddingError) as error:
        await embed_chunks(paper_chunks())
    assert "inconsistent dimensions" in str(error.value)


async def test_dimensions_must_stay_consistent_across_batches(fake_ollama):
    state = {"calls": 0}

    def handler(url, payload):
        state["calls"] += 1
        width = 4 if state["calls"] == 1 else 3
        return httpx.Response(
            200, json={"embeddings": [[1.0] * width for _ in payload["input"]]}
        )

    fake_ollama(handler)
    with pytest.raises(EmbeddingError) as error:
        await embed_chunks(paper_chunks(), batch_size=2)
    assert "inconsistent dimensions" in str(error.value)


async def test_non_json_response_is_rejected(fake_ollama):
    fake_ollama(lambda url, payload: httpx.Response(200, text="not json"))
    with pytest.raises(EmbeddingError):
        await embed_text("hello")


# --------------------------------------------------------------------------- #
# Ollama failure modes
# --------------------------------------------------------------------------- #


async def test_ollama_unavailable(fake_ollama):
    def handler(url, payload):
        raise httpx.ConnectError("connection refused")

    fake_ollama(handler)
    with pytest.raises(OllamaError) as error:
        await embed_text("hello")
    assert error.value.status_code == 503
    assert "ollama serve" in str(error.value)


async def test_embedding_model_not_installed(fake_ollama):
    def handler(url, payload):
        return httpx.Response(
            404, json={"error": "model 'nomic-embed-text' not found, try pulling it"}
        )

    fake_ollama(handler)
    with pytest.raises(OllamaError) as error:
        await embed_text("hello")
    assert error.value.status_code == 503
    assert "ollama pull" in str(error.value)


async def test_timeout(fake_ollama):
    def handler(url, payload):
        raise httpx.ReadTimeout("too slow")

    fake_ollama(handler)
    with pytest.raises(OllamaError) as error:
        await embed_text("hello")
    assert error.value.status_code == 504
    assert "timed out" in str(error.value)


async def test_server_error_is_surfaced(fake_ollama):
    fake_ollama(lambda url, payload: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(OllamaError) as error:
        await embed_text("hello")
    assert error.value.status_code == 502
    assert "boom" in str(error.value)


# --------------------------------------------------------------------------- #
# ranking
# --------------------------------------------------------------------------- #


async def test_ranking_is_ordered_by_similarity(fake_ollama):
    query = [1.0, 0.0, 0.0]
    fake_ollama(responds({"embeddings": [query]}))

    chunks = [
        make_embedded("far", [0.0, 1.0, 0.0], page_number=1),
        make_embedded("exact", [1.0, 0.0, 0.0], page_number=2),
        make_embedded("near", [0.9, 0.4, 0.0], page_number=3),
    ]
    ranked = await rank_chunks("q", chunks, top_k=3)

    assert [item.chunk_id for item in ranked] == ["exact", "near", "far"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[-1].score == pytest.approx(0.0)
    assert all(
        first.score >= second.score for first, second in zip(ranked, ranked[1:])
    )


async def test_ties_break_on_document_order_then_id(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))

    chunks = [
        make_embedded("z", [1.0, 0.0], page_number=3, start_char=10),
        make_embedded("a", [1.0, 0.0], page_number=1, start_char=50),
        make_embedded("m", [1.0, 0.0], page_number=1, start_char=10),
        make_embedded("b", [1.0, 0.0], page_number=1, start_char=10),
    ]
    ranked = await rank_chunks("q", chunks, top_k=4)

    assert [item.chunk_id for item in ranked] == ["b", "m", "a", "z"]


async def test_tie_breaking_is_stable_across_input_order(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))
    chunks = [
        make_embedded("z", [1.0, 0.0], page_number=3),
        make_embedded("a", [1.0, 0.0], page_number=1),
        make_embedded("m", [1.0, 0.0], page_number=2),
    ]
    first = await rank_chunks("q", chunks, top_k=3)
    second = await rank_chunks("q", list(reversed(chunks)), top_k=3)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


async def test_scores_within_float_noise_still_tie_break_deterministically(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))
    chunks = [
        make_embedded("second", [1.0, 1e-15], page_number=2),
        make_embedded("first", [1.0, 0.0], page_number=1),
    ]
    ranked = await rank_chunks("q", chunks, top_k=2)
    assert [item.chunk_id for item in ranked] == ["first", "second"]


async def test_question_is_embedded_exactly_once(fake_ollama):
    recorder = fake_ollama(responds({"embeddings": [[1.0, 0.0, 0.0]]}))
    chunks = [make_embedded("c{}".format(i), [1.0, 0.0, 0.0]) for i in range(25)]

    await rank_chunks("What did students use?", chunks, top_k=5)

    assert len(recorder.calls) == 1
    assert recorder.inputs == ["What did students use?"]


async def test_ranking_preserves_chunk_metadata(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))
    chunk = EmbeddedChunk(
        chunk_id="p0004-abc123",
        page_number=4,
        section="participants",
        text=(
            "Eighteen newly qualified nurses took part, recruited from three "
            "acute trusts in the north of England."
        ),
        start_char=120,
        end_char=147,
        embedding=[1.0, 0.0],
    )
    ranked = await rank_chunks("q", [chunk], top_k=1)

    assert ranked[0].chunk_id == "p0004-abc123"
    assert ranked[0].page_number == 4
    assert ranked[0].section == "participants"
    assert ranked[0].text == chunk.text
    assert ranked[0].start_char == 120
    assert ranked[0].end_char == 147


async def test_ranked_results_do_not_carry_the_vector(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))
    ranked = await rank_chunks("q", [make_embedded("a", [1.0, 0.0])], top_k=1)
    assert "embedding" not in ranked[0].model_dump()


# --------------------------------------------------------------------------- #
# top_k and empty input
# --------------------------------------------------------------------------- #


async def test_empty_chunk_list_returns_nothing_without_calling_ollama(fake_ollama):
    recorder = fake_ollama(responds({"embeddings": [[1.0]]}))
    assert await rank_chunks("anything", []) == []
    assert recorder.calls == []


@pytest.mark.parametrize("top_k,expected", [(0, 0), (-1, 0), (1, 1), (3, 3), (99, 4)])
async def test_top_k_bounds(fake_ollama, top_k, expected):
    recorder = fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))
    chunks = [
        make_embedded("c{}".format(i), [1.0, 0.0], page_number=i + 1) for i in range(4)
    ]
    ranked = await rank_chunks("q", chunks, top_k=top_k)

    assert len(ranked) == expected
    if top_k <= 0:
        assert recorder.calls == [], "no model call when nothing is requested"


async def test_default_top_k_is_five(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0]]}))
    chunks = [
        make_embedded("c{}".format(i), [1.0, 0.0], page_number=i + 1) for i in range(20)
    ]
    assert len(await rank_chunks("q", chunks)) == 5


async def test_question_dimension_must_match_the_chunks(fake_ollama):
    fake_ollama(responds({"embeddings": [[1.0, 0.0, 0.0, 0.0]]}))
    with pytest.raises(EmbeddingError) as error:
        await rank_chunks("q", [make_embedded("a", [1.0, 0.0])], top_k=1)
    assert "must be embedded by the same model" in str(error.value)


# --------------------------------------------------------------------------- #
# the semantic-retrieval scenario
# --------------------------------------------------------------------------- #


async def test_mobile_app_question_ranks_the_app_chunk_first(lexical):
    """End to end over one paper: embed the chunks, then ask a real question."""
    embedded = await embed_chunks(paper_chunks())
    assert len(embedded) == 4

    ranked = await rank_chunks(
        "What applications did students use?", embedded, top_k=4
    )

    assert ranked[0].section == "apps", [
        (item.section, round(item.score, 3)) for item in ranked
    ]
    assert ranked[0].score > ranked[1].score
    assert {item.section for item in ranked} == set(PAPER)


@pytest.mark.parametrize(
    "question,expected_section",
    [
        ("What applications did students use?", "apps"),
        ("How old were the participants?", "demographics"),
        ("How were the interviews analysed?", "methods"),
        ("What are the limitations of this study?", "limitations"),
    ],
)
async def test_each_question_retrieves_its_own_section(
    lexical, question, expected_section
):
    embedded = await embed_chunks(paper_chunks())
    ranked = await rank_chunks(question, embedded, top_k=1)
    assert ranked[0].section == expected_section


async def test_scores_are_within_the_cosine_range(lexical):
    embedded = await embed_chunks(paper_chunks())
    ranked = await rank_chunks("What applications did students use?", embedded, top_k=4)
    for item in ranked:
        assert -1.0 <= item.score <= 1.0
        assert not math.isnan(item.score)


async def test_ranking_is_reproducible(lexical):
    embedded = await embed_chunks(paper_chunks())
    first = await rank_chunks("What applications did students use?", embedded, top_k=4)
    second = await rank_chunks("What applications did students use?", embedded, top_k=4)
    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]


def test_ranked_chunk_round_trips_through_json():
    ranked = RankedChunk(
        chunk_id="a", page_number=1, section="apps", text="t",
        start_char=0, end_char=1, score=0.5,
    )
    assert RankedChunk.model_validate_json(ranked.model_dump_json()) == ranked
