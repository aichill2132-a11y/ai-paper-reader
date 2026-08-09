"""Tests for the developer retrieval demo. Ollama is stubbed; qwen3 is never called."""

import fitz
import httpx
import pytest

import ollama_client
import retrieval_demo
from embeddings import RankedChunk
from retrieval import RetrievalChunk
from retrieval_demo import (
    DEFAULT_QUESTIONS,
    DEFAULT_TOP_K,
    SNIPPET_CHARS,
    build_parser,
    describe_chunks,
    format_question,
    format_result,
    load_pages,
    run_demo,
    snippet,
)
from test_embeddings import FakeClient, lexical_handler

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def make_pdf(page_texts):
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        offset = 72
        for line in text.split("\n"):
            page.insert_text((60, offset), line)
            offset += 14
    data = document.tobytes()
    document.close()
    return data


PAPER = [
    "Mobile Assisted Language Learning in Higher Education\n"
    "R. Iyer and T. Nakamura\n"
    "Abstract\n"
    "This study examined how undergraduates use mobile applications.",
    "Participants\n"
    "Forty two undergraduate students took part, mean age twenty.\n"
    "Data collection and analysis\n"
    "Semi structured interviews were transcribed and examined with thematic analysis.",
    "Findings\n"
    "Students used the Duolingo and Memrise mobile applications daily.\n"
    "Limitations\n"
    "A limitation is the single site sample of language learning students.",
]


@pytest.fixture
def paper_pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(PAPER))
    return path


@pytest.fixture
def lexical_ollama(monkeypatch):
    """Route embedding calls to the lexical stub from the embeddings tests."""
    import embeddings

    embeddings.reset_capabilities()
    calls = []

    def handler(url, payload):
        calls.append(url)
        return lexical_handler(url, payload)

    monkeypatch.setattr(
        ollama_client.httpx, "AsyncClient", lambda **kw: FakeClient(handler)
    )
    return calls


def ranked(chunk_id="p0002-abc", score=0.5, page=2, section="participants", text="x"):
    return RankedChunk(
        chunk_id=chunk_id,
        page_number=page,
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        score=score,
    )


# --------------------------------------------------------------------------- #
# the built-in questions
# --------------------------------------------------------------------------- #


def test_default_questions_are_the_documented_six():
    assert DEFAULT_QUESTIONS == (
        "What applications did students use?",
        "How many participants were in the study?",
        "How were the data collected and analysed?",
        "What limitations did the authors identify?",
        "What role did teachers play in mobile learning?",
        "Did the paper report improved test scores?",
    )


def test_default_top_k_is_five():
    assert DEFAULT_TOP_K == 5


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #


def test_snippet_collapses_whitespace():
    assert snippet("a\n\n  b\tc  ") == "a b c"


def test_snippet_truncates_at_300_characters():
    result = snippet("word " * 200)
    assert len(result) == SNIPPET_CHARS + 3
    assert result.endswith("...")


def test_snippet_leaves_short_text_alone():
    assert snippet("short text") == "short text"
    assert "..." not in snippet("short text")


def test_format_result_shows_every_required_field():
    line = format_result(3, ranked(chunk_id="p0004-deadbeef", score=0.7431, page=4,
                                   section="methods", text="Interviews were recorded."))
    assert "3." in line
    assert "0.7431" in line
    assert "page 4" in line
    assert "section methods" in line
    assert "p0004-deadbeef" in line
    assert "Interviews were recorded." in line


def test_format_result_labels_an_unlabelled_section():
    assert "(unlabelled)" in format_result(1, ranked(section=""))


def test_format_question_numbers_results_from_one():
    block = format_question("Q?", [ranked(chunk_id="a"), ranked(chunk_id="b")])
    assert "Q: Q?" in block
    assert "  1. " in block and "  2. " in block


def test_format_question_handles_no_matches():
    assert "(no chunks matched)" in format_question("Q?", [])


def test_describe_chunks_summarises_the_corpus():
    chunks = [
        RetrievalChunk(chunk_id="a", page_number=1, section="abstract",
                       text="one two three", start_char=0, end_char=13),
        RetrievalChunk(chunk_id="b", page_number=2, section="methods",
                       text="four five", start_char=0, end_char=9),
    ]
    summary = describe_chunks(chunks)
    assert "2 chunks over 2 pages" in summary
    assert "abstract, methods" in summary


def test_describe_chunks_handles_an_empty_corpus():
    assert describe_chunks([]) == "  no chunks"


# --------------------------------------------------------------------------- #
# loading a PDF
# --------------------------------------------------------------------------- #


def test_load_pages_reuses_the_upload_extraction(paper_pdf):
    pages = load_pages(paper_pdf)
    assert [page.page_number for page in pages] == [1, 2, 3]
    assert "Duolingo" in pages[2].text


def test_missing_file_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        load_pages(tmp_path / "nope.pdf")
    assert "No such file" in str(exit_info.value)


def test_directory_argument_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        load_pages(tmp_path)
    assert "Not a file" in str(exit_info.value)


def test_unreadable_pdf_exits_cleanly(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"definitely not a pdf")
    with pytest.raises(SystemExit) as exit_info:
        load_pages(broken)
    assert "Could not read" in str(exit_info.value)


def test_scanned_pdf_exits_cleanly(tmp_path):
    document = fitz.open()
    document.new_page()
    blank = tmp_path / "scan.pdf"
    blank.write_bytes(document.tobytes())
    document.close()

    with pytest.raises(SystemExit) as exit_info:
        load_pages(blank)
    assert "No extractable text" in str(exit_info.value)


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def test_parser_takes_a_pdf_path():
    args = build_parser().parse_args(["/tmp/paper.pdf"])
    assert args.pdf == "/tmp/paper.pdf"
    assert args.top_k == DEFAULT_TOP_K
    assert args.questions is None


def test_parser_accepts_repeated_questions():
    args = build_parser().parse_args(
        ["p.pdf", "--question", "one?", "--question", "two?", "--top-k", "2"]
    )
    assert args.questions == ["one?", "two?"]
    assert args.top_k == 2


def test_parser_requires_a_path():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


# --------------------------------------------------------------------------- #
# the demo end to end
# --------------------------------------------------------------------------- #


async def test_demo_prints_ranked_results(paper_pdf, lexical_ollama, capsys):
    code = await run_demo(paper_pdf, DEFAULT_QUESTIONS, top_k=3)
    assert code == 0

    output = capsys.readouterr().out
    for question in DEFAULT_QUESTIONS:
        assert "Q: {}".format(question) in output
    assert "Embedding model:" in output
    assert "chunks over" in output
    assert "score " in output and "page " in output and "section " in output


async def test_demo_never_calls_the_generation_model(paper_pdf, lexical_ollama):
    await run_demo(paper_pdf, DEFAULT_QUESTIONS, top_k=3)
    assert lexical_ollama, "expected embedding calls"
    assert all(url.endswith(("/api/embed", "/api/embeddings")) for url in lexical_ollama)
    assert not any("/api/generate" in url for url in lexical_ollama)


async def test_demo_embeds_the_corpus_once_then_one_call_per_question(
    paper_pdf, lexical_ollama
):
    questions = DEFAULT_QUESTIONS[:3]
    await run_demo(paper_pdf, questions, top_k=2)
    # Chunks fit in a single batch here, so: 1 corpus call + 1 per question.
    assert len(lexical_ollama) == 1 + len(questions)


async def test_demo_respects_top_k(paper_pdf, lexical_ollama, capsys):
    await run_demo(paper_pdf, ("What applications did students use?",), top_k=2)
    block = capsys.readouterr().out.split("Q: ")[1]
    assert "  1. " in block and "  2. " in block
    assert "  3. " not in block


async def test_demo_finds_the_app_chunk_for_the_app_question(paper_pdf, lexical_ollama, capsys):
    await run_demo(paper_pdf, ("What applications did students use?",), top_k=1)
    block = capsys.readouterr().out.split("Q: ")[1]
    assert "Duolingo" in block or "findings" in block


async def test_demo_reports_embedding_failure(paper_pdf, monkeypatch, capsys):
    def handler(url, payload):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        ollama_client.httpx, "AsyncClient", lambda **kw: FakeClient(handler)
    )
    code = await run_demo(paper_pdf, DEFAULT_QUESTIONS, top_k=3)

    assert code == 1
    assert "Embedding failed" in capsys.readouterr().err


async def test_demo_reports_a_paper_with_no_chunks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(retrieval_demo, "build_retrieval_chunks", lambda *a, **k: [])
    path = tmp_path / "paper.pdf"
    path.write_bytes(make_pdf(PAPER))

    code = await run_demo(path, DEFAULT_QUESTIONS)
    assert code == 1
    assert "Nothing to rank" in capsys.readouterr().out
