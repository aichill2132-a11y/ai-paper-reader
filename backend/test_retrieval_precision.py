"""Retrieval precision: eligibility filtering, chunk granularity, diagnostics.

Ollama is stubbed with a hashing bag-of-words embedder, so ranking is earned
from real lexical overlap rather than hardcoded vectors. No generation model is
ever contacted.
"""

import pytest

from embeddings import EmbeddedChunk, rank_chunks, rank_with_diagnostics
from fixtures import LANGUAGE_RUNNING_HEAD, language_page_tuples
from metadata import repeated_lines
from retrieval import (
    MIN_EVIDENCE_WORDS,
    NON_EVIDENCE_SECTIONS,
    TARGET_MAX_WORDS,
    TARGET_MIN_WORDS,
    TARGET_WORDS,
    build_retrieval_chunks,
    chunks_by_page,
    eligible_chunks,
    exclusion_reason,
    exclusion_summary,
    has_labelled_sections,
    is_evidence_chunk,
)
from conftest import embedded_language_corpus, language_chunks, language_pages
from schemas import PageInput
from sections import parse_sections

def make_chunk(section, text, page_number=2, chunk_id="c1"):
    return EmbeddedChunk(
        chunk_id=chunk_id,
        page_number=page_number,
        section=section,
        text=text,
        start_char=0,
        end_char=len(text),
        embedding=[1.0, 0.0],
    )


LONG_ENOUGH = "This sentence carries enough words to count as retrievable evidence."


# --------------------------------------------------------------------------- #
# 1. eligibility rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("section", sorted(NON_EVIDENCE_SECTIONS))
def test_non_evidence_sections_are_excluded(section):
    chunk = make_chunk(section, LONG_ENOUGH)
    assert exclusion_reason(chunk) == section
    assert not is_evidence_chunk(chunk)


@pytest.mark.parametrize(
    "section", ["methods", "findings", "limitations", "participants",
                "data_collection_and_analysis", "discussion", "background",
                "abstract", "introduction", "research_question", "conclusion"]
)
def test_evidence_sections_remain_eligible(section):
    chunk = make_chunk(section, LONG_ENOUGH)
    assert exclusion_reason(chunk) is None
    assert is_evidence_chunk(chunk)


def test_front_matter_on_page_one_is_excluded():
    chunk = make_chunk("", LONG_ENOUGH, page_number=1)
    assert exclusion_reason(chunk) == "front-matter"


def test_unlabelled_text_after_page_one_is_kept():
    """Only page 1 carries front matter; unlabelled prose later is real text."""
    chunk = make_chunk("", LONG_ENOUGH, page_number=4)
    assert exclusion_reason(chunk) is None


def test_front_matter_rule_is_off_when_no_section_was_recognised():
    """A paper with no headings must not lose its first page."""
    chunk = make_chunk("", LONG_ENOUGH, page_number=1)
    assert exclusion_reason(chunk, document_has_sections=False) is None


def test_stub_chunks_are_excluded():
    chunk = make_chunk("findings", "Three themes.")
    assert exclusion_reason(chunk) == "too-short"
    assert len(chunk.text.split()) < MIN_EVIDENCE_WORDS


def test_section_matching_ignores_case_and_padding():
    assert exclusion_reason(make_chunk("  References ", LONG_ENOUGH)) == "references"


def test_has_labelled_sections():
    assert has_labelled_sections([make_chunk("methods", LONG_ENOUGH)])
    assert not has_labelled_sections([make_chunk("", LONG_ENOUGH)])
    assert not has_labelled_sections([])


def test_eligible_chunks_is_a_view_not_a_deletion():
    chunks = language_chunks()
    kept = eligible_chunks(chunks)

    assert len(kept) < len(chunks), "expected some exclusions"
    assert len(chunks) == len(language_chunks()), "the corpus itself must be intact"
    assert "references" in {chunk.section for chunk in chunks}
    assert "references" not in {chunk.section for chunk in kept}


def test_exclusion_summary_counts_each_rule():
    summary = exclusion_summary(language_chunks())
    assert summary.get("references") == 1
    assert summary.get("keywords") == 1
    assert summary.get("front-matter") == 1


# --------------------------------------------------------------------------- #
# 2. chunk granularity
# --------------------------------------------------------------------------- #

WORDS = (
    "publication language english metadata corpus indexed record subject field "
    "author country article journal share increase decline analysis window year "
    "citation policy science humanities biomedicine physical social regional"
).split()


def dense_page(word_count, seed=0):
    """Deterministic dense prose, wrapped into PDF-like lines."""
    import random

    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(word_count)]
    lines, line, length = [], [], 0
    for word in words:
        if length + len(word) + 1 > 95:
            lines.append(" ".join(line))
            line, length = [], 0
        line.append(word)
        length += len(word) + 1
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def dense_paper(pages=4, words_per_page=1200):
    headings = ["Introduction", "Methods", "Results", "Discussion"]
    return [
        PageInput(
            page_number=number,
            text="{}\n{}\n{}\n{}\n".format(
                LANGUAGE_RUNNING_HEAD,
                200 + number,
                headings[(number - 1) % 4],
                dense_page(words_per_page, seed=number),
            ),
        )
        for number in range(1, pages + 1)
    ]


@pytest.mark.parametrize("words_per_page", [700, 900, 1200, 1800, 2600, 4000])
def test_dense_chunks_centre_on_300_to_400_words(words_per_page):
    chunks = build_retrieval_chunks(dense_paper(pages=2, words_per_page=words_per_page))
    body = [chunk for chunk in chunks if chunk.section]
    sizes = [len(chunk.text.split()) for chunk in body]

    assert sizes
    mean = sum(sizes) / len(sizes)
    assert TARGET_MIN_WORDS <= mean <= TARGET_MAX_WORDS, sizes
    for size in sizes:
        assert size <= TARGET_MAX_WORDS, sizes


def test_chunks_are_no_longer_500_to_600_words():
    """The regression this task exists to fix."""
    chunks = build_retrieval_chunks(dense_paper(pages=3, words_per_page=1500))
    assert max(len(chunk.text.split()) for chunk in chunks) <= TARGET_MAX_WORDS


def test_a_section_slightly_over_the_ceiling_is_not_split_into_fragments():
    """410 words should stay whole rather than become two 235-word pieces."""
    pages = [PageInput(page_number=1, text="Methods\n" + dense_page(410))]
    sizes = [
        len(chunk.text.split())
        for chunk in build_retrieval_chunks(pages)
        if chunk.section
    ]
    assert len(sizes) == 1


def test_short_section_stubs_are_kept_as_is():
    pages = [
        PageInput(page_number=1, text="Abstract\n" + dense_page(30)),
        PageInput(page_number=2, text="Limitations\n" + dense_page(25)),
    ]
    sections = {chunk.section for chunk in build_retrieval_chunks(pages)}
    assert {"abstract", "limitations"} <= sections


def test_no_tiny_tail_chunks_in_dense_text():
    chunks = build_retrieval_chunks(dense_paper(pages=3, words_per_page=1400))
    for page, group in chunks_by_page(chunks).items():
        if len(group) > 1:
            tail = len(group[-1].text.split())
            assert tail >= TARGET_WORDS // 3, "page {} tail is {} words".format(page, tail)


def test_overlap_survives_the_smaller_target():
    chunks = build_retrieval_chunks(dense_paper(pages=2, words_per_page=1200))
    pairs = [
        (first, second)
        for first, second in zip(chunks, chunks[1:])
        if first.page_number == second.page_number and first.section == second.section
    ]
    assert pairs
    for first, second in pairs:
        assert second.start_char < first.end_char
        shared = " ".join(second.text.split()[:5])
        assert shared in first.text


def test_overlap_is_not_larger_than_a_chunk():
    chunks = build_retrieval_chunks(dense_paper(pages=2, words_per_page=1200))
    for first, second in zip(chunks, chunks[1:]):
        if first.page_number != second.page_number or first.section != second.section:
            continue
        overlap_chars = first.end_char - second.start_char
        assert 0 < overlap_chars < (first.end_char - first.start_char)


def test_page_and_section_metadata_survive_the_retune():
    pages = language_pages()
    parsed = parse_sections(language_page_tuples(), drop_lines=repeated_lines(language_page_tuples()))
    expected = {(section.name, page) for section in parsed for page in section.pages}

    for chunk in build_retrieval_chunks(pages, "language.pdf"):
        source = next(p for p in pages if p.page_number == chunk.page_number)
        assert 0 <= chunk.start_char < chunk.end_char <= len(source.text)
        if chunk.section:
            assert (chunk.section, chunk.page_number) in expected


def test_chunk_ids_are_still_deterministic():
    first = build_retrieval_chunks(language_pages(), "language.pdf")
    second = build_retrieval_chunks(language_pages(), "language.pdf")
    assert [chunk.model_dump() for chunk in first] == [
        chunk.model_dump() for chunk in second
    ]
    assert len({chunk.chunk_id for chunk in first}) == len(first)


# --------------------------------------------------------------------------- #
# 3. ranking with filtering
# --------------------------------------------------------------------------- #


async def test_references_never_reach_the_results(hashing_ollama):
    embedded = await embedded_language_corpus()
    for question in (
        "What did the study find about English?",
        "Which journals were cited?",
        "bibliometrics language citation scientometrics",
    ):
        results = await rank_chunks(question, embedded, top_k=8)
        assert "references" not in {item.section for item in results}, question


async def test_front_matter_never_reaches_the_results(hashing_ollama):
    embedded = await embedded_language_corpus()
    results = await rank_chunks(
        "The language of future scientific communication", embedded, top_k=8
    )
    assert all(item.section for item in results)
    assert "keywords" not in {item.section for item in results}


async def test_evidence_only_can_be_turned_off(hashing_ollama):
    embedded = await embedded_language_corpus()
    unfiltered = await rank_chunks(
        "bibliometrics language citation", embedded, top_k=20, evidence_only=False
    )
    assert "references" in {item.section for item in unfiltered}


async def test_top_k_order_is_unchanged_for_eligible_chunks(hashing_ollama):
    """Filtering removes candidates; it must not reorder the survivors."""
    embedded = await embedded_language_corpus()
    question = "How was publication language determined?"

    filtered = await rank_chunks(question, embedded, top_k=20)
    unfiltered = await rank_chunks(question, embedded, top_k=20, evidence_only=False)

    eligible_ids = {chunk.chunk_id for chunk in eligible_chunks(embedded)}
    survivors = [
        item.chunk_id for item in unfiltered if item.chunk_id in eligible_ids
    ]
    assert [item.chunk_id for item in filtered] == survivors


async def test_multiple_pieces_of_evidence_are_returned(hashing_ollama):
    """The future generator needs several chunks, not just rank 1."""
    embedded = await embedded_language_corpus()
    results = await rank_chunks("How was the corpus assembled?", embedded, top_k=3)
    assert len(results) == 3
    assert len({item.chunk_id for item in results}) == 3


async def test_ranking_is_still_deterministic(hashing_ollama):
    embedded = await embedded_language_corpus()
    first = await rank_chunks("What changed between 2000 and 2020?", embedded)
    second = await rank_chunks("What changed between 2000 and 2020?", embedded)
    assert [item.model_dump() for item in first] == [
        item.model_dump() for item in second
    ]


# --------------------------------------------------------------------------- #
# 4. diagnostics
# --------------------------------------------------------------------------- #


async def test_diagnostics_report_the_score_shape(hashing_ollama):
    embedded = await embedded_language_corpus()
    ranking = await rank_with_diagnostics(
        "How was publication language determined?", embedded, top_k=3
    )
    diagnostics = ranking.diagnostics

    assert diagnostics.returned == 3
    assert diagnostics.considered == len(eligible_chunks(embedded))
    assert diagnostics.top_score == pytest.approx(ranking.results[0].score)
    assert diagnostics.second_score == pytest.approx(ranking.results[1].score)
    assert diagnostics.score_gap == pytest.approx(
        ranking.results[0].score - ranking.results[1].score
    )
    assert diagnostics.mean_top_k == pytest.approx(
        sum(item.score for item in ranking.results) / 3
    )


async def test_diagnostics_report_what_was_excluded(hashing_ollama):
    embedded = await embedded_language_corpus()
    ranking = await rank_with_diagnostics("anything at all", embedded, top_k=3)

    assert ranking.diagnostics.excluded == 3
    assert ranking.diagnostics.excluded_by_reason == {
        "references": 1,
        "keywords": 1,
        "front-matter": 1,
    }


async def test_diagnostics_on_a_single_result(hashing_ollama):
    embedded = await embedded_language_corpus()
    ranking = await rank_with_diagnostics("English", embedded, top_k=1)
    assert ranking.diagnostics.second_score is None
    assert ranking.diagnostics.score_gap is None
    assert ranking.diagnostics.mean_top_k == pytest.approx(ranking.results[0].score)


async def test_diagnostics_on_an_empty_corpus(hashing_ollama):
    ranking = await rank_with_diagnostics("anything", [], top_k=5)
    assert ranking.results == []
    assert ranking.diagnostics.top_score is None
    assert ranking.diagnostics.returned == 0


async def test_diagnostics_when_everything_is_filtered_out(hashing_ollama):
    only_refs = [make_chunk("references", LONG_ENOUGH, page_number=5)]
    ranking = await rank_with_diagnostics("anything", only_refs, top_k=5)
    assert ranking.results == []
    assert ranking.diagnostics.excluded_by_reason == {"references": 1}


# --------------------------------------------------------------------------- #
# 5. real-paper regression
# --------------------------------------------------------------------------- #


async def test_publication_language_question_finds_the_study_design(hashing_ollama):
    embedded = await embedded_language_corpus()
    results = await rank_chunks(
        "How was the publication language of each article determined?",
        embedded,
        top_k=3,
    )
    assert results[0].page_number == 2
    assert results[0].section == "data_collection_and_analysis"
    assert "metadata" in results[0].text


async def test_english_increase_question_finds_the_results(hashing_ollama):
    embedded = await embedded_language_corpus()
    results = await rank_chunks(
        "By how much did the share of articles published in English increase?",
        embedded,
        top_k=3,
    )
    assert results[0].page_number == 3
    assert results[0].section == "findings"
    assert "94 per cent" in results[0].text


async def test_field_association_question_finds_the_results(hashing_ollama):
    embedded = await embedded_language_corpus()
    results = await rank_chunks(
        "Which subject fields were most associated with publishing in English?",
        embedded,
        top_k=3,
    )
    assert results[0].page_number == 3
    assert results[0].section == "findings"
    assert "biomedicine" in results[0].text


@pytest.mark.parametrize(
    "question",
    [
        "How was the publication language of each article determined?",
        "By how much did the share of articles published in English increase?",
        "Which subject fields were most associated with publishing in English?",
        "What are the limitations of this analysis?",
        "Which countries were under-represented?",
    ],
)
async def test_no_question_ever_retrieves_the_reference_list(hashing_ollama, question):
    embedded = await embedded_language_corpus()
    results = await rank_chunks(question, embedded, top_k=5)
    assert "references" not in {item.section for item in results}
    assert not any("Scientometrics" in item.text for item in results)


async def test_the_brazil_caveat_is_retrievable(hashing_ollama):
    """The caveat previously sat at rank 2 behind topically similar noise."""
    embedded = await embedded_language_corpus()
    results = await rank_chunks(
        "Why might Portuguese-language output be understated?", embedded, top_k=3
    )
    assert results[0].section == "limitations"
    assert "Brazil" in results[0].text


async def test_every_body_section_is_reachable(hashing_ollama):
    """Each body section can be retrieved by a question about its content.

    The questions ask about *what the section says*, not about the section by
    name. Under the lexical stub used here, a question phrased with a generic
    structural word ("what are the limitations?") is dominated by whichever
    chunk happens to share more content words, because "limitations" is one
    token among sixty. A trained embedding model weights that word far more
    heavily; see the caveats in the README before reading much into it.
    """
    embedded = await embedded_language_corpus()
    reached = set()
    for question in (
        "How was the publication language of each article determined?",
        "By how much did the share of articles published in English increase?",
        "Why might Portuguese-language output be understated?",
    ):
        results = await rank_chunks(question, embedded, top_k=1)
        reached.add(results[0].section)
    assert reached == {"data_collection_and_analysis", "findings", "limitations"}
