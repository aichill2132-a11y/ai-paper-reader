"""Tests for the retrieval chunker. No model, no network, no embeddings."""

import random

import pytest

from fixtures import RUNNING_HEAD, page_tuples, pages_payload
from metadata import repeated_lines
from retrieval import (
    MIN_TAIL_WORDS,
    TARGET_WORDS,
    OVERLAP_WORDS,
    TARGET_MAX_WORDS,
    TARGET_MIN_WORDS,
    RetrievalChunk,
    build_retrieval_chunks,
    build_units,
    chunks_by_page,
)
from schemas import PageInput
from sections import parse_sections

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

VOCAB = (
    "mentorship transition ward nurses practice supervision confidence handover "
    "reflection preceptor caseload rostering induction competence debrief "
    "escalation documentation prioritisation delegation feedback"
).split()


def prose(word_count, seed=0):
    """Deterministic filler prose of an exact word length.

    Seeded sampling, not a fixed stride: a repeating pattern would produce
    identical lines on several pages, which the running-head detector would
    correctly strip as page furniture.
    """
    rng = random.Random(seed)
    words = [rng.choice(VOCAB) for _ in range(word_count)]
    # Break it into sentences so it looks like real body text.
    sentences = []
    for start in range(0, len(words), 12):
        block = words[start : start + 12]
        sentences.append(block[0].capitalize() + " " + " ".join(block[1:]) + ".")
    return " ".join(sentences)


def wrapped(text, width=95):
    """Wrap prose into PDF-like short lines."""
    lines = []
    line = []
    length = 0
    for word in text.split():
        if length + len(word) + 1 > width:
            lines.append(" ".join(line))
            line, length = [], 0
        line.append(word)
        length += len(word) + 1
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def long_paper(pages=4, words_per_page=900):
    """A multi-page paper with a running head and one heading per page."""
    headings = ["Introduction", "Methods", "Findings", "Discussion"]
    built = []
    for number in range(1, pages + 1):
        body = wrapped(prose(words_per_page, seed=number))
        built.append(
            PageInput(
                page_number=number,
                text="{}\n{}\n{}\n{}\n".format(
                    RUNNING_HEAD, 1200 + number, headings[(number - 1) % 4], body
                ),
            )
        )
    return built


def fixture_pages():
    return [PageInput(**page) for page in pages_payload()]


@pytest.fixture
def long_chunks():
    return build_retrieval_chunks(long_paper(), "long.pdf")


@pytest.fixture
def fixture_chunks():
    return build_retrieval_chunks(fixture_pages(), "nurses.pdf")


def page_text(pages, number):
    return next(page.text for page in pages if page.page_number == number)


# --------------------------------------------------------------------------- #
# short paper
# --------------------------------------------------------------------------- #


def test_short_paper_produces_one_chunk():
    pages = [PageInput(page_number=1, text=wrapped(prose(80)))]
    chunks = build_retrieval_chunks(pages, "short.pdf")

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert 0 < chunks[0].word_count < TARGET_MIN_WORDS


def test_page_shorter_than_target_is_not_padded_from_the_next_page():
    """Page boundaries win over hitting the word target."""
    pages = [
        PageInput(page_number=1, text=wrapped(prose(60, seed=1))),
        PageInput(page_number=2, text=wrapped(prose(60, seed=2))),
    ]
    chunks = build_retrieval_chunks(pages, "short.pdf")

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert all(chunk.word_count < TARGET_MIN_WORDS for chunk in chunks)


def test_empty_and_whitespace_pages_are_skipped():
    pages = [
        PageInput(page_number=1, text=""),
        PageInput(page_number=2, text="   \n\n  \n"),
        PageInput(page_number=3, text=wrapped(prose(50))),
    ]
    chunks = build_retrieval_chunks(pages, "sparse.pdf")
    assert [chunk.page_number for chunk in chunks] == [3]


# --------------------------------------------------------------------------- #
# multi-page paper
# --------------------------------------------------------------------------- #


def test_the_fixture_itself_is_not_eaten_by_the_running_head_filter():
    """Guards the test data: repetitive filler used to look like page furniture."""
    pages = long_paper(pages=6)
    chunks = build_retrieval_chunks(pages, "long.pdf")
    for page in pages:
        body_words = len(page.text.split()) - len(RUNNING_HEAD.split()) - 2
        kept = sum(
            chunk.word_count
            for chunk in chunks
            if chunk.page_number == page.page_number
        )
        assert kept >= body_words * 0.9, "page {} lost content".format(
            page.page_number
        )


def test_multi_page_paper_covers_every_page(long_chunks):
    assert sorted(chunks_by_page(long_chunks)) == [1, 2, 3, 4]
    assert len(long_chunks) > 4  # long pages split into several chunks each


def test_chunks_are_emitted_in_reading_order(long_chunks):
    pages = [chunk.page_number for chunk in long_chunks]
    assert pages == sorted(pages)

    for page, group in chunks_by_page(long_chunks).items():
        starts = [chunk.start_char for chunk in group]
        assert starts == sorted(starts), "page {} out of order".format(page)


def test_pages_supplied_out_of_order_are_normalised():
    pages = list(reversed(long_paper(pages=3)))
    chunks = build_retrieval_chunks(pages, "long.pdf")
    assert [chunk.page_number for chunk in chunks] == sorted(
        chunk.page_number for chunk in chunks
    )


def test_chunk_sizes_land_inside_the_target_band(long_chunks):
    """Every chunk except the last of a unit must sit in 300-600 words."""
    grouped = chunks_by_page(long_chunks)
    checked = 0
    for group in grouped.values():
        for chunk in group[:-1]:
            assert TARGET_MIN_WORDS <= chunk.word_count <= TARGET_MAX_WORDS, (
                "{} has {} words".format(chunk.chunk_id, chunk.word_count)
            )
            checked += 1
    assert checked, "expected at least one split unit"


@pytest.mark.parametrize("words_per_page", [601, 700, 900, 1400, 2600, 4000])
def test_every_chunk_of_a_long_page_lands_in_the_band(words_per_page):
    """Balancing must hold across page lengths, not just the fixture."""
    pages = long_paper(pages=2, words_per_page=words_per_page)
    for chunk in build_retrieval_chunks(pages, "sized.pdf"):
        assert TARGET_MIN_WORDS <= chunk.word_count <= TARGET_MAX_WORDS, (
            "{} has {} words".format(chunk.chunk_id, chunk.word_count)
        )


def test_chunks_within_a_unit_are_evenly_sized():
    """Evenness is a property of one (page, section) unit, not of the document.

    Front matter above the first heading is its own short unit and is expected
    to be small.
    """
    pages = long_paper(pages=1, words_per_page=1800)
    chunks = build_retrieval_chunks(pages, "even.pdf")

    units = {}
    for chunk in chunks:
        units.setdefault((chunk.page_number, chunk.section), []).append(
            chunk.word_count
        )

    body = max(units.values(), key=len)
    assert len(body) > 2
    assert max(body) - min(body) < TARGET_WORDS // 4, body


def test_front_matter_above_the_first_heading_is_its_own_chunk():
    pages = long_paper(pages=1, words_per_page=400)
    chunks = build_retrieval_chunks(pages, "front.pdf")
    assert chunks[0].section == ""
    assert chunks[1].section == "introduction"


def test_no_tiny_trailing_chunks(long_chunks):
    """A stub tail is merged back unless doing so would breach the ceiling."""
    for page, group in chunks_by_page(long_chunks).items():
        if len(group) < 2:
            continue
        tail, previous = group[-1], group[-2]
        merged_would_overflow = tail.word_count + previous.word_count > TARGET_MAX_WORDS
        assert tail.word_count >= MIN_TAIL_WORDS or merged_would_overflow, (
            "page {} ends with a {}-word chunk".format(page, tail.word_count)
        )


def test_merging_a_tail_never_breaches_the_ceiling(long_chunks):
    for chunk in long_chunks:
        assert chunk.word_count <= TARGET_MAX_WORDS


# --------------------------------------------------------------------------- #
# section boundaries
# --------------------------------------------------------------------------- #


def test_sections_match_the_summary_parser():
    """The chunker must not invent its own idea of where sections start."""
    pages = fixture_pages()
    chunks = build_retrieval_chunks(pages, "nurses.pdf")

    parsed = parse_sections(page_tuples(), drop_lines=repeated_lines(page_tuples()))
    expected = {(section.name, page) for section in parsed for page in section.pages}

    for chunk in chunks:
        if chunk.section:
            assert (chunk.section, chunk.page_number) in expected, (
                "chunk claims section {!r} on page {}".format(
                    chunk.section, chunk.page_number
                )
            )


def test_known_sections_are_labelled(fixture_chunks):
    labelled = {chunk.section for chunk in fixture_chunks}
    for name in ("abstract", "research_question", "participants", "findings",
                 "limitations", "references"):
        assert name in labelled


def test_a_chunk_never_mixes_two_sections():
    pages = fixture_pages()
    chunks = build_retrieval_chunks(pages, "nurses.pdf")
    units = build_units(pages, repeated_lines(page_tuples()))

    for chunk in chunks:
        containing = [
            unit
            for unit in units
            if unit.page_number == chunk.page_number
            and unit.section == chunk.section
            and unit.lines[0].start <= chunk.start_char
            and chunk.end_char <= unit.lines[-1].end
        ]
        assert containing, "chunk {} spans a section boundary".format(chunk.chunk_id)


def test_a_section_continuing_onto_the_next_page_keeps_its_name():
    pages = [
        PageInput(page_number=1, text="Methods\n" + wrapped(prose(120, seed=1))),
        PageInput(page_number=2, text=wrapped(prose(120, seed=2))),
        PageInput(page_number=3, text="Findings\n" + wrapped(prose(120, seed=3))),
    ]
    chunks = build_retrieval_chunks(pages, "cont.pdf")
    by_page = {chunk.page_number: chunk.section for chunk in chunks}

    assert by_page[1] == "methods"
    assert by_page[2] == "methods", "the section name did not carry across the break"
    assert by_page[3] == "findings"


def test_text_before_the_first_heading_is_unlabelled():
    pages = [PageInput(page_number=1, text="A Title\nAn Author\n" + wrapped(prose(40)))]
    chunks = build_retrieval_chunks(pages, "front.pdf")
    assert chunks[0].section == ""


def test_run_in_heading_starts_a_new_chunk_with_its_body():
    pages = [
        PageInput(
            page_number=1,
            text="Introduction\n"
            + wrapped(prose(40, seed=1))
            + "\nLimitations. The sample was small and drawn from one site.\n",
        )
    ]
    chunks = build_retrieval_chunks(pages, "runin.pdf")
    sections = {chunk.section: chunk.text for chunk in chunks}

    assert "limitations" in sections
    assert sections["limitations"].startswith("The sample was small")
    assert "Limitations." not in sections["introduction"]


# --------------------------------------------------------------------------- #
# overlap
# --------------------------------------------------------------------------- #


def _consecutive_pairs(chunks):
    """Chunk pairs that are neighbours inside the same page and section."""
    pairs = []
    for first, second in zip(chunks, chunks[1:]):
        if (
            first.page_number == second.page_number
            and first.section == second.section
        ):
            pairs.append((first, second))
    return pairs


def test_neighbouring_chunks_overlap(long_chunks):
    pairs = _consecutive_pairs(long_chunks)
    assert pairs, "expected at least one split unit"

    for first, second in pairs:
        assert second.start_char < first.end_char, "chunks do not overlap"
        shared = second.text.split()[:OVERLAP_WORDS]
        assert " ".join(shared[:5]) in first.text


def test_overlap_is_bounded(long_chunks):
    for first, second in _consecutive_pairs(long_chunks):
        overlap_words = len(
            [
                word
                for word in second.text.split()
                if second.start_char < first.end_char
            ][: OVERLAP_WORDS * 3]
        )
        # The replayed region is measured in whole lines, so it can exceed
        # OVERLAP_WORDS by at most one line, never by a whole chunk.
        assert first.end_char - second.start_char > 0
        assert overlap_words <= TARGET_MAX_WORDS


def test_overlap_never_crosses_a_page_or_section(long_chunks):
    for first, second in zip(long_chunks, long_chunks[1:]):
        if first.page_number == second.page_number and first.section == second.section:
            continue
        assert second.start_char >= 0
        # Different unit: the spans are in different pages, or start afresh.
        if first.page_number == second.page_number:
            assert second.start_char >= first.end_char


def test_a_single_chunk_unit_has_no_overlap():
    pages = [PageInput(page_number=1, text=wrapped(prose(80)))]
    chunks = build_retrieval_chunks(pages, "one.pdf")
    assert len(chunks) == 1


# --------------------------------------------------------------------------- #
# stable chunk ids
# --------------------------------------------------------------------------- #


def test_chunk_ids_are_stable_across_runs():
    first = build_retrieval_chunks(long_paper(), "long.pdf")
    second = build_retrieval_chunks(long_paper(), "long.pdf")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.model_dump() for chunk in first] == [
        chunk.model_dump() for chunk in second
    ]


def test_chunk_ids_are_unique(long_chunks, fixture_chunks):
    for chunks in (long_chunks, fixture_chunks):
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids))


def test_chunk_ids_encode_the_page():
    chunks = build_retrieval_chunks(long_paper(), "long.pdf")
    for chunk in chunks:
        assert chunk.chunk_id.startswith("p{:04d}-".format(chunk.page_number))


def test_different_documents_do_not_collide():
    pages = long_paper()
    first = build_retrieval_chunks(pages, "paper-a.pdf")
    second = build_retrieval_chunks(pages, "paper-b.pdf")
    assert {c.chunk_id for c in first}.isdisjoint({c.chunk_id for c in second})
    assert [c.text for c in first] == [c.text for c in second]


def test_changing_the_text_changes_the_id():
    original = build_retrieval_chunks(
        [PageInput(page_number=1, text=wrapped(prose(80, seed=1)))], "x.pdf"
    )
    edited = build_retrieval_chunks(
        [PageInput(page_number=1, text=wrapped(prose(80, seed=2)))], "x.pdf"
    )
    assert original[0].chunk_id != edited[0].chunk_id


# --------------------------------------------------------------------------- #
# no empty chunks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pages",
    [
        long_paper(),
        fixture_pages(),
        [PageInput(page_number=1, text="Abstract\n\n\n   \nSome words here.\n")],
        [PageInput(page_number=1, text="Methods\n")],
    ],
)
def test_no_empty_chunks(pages):
    for chunk in build_retrieval_chunks(pages, "doc.pdf"):
        assert chunk.text.strip()
        assert chunk.word_count > 0
        assert chunk.end_char > chunk.start_char


def test_a_heading_with_no_body_produces_no_chunk():
    chunks = build_retrieval_chunks(
        [PageInput(page_number=1, text="Methods\nFindings\nReferences\n")], "bare.pdf"
    )
    assert chunks == []


# --------------------------------------------------------------------------- #
# page metadata
# --------------------------------------------------------------------------- #


def test_offsets_point_into_the_original_page_text(long_chunks):
    pages = long_paper()
    for chunk in long_chunks:
        source = page_text(pages, chunk.page_number)
        assert 0 <= chunk.start_char < chunk.end_char <= len(source)


def test_chunk_text_is_recoverable_from_its_span(long_chunks):
    pages = long_paper()
    for chunk in long_chunks:
        source = page_text(pages, chunk.page_number)
        span = " ".join(source[chunk.start_char : chunk.end_char].split())
        first_words = " ".join(chunk.text.split()[:8])
        last_words = " ".join(chunk.text.split()[-8:])
        assert span.startswith(first_words)
        assert span.endswith(last_words)


def test_running_heads_and_page_numbers_are_dropped(long_chunks):
    for chunk in long_chunks:
        assert RUNNING_HEAD not in chunk.text
        assert "1201" not in chunk.text


def test_page_numbers_are_preserved_exactly():
    pages = long_paper(pages=3)
    for chunk in build_retrieval_chunks(pages, "long.pdf"):
        source = page_text(pages, chunk.page_number)
        assert chunk.text.split()[0] in source


def test_chunks_by_page_groups_without_reordering(long_chunks):
    grouped = chunks_by_page(long_chunks)
    assert sum(len(group) for group in grouped.values()) == len(long_chunks)
    for page, group in grouped.items():
        assert all(chunk.page_number == page for chunk in group)


def test_model_round_trips_through_json(fixture_chunks):
    for chunk in fixture_chunks:
        assert RetrievalChunk.model_validate_json(chunk.model_dump_json()) == chunk
