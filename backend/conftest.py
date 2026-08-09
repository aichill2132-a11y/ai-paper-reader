"""Shared test fixtures.

The embedding stub lives here so the three retrieval test modules share one
implementation instead of each defining its own. It is a hashing bag-of-words
embedder: cosine over its vectors tracks lexical overlap, so ranking in the
tests is earned from the text rather than hardcoded.
"""

import hashlib

import httpx
import pytest

import embeddings
import ollama_client
from fixtures import (
    interview_pages_payload,
    language_pages_payload,
    mobile_pages_payload,
)
from retrieval import build_retrieval_chunks
from schemas import PageInput

EMBED_DIMENSIONS = 512
_PUNCTUATION = str.maketrans({char: " " for char in "?.,;:()[]-'\"/"})


class FakeClient:
    """Stands in for httpx.AsyncClient inside ollama_client."""

    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        return self._handler(url, json)


def tokens(text):
    for token in text.lower().translate(_PUNCTUATION).split():
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        yield token


def hashed_vector(text):
    """A stable hashing bag-of-words vector. hashlib, not hash(), so it is
    identical across processes."""
    vector = [0.0] * EMBED_DIMENSIONS
    for token in tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[int.from_bytes(digest[:4], "big") % EMBED_DIMENSIONS] += 1.0
    if not any(vector):
        vector[0] = 0.001
    return vector


def hashing_handler(url, payload):
    assert url.endswith("/api/embed")
    return httpx.Response(
        200, json={"embeddings": [hashed_vector(text) for text in payload["input"]]}
    )


@pytest.fixture
def hashing_ollama(monkeypatch):
    """Route every embedding request to the hashing stub."""
    embeddings.reset_capabilities()
    monkeypatch.setattr(
        ollama_client.httpx, "AsyncClient", lambda **kw: FakeClient(hashing_handler)
    )


def language_pages():
    return [PageInput(**page) for page in language_pages_payload()]


def language_chunks():
    return build_retrieval_chunks(language_pages(), "language.pdf")


async def embedded_language_corpus():
    """The language-paper fixture, chunked and embedded with the stub."""
    return await embeddings.embed_chunks(language_chunks())


def mobile_pages():
    return [PageInput(**page) for page in mobile_pages_payload()]


def mobile_chunks():
    return build_retrieval_chunks(mobile_pages(), "mobile.pdf")


async def embedded_mobile_corpus():
    """The mobile-devices fixture, chunked and embedded with the stub."""
    return await embeddings.embed_chunks(mobile_chunks())


def interview_pages():
    return [PageInput(**page) for page in interview_pages_payload()]


def interview_chunks():
    return build_retrieval_chunks(interview_pages(), "interview.pdf")


async def embedded_interview_corpus():
    """The interview-study fixture, chunked and embedded with the stub."""
    return await embeddings.embed_chunks(interview_chunks())
