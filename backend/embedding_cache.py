"""A tiny bounded cache of embedded papers.

Chunking and embedding a paper costs seconds to minutes locally; ranking a
question against an already-embedded paper costs one embedding call. Without a
cache every question re-embeds the whole document, which makes Ask the Paper
unusable interactively.

This is deliberately the smallest thing that works:

* the key is a SHA-256 of the filename and every page's text, so two different
  PDFs can never collide and the same PDF always hits;
* it holds at most EMBEDDING_CACHE_SIZE papers, evicting least-recently-used;
* it stores only derived data, never the uploaded file;
* it is process-local and disappears on restart.

Replacing it with a persistent store later means swapping `get` and `put`.
"""

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

from config import EMBEDDING_CACHE_SIZE
from embeddings import EmbeddedChunk
from schemas import PageInput


def document_key(filename: str, pages: Sequence[PageInput]) -> str:
    """A content address for one paper. Identical input, identical key."""
    digest = hashlib.sha256()
    digest.update((filename or "").encode("utf-8"))
    for page in sorted(pages, key=lambda item: item.page_number):
        digest.update(b"\x00")
        digest.update(str(page.page_number).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(page.text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache(object):
    """Least-recently-used cache of embedded chunk lists."""

    def __init__(self, capacity: int = EMBEDDING_CACHE_SIZE) -> None:
        self.capacity = max(1, capacity)
        self._entries = OrderedDict()  # type: OrderedDict
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[List[EmbeddedChunk]]:
        if key not in self._entries:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return self._entries[key]

    def put(self, key: str, chunks: Sequence[EmbeddedChunk]) -> None:
        self._entries[key] = list(chunks)
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, int]:
        return {
            "papers": len(self._entries),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
        }


# One process-wide cache. Explicit so tests can clear it.
EMBEDDING_CACHE = EmbeddingCache()
