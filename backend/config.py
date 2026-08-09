"""Configuration loaded from backend/.env."""

import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent

load_dotenv(BACKEND_DIR / ".env")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# Local inference is slow, so we fail fast on connect but wait a long time to read.
OLLAMA_CONNECT_TIMEOUT = _float_env("OLLAMA_CONNECT_TIMEOUT", 10.0)
OLLAMA_READ_TIMEOUT = _float_env("OLLAMA_READ_TIMEOUT", 300.0)

# Map-reduce chunking limits.
CHUNK_CHAR_LIMIT = _int_env("SUMMARY_CHUNK_CHARS", 12000)
CHUNK_PAGE_LIMIT = _int_env("SUMMARY_CHUNK_PAGES", 6)

# Generation options tuned for extraction rather than creative writing.
OLLAMA_TEMPERATURE = _float_env("OLLAMA_TEMPERATURE", 0.0)
OLLAMA_TOP_P = _float_env("OLLAMA_TOP_P", 0.9)
# The structured JSON is long, so leave plenty of room for it.
OLLAMA_NUM_PREDICT_MAP = _int_env("OLLAMA_NUM_PREDICT_MAP", 2048)
OLLAMA_NUM_PREDICT_REDUCE = _int_env("OLLAMA_NUM_PREDICT_REDUCE", 3072)

# Embeddings for retrieval. A dedicated embedding model is strongly preferred
# over reusing the generation model: it is far smaller, much faster, and
# produces vectors actually trained for semantic similarity.
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Chunks per /api/embed request. Larger batches are faster but use more memory.
OLLAMA_EMBED_BATCH = _int_env("OLLAMA_EMBED_BATCH", 16)

# Grounded answer generation (Ask the Paper). The generation model is the same
# qwen3:8b used for summarisation; only the token budget differs, because an
# answer is short and its context is a handful of passages.
OLLAMA_NUM_PREDICT_ANSWER = _int_env("OLLAMA_NUM_PREDICT_ANSWER", 768)
# How many passages may be sent to the model for one question.
MAX_EVIDENCE_CHUNKS = _int_env("ASK_MAX_EVIDENCE_CHUNKS", 4)
# How many ranked passages evidence selection may choose from. Wider than the
# answerability window so that a strong direct-answer passage sitting just
# outside it can still be cited.
SELECTION_POOL_SIZE = _int_env("ASK_SELECTION_POOL", 12)
# How many embedded papers to keep in memory. Bounded so a long-running server
# cannot grow without limit.
EMBEDDING_CACHE_SIZE = _int_env("ASK_EMBEDDING_CACHE_SIZE", 4)

# Development diagnostics. Never logs paper text or long evidence excerpts.
SUMMARY_DEBUG = _bool_env("SUMMARY_DEBUG", False)
