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

# Development diagnostics. Never logs paper text or long evidence excerpts.
SUMMARY_DEBUG = _bool_env("SUMMARY_DEBUG", False)
