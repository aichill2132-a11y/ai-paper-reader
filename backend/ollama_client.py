"""Thin async client for the local Ollama HTTP API.

Only the /api/generate endpoint is used. The paper text itself is never logged;
debug logging is limited to sizes, timings, and field names.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_CONNECT_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT_MAP,
    OLLAMA_READ_TIMEOUT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    SUMMARY_DEBUG,
)
from diagnostics import debug

logger = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(
    connect=OLLAMA_CONNECT_TIMEOUT,
    read=OLLAMA_READ_TIMEOUT,
    write=OLLAMA_CONNECT_TIMEOUT,
    pool=OLLAMA_CONNECT_TIMEOUT,
)

# Reasoning models such as qwen3 may wrap their scratchpad in <think> tags.
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)

# Not every Ollama build / model accepts these options. Once one is rejected we
# stop sending it, so we only pay for the retry round trip a single time.
_CAPABILITIES = {"think": True, "json_schema": True}


def reset_capabilities() -> None:
    """Re-enable the optional request features. Used by the tests."""
    _CAPABILITIES.update(think=True, json_schema=True)


class OllamaError(Exception):
    """An error that should be surfaced to the client with a clear message."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def inline_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve $ref/$defs into a self-contained schema.

    Ollama turns the schema into a grammar, and nested $defs are the part most
    likely to trip that conversion up.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 20:
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1])
                if isinstance(target, dict):
                    return resolve(target, depth + 1)
            return {
                key: resolve(value, depth + 1)
                for key, value in node.items()
                if key != "$defs"
            }
        if isinstance(node, list):
            return [resolve(item, depth + 1) for item in node]
        return node

    resolved = resolve(schema)
    return resolved if isinstance(resolved, dict) else schema


def _extract_json(raw: str) -> Tuple[Dict[str, Any], List[str]]:
    """Parse the model's reply as JSON.

    Returns the object plus a list of the cleanups that were needed, so callers
    can report *how* messy the output was without logging the output itself.
    """
    cleanups: List[str] = []
    text = raw

    if THINK_BLOCK.search(text):
        text = THINK_BLOCK.sub("", text)
        cleanups.append("stripped-think-block")
    elif UNCLOSED_THINK.match(text) and "{" in text:
        text = text[text.find("{") :]
        cleanups.append("stripped-unclosed-think")

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        cleanups.append("stripped-code-fence")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise OllamaError(
                "The model did not return JSON. Try generating the summary again.",
                502,
            )
        cleanups.append("extracted-object-from-prose")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise OllamaError(
                "The model returned malformed JSON. Try generating the summary again.",
                502,
            )

    if not isinstance(parsed, dict):
        raise OllamaError("The model returned JSON that was not an object.", 502)
    return parsed, cleanups


async def _post(payload: Dict[str, Any]) -> httpx.Response:
    try:
        # trust_env=False: Ollama is a local service and must never be sent
        # through an HTTP(S)_PROXY picked up from the environment.
        async with httpx.AsyncClient(timeout=TIMEOUT, trust_env=False) as client:
            return await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
    except httpx.TimeoutException:
        raise OllamaError(
            "Ollama timed out. Local summarisation of long papers can be slow; "
            "try again or use a smaller model.",
            504,
        )
    except httpx.ConnectError:
        raise OllamaError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
            "Make sure Ollama is running (`ollama serve`).",
            503,
        )
    except httpx.HTTPError:
        raise OllamaError(f"Could not talk to Ollama at {OLLAMA_BASE_URL}.", 503)
    except Exception as exc:  # pragma: no cover - unexpected transport failure
        raise OllamaError(
            f"Unexpected error talking to Ollama: {type(exc).__name__}", 502
        )


def _error_text(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()
    if isinstance(body, dict):
        return str(body.get("error", "")).strip()
    return ""


def _build_payload(
    prompt: str,
    system: str,
    schema: Optional[Dict[str, Any]],
    num_ctx: int,
    num_predict: int,
) -> Dict[str, Any]:
    use_schema = schema is not None and _CAPABILITIES["json_schema"]
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": inline_schema_refs(schema) if use_schema else "json",
        "options": {
            # Extraction, not creative writing: keep it deterministic.
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if _CAPABILITIES["think"]:
        # Reasoning models such as qwen3 are much faster with thinking disabled.
        payload["think"] = False
    return payload


def _drop_rejected_options(payload: Dict[str, Any], detail: str) -> bool:
    """Remove options this Ollama build rejected. True if a retry is worthwhile.

    The capability is remembered globally, so the extra round trip is paid once
    per process rather than once per call.
    """
    retried = False
    if "think" in detail:
        _CAPABILITIES["think"] = False
        payload.pop("think", None)
        retried = True
    if "format" in detail or "schema" in detail:
        _CAPABILITIES["json_schema"] = False
        payload["format"] = "json"
        retried = True
    return retried


def _model_not_installed() -> OllamaError:
    return OllamaError(
        "The model '{0}' is not installed in Ollama. "
        "Install it with: ollama pull {0}".format(OLLAMA_MODEL),
        503,
    )


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 404:
        raise _model_not_installed()
    if response.status_code < 400:
        return

    detail = _error_text(response) or "HTTP {}".format(response.status_code)
    if "not found" in detail.lower() or "try pulling" in detail.lower():
        raise _model_not_installed()
    raise OllamaError("Ollama returned an error: {}".format(detail), 502)


async def generate_json(
    prompt: str,
    *,
    system: str,
    schema: Optional[Dict[str, Any]] = None,
    num_ctx: int = 8192,
    num_predict: int = OLLAMA_NUM_PREDICT_MAP,
    label: str = "call",
) -> Dict[str, Any]:
    """Ask the local model for a single JSON object and return it parsed."""
    payload = _build_payload(prompt, system, schema, num_ctx, num_predict)

    started = time.monotonic()
    response = await _post(payload)

    if response.status_code == 400:
        if _drop_rejected_options(payload, _error_text(response).lower()):
            debug(logger, "%s: retrying without rejected options", label)
            response = await _post(payload)

    _raise_for_status(response)

    try:
        body = response.json()
    except ValueError:
        raise OllamaError("Ollama returned a response that was not JSON.", 502)

    raw = str(body.get("response", "")).strip()
    elapsed = time.monotonic() - started

    if not raw:
        logger.warning(
            "%s: Ollama returned an empty response after %.1fs "
            "(prompt_chars=%d, num_predict=%d)",
            label,
            elapsed,
            len(prompt),
            num_predict,
        )
        raise OllamaError(
            "Ollama returned an empty response. The model may have run out of "
            "context; try a shorter paper.",
            502,
        )

    parsed, cleanups = _extract_json(raw)

    debug(
        logger,
        "%s: %.1fs, prompt_chars=%d, response_chars=%d, cleanup=%s, keys=%s",
        label,
        elapsed,
        len(prompt),
        len(raw),
        ",".join(cleanups) if cleanups else "none",
        sorted(parsed.keys()),
    )
    if cleanups and not SUMMARY_DEBUG:
        logger.info("%s: model output needed cleanup: %s", label, ",".join(cleanups))

    return parsed
