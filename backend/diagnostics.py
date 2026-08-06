"""Development diagnostics shared by the summarisation modules.

Enabled with SUMMARY_DEBUG=true in backend/.env. Debug output is limited to
page ranges, field names, timings and truncated values: the paper text itself
is never logged.
"""

import logging
from typing import Any

from config import SUMMARY_DEBUG

DEBUG_PREFIX = "[summary-debug] "
# Debug logs never print more than this much of a value.
MAX_DEBUG_VALUE_CHARS = 60


def debug(logger: logging.Logger, message: str, *args: Any) -> None:
    """Log a diagnostic line, but only when SUMMARY_DEBUG is on."""
    if SUMMARY_DEBUG:
        logger.info(DEBUG_PREFIX + message, *args)


def short(value: Any, limit: int = MAX_DEBUG_VALUE_CHARS) -> str:
    """Collapse whitespace and truncate, so debug output cannot leak prose."""
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
