import json
import re
import sys
from typing import TypeVar

from langchain_core.messages import AIMessage
from loguru import logger
from pydantic import BaseModel

STRUCTURED_OUTPUT_ERROR = "Sorry, the narrator failed to respond. Please try again."

T = TypeVar("T", bound=BaseModel)

LOGFILE = "logfile.log"
FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level> | "
    "<level>{extra}</level>"
)

# Remove default handler only if logger is already configured
# This prevents errors if logger.remove() is called multiple times
try:
    logger.remove()
except ValueError:
    pass  # Logger already has no handlers

logger.add(
    sys.stderr,
    format=FORMAT,
    level="DEBUG",
    colorize=True,
)
logger.add(
    LOGFILE,
    serialize=False,
    level="DEBUG",
    rotation="500 MB",
    retention="14 days",
)

_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def split_thinking(text: str) -> tuple[str | None, str]:
    """Return (thinking, visible_text).  thinking is None when no block is found."""
    if not isinstance(text, str):
        return None, str(text) if text else ""
    thinking = "\n\n".join(m.group(1).strip() for m in _RE.finditer(text)) or None
    visible = _RE.sub("", text).strip()
    return thinking, visible


def strip_thinking(text: str) -> str:
    return _RE.sub("", text).strip() if isinstance(text, str) else (str(text) if text else "")


def parse_last_json(messages: list) -> dict:
    """Parse JSON from the last message's content; return {} on any failure."""
    last = messages[-1] if messages else None
    try:
        return json.loads(getattr(last, "content", "{}"))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}


def message_text(msg) -> str:
    """Extract plain text from a message."""
    return getattr(msg, "content", "") or ""


async def invoke_structured(
    llm,
    schema: type[T],
    messages,
    *,
    retries: int = 1,
) -> T:
    """Call with_structured_output with retries; re-raise on failure."""
    structured = llm.with_structured_output(schema)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await structured.ainvoke(messages)
        except Exception as e:
            last_error = e
            logger.error(f"Structured output failed (attempt {attempt + 1}/{retries + 1}): {e}")
    raise last_error  # type: ignore[misc]


def visible_response(messages: list) -> str:
    """Extract user-visible text from the last AIMessage."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, "content", "") or ""
        return strip_thinking(content)
    return ""
