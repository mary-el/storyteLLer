import json
import re
import sys

from langchain_core.messages import AIMessage
from loguru import logger

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
    """Extract plain text from a message (JSON wire-format AIMessages use response field)."""
    if isinstance(msg, AIMessage):
        payload = parse_last_json([msg])
        if "response" in payload:
            return payload["response"] or ""
        return getattr(msg, "content", "") or ""
    return getattr(msg, "content", "") or ""


def visible_response(messages: list) -> str:
    """Extract user-visible text from the last AIMessage (JSON wire format or plain)."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = getattr(msg, "content", "") or ""
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and payload.get("response"):
                return strip_thinking(payload["response"])
        except (json.JSONDecodeError, TypeError):
            pass
        return strip_thinking(content)
    return ""
