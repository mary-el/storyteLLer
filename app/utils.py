import re
import sys

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
