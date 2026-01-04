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
