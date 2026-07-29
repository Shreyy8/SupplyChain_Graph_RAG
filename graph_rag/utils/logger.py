# graph_rag/utils/logger.py
#
# PURPOSE:
#   Configures and exposes a single shared loguru logger instance
#   used across every module in the FINSIGHT_GraphRAG pipeline.
#
# WHY LOGURU INSTEAD OF PYTHON'S BUILT-IN LOGGING?
#   Python's built-in logging module requires boilerplate setup in
#   every file — creating a logger, adding handlers, setting levels.
#   Loguru replaces all of that with a single import line and works
#   out of the box with sensible defaults.
#
#   Key advantages for a teaching demo:
#     - Colour-coded output in the terminal by log level
#       DEBUG=white, INFO=blue, SUCCESS=green, WARNING=yellow, ERROR=red
#     - Automatic inclusion of timestamp, level, module name, line number
#     - logger.success() level — perfect for confirming pipeline stages
#     - Zero configuration needed in each module — just import and use
#
# HOW TO USE IN ANY MODULE:
#   from loguru import logger
#   logger.info("Starting pipeline...")
#   logger.success("Stage complete")
#   logger.warning("No entities found")
#   logger.error("Connection failed")
#   logger.debug("Chunk id: abc123")
#
# NOTE:
#   You do not need to import from this file in every module.
#   Loguru's logger is a global singleton — configuring it once here
#   (by importing this module at startup) applies the configuration
#   everywhere. The FastAPI lifespan handler imports this at startup.
#
# DO YOU RUN THIS FILE?
#   No. Imported once at application startup via api/main.py.


import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_level: str = "INFO", log_to_file: bool = False) -> None:
    """
    Configures the loguru logger with console and optional file output.

    Call this once at application startup. After calling this function,
    any module that does `from loguru import logger` gets the configured
    logger automatically — no per-module setup needed.

    Args:
        log_level:    Minimum log level to display.
                      One of: DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL
                      Default INFO shows everything except DEBUG messages.
                      Use DEBUG during development to see per-chunk details.

        log_to_file:  If True, also writes logs to a file at logs/pipeline.log
                      Useful for capturing ingestion output for review.
                      Default False — console only for the demo.
    """

    # Remove the default loguru handler so we can add our own
    # with custom formatting. Without this we would get duplicate log lines.
    logger.remove()

    # --- Console handler ---
    # Writes colour-coded logs to stdout (the terminal)
    # Format breakdown:
    #   {time:HH:mm:ss}  -> timestamp in 24h format e.g. 14:32:07
    #   {level:<8}       -> log level left-padded to 8 chars e.g. "INFO    "
    #   {name}           -> module name e.g. "graph_rag.agents.entity_agent"
    #   {line}           -> line number in the source file
    #   {message}        -> the actual log message
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,       # enable colour codes in terminal output
        enqueue=False,       # synchronous logging (simpler for a demo)
    )

    # --- File handler (optional) ---
    # If log_to_file=True, also writes to logs/pipeline.log
    # rotation="10 MB" starts a new log file when the current one hits 10MB
    # retention="7 days" deletes log files older than 7 days
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)   # create logs/ folder if it does not exist

        logger.add(
            log_dir / "pipeline.log",
            level=log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level:<8} | "
                "{name}:{line} | "
                "{message}"
            ),
            rotation="10 MB",     # start new file at 10MB
            retention="7 days",   # delete files older than 7 days
            colorize=False,       # no colour codes in log files
            enqueue=True,         # async file writes to avoid blocking
        )

        logger.info(f"File logging enabled at: {log_dir / 'pipeline.log'}")


# -----------------------------------------------------------------------------
# AUTO-CONFIGURE ON IMPORT
# -----------------------------------------------------------------------------
# When this module is imported for the first time, configure the logger
# immediately with default settings (INFO level, console only).
#
# This means any module that imports logger.py (directly or transitively)
# will automatically have a properly configured logger available.
#
# The FastAPI lifespan in api/main.py imports config which imports this,
# so the logger is configured before the first request arrives.

setup_logger(log_level="INFO", log_to_file=False)