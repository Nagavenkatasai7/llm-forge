"""Structured logging module for llm-forge.

Provides Rich-formatted console logging, file logging for training runs,
and a factory function for creating module-specific loggers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_LOGGING_CONFIGURED = False

# Custom Rich theme for log output
_LOG_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "green",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
    }
)

_console = Console(theme=_LOG_THEME, stderr=True)

# Default log directory (can be overridden via config)
_DEFAULT_LOG_DIR = Path("logs")


# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------


class _FileFormatter(logging.Formatter):
    """Detailed formatter for file-based log output.

    Format: ``[2024-12-01 14:30:05.123 UTC] [INFO ] [llm_forge.training] message``
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        level = record.levelname.ljust(8)
        module = record.name
        msg = record.getMessage()
        formatted = f"[{ts} UTC] [{level}] [{module}] {msg}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            formatted += f"\n{record.exc_text}"
        if record.stack_info:
            formatted += f"\n{record.stack_info}"
        return formatted


class _CompactRichHandler(RichHandler):
    """Thin wrapper around RichHandler with llm-forge defaults."""

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level = record.levelname
        style = {
            "DEBUG": "dim cyan",
            "INFO": "green",
            "WARNING": "bold yellow",
            "ERROR": "bold red",
            "CRITICAL": "bold white on red",
        }.get(level, "")
        return Text(f" {level:<8}", style=style)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(
    verbose: bool = False,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    enable_file_logging: bool = True,
) -> None:
    """Configure the root ``llm_forge`` logger.

    Parameters
    ----------
    verbose:
        When *True*, sets the log level to ``DEBUG``; otherwise ``INFO``.
    log_dir:
        Directory for log files.  Defaults to ``./logs``.
    log_file:
        Explicit log filename.  When *None*, a timestamped filename is
        generated automatically (e.g. ``llm_forge_20241201_143005.log``).
    enable_file_logging:
        When *False*, only console logging is configured (useful for
        quick CLI invocations such as ``llm-forge info``).
    """

    global _LOGGING_CONFIGURED

    level = logging.DEBUG if verbose else logging.INFO

    # Root llm_forge logger
    root_logger = logging.getLogger("llm_forge")
    root_logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if _LOGGING_CONFIGURED:
        # Just update the level and return
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    # Prevent propagation to the root logger so we don't get double output
    root_logger.propagate = False

    # ---- Console handler (Rich) ----
    console_handler = _CompactRichHandler(
        console=_console,
        show_time=True,
        show_level=True,
        show_path=verbose,
        rich_tracebacks=True,
        tracebacks_show_locals=verbose,
        markup=True,
        log_time_format="[%H:%M:%S]",
    )
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # ---- File handler ----
    if enable_file_logging:
        log_path = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            if log_file is None:
                ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
                log_file = f"llm_forge_{ts}.log"
            file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # always capture everything in file
            file_handler.setFormatter(_FileFormatter())
            root_logger.addHandler(file_handler)
        except OSError:
            # If we cannot write logs (read-only FS, etc.), warn but continue
            root_logger.warning("Unable to create log file at %s; file logging disabled.", log_path)

    # Quieten noisy third-party loggers
    for noisy in (
        "transformers",
        "datasets",
        "torch",
        "accelerate",
        "bitsandbytes",
        "urllib3",
        "httpx",
        "httpcore",
        "filelock",
        "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``llm_forge`` namespace.

    Parameters
    ----------
    name:
        Dotted module name, e.g. ``"training.lora"``.  The returned logger
        will be named ``llm_forge.<name>``.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Examples
    --------
    >>> logger = get_logger("training.lora")
    >>> logger.info("Starting LoRA fine-tuning")
    """

    # Ensure logging is configured at least with defaults
    if not _LOGGING_CONFIGURED:
        setup_logging(verbose=False, enable_file_logging=False)

    qualified = name if name.startswith("llm_forge") else f"llm_forge.{name}"
    return logging.getLogger(qualified)


def get_console() -> Console:
    """Return the shared Rich console instance used by llm-forge logging."""
    return _console
