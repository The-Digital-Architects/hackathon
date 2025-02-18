"""Setup the logger."""
import logging
import os
import sys
from types import TracebackType
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "ml_pipeline",
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Set up logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

def log_exception(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None) -> None:
    """Log any uncaught exceptions except KeyboardInterrupts.

    Based on https://stackoverflow.com/a/16993115.

    :param exc_type: The type of the exception.
    :param exc_value: The exception instance.
    :param exc_traceback: The traceback.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("A wild %s appeared!", exc_type.__name__, exc_info=(exc_type, exc_value, exc_traceback))


def print_section_separator(title: str, spacing: int = 2) -> None:
    """Print a section separator.

    :param title: title of the section
    :param spacing: spacing between the sections
    """
    try:
        separator_length = os.get_terminal_size().columns
    except OSError:
        separator_length = 200
    separator_char = "="
    title_char = " "
    separator = separator_char * separator_length
    title_padding = (separator_length - len(title)) // 2
    centered_title = (
        f"{title_char * title_padding}{title}{title_char * title_padding}" if len(title) % 2 == 0 else f"{title_char * title_padding}{title}{title_char * (title_padding + 1)}"
    )
    print("\n" * spacing)  # noqa: T201
    print(f"{separator}\n{centered_title}\n{separator}")  # noqa: T201
    print("\n" * spacing)  # noqa: T201


sys.excepthook = log_exception
