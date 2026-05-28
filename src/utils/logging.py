"""
SKETCH: Structured logging configuration for the pipeline.

Adds a console handler plus an optional file handler under ./logs so each
run leaves an auditable trace.

TODO:
- Add a JSON formatter option for production runs.
- Add a Rich handler for nicer terminal output.
- Wire log rotation (RotatingFileHandler / TimedRotatingFileHandler).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from src.config import LOGS_DIR

LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    logs_dir: Path = LOGS_DIR,
) -> logging.Logger:
    """
    SKETCH: Configure the root logger with console + optional file handler.

    Returns the project logger so callers can use `logger.info(...)` directly.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"run_{timestamp}.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Replace any pre-existing handlers (e.g. from src.utils.io basicConfig);
    root.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return logging.getLogger("japan_real_estate")