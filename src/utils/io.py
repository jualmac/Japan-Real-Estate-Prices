"""
I/O helpers and the shared project logger.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import json
import logging
from pathlib import Path
from typing import Any

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def read_json(filepath: str | Path) -> dict:
    """
    Read a JSON file and return its contents as a dict.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def write_json(filepath: str | Path, data: Any) -> None:
    """
    Persist a Python object to disk as indented JSON.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


########################################################################################################################
#
# LOGGER (basic fallback - structured setup lives in src/utils/logging.py)
#
########################################################################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("japan_real_estate")
