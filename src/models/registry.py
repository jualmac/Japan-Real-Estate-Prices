"""
Persistence layer for the best hyperparameters discovered by OptimizeRegressor.

Wraps the parameters/ folder so callers don't need to know about file naming.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import Dict

from src.config import PARAMETERS_DIR
from src.utils.io import read_json, write_json


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def best_params_path(model_name: str, parameters_dir: Path = PARAMETERS_DIR) -> Path:
    """
    Return the canonical JSON path for a given model name.
    """
    return parameters_dir / f"best_params_{model_name}.json"


def load_best_params(model_name: str, parameters_dir: Path = PARAMETERS_DIR) -> Dict:
    """
    Read the persisted best parameters for `model_name`.
    """
    return read_json(best_params_path(model_name, parameters_dir))


def save_best_params(model_name: str, params: Dict, parameters_dir: Path = PARAMETERS_DIR) -> Path:
    """
    Persist hyperparameters to disk and return the file path.
    """
    path = best_params_path(model_name, parameters_dir)
    write_json(path, params)
    return path
