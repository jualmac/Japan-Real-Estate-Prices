"""
Central project configuration.

Holds reproducibility constants (SEED), filesystem paths, column groupings used
across modules, and default modeling hyperparameters. A config.yaml mirroring
these defaults is checked in for a planned runtime override loader; that loader
is not wired up yet, so these dataclass defaults are the single source of truth.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import numpy as np


########################################################################################################################
#
# PATHS
#
########################################################################################################################
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
TRADE_PRICES_DIR: Path = DATASETS_DIR / "trade_prices"
PREFECTURE_CODE_FILE: Path = DATASETS_DIR / "prefecture_code.csv"
LOCATIONS_FILE: Path = DATASETS_DIR / "locations.csv"
PARAMETERS_DIR: Path = PROJECT_ROOT / "parameters"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DB_FILE: Path = DATASETS_DIR / "jpHouses.db"


########################################################################################################################
#
# COLUMN GROUPINGS
#
########################################################################################################################
# Columns that leak the target or are unusable signal;
COLUMNS_TO_DROP: List[str] = ["Remarks", "UnitPrice", "PricePerTsubo", "TimeToNearestStation"]

# Property types that are bare land (no building, no station, etc.);
LAND_TYPES: List[str] = ["Residential Land(Land Only)", "Agricultural Land", "Forest Land"]
RURAL_TYPES: List[str] = ["Agricultural Land", "Forest Land"]
CONDO_TYPES: List[str] = ["Pre-owned Condominiums, etc."]

# Final selected features for the model;
NUMERICAL_FEATURES: List[str] = [
    "TimeToNearestStation",
    "Area",
    "BuildingYear",
    "CoverageRatio",
    "FloorAreaRatio",
    "Year",
    "Quarter_cos",
    "Quarter_sin",
    "Latitude",
    "Longitude",
]
CATEGORICAL_FEATURES: List[str] = ["Type"]
TARGET_COLUMN: str = "TradePrice"


########################################################################################################################
#
# REPRODUCIBILITY
#
########################################################################################################################
def set_seed(seed: int) -> None:
    """
    Configure deterministic seeds for Python and NumPy.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


########################################################################################################################
#
# CONFIG DATACLASS
#
########################################################################################################################
@dataclass(frozen=True)
class Config:
    """
    Immutable runtime configuration consumed by main.py and the pipeline modules.

    Notes on nested temporal CV cost
    --------------------------------
    Total model fits per pipeline run scale approximately as

        len(models) * (outer_folds * (n_trials * inner_folds + 1)
                       + (n_trials * inner_folds + 1))

    So `n_trials`, `outer_folds`, and `inner_folds` are the main knobs to tune
    runtime against statistical robustness of the reported metrics.
    """
    seed: int = 42
    val_quantile: float = 0.70
    test_quantile: float = 0.85
    target_transform: str = "log1p"
    use_gpu: bool = True
    log_level: str = "INFO"
    datasets_dir: Path = DATASETS_DIR
    parameters_dir: Path = PARAMETERS_DIR
    logs_dir: Path = LOGS_DIR
    numerical_features: Tuple[str, ...] = field(default_factory=lambda: tuple(NUMERICAL_FEATURES))
    categorical_features: Tuple[str, ...] = field(default_factory=lambda: tuple(CATEGORICAL_FEATURES))

    # Nested temporal cross-validation settings;
    # models: Tuple[str, ...] = ("xgb", "lgbm", "rf", "enet", "svr")
    models: Tuple[str, ...] = ("xgb",)
    nested_cv_enabled: bool = True
    n_trials: int = 30
    outer_folds: int = 3
    inner_folds: int = 2
    year_column: str = "Year"

CONFIG: Config = Config()