"""
Central project configuration.

Holds reproducibility constants (SEED), filesystem paths, column groupings used
across modules, and default modeling hyperparameters. Values can be overridden
at runtime by loading config.yaml on top of the defaults defined here.
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
PARAMETERS_DIR: Path = PROJECT_ROOT / "parameters"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DB_FILE: Path = DATASETS_DIR / "jpHouses.db"


########################################################################################################################
#
# COLUMN GROUPINGS
#
########################################################################################################################
CATEGORICAL_COLUMNS: List[str] = [
    "Type", "Region", "Prefecture", "Municipality", "DistrictName", "NearestStation",
    "TimeToNearestStation", "FloorPlan", "LandShape", "Structure", "Use",
    "Purpose", "Direction", "Classification", "CityPlanning", "Period",
    "Renovation", "Remarks",
]

NUMERICAL_COLUMNS: List[str] = [
    "No", "MunicipalityCode", "MinTimeToNearestStation", "MaxTimeToNearestStation",
    "TradePrice", "Area", "AreaIsGreaterFlag", "UnitPrice", "PricePerTsubo",
    "Frontage", "FrontageIsGreaterFlag", "TotalFloorArea",
    "TotalFloorAreaIsGreaterFlag", "BuildingYear", "PrewarBuilding",
    "CoverageRatio", "FloorAreaRatio", "Breadth", "Year", "Quarter",
]

# Columns that leak the target or are unusable signal;
COLUMNS_TO_DROP: List[str] = ["Remarks", "Renovation", "UnitPrice", "PricePerTsubo"]

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
    "Frontage",
    "Breadth",
]
# Type itself stays constant within a study, but these categorical attributes vary;
CATEGORICAL_FEATURES: List[str] = ["Structure", "Classification", "CityPlanning"]
TARGET_COLUMN: str = "TradePrice"


########################################################################################################################
#
# STUDIES
#
########################################################################################################################
@dataclass(frozen=True)
class Study:
    """
    One modeling study scoped to a single property type (or group of types).

    Each study runs the full pipeline independently so that heterogeneous
    property markets are not forced into a single regressor.
    """
    name: str
    property_types: Tuple[str, ...]


# The three residential studies; agricultural and forest land are excluded;
STUDIES: Tuple[Study, ...] = (
    Study(
        name="residential_land_and_building",
        property_types=("Residential Land(Land and Building)",),
    ),
    Study(
        name="residential_land_only",
        property_types=("Residential Land(Land Only)",),
    ),
    Study(
        name="pre_owned_condominiums",
        property_types=("Pre-owned Condominiums, etc.",),
    ),
)


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
    """
    seed: int = 42
    n_trials: int = 25
    val_quantile: float = 0.70
    test_quantile: float = 0.85
    target_transform: str = "log1p"
    use_gpu: bool = True
    models: Tuple[str, ...] = ("xgb", "lgbm", "rf")
    # models: Tuple[str, ...] = ("xgb", "lgbm", "rf", "enet", "svr")
    log_level: str = "INFO"
    datasets_dir: Path = DATASETS_DIR
    parameters_dir: Path = PARAMETERS_DIR
    logs_dir: Path = LOGS_DIR
    numerical_features: Tuple[str, ...] = field(default_factory=lambda: tuple(NUMERICAL_FEATURES))
    categorical_features: Tuple[str, ...] = field(default_factory=lambda: tuple(CATEGORICAL_FEATURES))
    studies: Tuple[Study, ...] = field(default_factory=lambda: tuple(STUDIES))


CONFIG: Config = Config()
