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
LOCATIONS_FILE: Path = PROJECT_ROOT / "locations.csv"
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

# Snake_case names used after one-hot encoding (some models can't handle special chars);
SNAKE_CASE_FEATURES: List[str] = [
    "time_to_nearest_station", "area", "building_year", "coverage_ratio",
    "floor_area_ratio", "year", "quarter_cos", "quarter_sin", "latitude",
    "longitude", "type_forest_land", "type_pre_owned_condominiums_etc",
    "type_residential_land_land_only", "type_residential_land_land_and_building",
]


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
    n_trials: int = 30
    val_quantile: float = 0.70
    test_quantile: float = 0.85
    target_transform: str = "log1p"
    models: Tuple[str, ...] = ("xgb", "lgbm")
    log_level: str = "INFO"
    datasets_dir: Path = DATASETS_DIR
    parameters_dir: Path = PARAMETERS_DIR
    logs_dir: Path = LOGS_DIR
    numerical_features: Tuple[str, ...] = field(default_factory=lambda: tuple(NUMERICAL_FEATURES))
    categorical_features: Tuple[str, ...] = field(default_factory=lambda: tuple(CATEGORICAL_FEATURES))


CONFIG: Config = Config()
