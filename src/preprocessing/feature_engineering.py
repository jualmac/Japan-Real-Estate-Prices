"""
Feature engineering: cyclical encodings, TimeToNearestStation rebuild,
location enrichment, target log-transformation.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from src.config import DB_FILE
from src.data.locations import load_geolonia_location_lookup, normalize_romaji_key

# TODO: Add rolling-window statistics on TradePrice per Municipality/Year;

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def nature_encode(df: pd.DataFrame, col: str, div_period: int) -> None:
    """
    Apply a cyclical (sin/cos) transformation in-place.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame on which the function will be applied.
    col : str
        Period column on which the function will be applied.
    div_period : int
        Amount of periods until the cycle restarts (e.g. month=12, week=7, etc).
    """
    df[col + "_sin"] = np.sin(2 * np.pi * df[col] / div_period)
    df[col + "_cos"] = np.cos(2 * np.pi * df[col] / div_period)


def rebuild_time_to_nearest_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild a clean numeric `TimeToNearestStation` from min/max walking times.

    Equal min/max values preserve exact minute observations; interval estimates
    are represented by their midpoint.
    """
    if {"MinTimeToNearestStation", "MaxTimeToNearestStation"}.issubset(df.columns):
        df["TimeToNearestStation"] = (
            df["MinTimeToNearestStation"] + df["MaxTimeToNearestStation"]
        ) / 2
    return df


def add_location_features(df: pd.DataFrame, database_file: Path = DB_FILE) -> pd.DataFrame:
    """
    Left-merge Latitude/Longitude from Geolonia-backed location tables.
    """
    df_locations = load_geolonia_location_lookup(database_file)
    df = df.copy()
    df["DistrictNameKey"] = df["DistrictName"].apply(normalize_romaji_key)
    merged = df.merge(
        df_locations[["MunicipalityCode", "DistrictNameKey", "Latitude", "Longitude"]],
        on=["MunicipalityCode", "DistrictNameKey"],
        how="left",
        validate="many_to_one",
    )
    return merged.drop(columns="DistrictNameKey")


def log_transform_target(y: pd.Series) -> pd.Series:
    """
    Apply log1p to the regression target to compress the heavy right tail.
    """
    return np.log1p(y)


def inverse_log_transform_target(y_log: pd.Series | np.ndarray) -> np.ndarray:
    """
    Invert log1p back to the original target scale.
    """
    return np.expm1(y_log)


########################################################################################################################
#
# PIPELINE
#
########################################################################################################################
def apply_feature_engineering(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Apply feature engineering to all three splits and log-transform the targets.

    Steps (applied independently per split because they are deterministic and
    do not learn anything from the data):
    1. Rebuild TimeToNearestStation from min/max.
    2. Cyclical encoding of Quarter.
    3. Merge Latitude/Longitude from the SQLite locations table.
    4. Log-transform the target series.
    """
    splits = []
    for split in (X_train, X_val, X_test):
        df = split.copy()
        df = rebuild_time_to_nearest_station(df)
        if "Quarter" in df.columns:
            nature_encode(df, "Quarter", 4)
        df = add_location_features(df)
        splits.append(df)

    y_train_t = log_transform_target(y_train)
    y_val_t = log_transform_target(y_val)
    y_test_t = log_transform_target(y_test)

    return splits[0], splits[1], splits[2], y_train_t, y_val_t, y_test_t
