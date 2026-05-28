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
import numpy as np
import pandas as pd
from src.config import DB_FILE
from src.data.locations import load_locations

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
    Rebuild a clean numeric `TimeToNearestStation` as the mean of min/max times.

    The raw column mixes numeric minutes and intervals like '1H30-2H', which
    breaks training. min/max columns are always numeric, so the mean is safe.
    """
    if {"MinTimeToNearestStation", "MaxTimeToNearestStation"}.issubset(df.columns):
        df["TimeToNearestStation"] = (
            df["MinTimeToNearestStation"] + df["MaxTimeToNearestStation"]
        ) / 2
    return df


def add_location_features(df: pd.DataFrame, database_file: Path = DB_FILE) -> pd.DataFrame:
    """
    Left-merge Latitude/Longitude from SQLite locations table onto df.
    """
    df_locations = load_locations(database_file)
    return df.merge(df_locations, on="DistrictName", how="left")


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
def feature_engineer_frame(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the deterministic, row-wise feature engineering steps to a single frame.

    Steps:
    1. Rebuild TimeToNearestStation from min/max.
    2. Cyclical encoding of Quarter.
    3. Merge Latitude/Longitude from the SQLite locations table.
    """
    df = X.copy()
    df = rebuild_time_to_nearest_station(df)
    if "Quarter" in df.columns:
        nature_encode(df, "Quarter", 4)
    df = add_location_features(df)
    return df