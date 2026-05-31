"""
Temporal train/validation/test split.

Splits happen before any preprocessing to avoid data leakage from the future
into the past. The cutoff years are derived from quantiles of the Year column.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from dataclasses import dataclass
from typing import Iterable, Tuple
import pandas as pd
from src.config import CONFIG, TARGET_COLUMN

########################################################################################################################
#
# DATACLASS
#
########################################################################################################################
@dataclass
class TemporalSplit:
    """
    Container for the three temporally-ordered slices of a regression dataset.
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    year_cutoff_val: int
    year_cutoff_test: int


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def filter_by_type(
    df: pd.DataFrame,
    property_types: Iterable[str],
    type_column: str = "Type",
) -> pd.DataFrame:
    """
    Restrict the dataset to the given property type(s) for a single study.
    """
    types = list(property_types)
    return df[df[type_column].isin(types)].copy()


def temporal_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    year_column: str = "Year",
    val_quantile: float = CONFIG.val_quantile,
    test_quantile: float = CONFIG.test_quantile,
    verbose: bool = True,
) -> TemporalSplit:
    """
    Split df into train / validation / test chronologically.

    Train holds the oldest periods, validation the next chunk, and test the
    most recent periods. Cutoffs are computed from the requested quantiles of
    the year_column so the proportions are roughly 70/15/15.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    year_cutoff_val = int(df[year_column].quantile(val_quantile))
    year_cutoff_test = int(df[year_column].quantile(test_quantile))

    mask_test = df[year_column] >= year_cutoff_test
    mask_val = (df[year_column] >= year_cutoff_val) & (df[year_column] < year_cutoff_test)
    mask_train = df[year_column] < year_cutoff_val

    split = TemporalSplit(
        X_train=X[mask_train],
        X_val=X[mask_val],
        X_test=X[mask_test],
        y_train=y[mask_train],
        y_val=y[mask_val],
        y_test=y[mask_test],
        year_cutoff_val=year_cutoff_val,
        year_cutoff_test=year_cutoff_test,
    )

    if verbose:
        total = len(X)
        print(f"Train cutoff -> validation : year >= {year_cutoff_val}")
        print(f"Validation cutoff -> test  : year >= {year_cutoff_test}")
        print(f"Train      : {split.X_train.shape[0]:>8} rows ({split.X_train.shape[0] / total * 100:.1f}%)")
        print(f"Validation : {split.X_val.shape[0]:>8} rows ({split.X_val.shape[0] / total * 100:.1f}%)")
        print(f"Test       : {split.X_test.shape[0]:>8} rows ({split.X_test.shape[0] / total * 100:.1f}%)")

    return split


def unpack_split(split: TemporalSplit) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Convenience helper that returns the six split arrays as a tuple.
    """
    return split.X_train, split.X_val, split.X_test, split.y_train, split.y_val, split.y_test
