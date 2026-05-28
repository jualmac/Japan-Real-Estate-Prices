"""
Per-column missing-value analysis grouped by property Type.

These functions are read-only; they only print/return summaries used to inform
the imputation rules in src/preprocessing/cleaning.py.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Iterable, List
import pandas as pd

DEFAULT_COLUMNS_TO_ANALYZE: List[str] = [
    "Renovation", "FloorPlan", "Purpose", "TotalFloorArea",
    "BuildingYear", "Use", "Structure", "Frontage", "Breadth",
    "Classification", "Region", "Direction", "LandShape",
    "FloorAreaRatio", "CoverageRatio", "MaxTimeToNearestStation",
    "MinTimeToNearestStation", "NearestStation", "CityPlanning",
]

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def analyze_missing_by_type(df: pd.DataFrame, column: str, type_column: str = "Type") -> pd.DataFrame:
    """
    Print missing-value statistics for a single column, broken down by property type.

    Returns a DataFrame with total rows, filled rows and the filled percentage
    per `type_column` value.
    """
    print("\n ============================================================")
    print(f"COLUMN: {column}")
    print("============================================================")

    total = len(df)
    n_missing = int(df[column].isna().sum())
    pct_missing = n_missing / total * 100
    print(f"Missing total: {n_missing} ({pct_missing:.1f}%)")

    print("\n--- Where the NaNs live (% per property type) ---")
    proportion_nan = (
        df.loc[df[column].isna(), type_column]
        .value_counts(dropna=False, normalize=True)
        .round(3)
        * 100
    )
    print(proportion_nan)

    print("\n--- Fill rate per property type ---")
    group = df.groupby(type_column)[column]
    summary = group.agg(total="size", filled="count")
    summary["pct_filled"] = (summary["filled"] / summary["total"] * 100).round(1)
    print(summary)
    return summary


def analyze_all(
    df: pd.DataFrame,
    columns: Iterable[str] = DEFAULT_COLUMNS_TO_ANALYZE,
    type_column: str = "Type",
) -> None:
    """
    Run analyze_missing_by_type for every column in `columns` that exists in df.
    """
    for column in columns:
        if column in df.columns:
            analyze_missing_by_type(df, column, type_column=type_column)
