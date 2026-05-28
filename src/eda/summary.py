"""
General dataset overview helpers: shape, dtypes, duplicates, NaN profile.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import pandas as pd

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def print_overview(df: pd.DataFrame, name: str = "Full dataset") -> None:
    """
    Print row/column counts and a small header for a DataFrame.
    """
    print(f"===== OVERVIEW: {name} =====")
    print(f"Number of rows    : {df.shape[0]}")
    print(f"Number of columns : {df.shape[1]}")
    print()


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Print and return the number of fully-duplicated rows.
    """
    n_duplicates = int(df.duplicated().sum())
    print("=== DUPLICATE ROW CHECK ===")
    print(f"Duplicated rows (all columns equal): {n_duplicates}")
    print()
    return n_duplicates


def nan_profile(df: pd.DataFrame, ascending: bool = False) -> pd.Series:
    """
    Return the fraction of NaNs per column, sorted by share.
    """
    print("=== NULL VALUES CHECK ===")
    profile = df.isna().mean().round(2).sort_values(ascending=ascending)
    print(profile)
    return profile


def dtype_audit(df: pd.DataFrame) -> pd.Series:
    """
    Print and return the dtype of each column (sorted alphabetically).
    """
    print("=== COLUMN DTYPES ===")
    dtypes = df.dtypes.sort_values()
    print(dtypes)
    return dtypes


def year_distribution(df: pd.DataFrame, year_column: str = "Year") -> pd.Series:
    """
    Print the row count per Year (useful before a temporal split).
    """
    print(f"=== ROW DISTRIBUTION BY {year_column} ===")
    counts = df[year_column].value_counts().sort_index()
    print(counts)
    return counts