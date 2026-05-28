"""
DataFrame loaders that assemble the full Japan Real Estate dataset from SQLite tables.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import Dict
import pandas as pd
from src.config import DB_FILE
from src.data.db import DATA_TYPES, DBConnection

N_PREFECTURES = 47

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def load_prefectures(database_file: Path = DB_FILE) -> Dict[str, pd.DataFrame]:
    """
    Load every per-prefecture SQLite table ("01" ... "47") into a dict keyed by name.
    """
    db = DBConnection(database_file)
    prefectures: Dict[str, pd.DataFrame] = {}
    for i in range(1, N_PREFECTURES + 1):
        table_name = f"{i:02d}"
        df = db.run_sql(f'SELECT * FROM "{table_name}"')
        if df is None:
            raise RuntimeError(f"Could not load prefecture table {table_name} from {database_file}.")
        prefectures[f"prefecture_{table_name}"] = _normalize_trade_price_dataframe(df)
    return prefectures


def load_all_prefectures(database_file: Path = DB_FILE) -> pd.DataFrame:
    """
    Concatenate all 47 prefecture SQLite tables into a single DataFrame indexed sequentially.
    """
    prefectures = load_prefectures(database_file)
    return pd.concat(prefectures.values(), ignore_index=True)


def _normalize_trade_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same basic dtype/null normalization used by the database reader.
    """
    df = df.copy()
    for column, dtype in DATA_TYPES.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(dtype)
    df.replace("", pd.NA, inplace=True)
    return df
