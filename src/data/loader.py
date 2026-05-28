"""
DataFrame loaders that assemble the full Japan Real Estate dataset from local CSV files.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import Dict
import pandas as pd
from src.config import PREFECTURE_CODE_FILE, TRADE_PRICES_DIR

N_PREFECTURES = 47

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def load_prefecture_codes(path: Path = PREFECTURE_CODE_FILE) -> pd.DataFrame:
    """
    Load the prefecture_code lookup table.
    """
    return pd.read_csv(path)


def load_prefectures(directory: Path = TRADE_PRICES_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load every per-prefecture CSV (01.csv ... 47.csv) into a dict keyed by name.
    """
    prefectures: Dict[str, pd.DataFrame] = {}
    for i in range(1, N_PREFECTURES + 1):
        file_name = f"{i:02d}.csv"
        path = directory / file_name
        prefectures[f"prefecture_{i:02d}"] = pd.read_csv(path, low_memory=False)
    return prefectures


def load_all_prefectures(directory: Path = TRADE_PRICES_DIR) -> pd.DataFrame:
    """
    Concatenate all 47 prefecture CSVs into a single DataFrame indexed sequentially.
    """
    prefectures = load_prefectures(directory)
    return pd.concat(prefectures.values(), ignore_index=True)
