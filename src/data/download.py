"""
Downloads the dataset in case it is not available on './datasets' directly from Kaggle.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
import kagglesdk.kaggle_env as kaggle_env

# KaggleHub weird shit;
if not hasattr(kaggle_env, "get_web_endpoint"):
    kaggle_env.get_web_endpoint = kaggle_env.get_endpoint

import kagglehub
from src.config import DATASETS_DIR, TRADE_PRICES_DIR

KAGGLE_DATASET = "nishiodens/japan-real-estate-transaction-prices"

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def _trade_prices_dir_exists(directory: Path) -> bool:
    """
    Return True when the trade_prices folder already exists.
    """
    return directory.is_dir()


def download_tradeprices() -> Path:
    """
    Download the Kaggle dataset only when ./datasets/trade_prices is missing.
    Skips the download when the local trade_prices directory already exists.
    """
    if _trade_prices_dir_exists(TRADE_PRICES_DIR):
        print(f"Data already present at {TRADE_PRICES_DIR.resolve()}, skipping download.")
        return TRADE_PRICES_DIR
    print(f"No data found at {TRADE_PRICES_DIR.resolve()}, downloading from Kaggle...")

    # Force re-download because KaggleHub can leave a .complete marker even after manual cleanup;
    downloaded_path = kagglehub.dataset_download(
        KAGGLE_DATASET,
        force_download=True,
        output_dir=str(DATASETS_DIR),
    )
    print("Path to dataset files:", downloaded_path)
    return TRADE_PRICES_DIR

########################################################################################################################
#
# MAIN
#
########################################################################################################################
if __name__ == "__main__":
    download_tradeprices()