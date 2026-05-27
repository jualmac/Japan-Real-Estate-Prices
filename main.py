"""
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path

# KaggleHub weirdness;
import kagglesdk.kaggle_env as kaggle_env

if not hasattr(kaggle_env, "get_web_endpoint"):
    kaggle_env.get_web_endpoint = kaggle_env.get_endpoint

import kagglehub
from db_reader import DBConnection

########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
DATA_ROOT = Path("./data")
TRADE_PRICES_DIR = DATA_ROOT / "trade_prices"
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
    Download the Kaggle dataset only when ./data/trade_prices is missing.
    Skips the download when the local trade_prices directory already exists.
    """
    if _trade_prices_dir_exists(TRADE_PRICES_DIR):
        print(f"Data already present at {TRADE_PRICES_DIR.resolve()}, skipping download.")
        return TRADE_PRICES_DIR
    print(f"No data found at {TRADE_PRICES_DIR.resolve()}, downloading from Kaggle...")

    # Force the download here because KaggleHub can leave a .complete marker even if it was removed or renamed;
    downloaded_path = kagglehub.dataset_download(
        KAGGLE_DATASET,
        force_download=True,
        output_dir=str(TRADE_PRICES_DIR.parent),
    )
    print("Path to dataset files:", downloaded_path)
    return TRADE_PRICES_DIR

########################################################################################################################
#
# MAIN
#
########################################################################################################################
if __name__ == "__main__":
    trade_prices_dir = download_tradeprices()
    DBConnection().initialize_database(trade_prices_dir)
    print("Done!")