"""
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path

import pandas as pd
from db_reader import DBConnection

########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
DATASETS_DIR = Path("./datasets")
DB_FILE = DATASETS_DIR / "jpHouses.db"

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def read_data() -> Path:
    """
    Read CSV files from data/datasets and initialize the SQLite database in the same folder.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(DATASETS_DIR.rglob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {DATASETS_DIR.resolve()}.")
        return DATASETS_DIR

    # Load each CSV so missing or unreadable files fail early before database creation;
    for csv_file in csv_files:
        pd.read_csv(csv_file)
        print(f"Read {csv_file.name}.")

    db = DBConnection(database_file=str(DB_FILE))
    db.initialize_database(DATASETS_DIR)
    return DATASETS_DIR

########################################################################################################################
#
# MAIN
#
########################################################################################################################
if __name__ == "__main__":
    read_data()
    print("Done!")