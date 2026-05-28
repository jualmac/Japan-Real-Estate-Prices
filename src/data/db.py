"""
Helper class to read and write to the SQLite database backing the Japan Real Estate dataset.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import sqlite3
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from src.config import DATASETS_DIR, DB_FILE

########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
DEFAULT_QUERY = "SELECT * FROM TokyoPrices"  # TODO: expand once SQL exposes all prefectures;

DATA_TYPES: Dict[str, type] = {
    "Breadth": float,
    "PricePerTsubo": float,
    "UnitPrice": float,
    "MinTimeToNearestStation": float,
    "TotalFloorArea": float,
    "CoverageRatio": float,
    "FloorAreaRatio": float,
    "BuildingYear": float,
    "MaxTimeToNearestStation": float,
    "Frontage": float,
}

########################################################################################################################
#
# FUNCTIONS / CLASSES
#
########################################################################################################################
def ensure_database() -> Path:
    """
    Make sure ./datasets exists and the SQLite file is initialized from local CSVs.
    Returns the datasets directory.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    DBConnection().initialize_database(DATASETS_DIR)
    return DATASETS_DIR


class DBConnection:
    """
    Handle SQLite operations for the Japan Real Estate Price database.
    """

    def __init__(self, database_file: str | Path = DB_FILE):
        self.database_file = str(database_file)

    def dataframe_creator(self, query: str = DEFAULT_QUERY) -> pd.DataFrame:
        """
        Run a SELECT query and return a typed DataFrame.
        """
        try:
            # sqlite3.connect creates the SQLite file when it does not exist;
            with sqlite3.connect(self.database_file) as conn:
                print("Connected to SQLite Version", sqlite3.version)
                dataframe = pd.read_sql_query(query, conn)

            for column, dtype in DATA_TYPES.items():
                if column in dataframe.columns:
                    dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce").astype(dtype)

            if "No" in dataframe.columns:
                dataframe.set_index("No", inplace=True)
            dataframe.replace("", pd.NA, inplace=True)
            return dataframe

        except sqlite3.Error as error:
            print("Error occurred - ", error)
            return pd.DataFrame()

    def run_sql(self, query: str) -> Optional[pd.DataFrame]:
        """
        Run an arbitrary SQL statement; returns a DataFrame for SELECTs, None otherwise.
        """
        try:
            # cursor.description is only populated for queries that return rows;
            with sqlite3.connect(self.database_file) as conn:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                conn.commit()

                if cursor.description is None:
                    return None

                columns = [column[0] for column in cursor.description]
                return pd.DataFrame(rows, columns=columns)

        except sqlite3.Error as error:
            print("Error occurred - ", error)
            return None

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = False,
    ) -> None:
        """
        Insert a DataFrame into the specified SQLite table.
        """
        try:
            with sqlite3.connect(self.database_file) as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=index)
        except sqlite3.Error as error:
            print("Error occurred - ", error)

    def initialize_database(self, data_directory: str | Path = DATASETS_DIR) -> None:
        """
        Populate SQLite tables from downloaded Kaggle CSV files when the database is empty.
        """
        data_directory = Path(data_directory)
        Path(self.database_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            with sqlite3.connect(self.database_file) as conn:
                existing_tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                    conn,
                )

                existing_table_names = set(existing_tables["name"])
                expected_prefecture_tables = {f"{i:02d}" for i in range(1, 48)}

                if expected_prefecture_tables.issubset(existing_table_names):
                    print("Database already has all prefecture tables, skipping initialization.")
                    return

                csv_files = sorted(data_directory.rglob("*.csv"))

                if not csv_files:
                    print(f"No CSV files found in {data_directory.resolve()}.")
                    return

                for csv_file in csv_files:
                    table_name = csv_file.stem

                    # Each Kaggle CSV becomes one SQLite table with the same base filename;
                    dataframe = pd.read_csv(csv_file, low_memory=False)
                    dataframe.to_sql(table_name, conn, if_exists="replace", index=False)
                    print(f"Created table {table_name} from {csv_file.name}.")

        except sqlite3.Error as error:
            print("Error occurred - ", error)