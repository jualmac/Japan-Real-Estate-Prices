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
from src.config import (
    DATASETS_DIR,
    DB_FILE,
    DISTRICT_CODE_FILE,
    MUNICIPALITY_CODE_FILE,
    PREFECTURE_CODE_FILE,
    TRADE_PRICES_DIR,
)

########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
DEFAULT_QUERY = "SELECT * FROM TokyoPrices"  # TODO: expand once SQL exposes all prefectures;

LOOKUP_TABLE_FILES: Dict[str, Path] = {
    "prefecture_code": PREFECTURE_CODE_FILE,
    "municipality_code": MUNICIPALITY_CODE_FILE,
    "district_code": DISTRICT_CODE_FILE,
}

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

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """
        Quote a SQLite identifier while preserving it as a single identifier.
        """
        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    def _sqlite_type(dtype) -> str:
        """
        Map a pandas dtype to a SQLite storage class for additive schema updates.
        """
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"
        if pd.api.types.is_float_dtype(dtype):
            return "REAL"
        if pd.api.types.is_bool_dtype(dtype):
            return "INTEGER"
        return "TEXT"

    def _add_missing_columns(self, conn: sqlite3.Connection, df: pd.DataFrame, table_name: str) -> None:
        """
        Add DataFrame columns that are missing from an existing SQLite table.
        """
        quoted_table = self._quote_identifier(table_name)
        existing_columns = pd.read_sql_query(f"PRAGMA table_info({quoted_table})", conn)
        existing_column_names = set(existing_columns["name"])

        for column_name, dtype in df.dtypes.items():
            if column_name in existing_column_names:
                continue

            quoted_column = self._quote_identifier(column_name)
            conn.execute(
                f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_column} {self._sqlite_type(dtype)}"
            )

    def dataframe_creator(self, query: str = DEFAULT_QUERY) -> pd.DataFrame:
        """
        Run a SELECT query and return a typed DataFrame.
        """
        try:
            # sqlite3.connect creates the SQLite file when it does not exist;
            with sqlite3.connect(self.database_file) as conn:
                print("Connected to SQLite Version", sqlite3.sqlite_version)
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
                if if_exists == "append":
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                        conn,
                        params=(table_name,),
                    )
                    if not tables.empty:
                        self._add_missing_columns(conn, df, table_name)

                df.to_sql(table_name, conn, if_exists=if_exists, index=index)
        except sqlite3.Error as error:
            print("Error occurred - ", error)

    def initialize_database(self, data_directory: str | Path = DATASETS_DIR) -> None:
        """
        Populate SQLite tables from local CSV files when expected tables are missing.

        Trade-price rows are loaded from datasets/trade_prices/01.csv ... 47.csv.
        Location lookup tables come from prefecture_code.csv, municipality_code.csv,
        and district_code.csv in datasets/.
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
                expected_data_tables = {f"{i:02d}" for i in range(1, 48)} | set(
                    LOOKUP_TABLE_FILES
                )

                if expected_data_tables.issubset(existing_table_names):
                    print("Database already has all expected data tables, skipping initialization.")
                    return

                csv_sources: list[tuple[str, Path]] = []
                trade_prices_dir = TRADE_PRICES_DIR if data_directory == DATASETS_DIR else data_directory / "trade_prices"

                for prefecture_index in range(1, 48):
                    table_name = f"{prefecture_index:02d}"
                    if table_name in existing_table_names:
                        continue

                    csv_path = trade_prices_dir / f"{table_name}.csv"
                    if csv_path.is_file():
                        csv_sources.append((table_name, csv_path))

                for table_name, csv_path in LOOKUP_TABLE_FILES.items():
                    if table_name in existing_table_names:
                        continue
                    if csv_path.is_file():
                        csv_sources.append((table_name, csv_path))
                    else:
                        print(f"Missing CSV for table {table_name!r}: {csv_path.resolve()}")

                if not csv_sources:
                    missing_tables = sorted(expected_data_tables - existing_table_names)
                    print(
                        "No CSV sources found for missing tables: "
                        + ", ".join(missing_tables)
                    )
                    return

                for table_name, csv_file in csv_sources:
                    dataframe = pd.read_csv(csv_file, low_memory=False)
                    dataframe.to_sql(table_name, conn, if_exists="replace", index=False)
                    print(f"Created table {table_name} from {csv_file.name}.")

        except sqlite3.Error as error:
            print("Error occurred - ", error)