"""
Helper function to read and write to the SQLite database;
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import sqlite3
from pathlib import Path
import pandas as pd

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
DATASETS_DIR = Path("./datasets")
DB_FILE = DATASETS_DIR / "jpHouses.db"

db_file = 'datasets/jpHouses.db'
sql_query = f"""
    SELECT * 
    FROM TokyoPrices
""" #TODO: Select all data, not just Tokyo;

data_types = {
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

def read_data() -> Path:
    """
    Read CSV files from data/datasets and initialize the SQLite database in the same folder.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    db = DBConnection(database_file=str(DB_FILE))
    db.initialize_database(DATASETS_DIR)
    return DATASETS_DIR

class DBConnection:
    """
    Handle SQLite operations for the Japan Real Estate Price Database;
    """

    def __init__(self, database_file: str = db_file):
        self.database_file = database_file

    def dataframe_creator(self, query: str = sql_query) -> pd.DataFrame:
        """
        Create a cleaned DataFrame from a SQL query.
        """
        try:
            # sqlite3.connect creates the SQLite file when it does not exist;
            with sqlite3.connect(self.database_file) as conn:
                print("Connected to SQLite Version", sqlite3.version)
                dataframe = pd.read_sql_query(query, conn)

            # Keep numeric database columns predictable for analysis and modeling;
            for column, dtype in data_types.items():
                if column in dataframe.columns:
                    dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').astype(dtype)

            if 'No' in dataframe.columns:
                dataframe.set_index('No', inplace=True)
            dataframe.replace('', pd.NA, inplace=True)
            return dataframe

        except sqlite3.Error as error:
            print('Error occurred - ', error)
            return pd.DataFrame()

    def run_sql(self, query: str) -> pd.DataFrame | None:
        """
        Run a SQL query on the SQLite database.
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
            print('Error occurred - ', error)
            return None

    def insert_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = False
    ) -> None:
        """
        Insert a DataFrame into the specified SQLite table.
        """
        try:
            with sqlite3.connect(self.database_file) as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=index)

        except sqlite3.Error as error:
            print('Error occurred - ', error)

    def initialize_database(self, data_directory: str | Path = "datasets") -> None:
        """
        Initialize SQLite tables from downloaded Kaggle CSV files when the database is empty.
        """
        data_directory = Path(data_directory)
        Path(self.database_file).parent.mkdir(parents=True, exist_ok=True)

        try:
            with sqlite3.connect(self.database_file) as conn:
                existing_tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                    conn
                )

                if not existing_tables.empty:
                    print("Database already has tables, skipping initialization.")
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
            print('Error occurred - ', error)