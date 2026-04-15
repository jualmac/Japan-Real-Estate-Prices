import sqlite3
import pandas as pd

db_file = 'data/jpHouses.db'
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

#TODO: Improve function bellow to also create the tables;
def dataframe_creator():
    try:
        # Creates SQLite file if it doesn't exist;
        with sqlite3.connect(db_file) as conn:
            print("Connected to SQLite Version", sqlite3.version)
            dataframe = pd.read_sql_query(sql_query, conn)

            for column, dtype in data_types.items():
                dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce').astype(dtype)
            
            dataframe.set_index('No', inplace=True)
            dataframe.replace('', pd.NA, inplace=True)

    except sqlite3.Error as error:
        print('Error occurred - ', error)

    return dataframe

#TODO: Create function that saves the csv -> SQLite into SQLite; 