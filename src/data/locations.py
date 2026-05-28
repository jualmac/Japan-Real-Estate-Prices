"""
Location lookup utilities for district latitude/longitude features.

The checked-in locations CSV is treated as source data and is loaded into the
SQLite database by `ensure_database()`. Regeneration is available as an explicit
script because geocoding depends on an external service.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from src.config import DB_FILE, LOCATIONS_FILE
from src.data.db import DBConnection

N_PREFECTURES = 47

MANUAL_LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Sakaecho": (35.753072, 139.702148),
    "Shinjuku": (35.6937632, 139.7036319),
    "Higashioizumi": (35.65326585, 139.7095950961917),
    "Omorikita": (35.5884735, 139.7279334),
    "Omorinishi": (35.5884735, 139.7279334),
    "Nishikamata": (35.7111592, 139.7527104),
    "Minamimagome": (35.6927095, 139.699506),
    "Higashigotanda": (35.65326585, 139.7095950961917),
    "Toyotamakita": (35.748652449999994, 139.62212183114752),
    "Minamioizumi": (35.6927095, 139.699506),
    "Higashishinkoiwa": (35.65326585, 139.7095950961917),
    "Higashiogu": (35.65326585, 139.7095950961917),
    "Sekimachikita": (35.725677, 139.569367),
    "Kitakarasuyama": (35.69736665, 139.8124867359133),
    "Sasazuka": (35.6736801, 139.6672321),
}

########################################################################################################################
#
# LOADERS
#
########################################################################################################################
def load_locations(database_file: Path = DB_FILE) -> pd.DataFrame:
    """
    Load the district location lookup table from SQLite.
    """
    df = DBConnection(database_file).run_sql('SELECT * FROM "locations"')
    if df is None or df.empty:
        raise RuntimeError(f"Could not load locations table from {database_file}.")

    for column in ("Latitude", "Longitude"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


########################################################################################################################
#
# GENERATION
#
########################################################################################################################
class LocationBuilder:
    """
    Build `locations.csv` by geocoding unique DistrictName values from SQLite.
    """

    def __init__(
        self,
        user_agent: str,
        output_path: Path = LOCATIONS_FILE,
        database_file: Path = DB_FILE,
        logger: logging.Logger | None = None,
    ):
        try:
            Nominatim = importlib.import_module("geopy.geocoders").Nominatim
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install geopy to regenerate locations.csv.") from exc

        self.geolocator = Nominatim(user_agent=user_agent)
        self.output_path = output_path
        self.database_file = database_file
        self.logger = logger or logging.getLogger(__name__)

    def get_districts(self) -> pd.DataFrame:
        """
        Return unique district names across all per-prefecture trade price tables.
        """
        db = DBConnection(self.database_file)
        district_frames = []

        for i in range(1, N_PREFECTURES + 1):
            table_name = f"{i:02d}"
            df = db.run_sql(f'SELECT DISTINCT DistrictName FROM "{table_name}"')
            if df is not None and "DistrictName" in df.columns:
                district_frames.append(df)

        if not district_frames:
            raise RuntimeError(f"No district names found in {self.database_file}.")

        districts = pd.concat(district_frames, ignore_index=True)
        districts["DistrictName"] = districts["DistrictName"].astype(str).str.split(",").str[0].str.strip()
        districts = districts.dropna().drop_duplicates().sort_values("DistrictName").reset_index(drop=True)
        return districts

    def get_coordinates(self, place: str) -> Tuple[float | None, float | None]:
        """
        Geocode one district name.
        """
        try:
            location = self.geolocator.geocode(f"{place}, Japan")
            if location is None:
                return None, None
            return location.latitude, location.longitude
        except Exception:
            return None, None

    def apply_manual_locations(self, df_locations: pd.DataFrame) -> pd.DataFrame:
        """
        Fill known locations that geocoding commonly misses.
        """
        df_locations = df_locations.copy()
        for district, coords in MANUAL_LOCATIONS.items():
            df_locations.loc[
                df_locations["DistrictName"] == district,
                ["Latitude", "Longitude"],
            ] = coords
        return df_locations

    def build(self) -> pd.DataFrame:
        """
        Generate locations, write the CSV, and update the SQLite table.
        """
        df_locations = self.get_districts()
        self.logger.info("Retrieved %d unique districts.", len(df_locations))

        df_locations["Coordinates"] = df_locations["DistrictName"].apply(self.get_coordinates)
        df_locations[["Latitude", "Longitude"]] = pd.DataFrame(
            df_locations["Coordinates"].tolist(),
            index=df_locations.index,
        )
        df_locations = df_locations.drop(columns="Coordinates")
        df_locations = self.apply_manual_locations(df_locations)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_locations.to_csv(self.output_path, index=False)
        DBConnection(self.database_file).insert_dataframe(df_locations, "locations", if_exists="replace")
        self.logger.info("Saved locations to %s and SQLite table locations.", self.output_path)
        return df_locations


def build_locations(
    user_agent: str,
    output_path: Path = LOCATIONS_FILE,
    database_file: Path = DB_FILE,
) -> pd.DataFrame:
    """
    Convenience wrapper for programmatic regeneration.
    """
    return LocationBuilder(
        user_agent=user_agent,
        output_path=output_path,
        database_file=database_file,
    ).build()


########################################################################################################################
#
# CLI
#
########################################################################################################################
def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate district latitude/longitude lookup data.")
    parser.add_argument("--user-agent", required=True, help="Nominatim user agent, usually an email or app name.")
    parser.add_argument("--output-path", type=Path, default=LOCATIONS_FILE)
    parser.add_argument("--database-file", type=Path, default=DB_FILE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_locations(
        user_agent=args.user_agent,
        output_path=args.output_path,
        database_file=args.database_file,
    )


if __name__ == "__main__":
    main()
