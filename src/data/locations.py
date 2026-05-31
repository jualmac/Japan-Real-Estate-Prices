"""
Location lookup utilities for district latitude/longitude features.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from src.config import DB_FILE
from src.data.db import DBConnection

GEOLONIA_LOCATION_TABLES = ("prefecture_code", "municipality_code", "district_code")

########################################################################################################################
#
# LOADERS
#
########################################################################################################################
def normalize_romaji_key(value: object) -> str:
    """
    Normalize romanized Japanese place names for joins across data sources.
    """
    if pd.isna(value):
        return ""
    text = str(value).split(",")[0].strip().lower()
    key = re.sub(r"[^a-z0-9]", "", text)
    key = re.sub(r"^(?:oaza|aza)", "", key)
    return key.replace("ou", "o").replace("oo", "o")


def parent_district_key(value: object) -> str:
    """
    Derive a parent town key from detailed Geolonia districts when possible.
    """
    key = normalize_romaji_key(value)
    match = re.match(r"^(.+?(?:cho|machi|mura))(?=[a-z0-9])", key)
    if match:
        return match.group(1)
    return ""


def load_geolonia_location_lookup(database_file: Path = DB_FILE) -> pd.DataFrame:
    """
    Load district coordinates from Geolonia translation tables in SQLite.

    `district_code` is chome-level, while the Kaggle house-price rows usually
    only keep the base district. We collapse those rows by municipality and
    base district using median coordinates.
    """
    query = """
        SELECT
            d.MunicipalityCodeInt AS MunicipalityCode,
            COALESCE(TRIM(p.EnName), d.PrefectureEN) AS Prefecture,
            m.MunicipalityEN AS Municipality,
            d.DistrictNameBaseEN AS DistrictName,
            d.Latitude,
            d.Longitude
        FROM district_code AS d
        LEFT JOIN municipality_code AS m
            ON d.MunicipalityCode = m.MunicipalityCode
        LEFT JOIN prefecture_code AS p
            ON CAST(d.PrefectureCode AS INTEGER) = CAST(p.Code AS INTEGER)
        WHERE d.MunicipalityCodeInt IS NOT NULL
          AND d.DistrictNameBaseEN IS NOT NULL
          AND d.Latitude IS NOT NULL
          AND d.Longitude IS NOT NULL
    """
    df = DBConnection(database_file).run_sql(query)
    if df is None or df.empty:
        raise RuntimeError(
            "Could not load Geolonia location lookup. "
            f"Expected SQLite tables: {', '.join(GEOLONIA_LOCATION_TABLES)}."
        )

    df["MunicipalityCode"] = pd.to_numeric(df["MunicipalityCode"], errors="coerce")
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df["DistrictNameKey"] = df["DistrictName"].apply(normalize_romaji_key)
    df = df.dropna(subset=["MunicipalityCode", "Latitude", "Longitude"])
    df = df[df["DistrictNameKey"] != ""]

    exact_lookup = (
        df.groupby(["MunicipalityCode", "DistrictNameKey"], as_index=False)
        .agg(
            Latitude=("Latitude", "median"),
            Longitude=("Longitude", "median"),
            Prefecture=("Prefecture", "first"),
            Municipality=("Municipality", "first"),
            DistrictName=("DistrictName", "first"),
            SourceRows=("Latitude", "size"),
        )
    )

    parent_rows = df.copy()
    parent_rows["DistrictNameKey"] = parent_rows["DistrictName"].apply(parent_district_key)
    parent_rows = parent_rows[parent_rows["DistrictNameKey"] != ""]
    parent_lookup = (
        parent_rows.groupby(["MunicipalityCode", "DistrictNameKey"], as_index=False)
        .agg(
            Latitude=("Latitude", "median"),
            Longitude=("Longitude", "median"),
            Prefecture=("Prefecture", "first"),
            Municipality=("Municipality", "first"),
            DistrictName=("DistrictName", "first"),
            SourceRows=("Latitude", "size"),
        )
    )

    lookup = (
        pd.concat([exact_lookup, parent_lookup], ignore_index=True)
        .groupby(["MunicipalityCode", "DistrictNameKey"], as_index=False)
        .agg(
            Latitude=("Latitude", "first"),
            Longitude=("Longitude", "first"),
            Prefecture=("Prefecture", "first"),
            Municipality=("Municipality", "first"),
            DistrictName=("DistrictName", "first"),
            SourceRows=("SourceRows", "sum"),
        )
    )
    return lookup
