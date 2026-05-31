"""
Load simple Japanese <-> romanized address translation tables.

Source:
https://geolonia.github.io/japanese-addresses/latest.csv
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_FILE


GEOLONIA_LATEST_CSV = "https://geolonia.github.io/japanese-addresses/latest.csv"
MUNICIPALITY_TABLE = "municipality_code"
DISTRICT_TABLE = "district_code"


def clean_romaji(value: object, *, remove_spaces: bool = False) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().title()
    text = re.sub(r"\s+", " ", text)
    if remove_spaces:
        text = text.replace(" ", "")
    return text


def kaggle_style_prefecture(value: object) -> str | None:
    text = clean_romaji(value)
    if text is None:
        return None
    return re.sub(r"\s+(To|Fu|Ken)$", "", text)


def kaggle_style_municipality(value: object) -> str | None:
    """
    Convert Geolonia romaji into the rough English style used by Kaggle/MLIT.

    Example: SAPPORO SHI CHUO KU -> Chuo Ward,Sapporo City
    """
    if pd.isna(value):
        return None

    text = str(value).strip().upper()
    parts = []
    city_match = re.search(r"(.+?)\s+SHI(?:\s+|$)", text)
    ward_match = re.search(r"(.+?)\s+KU$", text)

    if city_match and ward_match:
        city = clean_romaji(city_match.group(1))
        ward_text = text[city_match.end():].removesuffix(" KU")
        ward = clean_romaji(ward_text)
        if city and ward:
            return f"{ward} Ward,{city} City"

    replacements = [
        (r"\s+SHI$", " City"),
        (r"\s+KU$", " Ward"),
        (r"\s+MACHI$", " Town"),
        (r"\s+CHO$", " Town"),
        (r"\s+MURA$", " Village"),
        (r"\s+SON$", " Village"),
        (r"\s+GUN\s+", " District "),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement.upper(), text)
    return clean_romaji(text)


def district_base_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"\s+\d+$", "", str(value).strip())
    return clean_romaji(text, remove_spaces=True)


def load_geolonia_csv(source: str) -> pd.DataFrame:
    return pd.read_csv(source, dtype={"都道府県コード": str, "市区町村コード": str})


def build_tables(addresses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = addresses.copy()

    df["PrefectureCode"] = df["都道府県コード"].str.zfill(2)
    df["MunicipalityCode"] = df["市区町村コード"].str.zfill(5)
    df["MunicipalityCodeInt"] = pd.to_numeric(df["市区町村コード"], errors="coerce").astype("Int64")
    df["PrefectureJP"] = df["都道府県名"]
    df["PrefectureKana"] = df["都道府県名カナ"]
    df["PrefectureRomaji"] = df["都道府県名ローマ字"].map(clean_romaji)
    df["PrefectureEN"] = df["都道府県名ローマ字"].map(kaggle_style_prefecture)
    df["MunicipalityJP"] = df["市区町村名"]
    df["MunicipalityKana"] = df["市区町村名カナ"]
    df["MunicipalityRomaji"] = df["市区町村名ローマ字"].map(clean_romaji)
    df["MunicipalityEN"] = df["市区町村名ローマ字"].map(kaggle_style_municipality)
    df["DistrictNameJP"] = df["大字町丁目名"]
    df["DistrictNameKana"] = df["大字町丁目名カナ"]
    df["DistrictNameRomaji"] = df["大字町丁目名ローマ字"].map(clean_romaji)
    df["DistrictNameEN"] = df["大字町丁目名ローマ字"].map(
        lambda value: clean_romaji(value, remove_spaces=True)
    )
    df["DistrictNameBaseEN"] = df["大字町丁目名ローマ字"].map(district_base_name)
    df["KoazaAlias"] = df["小字・通称名"]
    df["Latitude"] = pd.to_numeric(df["緯度"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["経度"], errors="coerce")

    municipality_columns = [
        "PrefectureCode",
        "MunicipalityCode",
        "MunicipalityCodeInt",
        "PrefectureJP",
        "PrefectureKana",
        "PrefectureEN",
        "MunicipalityJP",
        "MunicipalityKana",
        "MunicipalityRomaji",
        "MunicipalityEN",
    ]
    district_columns = [
        "PrefectureCode",
        "MunicipalityCode",
        "MunicipalityCodeInt",
        "PrefectureJP",
        "PrefectureEN",
        "MunicipalityJP",
        "MunicipalityEN",
        "DistrictNameJP",
        "DistrictNameKana",
        "DistrictNameRomaji",
        "DistrictNameEN",
        "DistrictNameBaseEN",
        "KoazaAlias",
        "Latitude",
        "Longitude",
    ]

    municipalities = df[municipality_columns].drop_duplicates().reset_index(drop=True)
    districts = df[district_columns].drop_duplicates().reset_index(drop=True)
    return municipalities, districts


def write_tables(
    municipalities: pd.DataFrame,
    districts: pd.DataFrame,
    database_file: Path,
) -> None:
    database_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_file) as conn:
        municipalities.to_sql(MUNICIPALITY_TABLE, conn, if_exists="replace", index=False)
        districts.to_sql(DISTRICT_TABLE, conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Geolonia address translation tables into SQLite.")
    parser.add_argument("--source", default=GEOLONIA_LATEST_CSV)
    parser.add_argument("--database-file", type=Path, default=DB_FILE)
    args = parser.parse_args()

    addresses = load_geolonia_csv(args.source)
    municipalities, districts = build_tables(addresses)
    write_tables(municipalities, districts, args.database_file)

    print(f"Wrote {len(municipalities):,} rows to {MUNICIPALITY_TABLE!r}.")
    print(f"Wrote {len(districts):,} rows to {DISTRICT_TABLE!r}.")


if __name__ == "__main__":
    main()
