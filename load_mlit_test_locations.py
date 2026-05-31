"""
One-off loader for MLIT Location Reference Information CSV files.

Scans a nested MLIT extract directory, concatenates every CSV it finds, adds a
few normalized helper columns, and writes the result to SQLite as test_locations.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_FILE


DEFAULT_SOURCE_DIR = Path("/home/juju/Downloads/a")
DEFAULT_TABLE_NAME = "test_locations"
ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")


def read_mlit_csv(csv_file: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        for parser_kwargs in (
            {"low_memory": False},
            {"engine": "python", "on_bad_lines": "skip"},
        ):
            try:
                df = pd.read_csv(csv_file, dtype=str, encoding=encoding, **parser_kwargs)
                df["source_file"] = str(csv_file)
                df["source_encoding"] = encoding
                return df
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"Could not decode {csv_file} with {ENCODINGS}: {last_error}",
    )


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def column(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(pd.NA, index=df.index)

    df["Prefecture"] = column("都道府県名")
    df["Municipality"] = column("市区町村名")
    df["MunicipalityCode"] = column("市区町村コード")
    df["DistrictName"] = column("大字町丁目名").combine_first(column("大字・丁目名"))
    df["KoazaAlias"] = column("小字・通称名")
    df["BlockLot"] = column("街区符号・地番")
    df["Latitude"] = pd.to_numeric(column("緯度"), errors="coerce")
    df["Longitude"] = pd.to_numeric(column("経度"), errors="coerce")

    return df


def load_locations(source_dir: Path) -> pd.DataFrame:
    csv_files = sorted(source_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {source_dir}.")

    frames = [frame for csv_file in csv_files if not (frame := read_mlit_csv(csv_file)).empty]
    if not frames:
        raise RuntimeError(f"CSV files were found under {source_dir}, but none contained rows.")

    locations = pd.concat(frames, ignore_index=True, sort=False)
    return add_normalized_columns(locations)


def write_table(df: pd.DataFrame, database_file: Path, table_name: str) -> None:
    database_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_file) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load MLIT location CSVs into SQLite.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--database-file", type=Path, default=DB_FILE)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    args = parser.parse_args()

    locations = load_locations(args.source_dir)
    write_table(locations, args.database_file, args.table_name)

    print(
        f"Wrote {len(locations):,} rows from {locations['source_file'].nunique():,} CSV files "
        f"to {args.database_file} table {args.table_name!r}."
    )


if __name__ == "__main__":
    main()
