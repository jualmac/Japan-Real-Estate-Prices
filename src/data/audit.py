"""
SQLite persistence helpers for pipeline audit checkpoints.

The audit tables are append-only and keyed by run_id, which lets one pipeline
execution be inspected across its model input snapshots and evaluation metrics.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import pandas as pd

from src.config import CONFIG, DB_FILE
from src.data.db import DBConnection
from src.preprocessing.splitting import TemporalSplit

########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
AUDIT_PIPELINE_RUNS_TABLE = "audit_pipeline_runs"
AUDIT_MODEL_INPUT_SPLITS_TABLE = "audit_model_input_splits"
AUDIT_MODEL_METRICS_TABLE = "audit_model_metrics"

SplitData = Tuple[pd.DataFrame, pd.Series]

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def create_run_id(now: datetime | None = None) -> str:
    """
    Create a sortable identifier for one pipeline execution.
    """
    timestamp = now or datetime.now(timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    return timestamp.strftime("run_%Y%m%dT%H%M%S%fZ")


def build_model_input_splits(run_id: str, splits: Mapping[str, SplitData]) -> pd.DataFrame:
    """
    Combine model-ready feature matrices and log targets into one audit table.
    """
    frames = []
    for split_name, (X, y_log) in splits.items():
        frame = X.reset_index(drop=True).copy()
        frame.insert(0, "row_id", pd.Series(y_log.index, index=frame.index).to_numpy())
        frame.insert(0, "split_name", split_name)
        frame.insert(0, "run_id", run_id)
        frame["target_log"] = y_log.reset_index(drop=True).to_numpy()
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def save_model_input_splits(
    run_id: str,
    splits: Mapping[str, SplitData],
    database_file: str | Path = DB_FILE,
) -> int:
    """
    Persist train/validation/test model inputs for later inspection.
    """
    audit_df = build_model_input_splits(run_id, splits)
    DBConnection(database_file).insert_dataframe(
        audit_df,
        AUDIT_MODEL_INPUT_SPLITS_TABLE,
        if_exists="append",
        index=False,
    )
    return len(audit_df)


def save_pipeline_run_metadata(
    run_id: str,
    split: TemporalSplit,
    feature_columns: Iterable[str],
    database_file: str | Path = DB_FILE,
) -> None:
    """
    Persist one row describing the pipeline run and its split boundaries.
    """
    feature_columns = list(feature_columns)
    metadata = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": CONFIG.seed,
                "n_trials": CONFIG.n_trials,
                "models": json.dumps(list(CONFIG.models)),
                "val_quantile": CONFIG.val_quantile,
                "test_quantile": CONFIG.test_quantile,
                "year_cutoff_val": split.year_cutoff_val,
                "year_cutoff_test": split.year_cutoff_test,
                "train_rows": len(split.X_train),
                "validation_rows": len(split.X_val),
                "test_rows": len(split.X_test),
                "feature_count": len(feature_columns),
                "feature_columns": json.dumps(feature_columns),
            }
        ]
    )
    DBConnection(database_file).insert_dataframe(
        metadata,
        AUDIT_PIPELINE_RUNS_TABLE,
        if_exists="append",
        index=False,
    )


def build_metrics_audit_frame(
    run_id: str,
    metrics_by_split: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Convert model metric tables into one normalized audit DataFrame.
    """
    frames = []
    for split_name, metrics in metrics_by_split.items():
        frame = metrics.reset_index(names="model_name").copy()
        frame.insert(0, "split_name", split_name)
        frame.insert(0, "run_id", run_id)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def save_model_metrics(
    run_id: str,
    metrics_by_split: Mapping[str, pd.DataFrame],
    database_file: str | Path = DB_FILE,
) -> int:
    """
    Persist evaluation metrics for each model and split.
    """
    metrics_df = build_metrics_audit_frame(run_id, metrics_by_split)
    DBConnection(database_file).insert_dataframe(
        metrics_df,
        AUDIT_MODEL_METRICS_TABLE,
        if_exists="append",
        index=False,
    )
    return len(metrics_df)
