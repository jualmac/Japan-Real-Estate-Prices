"""
Performance metrics, model comparison, and nested CV aggregation.

Provides MAPE / RMSE / MAE / R^2 in the original (un-logged) target scale,
a tidy comparison DataFrame across models, and aggregation helpers that
summarize nested temporal CV outer-fold metrics.
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import pandas as pd

from src.config import PARAMETERS_DIR
from src.utils.io import logger, write_json

if TYPE_CHECKING:
    from src.models.temporal_cv import OuterFoldResult

########################################################################################################################
#
# NESTED CV AGGREGATION
#
########################################################################################################################
METRIC_COLUMNS: List[str] = ["mape", "rmse", "mae", "r2"]

def nested_metrics_summary(
    per_fold_records: List["OuterFoldResult"],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a (per-fold DataFrame, aggregate Series) pair from a list of
    :class:`~src.models.temporal_cv.OuterFoldResult`.

    The aggregate Series exposes ``<metric>_mean`` and ``<metric>_std`` entries.
    """
    if not per_fold_records:
        empty = pd.DataFrame(columns=["fold"] + METRIC_COLUMNS)
        return empty, pd.Series(dtype=float)

    rows = [record.as_record() for record in per_fold_records]
    per_fold_df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

    aggregate: Dict[str, float] = {}
    for metric in METRIC_COLUMNS:
        if metric in per_fold_df.columns:
            aggregate[f"{metric}_mean"] = float(per_fold_df[metric].mean())
            aggregate[f"{metric}_std"] = float(per_fold_df[metric].std(ddof=0))
    return per_fold_df, pd.Series(aggregate)


def persist_nested_cv_artifacts(
    model_name: str,
    per_fold_records: List["OuterFoldResult"],
    parameters_dir: Path = PARAMETERS_DIR,
) -> Tuple[Path, pd.DataFrame, pd.Series]:
    """
    Persist nested-CV artifacts for ``model_name`` and return the materialized
    objects:

    - ``parameters/nested_cv_<model>.json`` -> per-fold params + metrics
      (loaded back via :func:`src.utils.io.read_json`).
    - Returns the per-fold DataFrame and aggregate Series for further use.

    The aggregate CSV is appended to via :func:`append_nested_cv_csv`.
    """
    parameters_dir.mkdir(parents=True, exist_ok=True)
    per_fold_df, aggregate = nested_metrics_summary(per_fold_records)

    payload = {
        "model": model_name,
        "per_fold": [record.as_record() | {"best_params": record.best_params} for record in per_fold_records],
        "aggregate": aggregate.to_dict(),
    }
    json_path = parameters_dir / f"nested_cv_{model_name}.json"
    write_json(json_path, payload)
    logger.info("Nested CV details for %s written to %s", model_name, json_path)
    return json_path, per_fold_df, aggregate


def append_nested_cv_csv(
    model_name: str,
    aggregate: pd.Series,
    parameters_dir: Path = PARAMETERS_DIR,
    filename: str = "nested_cv_metrics.csv",
) -> Path:
    """
    Append (or create) ``parameters/nested_cv_metrics.csv`` with one row per
    model summarizing nested-CV mean / std metrics across outer folds.
    """
    parameters_dir.mkdir(parents=True, exist_ok=True)
    csv_path = parameters_dir / filename

    row = {"model": model_name, **aggregate.to_dict()}
    new_row_df = pd.DataFrame([row])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["model"] != model_name]
        combined = pd.concat([existing, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_csv(csv_path, index=False)
    logger.info("Nested CV aggregate row for %s persisted to %s", model_name, csv_path)
    return csv_path


def nested_comparison_table(
    parameters_dir: Path = PARAMETERS_DIR,
    filename: str = "nested_cv_metrics.csv",
) -> Optional[pd.DataFrame]:
    """
    Return the current nested CV comparison table sorted by ``rmse_mean``, or
    ``None`` when the CSV does not exist.
    """
    csv_path = parameters_dir / filename
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "rmse_mean" in df.columns:
        df = df.sort_values("rmse_mean", ascending=True).reset_index(drop=True)
    return df