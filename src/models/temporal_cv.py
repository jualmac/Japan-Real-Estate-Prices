"""
Nested temporal cross-validation with expanding windows.

Outer loop
----------
Each outer fold trains on year-blocks ``0..k`` and evaluates on year-block
``k+1``. Outer fold metrics constitute the canonical performance estimate
of the pipeline (replacing the previous single-test-set evaluation).

Inner loop
----------
For each outer fold, Optuna is run on the outer-train window using a smaller
expanding-window temporal CV. The best hyperparameters from the inner study
are then used to fit one model on the full outer-train and score it on the
outer-eval block.

Both loops share :func:`expanding_year_folds` so the splitting logic is
identical at every level of the nest.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.models.fit_predict import fit_model, predict
from src.models.optimization import OptimizeRegressor
from src.models.training import instantiate_model
from src.preprocessing.feature_engineering import inverse_log_transform_target
from src.preprocessing.pipeline import preprocess_fold
from src.utils.io import logger

CVSplits = List[Tuple[np.ndarray, np.ndarray]]


########################################################################################################################
#
# DATACLASSES
#
########################################################################################################################
@dataclass
class OuterFoldResult:
    """
    Result of one outer fold of the nested temporal CV.
    """
    fold: int
    train_year_min: int
    train_year_max: int
    eval_year_min: int
    eval_year_max: int
    n_train: int
    n_eval: int
    best_params: Dict
    metrics: Dict[str, float]

    def as_record(self) -> Dict[str, object]:
        """
        Flatten into a row suitable for a tidy DataFrame.
        """
        record: Dict[str, object] = {
            "fold": self.fold,
            "train_year_min": self.train_year_min,
            "train_year_max": self.train_year_max,
            "eval_year_min": self.eval_year_min,
            "eval_year_max": self.eval_year_max,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
        }
        record.update(self.metrics)
        return record


########################################################################################################################
#
# FOLD GENERATION
#
########################################################################################################################
def expanding_year_folds(years: pd.Series, n_folds: int) -> CVSplits:
    """
    Build expanding-window temporal CV folds from a Year-like Series.

    The unique years are sorted ascending and chunked into ``n_folds + 1``
    contiguous blocks. Fold ``k`` (zero-indexed, ``k`` in ``0..n_folds - 1``)
    has:

    - train indices  = rows whose Year falls in blocks ``0..k``,
    - eval indices   = rows whose Year falls in block ``k + 1``.

    The resulting indices are positional (compatible with ``DataFrame.iloc``).

    Raises
    ------
    ValueError
        If ``n_folds`` is non-positive, or if there are not enough distinct
        years to build the requested number of folds (need at least
        ``n_folds + 1`` unique years).
    """
    if n_folds <= 0:
        raise ValueError(f"n_folds must be positive, got {n_folds}.")

    years_array = np.asarray(years)
    unique_years = np.unique(years_array)
    if unique_years.size < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} distinct years to build {n_folds} expanding folds, "
            f"got {unique_years.size}."
        )

    year_blocks = np.array_split(unique_years, n_folds + 1)
    splits: CVSplits = []
    for fold_index in range(n_folds):
        train_years = np.concatenate(year_blocks[: fold_index + 1])
        eval_years = year_blocks[fold_index + 1]
        train_mask = np.isin(years_array, train_years)
        eval_mask = np.isin(years_array, eval_years)
        train_idx = np.flatnonzero(train_mask)
        eval_idx = np.flatnonzero(eval_mask)
        if train_idx.size == 0 or eval_idx.size == 0:
            raise ValueError(
                f"Fold {fold_index} produced empty train or eval indices; "
                "check that Year values are evenly distributed."
            )
        splits.append((train_idx, eval_idx))
    return splits


########################################################################################################################
#
# ORCHESTRATOR
#
########################################################################################################################
@dataclass
class NestedTemporalCV:
    """
    Orchestrate nested temporal cross-validation for a single regressor.
    """
    n_trials: int = field(default_factory=lambda: CONFIG.n_trials)
    outer_folds: int = field(default_factory=lambda: CONFIG.outer_folds)
    inner_folds: int = field(default_factory=lambda: CONFIG.inner_folds)
    year_column: str = field(default_factory=lambda: CONFIG.year_column)

    def run(
        self,
        model_name: str,
        X_raw: pd.DataFrame,
        y_raw: pd.Series,
    ) -> List[OuterFoldResult]:
        """
        Run nested temporal CV for ``model_name``.

        Returns one :class:`OuterFoldResult` per outer fold. Metrics are
        reported in the original (un-logged) target scale to stay consistent
        with the canonical evaluation in :mod:`src.evaluation.metrics`.
        """
        if self.year_column not in X_raw.columns:
            raise ValueError(
                f"X_raw must contain the '{self.year_column}' column to build temporal folds."
            )

        outer_splits = expanding_year_folds(X_raw[self.year_column], self.outer_folds)
        results: List[OuterFoldResult] = []

        for fold_index, (train_idx, eval_idx) in enumerate(outer_splits):
            X_outer_train_raw = X_raw.iloc[train_idx]
            X_outer_eval_raw = X_raw.iloc[eval_idx]
            y_outer_train_raw = y_raw.iloc[train_idx]
            y_outer_eval_raw = y_raw.iloc[eval_idx]

            train_years = X_outer_train_raw[self.year_column]
            eval_years = X_outer_eval_raw[self.year_column]

            logger.info(
                "[%s] outer fold %d/%d: train years %d-%d (n=%d), eval years %d-%d (n=%d)",
                model_name,
                fold_index + 1,
                self.outer_folds,
                int(train_years.min()),
                int(train_years.max()),
                int(X_outer_train_raw.shape[0]),
                int(eval_years.min()),
                int(eval_years.max()),
                int(X_outer_eval_raw.shape[0]),
            )

            inner_splits = expanding_year_folds(train_years, self.inner_folds)
            optimizer = OptimizeRegressor(
                model_name=model_name,
                n_trials=self.n_trials,
                X_train=X_outer_train_raw,
                y_train=y_outer_train_raw,
                cv_splits=inner_splits,
                year_column=self.year_column,
                study_name=f"{model_name}-outer{fold_index + 1}",
            )
            best_params = optimizer.optimize(persist=False)

            fold_data = preprocess_fold(
                X_outer_train_raw,
                X_outer_eval_raw,
                y_outer_train_raw,
                y_outer_eval_raw,
            )
            model = instantiate_model(model_name, best_params)
            fit_model(
                model,
                fold_data.X_train,
                fold_data.y_train,
                context=f"[{model_name}] outer-fold-{fold_index + 1}",
                log_backend=True,
            )

            metrics = _score_in_original_scale(model, fold_data.X_val, fold_data.y_val)
            logger.info(
                "[%s] outer fold %d metrics: %s",
                model_name,
                fold_index + 1,
                _format_metrics(metrics),
            )

            results.append(
                OuterFoldResult(
                    fold=fold_index + 1,
                    train_year_min=int(train_years.min()),
                    train_year_max=int(train_years.max()),
                    eval_year_min=int(eval_years.min()),
                    eval_year_max=int(eval_years.max()),
                    n_train=int(X_outer_train_raw.shape[0]),
                    n_eval=int(X_outer_eval_raw.shape[0]),
                    best_params=best_params,
                    metrics=metrics,
                )
            )

        return results


########################################################################################################################
#
# HELPERS
#
########################################################################################################################
def select_best_fold_params(per_fold_records: List[OuterFoldResult]) -> Dict:
    """
    Return the hyperparameters of the best-performing outer fold (lowest RMSE).

    These are reused to fit the final production model on the full dataset,
    instead of running a separate full-data tuning pass.
    """
    if not per_fold_records:
        raise ValueError("Cannot select params from empty nested-CV results.")
    best = min(per_fold_records, key=lambda record: record.metrics["rmse"])
    return dict(best.best_params)


def _score_in_original_scale(
    model,
    X_val: pd.DataFrame,
    y_val_log: pd.Series,
) -> Dict[str, float]:
    """
    Predict with ``model`` and report MAPE / RMSE / MAE / R^2 on the original
    (un-logged) target scale.
    """
    from sklearn.metrics import (
        mean_absolute_error,
        mean_absolute_percentage_error,
        mean_squared_error,
        r2_score,
    )

    y_pred_log = predict(model, X_val)
    y_true = inverse_log_transform_target(y_val_log)
    y_pred = inverse_log_transform_target(y_pred_log)
    return {
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
