"""
Leakage-free preprocessing for a single (train, val) fold.

This module exposes :func:`preprocess_fold`, the canonical entrypoint used by
the nested temporal cross-validation and by the final full-data fit. Every
stateful component (DataCleaner, post-merge imputer, CategoricalEncoder,
NumericScaler) is fit on the fold's training portion only and replayed on the
validation portion, guaranteeing that information from the held-out window
never bleeds into training statistics.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
from src.preprocessing.cleaning import DataCleaner
from src.preprocessing.encoding import CategoricalEncoder, to_snake_case_columns
from src.preprocessing.scaling import NumericScaler
from src.preprocessing.feature_engineering import (
    feature_engineer_frame,
    log_transform_target,
)

########################################################################################################################
#
# DATACLASSES
#
########################################################################################################################
@dataclass
class FoldPreprocessors:
    """
    Container holding the stateful preprocessing objects fit on a fold's
    training portion. Exposed so callers can apply the same transformation to
    additional frames if needed.
    """
    cleaner: DataCleaner
    encoder: CategoricalEncoder
    scaler: NumericScaler


@dataclass
class FoldPreprocessedData:
    """
    Result of :func:`preprocess_fold`. Carries the transformed train/val arrays
    and the fitted preprocessors so they can be reused (e.g. for refitting on
    full data and then predicting on a fresh frame).
    """
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    preprocessors: FoldPreprocessors


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def preprocess_fold(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    y_train_raw: pd.Series,
    y_val_raw: pd.Series,
) -> FoldPreprocessedData:
    """
    Apply the full preprocessing pipeline to a single fold.

    All stateful steps are fit on ``X_train_raw`` / ``y_train_raw`` only and
    then replayed on the validation portion.

    Steps
    -----
    1. Rule-based cleaning + DistrictName imputers (fit on train).
    2. Feature engineering: TimeToNearestStation rebuild, cyclical Quarter,
       location merge, log1p target.
    3. Post-merge imputers (Latitude / Longitude by MunicipalityCode, fit on
       train post-engineering).
    4. One-hot encoding (column layout learned on train).
    5. MinMax scaling (bounds learned on train).
    6. snake_case rename to satisfy LightGBM / XGBoost column constraints.

    Returns
    -------
    FoldPreprocessedData
        Transformed train/val frames + the fitted preprocessors.
    """
    cleaner = DataCleaner().fit(X_train_raw)
    X_train = cleaner.transform(X_train_raw)
    X_val = cleaner.transform(X_val_raw)

    X_train = feature_engineer_frame(X_train)
    X_val = feature_engineer_frame(X_val)
    y_train = log_transform_target(y_train_raw)
    y_val = log_transform_target(y_val_raw)

    cleaner.fit_post_merge(X_train)
    X_train = cleaner.transform_post_merge(X_train)
    X_val = cleaner.transform_post_merge(X_val)

    encoder = CategoricalEncoder().fit(X_train)
    X_train = encoder.transform(X_train)
    X_val = encoder.transform(X_val)

    scaler = NumericScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    X_train = to_snake_case_columns(X_train)
    X_val = to_snake_case_columns(X_val)

    return FoldPreprocessedData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        preprocessors=FoldPreprocessors(cleaner=cleaner, encoder=encoder, scaler=scaler),
    )


def preprocess_full(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    *,
    eval_X_raw: Optional[pd.DataFrame] = None,
    eval_y_raw: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series], FoldPreprocessors]:
    """
    Fit the preprocessing pipeline on the full training data, returning the
    transformed train arrays plus the fitted preprocessors. Optionally
    transforms an evaluation frame using the same fitted preprocessors.

    Used by the final training stage of the pipeline: nested CV gives the
    unbiased performance estimate, and after it we refit a single production
    model on all data using these preprocessors.
    """
    cleaner = DataCleaner().fit(X_raw)
    X_train = cleaner.transform(X_raw)
    X_train = feature_engineer_frame(X_train)
    y_train = log_transform_target(y_raw)

    cleaner.fit_post_merge(X_train)
    X_train = cleaner.transform_post_merge(X_train)

    encoder = CategoricalEncoder().fit(X_train)
    X_train = encoder.transform(X_train)

    scaler = NumericScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = to_snake_case_columns(X_train)

    preprocessors = FoldPreprocessors(cleaner=cleaner, encoder=encoder, scaler=scaler)

    if eval_X_raw is None or eval_y_raw is None:
        return X_train, y_train, None, None, preprocessors

    X_eval = cleaner.transform(eval_X_raw)
    X_eval = feature_engineer_frame(X_eval)
    X_eval = cleaner.transform_post_merge(X_eval)
    X_eval = encoder.transform(X_eval)
    X_eval = scaler.transform(X_eval)
    X_eval = to_snake_case_columns(X_eval)
    y_eval = log_transform_target(eval_y_raw)
    return X_train, y_train, X_eval, y_eval, preprocessors