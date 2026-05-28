"""
GPU-aware fit and predict helpers shared by optimization, training, and evaluation.

Centralizes the cuDF / cuML / sklearn branching; Callers do not need to know whether the underlying estimator runs on
CPU or GPU; they just pass pandas DataFrames / Series and get host-side numpy arrays back.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from src.models.gpu_data import (
    cudf_available,
    is_cuml_random_forest,
    to_cudf_dataframe,
    to_cudf_series,
    to_host_array,
    xgboost_uses_cuda,
)
from src.utils.io import logger

ModelType = Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, ElasticNet, LinearSVR]


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def fit_model(
    model: ModelType,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    context: str = "model",
    log_backend: bool = False,
) -> ModelType:
    """
    Fit ``model`` on ``(X, y)``, moving data to GPU memory when the estimator
    requires it. Falls back to pandas when cuDF is unavailable.
    """
    if isinstance(model, XGBRegressor) and xgboost_uses_cuda(model):
        if cudf_available():
            model.fit(to_cudf_dataframe(X), to_cudf_series(y))
            if log_backend:
                logger.info("%s fitted XGBoost with cuDF GPU data.", context)
        else:
            if log_backend:
                logger.warning(
                    "%s requested XGBoost CUDA but cuDF is unavailable; falling back to pandas CPU data.",
                    context,
                )
            model.fit(X, y)
    elif is_cuml_random_forest(model):
        if cudf_available():
            model.fit(to_cudf_dataframe(X), to_cudf_series(y))
            if log_backend:
                logger.info("%s fitted RandomForest with cuML GPU backend.", context)
        else:
            model.fit(X.astype("float32"), y.astype("float32"))
            if log_backend:
                logger.info("%s fitted RandomForest with cuML (host arrays).", context)
    else:
        model.fit(X, y)
        if log_backend and isinstance(model, RandomForestRegressor):
            logger.info("%s fitted RandomForest with sklearn CPU backend.", context)

    if log_backend and isinstance(model, LGBMRegressor):
        logger.info(
            "%s fitted LightGBM with device_type=%s",
            context,
            model.booster_.params.get("device_type", "cpu"),
        )
    return model


def predict(model: ModelType, X: pd.DataFrame) -> np.ndarray:
    """
    Predict with ``model`` and return a host-side numpy array regardless of the
    underlying backend.
    """
    if isinstance(model, XGBRegressor) and xgboost_uses_cuda(model):
        if cudf_available():
            return to_host_array(model.predict(to_cudf_dataframe(X)))
        return np.asarray(model.predict(X))
    if is_cuml_random_forest(model):
        if cudf_available():
            return to_host_array(model.predict(to_cudf_dataframe(X)))
        return to_host_array(model.predict(X.astype("float32")))
    return np.asarray(model.predict(X))