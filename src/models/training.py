"""
Train the final regression models using the best hyperparameters found by
OptimizeRegressor (or by nested temporal cross-validation).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Dict, Iterable, Union

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from src.config import CONFIG
from src.models.fit_predict import fit_model
from src.models.gpu_data import cuml_available, make_cuml_random_forest
from src.models.registry import load_best_params

ModelType = Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, ElasticNet, LinearSVR]


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def instantiate_model(model_name: str, params: Dict) -> ModelType:
    """
    Build a fresh regressor for ``model_name`` with sensible backend defaults
    injected on top of the loaded params.

    Public counterpart of the previous ``_instantiate`` helper. Used both by
    final training and by the nested temporal CV orchestrator when fitting the
    outer-fold model from its inner-CV best params.
    """
    params = dict(params)
    params.setdefault("random_state", CONFIG.seed)

    if model_name == "rf":
        if CONFIG.use_gpu and cuml_available():
            return make_cuml_random_forest(**params)
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)
    if model_name == "xgb":
        params.setdefault("n_jobs", -1)
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda" if CONFIG.use_gpu else "cpu")
        return XGBRegressor(**params)
    if model_name == "lgbm":
        params.setdefault("n_jobs", -1)
        params.setdefault("device_type", "gpu" if CONFIG.use_gpu else "cpu")
        params.setdefault("verbosity", -1)
        return LGBMRegressor(**params)
    if model_name == "enet":
        return ElasticNet(**params)
    if model_name == "svr":
        return LinearSVR(**params)

    raise ValueError(f"Unsupported model name: {model_name}")


def train_from_best_params(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ModelType:
    """
    Load the best parameters from parameters/ and fit the matching regressor.
    """
    params = load_best_params(model_name)
    model = instantiate_model(model_name, params)
    return fit_model(
        model,
        X_train,
        y_train,
        context=f"[{model_name}] final-training",
        log_backend=True,
    )


def train_all(
    model_names: Iterable[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, ModelType]:
    """
    Fit every requested model and return a dict keyed by model name.
    """
    return {name: train_from_best_params(name, X_train, y_train) for name in model_names}
