"""
Bayesian hyperparameter optimization (Optuna) for the candidate regressors.

# TODO: Add MLflow as a way to keep the models parameters instead of .json files;
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
import json
from pathlib import Path
from typing import Union

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from src.config import CONFIG, PARAMETERS_DIR
from src.models.gpu_data import (
    cudf_available,
    cuml_available,
    is_cuml_random_forest,
    make_cuml_random_forest,
    to_cudf_dataframe,
    to_cudf_series,
    to_host_array,
    xgboost_uses_cuda,
)
from src.utils.io import logger


########################################################################################################################
#
# CLASS
#
########################################################################################################################
class OptimizeRegressor:
    """
    Perform Bayesian Optimization on supported regression models.

    Attributes
    ----------
    model_name : str
        Name of the regression model to optimize ('rf', 'xgb', 'lgbm', 'enet', 'svr').
    n_trials : int
        Number of trials for the optimization process.
    X_train : DataFrame
        Training features.
    y_train : Series
        Training target.
    logger : Logger
        Shared project logger.
    file_name : Path
        Destination JSON file for the best hyperparameters.
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int,
        X_train: DataFrame,
        y_train: Series,
        parameters_dir: Path = PARAMETERS_DIR,
    ):
        self.model_name = model_name
        self.n_trials = 1 if model_name == "rf" else n_trials # sklearn RandomForest CV is very slow; always run a single trial.
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        parameters_dir = Path(parameters_dir)
        parameters_dir.mkdir(parents=True, exist_ok=True)
        self.file_name = parameters_dir / f"best_params_{model_name}.json"
        self._logged_xgb_cudf = False
        self._logged_xgb_cudf_unavailable = False
        self._logged_lgbm_device = False
        self._logged_rf_backend = False

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization (returns CV-averaged RMSE).
        """
        if self.model_name == "rf":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20, step=1)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, step=1)
            max_samples = trial.suggest_float("max_samples", 0.5, 1.0, step=0.05)

            rf_params = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_samples=max_samples,
                random_state=CONFIG.seed,
            )

            if CONFIG.use_gpu and cuml_available():
                return self.evaluate(make_cuml_random_forest(**rf_params))
            return self.evaluate(RandomForestRegressor(n_jobs=-1, **rf_params))

        if self.model_name == "xgb":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.005)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10, step=1)
            subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.05)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05)

            return self.evaluate(
                XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    tree_method="hist",
                    device="cuda" if CONFIG.use_gpu else "cpu",
                    n_jobs=-1,
                    random_state=CONFIG.seed,
                )
            )

        if self.model_name == "lgbm":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.005)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            num_leaves = trial.suggest_int("num_leaves", 32, 128, step=4)

            return self.evaluate(
                LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    device_type="gpu" if CONFIG.use_gpu else "cpu",
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=CONFIG.seed,
                )
            )

        if self.model_name == "enet":
            alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            max_iter = trial.suggest_int("max_iter", 1000, 10000, step=1000)
            tol = trial.suggest_float("tol", 1e-5, 1e-2, log=True)
            selection = trial.suggest_categorical("selection", ["cyclic", "random"])

            return self.evaluate(
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    tol=tol,
                    selection=selection,
                    random_state=CONFIG.seed,
                )
            )

        if self.model_name == "svr":
            C = trial.suggest_float("C", 1e-3, 100.0, log=True)
            epsilon = trial.suggest_float("epsilon", 1e-3, 1.0, log=True)
            loss = trial.suggest_categorical(
                "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]
            )

            return self.evaluate(
                LinearSVR(
                    C=C,
                    epsilon=epsilon,
                    loss=loss,
                    max_iter=100_000,
                    tol=1e-3,
                    random_state=CONFIG.seed,
                )
            )

        self.logger.error("Please provide a supported model: rf, xgb, lgbm, enet or svr")
        raise TypeError(f"Unsupported model name: {self.model_name}")

    def evaluate(
        self,
        model: Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, ElasticNet, LinearSVR],
    ) -> float:
        """
        Evaluate the model using K-Fold cross-validation; returns mean RMSE.
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG.seed)
        rmse_scores = []

        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
            y_tr, y_val = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            if isinstance(model, XGBRegressor) and xgboost_uses_cuda(model):
                if cudf_available():
                    model.fit(to_cudf_dataframe(X_tr), to_cudf_series(y_tr))
                    if not self._logged_xgb_cudf:
                        self.logger.info("XGBoost optimization fitted with cuDF GPU data.")
                        self._logged_xgb_cudf = True
                    y_pred = to_host_array(model.predict(to_cudf_dataframe(X_val)))
                else:
                    if not self._logged_xgb_cudf_unavailable:
                        self.logger.warning(
                            "cuDF is unavailable; XGBoost optimization is using pandas CPU data."
                        )
                        self._logged_xgb_cudf_unavailable = True
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
            elif is_cuml_random_forest(model):
                if cudf_available():
                    model.fit(to_cudf_dataframe(X_tr), to_cudf_series(y_tr))
                    y_pred = to_host_array(model.predict(to_cudf_dataframe(X_val)))
                else:
                    model.fit(X_tr.astype("float32"), y_tr.astype("float32"))
                    y_pred = to_host_array(model.predict(X_val.astype("float32")))
                if not self._logged_rf_backend:
                    self.logger.info("RandomForest optimization fitted with cuML GPU backend.")
                    self._logged_rf_backend = True
            else:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                if (
                    self.model_name == "rf"
                    and isinstance(model, RandomForestRegressor)
                    and not self._logged_rf_backend
                ):
                    self.logger.info("RandomForest optimization fitted with sklearn CPU backend.")
                    self._logged_rf_backend = True

            if isinstance(model, LGBMRegressor) and not self._logged_lgbm_device:
                self.logger.info(
                    "LightGBM optimization fitted with device_type=%s",
                    model.booster_.params.get("device_type", "cpu"),
                )
                self._logged_lgbm_device = True

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)

        return float(np.mean(rmse_scores))

    def optimize(self) -> dict:
        """
        Run optuna study; persist best params to JSON and return them.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=self.n_trials,
            timeout=900,
        )

        with open(self.file_name, "w") as f:
            json.dump(study.best_params, f, indent=4)

        self.logger.info(f"Best parameters saved to {self.file_name}")
        return study.best_params
