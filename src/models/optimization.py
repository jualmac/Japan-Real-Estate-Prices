"""
Bayesian hyperparameter optimization (Optuna) for the candidate regressors.

The optimizer receives RAW (pre-preprocessing) X / y and a list of CV splits.
Every Optuna trial refits the full preprocessing pipeline inside each split,
guaranteeing zero leakage between the inner-train portion (used to fit
preprocessing + the candidate model) and the inner-val portion (used to score
the trial). This is the building block of the nested temporal CV in
``src/models/temporal_cv.py``.

# TODO: Add MLflow as a way to keep the models parameters instead of .json files;
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from src.config import CONFIG, PARAMETERS_DIR
from src.models.fit_predict import fit_model, predict
from src.models.gpu_data import cuml_available, make_cuml_random_forest
from src.models.registry import save_best_params
from src.preprocessing.pipeline import preprocess_fold
from src.utils.io import logger

ModelType = Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, ElasticNet, LinearSVR]
CVSplits = Sequence[Tuple[np.ndarray, np.ndarray]]


########################################################################################################################
#
# CLASS
#
########################################################################################################################
class OptimizeRegressor:
    """
    Perform Bayesian Optimization on supported regression models with
    leakage-free CV.

    Parameters
    ----------
    model_name : str
        Name of the regression model to optimize ('rf', 'xgb', 'lgbm', 'enet',
        'svr').
    n_trials : int
        Number of Optuna trials.
    X_train : pd.DataFrame
        RAW training features (preprocessing is applied inside each CV fold).
    y_train : pd.Series
        RAW target series (log transformation is applied inside each CV fold).
    cv_splits : sequence of (train_idx, val_idx) numpy arrays, optional
        Fold indices into ``X_train`` / ``y_train``. When omitted, falls back
        to an expanding-window temporal split built from ``X_train[year_column]``
        with ``CONFIG.inner_folds`` folds.
    year_column : str, optional
        Column name used to build default temporal folds. Defaults to
        ``CONFIG.year_column``.
    study_name : str, optional
        Human-readable label included in log messages. Defaults to
        ``model_name``.
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splits: Optional[CVSplits] = None,
        year_column: Optional[str] = None,
        study_name: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.n_trials = n_trials
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)
        self.file_name = PARAMETERS_DIR / f"best_params_{model_name}.json"
        self.year_column = year_column or CONFIG.year_column
        self.study_name = study_name or model_name
        self.cv_splits: CVSplits = list(cv_splits) if cv_splits is not None else self._default_cv_splits()
        self._logged_backends: bool = False

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization (returns CV-averaged RMSE).
        """
        params = self._suggest_params(trial)
        return self.evaluate(lambda: self._instantiate(params))

    def evaluate(self, model_factory: Callable[[], ModelType]) -> float:
        """
        Score one candidate configuration via the configured CV splits.

        ``model_factory`` is a zero-argument callable that returns a freshly
        constructed estimator. A new estimator is built for every fold so that
        fits from previous folds don't contaminate later ones.
        """
        rmse_scores: List[float] = []
        for fold_index, (train_idx, val_idx) in enumerate(self.cv_splits):
            X_train_raw = self.X_train.iloc[train_idx]
            X_val_raw = self.X_train.iloc[val_idx]
            y_train_raw = self.y_train.iloc[train_idx]
            y_val_raw = self.y_train.iloc[val_idx]

            fold_data = preprocess_fold(X_train_raw, X_val_raw, y_train_raw, y_val_raw)
            model = model_factory()
            log_backend = not self._logged_backends and fold_index == 0
            fit_model(
                model,
                fold_data.X_train,
                fold_data.y_train,
                context=f"[{self.study_name}] inner-fold-{fold_index}",
                log_backend=log_backend,
            )
            if log_backend:
                self._logged_backends = True

            y_pred = predict(model, fold_data.X_val)
            rmse = float(np.sqrt(mean_squared_error(fold_data.y_val, y_pred)))
            rmse_scores.append(rmse)

        return float(np.mean(rmse_scores))

    def optimize(self, persist: bool = True) -> dict:
        """
        Run an Optuna study; optionally persist the best parameters to JSON.

        Parameters
        ----------
        persist : bool, default True
            When True, write the resulting best parameters to the canonical
            ``parameters/best_params_<model>.json`` file. Set to False when the
            study is part of a nested-CV inner tuning pass so that the
            canonical file is only overwritten by the final full-data tuning.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials)

        if persist:
            save_best_params(self.model_name, study.best_params)
            self.logger.info("Best parameters saved to %s", self.file_name)
        return dict(study.best_params)

    def _default_cv_splits(self) -> CVSplits:
        """
        Build default expanding-window temporal CV folds from the year column.
        Imported lazily to avoid a circular import with temporal_cv.
        """
        from src.models.temporal_cv import expanding_year_folds

        if self.year_column not in self.X_train.columns:
            raise ValueError(
                f"OptimizeRegressor requires the '{self.year_column}' column in X_train "
                "to build default temporal CV splits, or an explicit cv_splits argument."
            )
        return expanding_year_folds(self.X_train[self.year_column], CONFIG.inner_folds)

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Suggest hyperparameters for ``self.model_name`` from the trial.
        """
        if self.model_name == "rf":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, step=1),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10, step=1),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0, step=0.05),
            }
        if self.model_name == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.005),
                "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10, step=1),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),
            }
        if self.model_name == "lgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, step=0.005),
                "max_depth": trial.suggest_int("max_depth", 3, 12, step=1),
                "num_leaves": trial.suggest_int("num_leaves", 32, 128, step=4),
            }
        if self.model_name == "enet":
            return {
                "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000, step=1000),
                "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
                "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
            }
        if self.model_name == "svr":
            return {
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                "loss": trial.suggest_categorical(
                    "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]
                ),
                "max_iter": trial.suggest_int("max_iter", 2000, 20000, step=2000),
                "tol": trial.suggest_float("tol", 1e-5, 1e-2, log=True),
            }
        self.logger.error("Please provide a supported model: rf, xgb, lgbm, enet or svr")
        raise TypeError(f"Unsupported model name: {self.model_name}")

    def _instantiate(self, params: dict) -> ModelType:
        """
        Build a fresh estimator for the current trial using ``params`` plus the
        default backend wiring (GPU vs CPU, seed, n_jobs).
        """
        params = dict(params)
        if self.model_name == "rf":
            rf_params = dict(params, random_state=CONFIG.seed)
            if CONFIG.use_gpu and cuml_available():
                return make_cuml_random_forest(**rf_params)
            return RandomForestRegressor(n_jobs=-1, **rf_params)
        if self.model_name == "xgb":
            return XGBRegressor(
                tree_method="hist",
                device="cuda" if CONFIG.use_gpu else "cpu",
                n_jobs=-1,
                random_state=CONFIG.seed,
                **params,
            )
        if self.model_name == "lgbm":
            return LGBMRegressor(
                device_type="gpu" if CONFIG.use_gpu else "cpu",
                verbosity=-1,
                n_jobs=-1,
                random_state=CONFIG.seed,
                **params,
            )
        if self.model_name == "enet":
            return ElasticNet(random_state=CONFIG.seed, **params)
        if self.model_name == "svr":
            return LinearSVR(random_state=CONFIG.seed, **params)
        raise TypeError(f"Unsupported model name: {self.model_name}")