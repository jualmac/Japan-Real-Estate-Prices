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
from typing import Union

import numpy as np
import optuna
from lightgbm import LGBMRegressor
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from src.config import CONFIG, PARAMETERS_DIR
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
        Name of the regression model to optimize ('rf', 'xgb', 'lgbm').
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

    def __init__(self, model_name: str, n_trials: int, X_train: DataFrame, y_train: Series):
        self.model_name = model_name
        self.n_trials = n_trials
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)
        self.file_name = PARAMETERS_DIR / f"best_params_{model_name}.json"

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization (returns CV-averaged RMSLE).
        """
        if self.model_name == "rf":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20, step=1)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, step=1)
            max_samples = trial.suggest_float("max_samples", 0.5, 1.0, step=0.05)

            return self.evaluate(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_samples=max_samples,
                    n_jobs=-1,
                    random_state=CONFIG.seed,
                )
            )

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
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=CONFIG.seed,
                )
            )

        self.logger.error("Please provide a supported model: rf, xgb or lgbm")
        raise TypeError(f"Unsupported model name: {self.model_name}")

    def evaluate(self, model: Union[RandomForestRegressor, XGBRegressor, LGBMRegressor]) -> float:
        """
        Evaluate the model using K-Fold cross-validation; returns mean RMSLE.
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=CONFIG.seed)
        rmsle_scores = []

        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
            y_tr, y_val = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
            rmsle_scores.append(rmsle)

        return float(np.mean(rmsle_scores))

    def optimize(self) -> dict:
        """
        Run optuna study; persist best params to JSON and return them.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials)

        with open(self.file_name, "w") as f:
            json.dump(study.best_params, f, indent=4)

        self.logger.info(f"Best parameters saved to {self.file_name}")
        return study.best_params
