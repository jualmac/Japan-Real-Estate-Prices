#TODO: Add Mlflow as a way to keep the models parameters instead of .json files;

import json
from typing import Union
import optuna
import numpy as np
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from utils import logger


class OptimizeRegressor:
    """
    Performs Bayesian Optimization on specified regression models.

    Attributes
    ----------
    model_name : str
        Name of the regression model to optimize ('rf', 'xgb', 'lgbm').
    n_trials : int
        Number of trials for the optimization process.
    X_train : DataFrame
        Training data features.
    y_train : Series
        Training data target variable.
    logger : Logger
        Logger for logging messages.
    file_name : str
        Name of the file to save the best hyperparameters.
    """

    def __init__(
        self, model_name: str, n_trials: int, X_train: DataFrame, y_train: Series
    ):
        """
        Initializes the OptimizeRegressor class with the specified model and parameters.

        Parameters
        ----------
        model_name : str
            Name of the regression model to optimize ('rf', 'xgb', 'lgbm').
        n_trials : int
            Number of trials for the optimization process.
        X_train : DataFrame
            Training data features.
        y_train : Series
            Training data target variable.
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.X_train = X_train
        self.y_train = y_train
        self.logger = logger
        self.file_name = f"/parameters/best_params_{model_name}.json"

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Bayesian Optimization to tune hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            A single trial of an optimization experiment.

        Returns
        -------
        float
            The RMSLE score for the given trial.
        """
        if self.model_name == "rf":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20, step=1)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, step=1)
            max_samples = trial.suggest_float("max_samples", 0.5, 1.0, step=0.05)

            rmsle = self.evaluate(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_samples=max_samples,
                    verbose=-1,
                    n_jobs=-1,
                    random_state=42,
                )
            )

            return rmsle

        elif self.model_name == "xgb":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.005)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10, step=1)
            subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.05)
            colsample_bytree = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0, step=0.05
            )

            rmsle = self.evaluate(
                XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    n_jobs=-1,
                    random_state=42,
                )
            )

            return rmsle

        elif self.model_name == "lgbm":
            n_estimators = trial.suggest_int("n_estimators", 500, 3000, step=100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.005)
            max_depth = trial.suggest_int("max_depth", 3, 12, step=1)
            num_leaves = trial.suggest_int("num_leaves", 32, 128, step=4)

            rmsle = self.evaluate(
                LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    num_leaves=num_leaves,
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=42,
                )
            )

            return rmsle

        else:
            self.logger.error("Please provide a supported model: rf, xgb or lgbm")
            raise TypeError()

    def evaluate(
        self, model: Union[RandomForestRegressor, XGBRegressor, LGBMRegressor]
    ) -> float:
        """
        Evaluates the model using K-Fold cross-validation and returns the average RMSLE.

        Parameters
        ----------
        model : Union[RandomForestRegressor, XGBRegressor, LGBMRegressor]
            The regression model to evaluate.

        Returns
        -------
        float
            The average RMSLE score across the K-Fold splits.
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmsle_scores = []

        for train_index, val_index in kf.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_index], self.X_train.iloc[val_index]
            y_tr, y_val = self.y_train.iloc[train_index], self.y_train.iloc[val_index]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
            rmsle_scores.append(rmsle)

        rmsle = np.mean(rmsle_scores)
        return rmsle

    def optimize(self):
        """
        Conducts Bayesian optimization to find the best hyperparameters for the specified model.
        Saves the best hyperparameters to a JSON file.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self.objective(trial), n_trials=self.n_trials)

        with open(self.file_name, "w") as f:
            json.dump(study.best_params, f, indent=4)

        self.logger.info(f"Best parameters saved to {self.file_name}")


