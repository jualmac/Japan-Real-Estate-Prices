"""
Numeric scaling. Keeps a separate MinMaxScaler per column so each feature can
be inverted independently and so the same fit can be replayed on val/test.

Tree-based regressors (Random Forest, XGBoost, LightGBM) do not require scaling,
but we keep this step because it is useful when comparing them with neural networks
or distance-based models in the spot-check stage.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
DEFAULT_SCALED_COLUMNS: List[str] = [
    "TimeToNearestStation",
    "Area",
    "BuildingYear",
    "CoverageRatio",
    "FloorAreaRatio",
    "Year",
    "Latitude",
    "Longitude",
]

########################################################################################################################
#
# CLASS
#
########################################################################################################################
@dataclass
class NumericScaler:
    """
    Fit one MinMaxScaler per column on the training split; replay on val/test.
    """
    columns: List[str] = field(default_factory=lambda: list(DEFAULT_SCALED_COLUMNS))
    scalers: Dict[str, MinMaxScaler] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "NumericScaler":
        for column in self.columns:
            if column not in X_train.columns:
                continue
            scaler = MinMaxScaler()
            scaler.fit(X_train[[column]])
            self.scalers[column] = scaler
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("NumericScaler.transform called before fit.")
        df = X.copy()
        for column, scaler in self.scalers.items():
            if column in df.columns:
                df[column] = scaler.transform(df[[column]])
        return df

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)
