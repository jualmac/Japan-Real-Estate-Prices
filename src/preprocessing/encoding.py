"""
Categorical encoding (one-hot) with consistent column layout across splits.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


########################################################################################################################
#
# CLASS
#
########################################################################################################################
@dataclass
class CategoricalEncoder:
    """
    One-hot encode the categorical features, learn the output column layout on train,
    and replay the exact same columns on validation/test (filling missing levels with 0).
    """
    categorical_columns: List[str] = field(default_factory=lambda: list(CATEGORICAL_FEATURES))
    numerical_columns: List[str] = field(default_factory=lambda: list(NUMERICAL_FEATURES))
    final_columns: List[str] = field(default_factory=list)
    fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "CategoricalEncoder":
        encoded = self._encode(X_train)
        self.final_columns = list(encoded.columns)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("CategoricalEncoder.transform called before fit.")
        encoded = self._encode(X)
        encoded = encoded.reindex(columns=self.final_columns, fill_value=0)
        return encoded

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_columns = [c for c in self.numerical_columns if c in X.columns]
        df = X[keep_columns].copy()
        for column in self.categorical_columns:
            if column not in X.columns:
                continue
            one_hot = pd.get_dummies(X[column], drop_first=True, prefix=column, dtype=int)
            df = df.join(one_hot)
        return df


########################################################################################################################
#
# CONVENIENCE
#
########################################################################################################################
def _to_snake_case(name: str) -> str:
    """
    Convert an arbitrary column name (camelCase, one-hot, with punctuation) to
    snake_case while preserving word boundaries.
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    s = s.lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def to_snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the encoded feature columns to snake_case to satisfy LightGBM/XGBoost.

    Names are derived from the actual (encoder-produced) column labels, so the
    layout adapts to whatever one-hot columns the fitted encoder emitted instead
    of relying on a fixed positional constant.
    """
    df = df.copy()
    df.columns = [_to_snake_case(column) for column in df.columns]
    return df
