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
from typing import List, Tuple

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
def encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit a CategoricalEncoder on X_train, transform all three splits.
    """
    encoder = CategoricalEncoder().fit(X_train)
    return encoder.transform(X_train), encoder.transform(X_val), encoder.transform(X_test)


def _to_snake_case(name: str) -> str:
    """
    Convert an arbitrary column name to a snake_case, special-char-free token.

    Handles CamelCase boundaries (``FloorAreaRatio`` -> ``floor_area_ratio``) and
    one-hot suffixes containing spaces or punctuation
    (``Classification_City Road`` -> ``classification_city_road``).
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


def to_snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the encoded feature columns to snake_case to satisfy LightGBM/XGBoost.

    The number of one-hot columns depends on the categorical levels present in
    each study's training split, so names are derived dynamically rather than
    from a fixed list. Duplicate results are disambiguated with a numeric suffix.
    """
    df = df.copy()
    new_columns: List[str] = []
    seen: dict[str, int] = {}
    for column in df.columns:
        snake = _to_snake_case(str(column))
        if snake in seen:
            seen[snake] += 1
            snake = f"{snake}_{seen[snake]}"
        else:
            seen[snake] = 0
        new_columns.append(snake)
    df.columns = new_columns
    return df
