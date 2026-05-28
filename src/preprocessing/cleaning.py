"""
DataCleaner: column dropping, dtype coercion, and rule-based NaN imputation.

The cleaner is fit on the training split only and replays the learned
group-wise statistics on the validation/test splits. This prevents the
data leakage that existed in the original aaa.py monolith (where imputers were
fitted using X_test medians).

Imputation rules (informed by the EDA in src/eda/missing.py):
- Land-only types receive 'Not Applicable' for FloorPlan / Renovation / Structure.
- Condominium and rural types receive zero for Frontage / Breadth.
- Condominium and rural types receive 'Not Applicable' for Classification.
- Rural types receive 'Not Applicable' for NearestStation.
- When MinTimeToNearestStation == 120 and MaxTimeToNearestStation is NaN,
  MaxTimeToNearestStation is filled with 120.
- Numerical features are imputed by DistrictName group statistics learned on
  train, falling back to column-wide medians.
- Latitude / Longitude are populated by ``add_location_features`` and may end
  up NaN whenever the DistrictName merge misses; ``fit_post_merge`` /
  ``transform_post_merge`` learn MunicipalityCode group medians on train and
  replay them on val/test after feature engineering.
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
from src.config import COLUMNS_TO_DROP, CONDO_TYPES, LAND_TYPES, RURAL_TYPES


########################################################################################################################
#
# CONSTANTS
#
########################################################################################################################
NUMERIC_DISTRICT_FILLERS: Dict[str, str] = {
    "BuildingYear": "mean",
    "MinTimeToNearestStation": "median",
    "MaxTimeToNearestStation": "median",
    "CoverageRatio": "median",
    "FloorAreaRatio": "median",
}

NUMERIC_MUNICIPALITY_FILLERS: Dict[str, str] = {
    "Latitude": "median",
    "Longitude": "median",
}

########################################################################################################################
#
# CLASS
#
########################################################################################################################
@dataclass
class DataCleaner:
    """
    Stateful cleaner that learns imputation values on train and applies them everywhere.
    """
    columns_to_drop: List[str] = field(default_factory=lambda: list(COLUMNS_TO_DROP))
    district_group_medians: Dict[str, pd.Series] = field(default_factory=dict)
    municipality_group_medians: Dict[str, pd.Series] = field(default_factory=dict)
    column_medians: Dict[str, float] = field(default_factory=dict)
    fitted: bool = False
    post_merge_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "DataCleaner":
        """
        Learn group-wise statistics from the training set only.
        """
        df = X_train.copy()
        df = self._apply_rule_imputations(df)

        for column, agg in NUMERIC_DISTRICT_FILLERS.items():
            if column in df.columns and "DistrictName" in df.columns:
                self.district_group_medians[column] = df.groupby("DistrictName")[column].agg(agg)

        for column, agg in NUMERIC_MUNICIPALITY_FILLERS.items():
            if column in df.columns and "MunicipalityCode" in df.columns:
                self.municipality_group_medians[column] = df.groupby("MunicipalityCode")[column].agg(agg)

        for column in list(NUMERIC_DISTRICT_FILLERS) + list(NUMERIC_MUNICIPALITY_FILLERS):
            if column in df.columns:
                self.column_medians[column] = float(df[column].median())

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned rules and impute remaining NaNs.
        """
        if not self.fitted:
            raise RuntimeError("DataCleaner.transform called before fit.")

        df = X.copy()
        df = self._drop_columns(df)
        df = self._apply_rule_imputations(df)

        # Fill numeric NaNs via the train-fitted group medians, with column-wide medians as fallback;
        for column, agg in NUMERIC_DISTRICT_FILLERS.items():
            if column not in df.columns:
                continue
            if column in self.district_group_medians:
                df[column] = df[column].fillna(df["DistrictName"].map(self.district_group_medians[column]))
            df[column] = df[column].fillna(self.column_medians.get(column))

        for column, agg in NUMERIC_MUNICIPALITY_FILLERS.items():
            if column not in df.columns:
                continue
            if column in self.municipality_group_medians:
                df[column] = df[column].fillna(df["MunicipalityCode"].map(self.municipality_group_medians[column]))
            df[column] = df[column].fillna(self.column_medians.get(column))
        return df

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    def fit_post_merge(self, X_train: pd.DataFrame) -> "DataCleaner":
        """
        Learn MunicipalityCode group medians for columns that only exist after
        ``add_location_features`` runs (Latitude / Longitude).

        Must be called on the training split AFTER ``apply_feature_engineering``.
        """
        df = X_train
        for column, agg in NUMERIC_MUNICIPALITY_FILLERS.items():
            if column not in df.columns:
                continue
            if "MunicipalityCode" in df.columns:
                self.municipality_group_medians[column] = (
                    df.groupby("MunicipalityCode")[column].agg(agg)
                )
            self.column_medians[column] = float(df[column].median())
        self.post_merge_fitted = True
        return self

    def transform_post_merge(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the post-merge MunicipalityCode imputers to Latitude / Longitude,
        with a column-wide median fallback. Safe to call on any split.
        """
        if not self.post_merge_fitted:
            raise RuntimeError("DataCleaner.transform_post_merge called before fit_post_merge.")

        df = X.copy()
        for column in NUMERIC_MUNICIPALITY_FILLERS:
            if column not in df.columns:
                continue
            if column in self.municipality_group_medians and "MunicipalityCode" in df.columns:
                df[column] = df[column].fillna(
                    df["MunicipalityCode"].map(self.municipality_group_medians[column])
                )
            df[column] = df[column].fillna(self.column_medians.get(column))
        return df

    ####################################################################################################################
    # Private helpers
    ####################################################################################################################
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[c for c in self.columns_to_drop if c in df.columns])

    def _apply_rule_imputations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deterministic, leakage-free rules derived from the EDA notes.
        """
        # Bare-land types: floor plan / renovation / structure are not applicable;
        if "Type" in df.columns:
            land_mask = df["Type"].isin(LAND_TYPES)
            for column in ("FloorPlan", "Renovation", "Structure"):
                if column in df.columns:
                    df.loc[land_mask, column] = df.loc[land_mask, column].fillna("Not Applicable")

            # Condominiums and rural plots: no frontage / breadth measurements;
            zero_mask = df["Type"].isin(CONDO_TYPES + RURAL_TYPES)
            for column in ("Frontage", "Breadth"):
                if column in df.columns:
                    df.loc[zero_mask, column] = df.loc[zero_mask, column].fillna(0.0)

            # Condominiums and rural plots: classification is also not applicable;
            for column in ("Classification",):
                if column in df.columns:
                    df.loc[zero_mask, column] = df.loc[zero_mask, column].fillna("Not Applicable")

            # Rural plots have no nearest station;
            rural_mask = df["Type"].isin(RURAL_TYPES)
            if "NearestStation" in df.columns:
                df.loc[rural_mask, "NearestStation"] = df.loc[rural_mask, "NearestStation"].fillna("Not Applicable")

        # If we know the minimum walking time saturates at 120min, the max is also 120min;
        if {"MinTimeToNearestStation", "MaxTimeToNearestStation"}.issubset(df.columns):
            saturation_mask = df["MinTimeToNearestStation"] == 120
            df.loc[saturation_mask, "MaxTimeToNearestStation"] = df.loc[saturation_mask, "MaxTimeToNearestStation"].fillna(120)
        return df
