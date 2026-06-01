"""
SHAP-based interpretation of the best model.

These plots support the "Interpretação e análise crítica" section of the
assignment: they show how each attribute drives both the global ranking and
individual predictions.

The explainer is chosen automatically from the model type:
- Tree models (XGB / LGBM / sklearn RF) use the fast, exact TreeExplainer.
- Linear models (ElasticNet / LinearSVR) use LinearExplainer.
- Anything else (e.g. a cuML RandomForest) falls back to the model-agnostic
  permutation-based shap.Explainer.

NOTE: the pipeline trains on a log1p-transformed target (see
src/evaluation/metrics.py), so the SHAP values produced here are expressed in
the log1p price scale, not raw yen.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from src.utils.io import logger

_TREE_MODELS = (XGBRegressor, LGBMRegressor, RandomForestRegressor)
_LINEAR_MODELS = (ElasticNet, LinearSVR)


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def subsample_for_shap(X: pd.DataFrame, n: int = 5000, random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Keep SHAP computations tractable on a 1M+ row dataset by drawing a sample.
    """
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=random_state)


def _build_explainer(model, X_background: pd.DataFrame) -> shap.Explainer:
    """
    Pick the most appropriate SHAP explainer for the given fitted model.
    """
    if isinstance(model, _TREE_MODELS):
        return shap.TreeExplainer(model)
    if isinstance(model, _LINEAR_MODELS):
        return shap.LinearExplainer(model, X_background)
    return shap.Explainer(model.predict, X_background)


def _as_single_output_explanation(shap_values: shap.Explanation) -> shap.Explanation:
    """
    Collapse SHAP's extra singleton output axis for single-target regressors.

    Different explainers attach the spurious singleton axis to different
    attributes. TreeExplainer can return 3-D ``values`` of shape
    ``(n, features, 1)``, whereas LinearExplainer keeps ``values`` 2-D but
    returns ``base_values`` of shape ``(n, 1)``. Either form breaks the slicer
    used by ``shap.plots.scatter``/``waterfall``, so we normalise all of them
    back to single-output shapes here.
    """
    values = np.asarray(shap_values.values)
    if values.ndim == 3 and 1 in values.shape[1:]:
        values = values[:, 0, :] if values.shape[1] == 1 else values[:, :, 0]

    base_values = shap_values.base_values
    if base_values is not None:
        base_values = np.asarray(base_values)
        if base_values.ndim == 2 and base_values.shape[1] == 1:
            base_values = base_values[:, 0]

    data = shap_values.data
    if data is not None:
        data = np.asarray(data)
        if data.ndim == 3 and 1 in data.shape[1:]:
            data = data[:, 0, :] if data.shape[1] == 1 else data[:, :, 0]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=shap_values.feature_names,
    )


def compute_shap_values(model, X_sample: pd.DataFrame) -> shap.Explanation:
    """
    Build the explainer and return a SHAP Explanation object for `X_sample`.
    """
    explainer = _build_explainer(model, X_sample)
    return _as_single_output_explanation(explainer(X_sample))



def _save_current_figure(output_dir: Union[Path, str], filename: str) -> Path:
    """
    Persist the active matplotlib figure to `output_dir/filename` and close it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def shap_summary(
    shap_values: shap.Explanation,
    output_dir: Union[Path, str] = "plots",
    filename: str = "shap_summary.png",
    max_display: int = 15,
) -> Path:
    """
    SHAP beeswarm summary plot for the top `max_display` features.
    """
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    return _save_current_figure(output_dir, filename)


def shap_bar(
    shap_values: shap.Explanation,
    output_dir: Union[Path, str] = "plots",
    filename: str = "shap_bar.png",
    max_display: int = 15,
) -> Path:
    """
    Global feature ranking by mean absolute SHAP value.
    """
    plt.figure()
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    return _save_current_figure(output_dir, filename)


def shap_dependence(
    shap_values: shap.Explanation,
    feature: str,
    output_dir: Union[Path, str] = "plots",
    filename: Optional[str] = None,
) -> Path:
    """
    Dependence (scatter) plot showing how `feature` drives its own contribution.
    """
    filename = filename or f"shap_dependence_{feature}.png"
    feature_names = list(shap_values.feature_names)
    if feature not in feature_names:
        raise ValueError(f"Feature {feature!r} is not present in SHAP feature names.")
    feature_idx = feature_names.index(feature)

    plt.figure()
    shap.plots.scatter(shap_values[:, feature_idx], show=False)
    return _save_current_figure(output_dir, filename)


def shap_waterfall(
    shap_values: shap.Explanation,
    row_index: int = 0,
    output_dir: Union[Path, str] = "plots",
    filename: Optional[str] = None,
) -> Path:
    """
    Local explanation (waterfall plot) for a single observation.
    """
    filename = filename or f"shap_waterfall_row_{row_index}.png"
    plt.figure()
    shap.plots.waterfall(shap_values[row_index], show=False)
    return _save_current_figure(output_dir, filename)


def _top_feature(shap_values: shap.Explanation) -> str:
    """
    Return the feature with the largest mean absolute SHAP value.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    return str(shap_values.feature_names[int(np.argmax(mean_abs))])


def run_shap_analysis(
    model,
    X: pd.DataFrame,
    n_sample: int = 5000,
    output_dir: Union[Path, str] = "plots",
    dependence_feature: Optional[str] = None,
) -> List[Path]:
    """
    Full SHAP report for `model`: subsample, explain, and save every plot.

    Returns the list of saved figure paths (summary, bar, dependence, waterfall).
    """
    X_sample = subsample_for_shap(X, n=n_sample)
    logger.info("Computing SHAP values on %d rows...", len(X_sample))
    shap_values = compute_shap_values(model, X_sample)

    feature = dependence_feature or _top_feature(shap_values)
    logger.info("SHAP dependence plot uses the most influential feature: %s", feature)

    paths = [
        shap_summary(shap_values, output_dir=output_dir),
        shap_bar(shap_values, output_dir=output_dir),
        shap_dependence(shap_values, feature, output_dir=output_dir),
        shap_waterfall(shap_values, row_index=0, output_dir=output_dir),
    ]
    logger.info("SHAP plots saved: %s", ", ".join(str(p) for p in paths))
    return paths
