"""
SKETCH: SHAP-based interpretation of the best model.

These plots are required by the "Interpretação e análise crítica" section of
the assignment to show how attributes drive each individual prediction.

TODO before running on the full dataset:
- Sub-sample X (e.g. 5_000 rows) before calling shap.Explainer; the dataset is too
  large for an exact computation.
- Cache the Explainer object to disk to avoid recomputation between runs.
- Decide whether to use TreeExplainer (fast, exact for tree models) or the
  generic shap.Explainer (slower but model-agnostic).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Optional

import pandas as pd


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def shap_summary(model, X_sample: pd.DataFrame, max_display: int = 15) -> None:
    """
    SKETCH: SHAP beeswarm summary plot for the top `max_display` features.

    Implementation outline (uncomment after `pip install shap`):

        import shap
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample, max_display=max_display)
    """
    raise NotImplementedError("Install shap and uncomment the body of shap_summary().")


def shap_dependence(model, X_sample: pd.DataFrame, feature: str) -> None:
    """
    SKETCH: SHAP dependence plot for a single feature.

    Useful to inspect non-linear relationships between a feature and its
    contribution to the prediction.
    """
    raise NotImplementedError("Install shap and implement the body of shap_dependence().")


def force_plot_for_row(model, X_sample: pd.DataFrame, row_index: int) -> None:
    """
    SKETCH: Local explanation (force plot) for a single observation.

    Useful in the article to walk the reader through one specific prediction.
    """
    raise NotImplementedError("Install shap and implement the body of force_plot_for_row().")


def subsample_for_shap(X: pd.DataFrame, n: int = 5000, random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Helper to keep SHAP computations tractable on a 1M+ row dataset.
    """
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=random_state)
