"""
SKETCH: Feature importance helpers (built-in attributes + permutation importance).

Used to answer the assignment requirement (iv) "Interpretação e análise crítica":
identify which attributes have the largest impact on the best model's predictions.

TODO:
- Decide a consistent visual style (palette, top_n) for the article.
- Persist the importance DataFrames to parameters/ for reproducibility.
- Add support for partial dependence plots (sklearn.inspection.PartialDependenceDisplay).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def builtin_importance(model, feature_names: Sequence[str]) -> pd.DataFrame:
    """
    SKETCH: Return the model's built-in importance signal sorted descending.

    Works out-of-the-box for tree-based models (RF / XGB / LGBM) via
    `feature_importances_`, and falls back to `|coef_|` for linear models
    (ElasticNet / LinearSVR).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))
    else:
        raise AttributeError("Model exposes neither feature_importances_ nor coef_.")
    df = pd.DataFrame({
        "feature": list(feature_names),
        "importance": importances,
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_importance_report(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """
    SKETCH: Compute permutation importance and return a tidy DataFrame.

    Model-agnostic alternative to built-in importance. Slower but more reliable
    when features are correlated.
    """
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    df = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_importance(model, feature_names: Sequence[str], top_n: int = 15) -> None:
    """
    SKETCH: Bar chart of the top_n most important features.

    TODO: switch to seaborn / consistent figure styling once it is defined.
    """
    importance_df = builtin_importance(model, feature_names).head(top_n)
    plt.figure(figsize=(10, 0.4 * top_n + 2))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.title(f"Top {top_n} feature importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()
