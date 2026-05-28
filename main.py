"""
Principal entrypoint for the Japan Real Estate Prices ML pipeline.

Orchestrates the full workflow:
1.  Reproducibility setup (seeds + logging);
2.  Data acquisition (Kaggle download) and loading (CSV concat);
3.  Exploratory data analysis (overview + missing profile);
4.  Nested temporal cross-validation (expanding-window outer + inner folds)
    per candidate model, with leakage-free preprocessing fit per fold. The
    best outer fold's hyperparameters are persisted as the canonical params.
5.  Final training on the full dataset using those nested-CV best params.
6.  Nested CV metric reporting (mean +/- std across outer folds) and best
    model selection by mean outer-fold RMSE from the current run.
7.  Interpretability (feature importance + SHAP) on the best model.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Dict

from src.config import CONFIG, TARGET_COLUMN, set_seed
from src.data.db import ensure_database
from src.data.download import download_tradeprices
from src.data.loader import load_all_prefectures
from src.eda.summary import check_duplicates, nan_profile, print_overview
from src.evaluation.metrics import (
    append_nested_cv_csv,
    nested_comparison_table,
    persist_nested_cv_artifacts,
)
from src.interpretability.feature_importance import plot_importance
from src.models.registry import save_best_params
from src.models.temporal_cv import NestedTemporalCV, select_best_fold_params
from src.models.training import train_all
from src.preprocessing.pipeline import preprocess_full
from src.utils.logging import setup_logging
from src.visualization.plots import year_distribution

########################################################################################################################
#
# PIPELINE
#
########################################################################################################################
def main() -> None:
    logger = setup_logging(level=CONFIG.log_level)
    set_seed(CONFIG.seed)
    logger.info(
        "Pipeline started (seed=%s, n_trials=%s, outer_folds=%s, inner_folds=%s).",
        CONFIG.seed,
        CONFIG.n_trials,
        CONFIG.outer_folds,
        CONFIG.inner_folds,
    )

    download_tradeprices()
    ensure_database()
    df = load_all_prefectures()

    print_overview(df)
    check_duplicates(df)
    nan_profile(df)
    year_distribution(df)

    X_raw = df.drop(columns=[TARGET_COLUMN])
    y_raw = df[TARGET_COLUMN]

    ####################################################################################################################
    # Nested temporal CV: canonical performance estimate per model; the best
    # outer fold's hyperparameters become the canonical params for the final fit;
    ####################################################################################################################
    nested_cv = NestedTemporalCV()
    run_scores: Dict[str, float] = {}
    for model_name in CONFIG.models:
        logger.info("Running nested temporal CV for %s...", model_name)
        per_fold_records = nested_cv.run(model_name, X_raw, y_raw)
        _, _, aggregate = persist_nested_cv_artifacts(model_name, per_fold_records)
        append_nested_cv_csv(model_name, aggregate)
        run_scores[model_name] = float(aggregate["rmse_mean"])
        save_best_params(model_name, select_best_fold_params(per_fold_records))

    ####################################################################################################################
    # Refit production models on the full dataset using nested-CV best params;
    ####################################################################################################################
    X_full, y_full, _, _, _ = preprocess_full(X_raw, y_raw)
    models = train_all(CONFIG.models, X_full, y_full)

    ####################################################################################################################
    # Report and interpret; select the best model from THIS run, not stale CSV rows;
    ####################################################################################################################
    comparison = nested_comparison_table()
    if comparison is not None:
        logger.info("Nested CV comparison (sorted by mean RMSE):\n%s", comparison)

    best_name = min(run_scores, key=run_scores.get)
    logger.info("Best model by mean outer-fold RMSE (this run): %s", best_name)
    plot_importance(models[best_name], X_full.columns)
    # TODO: shap_summary(models[best_name], subsample_for_shap(X_full))

########################################################################################################################
#
# MAIN
#
########################################################################################################################
if __name__ == "__main__":
    main()
    print("Done!")