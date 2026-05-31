"""
Principal entrypoint for the Japan Real Estate Prices ML pipeline.

Orchestrates the full workflow:
1.  Reproducibility setup (seeds + logging);
2.  Data acquisition (Kaggle download) and loading (CSV concat);
3.  Exploratory data analysis (overview + missing profile);
4.  Temporal train/validation/test split (leak-free);
5.  Cleaning (rule-based + train-fitted imputers);
6.  Feature engineering (cyclical Quarter, time-to-station rebuild, locations);
7.  Categorical encoding + numeric scaling (train-fitted, replayed elsewhere).
8.  Hyperparameter optimization (Optuna) per candidate model.
9.  Final training with the best parameters.
10. Evaluation (MAPE / RMSE / MAE / R^2).
11. Interpretability (feature importance + SHAP) on the best model.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path

from src.config import CONFIG, PARAMETERS_DIR, Study, set_seed
from src.data.audit import (
    create_run_id,
    save_model_input_splits,
    save_model_metrics,
    save_pipeline_run_metadata,
)
from src.data.db import ensure_database
from src.data.download import download_tradeprices
from src.data.loader import load_all_prefectures
from src.eda.summary import check_duplicates, nan_profile, print_overview
from src.evaluation.metrics import evaluate_all
from src.interpretability.feature_importance import plot_importance
from src.interpretability.shap_analysis import run_shap_analysis
from src.models.optimization import OptimizeRegressor
from src.models.training import train_all
from src.preprocessing.cleaning import DataCleaner
from src.preprocessing.encoding import encode_categoricals, to_snake_case_columns
from src.preprocessing.feature_engineering import apply_feature_engineering
from src.preprocessing.scaling import scale_splits
from src.preprocessing.splitting import filter_by_type, temporal_split, unpack_split
from src.utils.logging import setup_logging
from src.visualization.plots import (
    plot_boosting_training_curves,
    plot_test_predictions,
    plot_train_validation_loss,
    year_distribution,
)

########################################################################################################################
#
# PIPELINE
#
########################################################################################################################
def run_study(study: Study, df, run_id: str, logger) -> None:
    """
    Run the full modeling pipeline for a single property-type study.

    Artifacts are isolated per study: parameters land in parameters/<study>/ and
    plots in plots/<study>/, while audit rows are tagged with the study name.
    """
    logger.info("=== Study '%s' (types=%s) ===", study.name, ", ".join(study.property_types))
    parameters_dir = PARAMETERS_DIR / study.name
    plots_dir = Path("plots") / study.name

    study_df = filter_by_type(df, study.property_types)
    logger.info("Study '%s' has %d rows after type filtering.", study.name, len(study_df))
    year_distribution(study_df, output_dir=plots_dir)

    split = temporal_split(study_df)
    X_tr, X_val, X_te, y_tr, y_val, y_te = unpack_split(split)

    cleaner = DataCleaner().fit(X_tr)
    X_tr, X_val, X_te = cleaner.transform(X_tr), cleaner.transform(X_val), cleaner.transform(X_te)

    X_tr, X_val, X_te, y_tr, y_val, y_te = apply_feature_engineering(
        X_tr, X_val, X_te, y_tr, y_val, y_te
    )

    cleaner.fit_post_merge(X_tr)
    X_tr = cleaner.transform_post_merge(X_tr)
    X_val = cleaner.transform_post_merge(X_val)
    X_te = cleaner.transform_post_merge(X_te)

    X_tr, X_val, X_te = encode_categoricals(X_tr, X_val, X_te)
    X_tr, X_val, X_te = scale_splits(X_tr, X_val, X_te)

    X_tr = to_snake_case_columns(X_tr)
    X_val = to_snake_case_columns(X_val)
    X_te = to_snake_case_columns(X_te)

    save_pipeline_run_metadata(run_id, study.name, split, X_tr.columns)
    saved_split_rows = save_model_input_splits(
        run_id,
        study.name,
        {
            "train": (X_tr, y_tr),
            "validation": (X_val, y_val),
            "test": (X_te, y_te),
        },
    )
    logger.info(
        "Saved %d model input audit rows for run_id=%s study=%s.",
        saved_split_rows, run_id, study.name,
    )

    for model_name in CONFIG.models:
        logger.info("Optimizing %s with %d trials...", model_name, CONFIG.n_trials)
        OptimizeRegressor(
            model_name=model_name,
            n_trials=CONFIG.n_trials,
            X_train=X_tr,
            y_train=y_tr,
            parameters_dir=parameters_dir,
        ).optimize()

    models = train_all(CONFIG.models, X_tr, y_tr, X_val, y_val, parameters_dir=parameters_dir)
    training_curve_path = plot_boosting_training_curves(models, output_dir=plots_dir)
    logger.info("Saved boosting training curves to %s.", training_curve_path)

    train_results = evaluate_all(models, X_tr, y_tr)
    logger.info("Model comparison on the training split:\n%s", train_results)

    validation_results = evaluate_all(models, X_val, y_val)
    logger.info("Model comparison on the validation split:\n%s", validation_results)
    loss_plot_path = plot_train_validation_loss(train_results, validation_results, output_dir=plots_dir)
    logger.info("Saved training-vs-validation loss plot to %s.", loss_plot_path)

    results = evaluate_all(models, X_te, y_te)
    logger.info("Model comparison on the test split:\n%s", results)
    prediction_plot_path = plot_test_predictions(models, X_te, y_te, output_dir=plots_dir)
    logger.info("Saved test prediction performance plot to %s.", prediction_plot_path)
    saved_metric_rows = save_model_metrics(
        run_id,
        study.name,
        {
            "train": train_results,
            "validation": validation_results,
            "test": results,
        },
    )
    logger.info(
        "Saved %d model metric audit rows for run_id=%s study=%s.",
        saved_metric_rows, run_id, study.name,
    )

    best_name = results["rmse"].idxmin()
    logger.info("Best model by RMSE for study '%s': %s", study.name, best_name)
    plot_importance(models[best_name], X_tr.columns, output_dir=plots_dir)
    run_shap_analysis(models[best_name], X_te, output_dir=plots_dir)


def main() -> None:
    logger = setup_logging(level=CONFIG.log_level)
    set_seed(CONFIG.seed)
    run_id = create_run_id()
    logger.info(
        "Pipeline started (run_id=%s, seed=%s, n_trials=%s, studies=%d).",
        run_id, CONFIG.seed, CONFIG.n_trials, len(CONFIG.studies),
    )

    download_tradeprices()
    ensure_database()
    df = load_all_prefectures()

    print_overview(df)
    check_duplicates(df)
    nan_profile(df)

    for study in CONFIG.studies:
        run_study(study, df, run_id, logger)

########################################################################################################################
#
# MAIN
#
########################################################################################################################
if __name__ == "__main__":
    main()
    print("Done!")