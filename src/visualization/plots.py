"""
Shared plotting helpers (heatmaps, distributions, etc.).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import Mapping, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.frame import DataFrame

from src.evaluation.metrics import predict_original_scale
from src.preprocessing.feature_engineering import inverse_log_transform_target


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def plot_heatmap(num_attributes: DataFrame, figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Plot a correlation heatmap (lower triangular) for the numerical attributes.
    """
    corr = num_attributes.corr(method="pearson")
    mask = np.triu(corr)
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, mask=mask)
    plt.tight_layout()
    plt.show()


def year_distribution(
    df: pd.DataFrame,
    year_column: str = "Year",
    building_year_column: str = "BuildingYear",
    output_dir: Path | str = "plots",
    filename: str = "year_distribution.png",
    figsize: Tuple[int, int] = (16, 6),
) -> Path:
    """
    Plot the row distributions by year and building year, then save them.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.histplot(data=df, x=year_column, discrete=True, shrink=0.8, ax=axes[0])
    axes[0].set_title(f"Row Distribution by {year_column}")
    axes[0].set_xlabel(year_column)
    axes[0].set_ylabel("Row Count")

    sns.histplot(data=df, x=building_year_column, discrete=True, shrink=0.8, ax=axes[1])
    axes[1].set_title(f"Row Distribution by {building_year_column}")
    axes[1].set_xlabel(building_year_column)
    axes[1].set_ylabel("Row Count")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_train_validation_loss(
    train_results: pd.DataFrame,
    validation_results: pd.DataFrame,
    metric: str = "rmse",
    output_dir: Path | str = "plots",
    filename: str = "train_validation_loss.png",
    figsize: Tuple[int, int] = (12, 6),
) -> Path:
    """
    Save a train-vs-validation loss chart for diagnosing underfitting/overfitting.
    """
    if metric not in train_results.columns or metric not in validation_results.columns:
        raise ValueError(f"Metric '{metric}' must exist in both result DataFrames.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    loss_df = pd.DataFrame({
        "Train": train_results[metric],
        "Validation": validation_results[metric],
    }).dropna()
    loss_df = loss_df.loc[validation_results.index.intersection(loss_df.index)]

    fig, ax = plt.subplots(figsize=figsize)
    loss_df.plot(kind="bar", ax=ax)
    ax.set_title(f"Training vs Validation Loss ({metric.upper()})")
    ax.set_xlabel("Model")
    ax.set_ylabel(f"{metric.upper()} on Original TradePrice Scale (Yen)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Split")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def _metric_curve(history: Mapping, split_candidates: Tuple[str, ...]) -> np.ndarray | None:
    """
    Extract the first available metric curve for a split from a model eval history.
    """
    for split_name in split_candidates:
        split_history = history.get(split_name)
        if not split_history:
            continue

        metric_name = "rmse" if "rmse" in split_history else next(iter(split_history))
        return np.asarray(split_history[metric_name], dtype=float)

    return None


def plot_boosting_training_curves(
    models: Mapping[str, object],
    output_dir: Path | str = "plots",
    filename: str = "boosting_training_curves.png",
    figsize_per_plot: Tuple[int, int] = (7, 5),
) -> Path:
    """
    Save train-vs-validation RMSE curves for boosting models with eval history.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    histories = {}
    for model_name in ("xgb", "lgbm"):
        model = models.get(model_name)
        if model is None:
            continue
        if hasattr(model, "evals_result") and callable(model.evals_result):
            histories[model_name] = model.evals_result()
        elif hasattr(model, "evals_result_"):
            histories[model_name] = model.evals_result_

    model_count = max(len(histories), 1)
    fig, axes = plt.subplots(
        1,
        model_count,
        figsize=(figsize_per_plot[0] * model_count, figsize_per_plot[1]),
        squeeze=False,
    )

    if not histories:
        axes[0, 0].text(
            0.5,
            0.5,
            "No XGBoost or LightGBM evaluation history was recorded.",
            ha="center",
            va="center",
        )
        axes[0, 0].axis("off")
    else:
        for ax, (model_name, history) in zip(axes.ravel(), histories.items()):
            train_curve = _metric_curve(history, ("train", "training", "validation_0"))
            validation_curve = _metric_curve(history, ("validation", "valid_1", "validation_1"))

            if train_curve is None or validation_curve is None:
                ax.text(0.5, 0.5, f"No complete history for {model_name}.", ha="center", va="center")
                ax.axis("off")
                continue

            ax.plot(train_curve, label="Train")
            ax.plot(validation_curve, label="Validation")
            ax.set_title(f"{model_name}: Training Curve")
            ax.set_xlabel("Boosting Iteration")
            ax.set_ylabel("RMSE on Log-Transformed TradePrice")
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.legend(title="Split")

    fig.suptitle("Boosting Model Training Curves", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_test_predictions(
    models: Mapping[str, object],
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
    output_dir: Path | str = "plots",
    filename: str = "test_actual_vs_predicted.png",
    max_points: int | None = None,
    random_state: int = 42,
    figsize_per_plot: Tuple[int, int] = (6, 5),
) -> Path:
    """
    Save actual-vs-predicted scatter plots for every fitted model on the test split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if max_points is not None and len(X_test) > max_points:
        rng = np.random.default_rng(random_state)
        row_positions = np.sort(rng.choice(len(X_test), size=max_points, replace=False))
        X_plot = X_test.iloc[row_positions]
        y_plot_log = y_test_log.iloc[row_positions]
    else:
        X_plot = X_test
        y_plot_log = y_test_log

    y_true = np.asarray(inverse_log_transform_target(y_plot_log))
    model_count = len(models)
    ncols = min(3, model_count)
    nrows = int(np.ceil(model_count / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False,
    )

    for ax, (model_name, model) in zip(axes.ravel(), models.items()):
        y_pred = np.asarray(predict_original_scale(model, X_plot))
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        actual = y_true[finite_mask]
        predicted = y_pred[finite_mask]

        if actual.size == 0:
            ax.set_title(f"{model_name}: no finite predictions")
            ax.axis("off")
            continue

        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())

        ax.scatter(actual, predicted, s=8, alpha=0.25, edgecolors="none", rasterized=True)
        ax.plot([lower, upper], [lower, upper], color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{model_name}: Test Actual vs Predicted")
        ax.set_xlabel("Actual TradePrice (Yen)")
        ax.set_ylabel("Predicted TradePrice (Yen)")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

    for ax in axes.ravel()[model_count:]:
        ax.axis("off")

    fig.suptitle("Model Predictions on the Test Dataset", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path
