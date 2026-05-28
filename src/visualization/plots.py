"""
Shared plotting helpers (heatmaps, distributions, etc.).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.frame import DataFrame


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
