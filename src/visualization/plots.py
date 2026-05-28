"""
Shared plotting helpers (heatmaps, distributions, etc.).
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
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
