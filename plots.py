from typing import Tuple
from pandas.core.frame import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(num_attributes: DataFrame, figsize: Tuple = (15, 8)):
    """
    Plots Heatmap.

    Parameters
    ----------
    num_attributes: DataFrame
        DataFrame with numerical attributes.
    figsize: Tuple, defaults to (15, 8)
        Figure size.
    """

    corr = num_attributes.corr(method="pearson").corr()
    mask = np.triu(corr)
    plt.figure(figsize=figsize)

    sns.heatmap(corr, annot=True, mask=mask)
