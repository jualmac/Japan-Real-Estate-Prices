"""
Optional cuDF helpers for GPU-backed model input.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

try:
    import cudf
except ImportError:
    cudf = None

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def cudf_available() -> bool:
    """
    Return whether RAPIDS cuDF is importable in the current environment.
    """
    return cudf is not None


def to_cudf_dataframe(X: pd.DataFrame) -> Any:
    """
    Copy a pandas DataFrame to GPU memory as a cuDF DataFrame.
    """
    if cudf is None:
        raise ImportError("cuDF is not installed in this environment.")
    return cudf.from_pandas(X.astype("float32", copy=False), nan_as_null=False)


def to_cudf_series(y: pd.Series) -> Any:
    """
    Copy a pandas Series to GPU memory as a cuDF Series.
    """
    if cudf is None:
        raise ImportError("cuDF is not installed in this environment.")
    return cudf.from_pandas(y.astype("float32", copy=False), nan_as_null=False)


def to_host_array(values: Any) -> np.ndarray:
    """
    Copy GPU-backed arrays back to host memory for sklearn metrics and reporting.
    """
    if hasattr(values, "get"):
        return values.get()
    if cudf is not None and isinstance(values, (cudf.Series, cudf.DataFrame)):
        return values.to_pandas().to_numpy()
    return np.asarray(values)


def xgboost_uses_cuda(model: Any) -> bool:
    """
    Return whether an XGBoost sklearn estimator is configured for CUDA.
    """
    device = str(model.get_params().get("device", "")).lower()
    return device.startswith("cuda")
