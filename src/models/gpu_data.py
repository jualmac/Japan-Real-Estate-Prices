"""
Optional cuDF / cuML helpers for GPU-backed model input.
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

try:
    from cuml.ensemble import RandomForestRegressor as _CuMLRandomForestRegressor
except ImportError:
    _CuMLRandomForestRegressor = None

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


def cuml_available() -> bool:
    """
    Return whether RAPIDS cuML RandomForestRegressor is importable.
    """
    return _CuMLRandomForestRegressor is not None


def make_cuml_random_forest(**params: Any) -> Any:
    """
    Instantiate a cuML RandomForestRegressor, filtering out sklearn-only kwargs.

    cuML's RF does not accept ``n_jobs`` (it uses ``n_streams``) and does not
    support sklearn's ``criterion`` string; unsupported keys are dropped so that
    the Optuna search space stays shared between the sklearn and cuML backends.
    """
    if _CuMLRandomForestRegressor is None:
        raise ImportError("cuML is not installed in this environment.")
    sklearn_only = {"n_jobs", "criterion"}
    cleaned = {k: v for k, v in params.items() if k not in sklearn_only}
    return _CuMLRandomForestRegressor(**cleaned)


def is_cuml_random_forest(model: Any) -> bool:
    """
    Return whether ``model`` is a cuML RandomForestRegressor instance.
    """
    if _CuMLRandomForestRegressor is None:
        return False
    return isinstance(model, _CuMLRandomForestRegressor)
