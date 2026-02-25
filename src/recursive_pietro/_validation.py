"""Input validation utilities for recursive_pietro."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def validate_data(X: pd.DataFrame | np.ndarray, column_names: list[str] | None = None) -> pd.DataFrame:
    """Convert X to a DataFrame with proper column names.

    Args:
        X: Feature matrix as DataFrame or 2D ndarray.
        column_names: Optional override for column names.

    Returns:
        DataFrame with string column names.

    Raises:
        TypeError: If X is not a DataFrame or 2D ndarray.
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, np.ndarray) and X.ndim == 2:
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    else:
        raise TypeError("X must be a pandas DataFrame or a 2D numpy array.")

    if column_names is not None:
        if len(column_names) != df.shape[1]:
            raise ValueError(
                f"column_names has {len(column_names)} elements but X has {df.shape[1]} columns."
            )
        df.columns = column_names

    # Ensure all column names are strings
    df.columns = [str(c) for c in df.columns]
    return df


def validate_target(y: pd.Series | np.ndarray | list, index: pd.Index) -> pd.Series:
    """Convert y to a Series aligned with X's index.

    Args:
        y: Target vector.
        index: Index from the validated X DataFrame.

    Returns:
        pd.Series aligned to X.
    """
    if isinstance(y, pd.Series):
        return y.reindex(index)
    return pd.Series(y, index=index, name="target")


def validate_sample_weight(
    sample_weight: pd.Series | np.ndarray | list | None,
    index: pd.Index,
) -> pd.Series | None:
    """Convert sample_weight to a Series aligned with X's index, or return None."""
    if sample_weight is None:
        return None
    if isinstance(sample_weight, pd.Series):
        return sample_weight.reindex(index)
    return pd.Series(sample_weight, index=index, name="sample_weight")


def validate_step(step: int | float) -> int | float:
    """Validate the step parameter."""
    if not isinstance(step, (int, float)) or step <= 0:
        raise ValueError(f"step must be a positive int or float, got {step!r}.")
    if isinstance(step, float) and step >= 1.0:
        raise ValueError(f"When step is a float, it must be in (0, 1), got {step}.")
    return step


def validate_min_features(min_features: int) -> int:
    """Validate the min_features_to_select parameter."""
    if not isinstance(min_features, int) or min_features < 1:
        raise ValueError(f"min_features_to_select must be a positive int, got {min_features!r}.")
    return min_features


def validate_columns_to_keep(
    columns_to_keep: list[str] | None,
    all_columns: list[str],
) -> list[str] | None:
    """Validate columns_to_keep against available columns."""
    if columns_to_keep is None:
        return None
    if not all(isinstance(c, str) for c in columns_to_keep):
        raise ValueError("All elements in columns_to_keep must be strings.")
    missing = set(columns_to_keep) - set(all_columns)
    if missing:
        raise ValueError(f"columns_to_keep contains columns not in X: {sorted(missing)}")
    return list(columns_to_keep)


def validate_groups(groups, index: pd.Index, cv_splitter) -> np.ndarray | None:
    """Validate and convert groups for group-aware CV splitters.

    Raises ValueError when *groups* is provided but the CV splitter does not
    consume them, or vice-versa.
    """
    # Detect whether the splitter expects groups.
    # Group-aware splitters define get_n_splits(X, y, groups) and will fail
    # without groups.  A reliable check: the splitter's class lives in a
    # module path containing "group" or it has ``groups`` in its split signature.
    _GROUP_SPLITTERS = set()
    try:
        from sklearn.model_selection import (
            GroupKFold,
            GroupShuffleSplit,
            LeaveOneGroupOut,
            LeavePGroupsOut,
            StratifiedGroupKFold,
        )

        _GROUP_SPLITTERS = {GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, LeavePGroupsOut, StratifiedGroupKFold}
    except ImportError:
        pass

    splitter_is_group_aware = type(cv_splitter) in _GROUP_SPLITTERS

    if groups is None:
        if splitter_is_group_aware:
            raise ValueError(
                f"CV splitter {type(cv_splitter).__name__} requires groups, "
                "but groups=None was passed to fit()."
            )
        return None

    if not splitter_is_group_aware:
        warnings.warn(
            f"groups was provided but CV splitter {type(cv_splitter).__name__} "
            "does not use groups. The groups argument will be ignored. "
            "Pass a group-aware splitter (e.g. GroupKFold) to use groups."
        )

    if isinstance(groups, pd.Series):
        return groups.reindex(index).values
    return np.asarray(groups)


def validate_shap_variance_penalty(factor: float | int | None) -> float:
    """Validate and normalize the SHAP variance penalty factor."""
    if factor is None:
        return 0.0
    if not isinstance(factor, (float, int)):
        warnings.warn("shap_variance_penalty_factor must be None, int, or float. Using 0.")
        return 0.0
    if factor < 0:
        warnings.warn("shap_variance_penalty_factor must be >= 0. Using 0.")
        return 0.0
    return float(factor)
