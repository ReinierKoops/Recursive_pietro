"""SHAP value computation and aggregation helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from shap import Explainer
from shap.explainers import TreeExplainer
from shap.utils import sample
from sklearn.pipeline import Pipeline


def compute_shap_values(
    model,
    X: pd.DataFrame,
    *,
    approximate: bool = False,
    check_additivity: bool = True,
    random_state: int | None = None,
    sample_size: int = 100,
    verbose: int = 0,
    **shap_kwargs,
) -> np.ndarray:
    """Compute SHAP values for a fitted model on dataset X.

    Args:
        model: A fitted sklearn-compatible model.
        X: Feature matrix (validation data).
        approximate: Use approximate SHAP (TreeExplainer only, faster).
        check_additivity: Check SHAP additivity (TreeExplainer only).
        random_state: Seed for background data sampling.
        sample_size: Number of background samples for non-tree explainers.
        verbose: Verbosity level.
        **shap_kwargs: Additional kwargs for shap.Explainer.

    Returns:
        SHAP values array with shape (n_samples, n_features) for binary/regression,
        or (n_classes, n_samples, n_features) for multiclass.
    """
    if isinstance(model, Pipeline):
        raise TypeError(
            "Pipeline models are not supported because shap.Explainer cannot introspect "
            "pipeline stages. Apply data transformations before calling recursive_pietro."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if verbose <= 1 else "default")

        # For tree explainers with tree_path_dependent or categorical features,
        # do not pass a masker (SHAP issue #480)
        has_categoricals = X.select_dtypes("category").shape[1] > 0
        if shap_kwargs.get("feature_perturbation") == "tree_path_dependent" or has_categoricals:
            explainer = Explainer(model, seed=random_state, **shap_kwargs)
        else:
            # Create background data for non-tree explainers
            n_bg = min(sample_size, max(1, int(np.ceil(X.shape[0] * 0.2)))) if X.shape[0] < sample_size else sample_size
            mask = sample(X, n_bg, random_state=random_state)
            explainer = Explainer(model, seed=random_state, masker=mask, **shap_kwargs)

        if isinstance(explainer, TreeExplainer):
            shap_values = explainer.shap_values(
                X, check_additivity=check_additivity, approximate=approximate
            )
            # SHAP >= 0.43: binary classification returns 3D array (n_samples, n_features, 2)
            if not isinstance(shap_values, list) and shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        else:
            shap_values = explainer.shap_values(X)

        # For binary classification, some models return a list of [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

    return shap_values


def aggregate_shap_importance(
    shap_values: np.ndarray,
    columns: list[str],
    penalty_factor: float = 0.0,
) -> pd.DataFrame:
    """Compute mean absolute SHAP importance per feature, sorted descending.

    Args:
        shap_values: SHAP values array. Shape (n_samples, n_features) for
            binary/regression, or (n_classes, n_samples, n_features) for multiclass.
        columns: Feature names corresponding to the columns axis.
        penalty_factor: Penalize high-variance features:
            penalized_mean = mean_abs_shap - (std_abs_shap * penalty_factor).
            Used only for sorting; the penalized column is dropped from output.

    Returns:
        DataFrame indexed by feature name with columns
        'mean_abs_shap_value' and 'mean_shap_value', sorted by importance descending.
    """
    abs_shap = np.abs(shap_values)

    if shap_values.ndim > 2:
        # Multiclass: shape (n_classes, n_samples, n_features) → sum across classes first
        sum_abs = np.sum(abs_shap, axis=0)
        sum_raw = np.sum(shap_values, axis=0)
        mean_abs = np.mean(sum_abs, axis=0)
        mean_raw = np.mean(sum_raw, axis=0)
        std_abs = np.std(sum_abs, axis=0)
    else:
        mean_abs = np.mean(abs_shap, axis=0)
        mean_raw = np.mean(shap_values, axis=0)
        std_abs = np.std(abs_shap, axis=0)

    penalized = mean_abs - (std_abs * penalty_factor)

    df = pd.DataFrame(
        {
            "mean_abs_shap_value": mean_abs,
            "mean_shap_value": mean_raw,
            "_penalized": penalized,
        },
        index=columns,
    ).astype(float)

    df = df.sort_values("_penalized", ascending=False)
    df = df.drop(columns=["_penalized"])
    return df
