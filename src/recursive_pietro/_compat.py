"""Early stopping compatibility layer for boosted tree models."""

from __future__ import annotations

from sklearn.model_selection._search import BaseSearchCV


def _get_base_model(model):
    """Unwrap a SearchCV to get the base estimator."""
    if isinstance(model, BaseSearchCV):
        return model.estimator
    return model


def is_early_stopping_compatible(model) -> bool:
    """Check if the model supports early stopping (LightGBM, XGBoost, CatBoost)."""
    base = _get_base_model(model)
    for lib, class_name in _SUPPORTED_LIBRARIES:
        try:
            module = __import__(lib, fromlist=[class_name])
            if isinstance(base, getattr(module, class_name)):
                return True
        except ImportError:
            pass
    return False


_SUPPORTED_LIBRARIES = [
    ("lightgbm", "LGBMModel"),
    ("xgboost.sklearn", "XGBModel"),
    ("catboost", "CatBoost"),
]


def get_early_stopping_fit_params(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    early_stopping_rounds: int,
    eval_metric: str | None,
    sample_weight=None,
    train_index=None,
    val_index=None,
    verbose: int = 0,
) -> dict:
    """Build fit keyword arguments for early-stopping-compatible models.

    Returns a dict suitable for ``model.fit(**params)``. Also mutates ``model``
    in-place where needed (e.g. XGBoost ``set_params``).
    """
    # Try LightGBM first
    try:
        from lightgbm import LGBMModel

        if isinstance(model, LGBMModel):
            return _fit_params_lightgbm(
                model, X_train, y_train, X_val, y_val,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
                sample_weight=sample_weight,
                train_index=train_index,
                val_index=val_index,
                verbose=verbose,
            )
    except ImportError:
        pass

    # Try XGBoost
    try:
        from xgboost.sklearn import XGBModel

        if isinstance(model, XGBModel):
            return _fit_params_xgboost(
                model, X_train, y_train, X_val, y_val,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
                sample_weight=sample_weight,
                train_index=train_index,
                val_index=val_index,
            )
    except ImportError:
        pass

    # Try CatBoost
    try:
        from catboost import CatBoost

        if isinstance(model, CatBoost):
            return _fit_params_catboost(
                model, X_train, y_train, X_val, y_val,
                early_stopping_rounds=early_stopping_rounds,
                sample_weight=sample_weight,
                train_index=train_index,
                val_index=val_index,
            )
    except ImportError:
        pass

    raise ValueError(
        f"Model type {type(model).__name__} is not supported for early stopping. "
        "Only LightGBM, XGBoost, and CatBoost models are supported."
    )


def _fit_params_lightgbm(
    model, X_train, y_train, X_val, y_val, *,
    early_stopping_rounds, eval_metric, sample_weight, train_index, val_index, verbose,
) -> dict:
    from lightgbm import early_stopping, log_evaluation

    params = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
        "eval_metric": eval_metric,
        "callbacks": [
            early_stopping(early_stopping_rounds, first_metric_only=True),
            log_evaluation(1 if verbose >= 2 else 0),
        ],
    }
    if sample_weight is not None:
        params["sample_weight"] = sample_weight.iloc[train_index]
        params["eval_sample_weight"] = [sample_weight.iloc[val_index]]
    return params


def _fit_params_xgboost(
    model, X_train, y_train, X_val, y_val, *,
    early_stopping_rounds, eval_metric, sample_weight, train_index, val_index,
) -> dict:
    model.set_params(eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds)
    params = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
    }
    if sample_weight is not None:
        params["sample_weight"] = sample_weight.iloc[train_index]
        params["eval_sample_weight"] = [sample_weight.iloc[val_index]]
    return params


def _fit_params_catboost(
    model, X_train, y_train, X_val, y_val, *,
    early_stopping_rounds, sample_weight, train_index, val_index,
) -> dict:
    from catboost import Pool

    cat_features = list(X_train.select_dtypes(include=["category"]).columns)
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    eval_pool = Pool(X_val, y_val, cat_features=cat_features)

    if sample_weight is not None:
        train_pool.set_weight(sample_weight.iloc[train_index])
        eval_pool.set_weight(sample_weight.iloc[val_index])

    model.set_params(early_stopping_rounds=early_stopping_rounds)

    return {
        "X": train_pool,
        "eval_set": eval_pool,
    }
