# recursive-pietro: SHAP Recursive Feature Elimination with Early Stopping

## Overview

A focused, modern Python package that performs **backward recursive feature elimination using SHAP importance values** with cross-validation and optional early stopping. Inspired by Probatus's `ShapRFECV` but rebuilt from scratch to be faster, cleaner, and sklearn-compatible.

## Motivation

Probatus (`ing-bank/probatus`) provides `ShapRFECV` which is the most popular SHAP-based RFE implementation. However, it has several pain points:

1. **Outdated constraints** — pinned to `numpy<2.0.0`, old dependency stack
2. **Slow** — no caching, rebuilds SHAP explainer every fold, uses `pd.concat` in a loop
3. **Poor sklearn integration** — custom `BaseFitComputePlotClass` ABC instead of sklearn's `BaseEstimator`/`SelectorMixin`, meaning it doesn't work with sklearn pipelines
4. **Code duplication** — near-identical methods for early-stopping vs. standard path
5. **Heavy imports** — requires `loguru`, `numba`, `matplotlib` at import time even when unused
6. **No progress feedback** — no progress bars for long-running elimination
7. **Deprecated cruft** — ships a deprecated `EarlyStoppingShapRFECV` class

## Package Name

`recursive_pietro`

## Core Feature

**One thing, done well**: SHAP-based Recursive Feature Elimination with Cross-Validation (with optional early stopping for boosted tree models).

## Algorithm

```
Input: X (features), y (target), model, step, min_features_to_select
Output: report DataFrame, optimal feature set

1. Start with all features
2. WHILE num_features > min_features_to_select:
   a. (Optional) Tune hyperparameters via sklearn SearchCV
   b. For each CV fold (parallelized via joblib):
      - Split into train/validation
      - Fit model on train (with early stopping if configured)
      - Score model on train and validation sets
      - Compute SHAP values on validation set
   c. Aggregate SHAP values across folds → mean |SHAP| per feature
      (optionally penalized by std for coherent selection)
   d. Remove `step` lowest-importance features
      (respecting columns_to_keep and min_features_to_select)
   e. Record round results (num_features, scores, feature sets)
3. Return report + fitted selector state
```

## API Design

### Main Class: `ShapFeatureElimination`

```python
from recursive_pietro import ShapFeatureElimination

selector = ShapFeatureElimination(
    model,                          # sklearn-compatible estimator or SearchCV
    step=0.2,                       # int (absolute) or float (fraction) of features to remove per round
    min_features_to_select=1,       # stopping criterion
    cv=5,                           # int, CV splitter, or iterable
    scoring="roc_auc",              # sklearn scoring string or callable
    n_jobs=-1,                      # parallelism for CV folds
    random_state=42,                # reproducibility
    # Early stopping (optional — for LightGBM, XGBoost, CatBoost)
    early_stopping_rounds=None,     # int or None
    eval_metric=None,               # str — model's eval_metric for early stopping
    # SHAP control
    shap_variance_penalty_factor=None,  # float — penalize high-variance SHAP features
    shap_fast_mode=False,           # if True, use approximate SHAP (TreeExplainer only)
    # UX
    verbose=0,                      # 0=silent, 1=progress bar, 2=detailed logging
)

# sklearn-compatible API
selector.fit(X, y, groups=None, sample_weight=None, columns_to_keep=None)
X_reduced = selector.transform(X)
X_reduced = selector.fit_transform(X, y)

# Results
report_df = selector.report_                    # DataFrame with per-round metrics
selected_features = selector.selected_features_ # list[str] — best feature set
support = selector.support_                     # bool array like sklearn RFE
ranking = selector.ranking_                     # int array like sklearn RFE

# Feature set selection strategies
features = selector.get_feature_set(method="best")              # highest val score
features = selector.get_feature_set(method="best_coherent")     # lowest std within threshold
features = selector.get_feature_set(method="best_parsimonious") # fewest features within threshold
features = selector.get_feature_set(method=10)                  # exactly 10 features

# Plotting
fig = selector.plot(show=True)
```

### sklearn Compatibility

The class inherits from `sklearn.base.BaseEstimator` and `sklearn.feature_selection.SelectorMixin`, which means:

- Works in `sklearn.pipeline.Pipeline`
- Supports `get_params()` / `set_params()`
- Supports `get_support()` / `get_feature_names_out()`
- Compatible with `sklearn.model_selection.cross_val_score` etc.

## Project Structure

```
recursive_pietro/
├── pyproject.toml
├── LICENSE
├── README.md                          # (optional, minimal)
├── src/
│   └── recursive_pietro/
│       ├── __init__.py                # Public API exports
│       ├── _elimination.py            # ShapFeatureElimination class
│       ├── _shap.py                   # SHAP value computation helpers
│       ├── _validation.py             # Input validation utilities
│       └── _compat.py                 # Early stopping compatibility (LightGBM/XGBoost/CatBoost)
└── tests/
    ├── __init__.py
    ├── conftest.py                    # Shared fixtures (datasets, models)
    ├── test_elimination.py            # Core elimination tests
    ├── test_early_stopping.py         # Early stopping integration tests
    ├── test_sklearn_compat.py         # Pipeline / sklearn API tests
    └── test_shap.py                   # SHAP helper tests
```

## Dependencies

### Required (minimal)
```
python >= 3.10
scikit-learn >= 1.3
shap >= 0.43
numpy >= 1.24
pandas >= 2.0
joblib >= 1.3
```

### Optional
```
lightgbm >= 4.0      # for early stopping with LGBM
xgboost >= 2.0       # for early stopping with XGB
catboost >= 1.2       # for early stopping with CatBoost
matplotlib >= 3.7     # for plot() method
tqdm >= 4.60          # for progress bars (verbose=1)
```

### Dev
```
pytest >= 7.0
pytest-cov
ruff
mypy
```

## Key Improvements Over Probatus

### 1. sklearn-native design
- Inherits `BaseEstimator` + `SelectorMixin` instead of custom ABC
- Works natively in `Pipeline`, `cross_val_score`, `GridSearchCV`
- Standard attributes: `support_`, `ranking_`, `n_features_`

### 2. Unified early stopping
- Single class handles both standard and early-stopping paths
- No deprecated `EarlyStoppingShapRFECV` needed
- Clean dispatcher pattern for LightGBM/XGBoost/CatBoost fit params

### 3. Performance improvements
- Pre-allocate report list, convert to DataFrame once at the end (not `pd.concat` in loop)
- Reuse SHAP explainer type detection across folds
- `shap_fast_mode=True` flag for approximate SHAP (fast on large datasets)
- No unnecessary imports at module level (matplotlib, tqdm lazy-loaded)

### 4. Modern Python
- Python 3.10+ (use `X | Y` union types, match/case where appropriate)
- Full type annotations on public API
- `src/` layout per modern packaging standards
- `pyproject.toml` only (no setup.py/setup.cfg)

### 5. Better UX
- Progress bars via tqdm when `verbose=1`
- Structured logging via stdlib `logging` when `verbose=2`
- Clear error messages with suggestions

### 6. Cleaner dependencies
- No `loguru`, no `numba` required
- `matplotlib` only needed if you call `plot()`
- `tqdm` only needed if `verbose=1`

## Module Details

### `_elimination.py` — Core Class

The `ShapFeatureElimination` class:

- `__init__`: Store params, validate step/min_features
- `fit(X, y, ...)`: Main elimination loop
  - Validate inputs (X→DataFrame, y→Series, check columns_to_keep)
  - Set up CV splitter via `sklearn.model_selection.check_cv`
  - Loop: at each round, parallel CV via joblib → aggregate SHAP → remove features → record results
  - After loop: set `fitted_=True`, compute `support_`, `ranking_`, `selected_features_`
- `transform(X)`: Return `X[selected_features_]`
- `get_feature_set(method, threshold)`: Feature selection strategies
- `plot(show)`: Lazy-import matplotlib, plot train/val curves with std bands
- `_get_support_mask()`: Required by `SelectorMixin`

### `_shap.py` — SHAP Helpers

- `compute_shap_values(model, X, ...)`: Compute SHAP values for fitted model on dataset
  - Handle TreeExplainer vs. generic Explainer
  - Handle binary class (pick class 1), multiclass (sum across classes), regression
  - Support `approximate=True` for TreeExplainer
  - Mask/background data sampling
- `aggregate_shap_importance(shap_values, columns, penalty_factor)`: Mean |SHAP| per feature with optional variance penalty

### `_validation.py` — Input Validation

- `validate_data(X, column_names)` → `pd.DataFrame` with proper column names
- `validate_target(y, index)` → `pd.Series` aligned to X
- `validate_sample_weight(sample_weight, index)` → `pd.Series` or None
- `validate_step(step)` → validated step
- `validate_min_features(min_features)` → validated int

### `_compat.py` — Early Stopping Support

- `is_early_stopping_compatible(model)` → bool (check if LGBM/XGB/CatBoost)
- `get_early_stopping_fit_params(model, X_train, y_train, X_val, y_val, ...)` → dict
  - Dispatch to correct fit param format per model type
  - Handle LightGBM callbacks (`early_stopping()`, `log_evaluation()`)
  - Handle XGBoost params (`eval_metric`, `early_stopping_rounds` on model)
  - Handle CatBoost Pool objects

## Test Plan

### `test_elimination.py`
- Fit with RandomForestClassifier, verify report has correct shape
- Fit with step=int vs step=float, verify different elimination speeds
- Verify `columns_to_keep` are never eliminated
- Verify `min_features_to_select` is respected
- Verify `transform()` returns correct subset of columns
- Verify `get_feature_set()` with all methods (best, best_coherent, best_parsimonious, int)
- Test with multiclass target
- Test with regression model

### `test_early_stopping.py`
- Fit with LGBMClassifier + early stopping params
- Verify incompatible model raises ValueError
- Verify `early_stopping_rounds` without `eval_metric` warns

### `test_sklearn_compat.py`
- Use in `sklearn.pipeline.Pipeline`
- `get_params()` / `set_params()` round-trip
- `get_support()` returns correct bool mask
- `get_feature_names_out()` returns correct feature names
- `clone()` works correctly

### `test_shap.py`
- SHAP values computed correctly for tree model
- SHAP values computed correctly for linear model
- `shap_fast_mode` uses approximate computation
- Multiclass SHAP aggregation
- Variance penalty reduces selection of noisy features

## Build & Distribution

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "recursive-pietro"
version = "0.1.0"
requires-python = ">=3.10"
```

## Out of Scope (intentionally excluded)

- Sample similarity analysis (Probatus `sample_similarity`)
- Model interpretation beyond feature elimination (Probatus `interpret`)
- Custom Scorer class (use sklearn's `make_scorer` directly)
- Notebook/documentation generation tooling
- Polars support (may be added later)
