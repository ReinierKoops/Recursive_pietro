# recursive-pietro

SHAP-based recursive feature elimination with cross-validation and early stopping.

Drop-in sklearn-compatible replacement for Probatus `ShapRFECV` — faster, cleaner, and works in pipelines.

## Installation

```bash
pip install recursive-pietro
```

With optional boosting-library support:

```bash
pip install recursive-pietro[lightgbm]   # LightGBM early stopping
pip install recursive-pietro[xgboost]    # XGBoost early stopping
pip install recursive-pietro[catboost]   # CatBoost early stopping
pip install recursive-pietro[plot]       # matplotlib plotting
pip install recursive-pietro[all]        # everything
```

## Quick start

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from recursive_pietro import ShapFeatureElimination

X, y = make_classification(n_samples=200, n_features=15, n_informative=5, random_state=42)

selector = ShapFeatureElimination(
    RandomForestClassifier(n_estimators=50, random_state=42),
    step=0.2,
    cv=3,
    scoring="roc_auc",
    random_state=42,
)

selector.fit(X, y)

# Selected features
print(selector.selected_features_)

# Use in transform
X_reduced = selector.transform(X)
```

## Early stopping (LightGBM / XGBoost / CatBoost)

```python
from lightgbm import LGBMClassifier

selector = ShapFeatureElimination(
    LGBMClassifier(n_estimators=500, random_state=42),
    step=0.2,
    cv=5,
    scoring="roc_auc",
    early_stopping_rounds=50,
    eval_metric="auc",
)

selector.fit(X, y)
```

## Feature set selection strategies

After fitting, choose different feature sets from the elimination report:

```python
selector.get_feature_set(method="best")              # highest validation score
selector.get_feature_set(method="best_parsimonious")  # fewest features within threshold
selector.get_feature_set(method="best_coherent")      # lowest std within threshold
selector.get_feature_set(method=10)                   # exactly 10 features
```

## sklearn pipeline support

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("feature_selection", ShapFeatureElimination(
        RandomForestClassifier(n_estimators=50, random_state=42),
        step=1, cv=3, scoring="roc_auc",
    )),
    ("classifier", LogisticRegression()),
])

pipe.fit(X, y)
```

## Plotting

```bash
pip install recursive-pietro[plot]
```

```python
selector.plot()
```

## License

MIT
