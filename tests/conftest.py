"""Shared fixtures for recursive_pietro tests."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@pytest.fixture
def binary_dataset():
    """Binary classification dataset with 10 features (4 informative)."""
    feature_names = [f"f{i}" for i in range(1, 11)]
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=4,
        n_redundant=4,
        n_clusters_per_class=1,
        class_sep=0.05,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def multiclass_dataset():
    """Multiclass (3 classes) classification dataset."""
    feature_names = [f"f{i}" for i in range(1, 9)]
    X, y = make_classification(
        n_samples=100,
        n_features=8,
        n_informative=4,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def regression_dataset():
    """Regression dataset."""
    feature_names = [f"f{i}" for i in range(1, 9)]
    X, y = make_regression(
        n_samples=100,
        n_features=8,
        n_informative=4,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def rf_classifier():
    """Small random forest classifier for testing."""
    return RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=42,
    )


@pytest.fixture
def rf_regressor():
    """Small random forest regressor for testing."""
    return RandomForestRegressor(
        n_estimators=5,
        max_depth=3,
        random_state=42,
    )
