"""Tests for sklearn API compatibility."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recursive_pietro import ShapFeatureElimination


class TestGetSetParams:
    """Test sklearn get_params / set_params."""

    def test_get_params(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=5)
        params = sel.get_params()
        assert params["step"] == 3
        assert params["cv"] == 5
        assert params["model"] is rf_classifier

    def test_set_params(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3)
        sel.set_params(step=5, cv=10)
        assert sel.step == 5
        assert sel.cv == 10

    def test_clone(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=5, scoring="roc_auc")
        sel_clone = clone(sel)
        assert sel_clone.step == 3
        assert sel_clone.cv == 5
        assert sel_clone is not sel


class TestSupport:
    """Test support mask and feature names."""

    def test_get_support(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=1)
        X, y = binary_dataset
        sel.fit(X, y)
        support = sel.get_support()
        assert isinstance(support, np.ndarray)
        assert support.dtype == bool
        assert support.shape == (X.shape[1],)
        assert support.sum() == sel.n_features_

    def test_get_feature_names_out(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=1)
        X, y = binary_dataset
        sel.fit(X, y)
        names = sel.get_feature_names_out()
        assert len(names) == sel.n_features_
        assert all(n in X.columns for n in names)


class TestPipeline:
    """Test usage within sklearn Pipeline."""

    def test_pipeline_fit_transform(self, binary_dataset, rf_classifier):
        """ShapFeatureElimination works as a step in a Pipeline.

        Note: The scaler runs first, then the selector receives numeric data.
        We use the selector as the last step so Pipeline.fit_transform works.
        """
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=1)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", sel),
        ])
        X, y = binary_dataset
        X_out = pipe.fit_transform(X, y)
        assert X_out.shape[1] <= X.shape[1]
        assert X_out.shape[0] == X.shape[0]


class TestNotFittedError:
    """Test that accessing attributes before fit raises errors."""

    def test_transform_before_fit(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, n_jobs=1)
        X, _ = binary_dataset
        with pytest.raises(Exception):  # NotFittedError
            sel.transform(X)

    def test_get_support_before_fit(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, n_jobs=1)
        with pytest.raises(Exception):  # NotFittedError
            sel.get_support()
