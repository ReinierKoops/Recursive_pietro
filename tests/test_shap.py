"""Tests for SHAP computation helpers."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from recursive_pietro._shap import aggregate_shap_importance, compute_shap_values


class TestComputeShapValues:
    """Test SHAP value computation."""

    def test_basic_shap_values(self, binary_dataset):
        X, y = binary_dataset
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        shap_vals = compute_shap_values(model, X)
        assert isinstance(shap_vals, np.ndarray)
        assert shap_vals.shape == X.shape

    def test_approximate_mode(self, binary_dataset):
        X, y = binary_dataset
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        shap_vals = compute_shap_values(model, X, approximate=True)
        assert shap_vals.shape == X.shape

    def test_pipeline_raises(self, binary_dataset):
        X, y = binary_dataset
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
        ])
        pipe.fit(X, y)
        with pytest.raises(TypeError, match="Pipeline"):
            compute_shap_values(pipe, X)


class TestAggregateShapImportance:
    """Test SHAP importance aggregation."""

    def test_output_shape(self):
        rng = np.random.default_rng(42)
        shap_values = rng.standard_normal((100, 5))
        columns = ["a", "b", "c", "d", "e"]
        result = aggregate_shap_importance(shap_values, columns)
        assert result.shape == (5, 2)
        assert list(result.columns) == ["mean_abs_shap_value", "mean_shap_value"]
        assert list(result.index) == columns or set(result.index) == set(columns)

    def test_sorted_descending(self):
        rng = np.random.default_rng(42)
        shap_values = rng.standard_normal((100, 5))
        columns = ["a", "b", "c", "d", "e"]
        result = aggregate_shap_importance(shap_values, columns)
        values = result["mean_abs_shap_value"].tolist()
        # Should be sorted descending (with penalty_factor=0, order is by mean_abs)
        assert values == sorted(values, reverse=True)

    def test_penalty_changes_order(self):
        """A feature with high mean but high variance should drop with penalty."""
        # Feature 0: consistently important (low variance, mean_abs=1.0, std=0.0)
        # Feature 1: higher mean but very high variance (mean_abs=3.5, std=1.118)
        shap_values = np.array([
            [1.0, 5.0],
            [1.0, -3.0],
            [1.0, 4.0],
            [1.0, -2.0],
        ])
        columns = ["consistent", "noisy"]
        result_no_penalty = aggregate_shap_importance(shap_values, columns, penalty_factor=0.0)
        result_penalty = aggregate_shap_importance(shap_values, columns, penalty_factor=3.0)

        # Without penalty, noisy has higher mean_abs_shap
        assert result_no_penalty.index[0] == "noisy"
        # With high penalty, consistent should rank higher (penalized: 1.0 vs ~0.15)
        assert result_penalty.index[0] == "consistent"

    def test_multiclass_shap_values(self):
        """Test with 3D shap values (multiclass)."""
        rng = np.random.default_rng(42)
        # Shape: (n_classes=3, n_samples=50, n_features=4)
        shap_values = rng.standard_normal((3, 50, 4))
        columns = ["a", "b", "c", "d"]
        result = aggregate_shap_importance(shap_values, columns)
        assert result.shape == (4, 2)
