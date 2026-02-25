"""Tests for the core ShapFeatureElimination class."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from recursive_pietro import ShapFeatureElimination

# Use n_jobs=1 in tests for speed and determinism.
NJ = 1


class TestBasicElimination:
    """Test the basic elimination workflow."""

    def test_fit_returns_self(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        result = sel.fit(X, y)
        assert result is sel

    def test_report_shape(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        assert isinstance(sel.report_, pd.DataFrame)
        assert len(sel.report_) > 0
        expected_cols = {
            "num_features", "features_set", "eliminated_features",
            "train_metric_mean", "train_metric_std",
            "val_metric_mean", "val_metric_std",
        }
        assert expected_cols == set(sel.report_.columns)

    def test_first_round_uses_all_features(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        assert sel.report_.iloc[0]["num_features"] == 10

    def test_features_decrease_each_round(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        n_features = sel.report_["num_features"].tolist()
        for i in range(1, len(n_features)):
            assert n_features[i] < n_features[i - 1]

    def test_min_features_respected(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(
            rf_classifier, step=3, min_features_to_select=3, cv=2, scoring="roc_auc", n_jobs=NJ,
        )
        X, y = binary_dataset
        sel.fit(X, y)
        last_round = sel.report_.iloc[-1]
        remaining_after_last = len(last_round["features_set"]) - len(last_round["eliminated_features"])
        assert remaining_after_last >= 3

    def test_step_float(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=0.3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        # With 10 features and step=0.3, first round removes floor(10*0.3)=3 features
        first_removed = len(sel.report_.iloc[0]["eliminated_features"])
        assert first_removed == 3

    def test_step_int(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        first_removed = len(sel.report_.iloc[0]["eliminated_features"])
        assert first_removed == 3


class TestTransform:
    """Test transform and feature selection output."""

    def test_transform_returns_subset(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        X_out = sel.transform(X)
        assert X_out.shape[1] == sel.n_features_
        assert X_out.shape[1] <= X.shape[1]

    def test_fit_transform(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        X_out = sel.fit_transform(X, y)
        assert X_out.shape[1] == sel.n_features_

    def test_selected_features_match_support(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        support_features = [c for c, s in zip(sel.column_names_, sel.support_) if s]
        assert support_features == sel.selected_features_


class TestColumnsToKeep:
    """Test the columns_to_keep parameter."""

    def test_kept_columns_never_eliminated(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        keep = ["f1", "f2"]
        sel.fit(X, y, columns_to_keep=keep)

        for _, row in sel.report_.iterrows():
            eliminated = row["eliminated_features"]
            for col in keep:
                assert col not in eliminated

    def test_invalid_columns_to_keep_raises(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        with pytest.raises(ValueError, match="not in X"):
            sel.fit(X, y, columns_to_keep=["nonexistent_column"])


class TestGetFeatureSet:
    """Test the get_feature_set method."""

    def test_best(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        features = sel.get_feature_set(method="best")
        assert isinstance(features, list)
        assert len(features) > 0

    def test_best_parsimonious(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        features = sel.get_feature_set(method="best_parsimonious", threshold=0.1)
        best_features = sel.get_feature_set(method="best")
        assert len(features) <= len(best_features)

    def test_int_method(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        n = sel.report_.iloc[0]["num_features"]
        features = sel.get_feature_set(method=n)
        assert len(features) == n

    def test_invalid_int_raises(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        with pytest.raises(ValueError, match="No round had exactly"):
            sel.get_feature_set(method=999)

    def test_invalid_method_raises(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        with pytest.raises(ValueError, match="method must be"):
            sel.get_feature_set(method="invalid")


class TestRegression:
    """Test with regression models."""

    def test_regression_fit(self, regression_dataset, rf_regressor):
        sel = ShapFeatureElimination(
            rf_regressor, step=3, cv=2, scoring="neg_mean_squared_error", n_jobs=NJ,
        )
        X, y = regression_dataset
        sel.fit(X, y)
        assert len(sel.report_) > 0
        assert sel.n_features_ > 0


class TestMulticlass:
    """Test with multiclass classification."""

    def test_multiclass_fit(self, multiclass_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="accuracy", n_jobs=NJ)
        X, y = multiclass_dataset
        sel.fit(X, y)
        assert len(sel.report_) > 0


class TestNumpyInput:
    """Test with numpy array input (no column names)."""

    def test_numpy_array_input(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X.values, y.values)
        assert sel.column_names_[0] == "feature_0"
        assert len(sel.report_) > 0


class TestValidation:
    """Test parameter validation."""

    def test_invalid_step_zero(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=0)
        with pytest.raises(ValueError, match="step must be"):
            sel.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))

    def test_invalid_step_negative(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=-1)
        with pytest.raises(ValueError, match="step must be"):
            sel.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))

    def test_invalid_step_float_ge_1(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=1.5)
        with pytest.raises(ValueError, match="must be in"):
            sel.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))

    def test_invalid_min_features(self, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, min_features_to_select=0)
        with pytest.raises(ValueError, match="min_features_to_select"):
            sel.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))

    def test_invalid_early_stopping_rounds(self, rf_classifier):
        sel = ShapFeatureElimination(
            rf_classifier, early_stopping_rounds=-5, eval_metric="auc",
        )
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            sel.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))

    def test_early_stopping_incompatible_model(self, rf_classifier):
        sel = ShapFeatureElimination(
            rf_classifier, early_stopping_rounds=10, eval_metric="auc",
        )
        with pytest.raises(ValueError, match="not compatible"):
            sel.fit(pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]))


class TestRanking:
    """Test feature ranking."""

    def test_ranking_shape(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        assert len(sel.ranking_) == X.shape[1]

    def test_selected_features_have_rank_1(self, binary_dataset, rf_classifier):
        sel = ShapFeatureElimination(rf_classifier, step=3, cv=2, scoring="roc_auc", n_jobs=NJ)
        X, y = binary_dataset
        sel.fit(X, y)
        for i, col in enumerate(sel.column_names_):
            if col in sel.selected_features_:
                assert sel.ranking_[i] == 1
