"""Tests for early stopping integration with boosted tree models."""

import pytest

from recursive_pietro import ShapFeatureElimination

NJ = 1


class TestEarlyStoppingLightGBM:
    """Test early stopping with LightGBM (if installed)."""

    @pytest.fixture
    def lgbm_model(self):
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)

    def test_fit_with_early_stopping(self, binary_dataset, lgbm_model):
        sel = ShapFeatureElimination(
            lgbm_model,
            step=3,
            cv=2,
            scoring="roc_auc",
            early_stopping_rounds=5,
            eval_metric="auc",
            n_jobs=NJ,
        )
        X, y = binary_dataset
        sel.fit(X, y)
        assert len(sel.report_) > 0
        assert sel.n_features_ > 0

    def test_transform_after_early_stopping(self, binary_dataset, lgbm_model):
        sel = ShapFeatureElimination(
            lgbm_model,
            step=3,
            cv=2,
            scoring="roc_auc",
            early_stopping_rounds=5,
            eval_metric="auc",
            n_jobs=NJ,
        )
        X, y = binary_dataset
        sel.fit(X, y)
        X_out = sel.transform(X)
        assert X_out.shape[1] == sel.n_features_


class TestEarlyStoppingXGBoost:
    """Test early stopping with XGBoost (if installed)."""

    @pytest.fixture
    def xgb_model(self):
        pytest.importorskip("xgboost")
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=50, max_depth=3, random_state=42, verbosity=0,
        )

    def test_fit_with_early_stopping(self, binary_dataset, xgb_model):
        sel = ShapFeatureElimination(
            xgb_model,
            step=3,
            cv=2,
            scoring="roc_auc",
            early_stopping_rounds=5,
            eval_metric="logloss",
            n_jobs=NJ,
        )
        X, y = binary_dataset
        sel.fit(X, y)
        assert len(sel.report_) > 0


class TestEarlyStoppingValidation:
    """Test validation of early stopping parameters."""

    def test_warning_without_eval_metric(self, binary_dataset):
        pytest.importorskip("lightgbm")
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=50, verbose=-1)
        sel = ShapFeatureElimination(
            model,
            step=3,
            cv=2,
            scoring="roc_auc",
            early_stopping_rounds=5,
            eval_metric=None,
            n_jobs=NJ,
        )
        X, y = binary_dataset
        with pytest.warns(UserWarning, match="eval_metric is None"):
            sel.fit(X, y)
