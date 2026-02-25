"""Tests for seed stability and column-order invariance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from recursive_pietro import ShapFeatureElimination

NJ = 1


def _make_data(random_state=42):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=4,
        n_redundant=4, n_clusters_per_class=1, class_sep=0.05,
        random_state=random_state,
    )
    columns = [f"f{i}" for i in range(1, 11)]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


class TestSeedStability:
    """Two runs with the same random_state must produce identical results."""

    def test_same_seed_same_results(self):
        X, y = _make_data()
        kwargs = dict(step=3, cv=2, scoring="roc_auc", n_jobs=NJ, random_state=42)

        sel_a = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), **kwargs)
        sel_a.fit(X, y)

        sel_b = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), **kwargs)
        sel_b.fit(X, y)

        assert sel_a.selected_features_ == sel_b.selected_features_
        np.testing.assert_array_equal(sel_a.ranking_, sel_b.ranking_)
        pd.testing.assert_frame_equal(sel_a.report_, sel_b.report_)

    def test_different_seed_may_differ(self):
        X, y = _make_data()
        kwargs = dict(step=3, cv=2, scoring="roc_auc", n_jobs=NJ)

        sel_a = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), random_state=1, **kwargs)
        sel_a.fit(X, y)

        sel_b = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), random_state=99, **kwargs)
        sel_b.fit(X, y)

        # With different seeds, at least the scores should differ
        a_scores = sel_a.report_["val_metric_mean"].tolist()
        b_scores = sel_b.report_["val_metric_mean"].tolist()
        assert a_scores != b_scores

    def test_model_without_random_state_still_seeded(self):
        """Even if the user's model has no explicit random_state, we seed it."""
        X, y = _make_data()
        # Model with random_state=None — _seed_model should set it from fold_seed
        model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=None)
        kwargs = dict(step=3, cv=2, scoring="roc_auc", n_jobs=NJ, random_state=42)

        sel_a = ShapFeatureElimination(model, **kwargs)
        sel_a.fit(X, y)

        sel_b = ShapFeatureElimination(model, **kwargs)
        sel_b.fit(X, y)

        assert sel_a.selected_features_ == sel_b.selected_features_
        pd.testing.assert_frame_equal(sel_a.report_, sel_b.report_)


class TestColumnOrderInvariance:
    """Results must be identical regardless of the input column order."""

    def test_shuffled_columns_same_result(self):
        X, y = _make_data()
        kwargs = dict(step=3, cv=2, scoring="roc_auc", n_jobs=NJ, random_state=42)

        sel_a = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), **kwargs)
        sel_a.fit(X, y)

        # Shuffle columns
        rng = np.random.RandomState(99)
        shuffled_cols = list(X.columns)
        rng.shuffle(shuffled_cols)
        X_shuffled = X[shuffled_cols]

        sel_b = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), **kwargs)
        sel_b.fit(X_shuffled, y)

        assert sorted(sel_a.selected_features_) == sorted(sel_b.selected_features_)
        # column_names_ should be sorted identically
        assert sel_a.column_names_ == sel_b.column_names_
        pd.testing.assert_frame_equal(sel_a.report_, sel_b.report_)

    def test_sort_columns_disabled(self):
        """When sort_columns=False, column order is preserved as-is."""
        X, y = _make_data()
        rng = np.random.RandomState(99)
        shuffled_cols = list(X.columns)
        rng.shuffle(shuffled_cols)
        X_shuffled = X[shuffled_cols]

        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3, cv=2, scoring="roc_auc", n_jobs=NJ, random_state=42,
            sort_columns=False,
        )
        sel.fit(X_shuffled, y)

        # column_names_ should match the input order, not sorted
        assert sel.column_names_ == shuffled_cols
