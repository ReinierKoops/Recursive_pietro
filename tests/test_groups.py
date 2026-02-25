"""Tests for group-aware cross-validation splits."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from recursive_pietro import ShapFeatureElimination

NJ = 1


def _make_grouped_data(random_state=42):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=4,
        n_redundant=4,
        n_clusters_per_class=1,
        class_sep=0.05,
        random_state=random_state,
    )
    columns = [f"f{i}" for i in range(1, 11)]
    X = pd.DataFrame(X, columns=columns)
    y = pd.Series(y, name="target")
    # 6 groups of 20 samples each
    groups = np.repeat(np.arange(6), 20)
    return X, y, groups


class TestGroupKFold:
    """Test that GroupKFold works end-to-end."""

    def test_fit_with_group_kfold(self):
        X, y, groups = _make_grouped_data()
        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3,
            cv=GroupKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=NJ,
            random_state=42,
        )
        sel.fit(X, y, groups=groups)
        assert len(sel.report_) > 0
        assert sel.n_features_ > 0

    def test_transform_after_group_fit(self):
        X, y, groups = _make_grouped_data()
        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3,
            cv=GroupKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=NJ,
            random_state=42,
        )
        sel.fit(X, y, groups=groups)
        X_out = sel.transform(X)
        assert X_out.shape[1] == sel.n_features_

    def test_groups_seed_stable(self):
        X, y, groups = _make_grouped_data()
        kwargs = dict(
            step=3,
            cv=GroupKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=NJ,
            random_state=42,
        )

        sel_a = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), **kwargs)
        sel_a.fit(X, y, groups=groups)

        sel_b = ShapFeatureElimination(RandomForestClassifier(n_estimators=5, max_depth=3), **kwargs)
        sel_b.fit(X, y, groups=groups)

        assert sel_a.selected_features_ == sel_b.selected_features_
        pd.testing.assert_frame_equal(sel_a.report_, sel_b.report_)


class TestStratifiedGroupKFold:
    """Test that StratifiedGroupKFold works."""

    def test_fit_with_stratified_group_kfold(self):
        X, y, groups = _make_grouped_data()
        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3,
            cv=StratifiedGroupKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=NJ,
            random_state=42,
        )
        sel.fit(X, y, groups=groups)
        assert len(sel.report_) > 0


class TestGroupValidation:
    """Test that groups are validated correctly."""

    def test_group_splitter_without_groups_raises(self):
        X, y, _ = _make_grouped_data()
        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3,
            cv=GroupKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=NJ,
        )
        with pytest.raises(ValueError, match="requires groups"):
            sel.fit(X, y)

    def test_groups_with_non_group_splitter_warns(self):
        X, y, groups = _make_grouped_data()
        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3,
            cv=2,
            scoring="roc_auc",
            n_jobs=NJ,
            random_state=42,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sel.fit(X, y, groups=groups)
            group_warnings = [x for x in w if "does not use groups" in str(x.message)]
            assert len(group_warnings) == 1

    def test_groups_as_series(self):
        X, y, groups = _make_grouped_data()
        groups_series = pd.Series(groups, name="group")
        sel = ShapFeatureElimination(
            RandomForestClassifier(n_estimators=5, max_depth=3),
            step=3,
            cv=GroupKFold(n_splits=3),
            scoring="roc_auc",
            n_jobs=NJ,
            random_state=42,
        )
        sel.fit(X, y, groups=groups_series)
        assert len(sel.report_) > 0
