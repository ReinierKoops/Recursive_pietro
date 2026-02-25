"""ShapFeatureElimination — SHAP-based recursive feature elimination with CV."""

from __future__ import annotations

import logging
import warnings
from math import floor

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils.validation import check_is_fitted

from recursive_pietro._compat import (
    get_early_stopping_fit_params,
    is_early_stopping_compatible,
)
from recursive_pietro._shap import aggregate_shap_importance, compute_shap_values
from recursive_pietro._validation import (
    validate_columns_to_keep,
    validate_data,
    validate_groups,
    validate_min_features,
    validate_sample_weight,
    validate_shap_variance_penalty,
    validate_step,
    validate_target,
)

logger = logging.getLogger(__name__)


def _seed_model(model, seed: int | None) -> None:
    """Set ``random_state`` on *model* in-place if it accepts one and *seed* is not None."""
    if seed is None:
        return
    try:
        params = model.get_params(deep=False)
    except AttributeError:
        return
    if "random_state" in params:
        model.set_params(random_state=seed)


class ShapFeatureElimination(SelectorMixin, BaseEstimator):
    """Backward recursive feature elimination using SHAP importance with cross-validation.

    At each round the algorithm:

    1. (Optional) Tunes hyperparameters via an sklearn SearchCV wrapper.
    2. Runs cross-validation, fitting the model on train folds and computing
       SHAP feature importance on validation folds (with optional early stopping
       for LightGBM / XGBoost / CatBoost).
    3. Removes the ``step`` lowest-importance features.

    The process repeats until ``min_features_to_select`` features remain.
    Results from every round are stored in ``report_`` so the user can choose
    the optimal feature set post-hoc.

    This class is fully sklearn-compatible: it inherits from
    ``BaseEstimator`` and ``SelectorMixin`` so it works in ``Pipeline``,
    supports ``get_params`` / ``set_params``, ``get_support``, and
    ``get_feature_names_out``.

    Parameters
    ----------
    model : estimator or sklearn SearchCV
        A sklearn-compatible classifier or regressor, or a hyperparameter
        search wrapper (``GridSearchCV``, ``RandomizedSearchCV``, etc.).
    step : int or float, default=1
        Features to remove per round.  If int, remove exactly that many.
        If float in (0, 1), remove that fraction of remaining features
        (rounded down, minimum 1).
    min_features_to_select : int, default=1
        Minimum features to keep.  The loop stops when this is reached.
    cv : int, CV splitter, or iterable, default=5
        Cross-validation strategy (passed to ``sklearn.model_selection.check_cv``).
    scoring : str or callable, default="roc_auc"
        Metric used to evaluate model performance per fold.
    n_jobs : int, default=-1
        Number of parallel jobs for CV folds (``joblib``).
    random_state : int or None, default=None
        Random seed for reproducibility.
    early_stopping_rounds : int or None, default=None
        Boosting early-stopping patience (LightGBM / XGBoost / CatBoost only).
    eval_metric : str or None, default=None
        Metric for the model's internal early-stopping evaluation.
    shap_variance_penalty_factor : float or None, default=None
        Penalise high-variance SHAP features when ranking.
        ``penalized = mean_abs_shap - std_abs_shap * factor``.
    shap_fast_mode : bool, default=False
        Use approximate SHAP values (``TreeExplainer`` only).
    sort_columns : bool, default=True
        Sort feature columns alphabetically before fitting so that results
        are invariant to the input column order.
    verbose : int, default=0
        0 = silent, 1 = progress bar, 2 = detailed logging per round.

    Attributes
    ----------
    report_ : pd.DataFrame
        Per-round results (num_features, scores, feature sets).
    selected_features_ : list[str]
        Feature names corresponding to the best validation score.
    support_ : np.ndarray of bool
        Boolean mask over the original feature set.
    ranking_ : np.ndarray of int
        Feature ranking (1 = selected, higher = eliminated earlier).
    n_features_ : int
        Number of selected features.
    column_names_ : list[str]
        Original feature names seen during ``fit``.
    """

    def __init__(
        self,
        model,
        *,
        step: int | float = 1,
        min_features_to_select: int = 1,
        cv: int | object = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        random_state: int | None = None,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        shap_variance_penalty_factor: float | None = None,
        shap_fast_mode: bool = False,
        sort_columns: bool = True,
        verbose: int = 0,
    ):
        self.model = model
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.shap_variance_penalty_factor = shap_variance_penalty_factor
        self.shap_fast_mode = shap_fast_mode
        self.sort_columns = sort_columns
        self.verbose = verbose

    # ------------------------------------------------------------------
    # sklearn SelectorMixin interface
    # ------------------------------------------------------------------

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def transform(self, X):
        """Reduce X to the selected features.

        When ``sort_columns=True`` (default), columns are sorted to match
        the order seen during ``fit`` before applying the feature mask.
        """
        check_is_fitted(self, "support_")
        if self.sort_columns and isinstance(X, pd.DataFrame):
            X = X[sorted(X.columns)]
        return super().transform(X)

    # ------------------------------------------------------------------
    # fit / transform
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        y,
        *,
        groups=None,
        sample_weight=None,
        columns_to_keep: list[str] | None = None,
        column_names: list[str] | None = None,
        **shap_kwargs,
    ):
        """Run the recursive feature elimination.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : array-like
            Target vector.
        groups : array-like or None
            Group labels for group-aware CV splitters.
        sample_weight : array-like or None
            Per-sample weights passed to the model's ``fit``.
        columns_to_keep : list[str] or None
            Columns that must never be eliminated.
        column_names : list[str] or None
            Override column names for X.
        **shap_kwargs
            Extra keyword arguments forwarded to ``shap.Explainer``.
            Supports ``check_additivity`` and ``feature_perturbation``.

        Returns
        -------
        self
        """
        # --- Validate constructor params ---
        step = validate_step(self.step)
        min_features = validate_min_features(self.min_features_to_select)
        penalty_factor = validate_shap_variance_penalty(self.shap_variance_penalty_factor)

        if self.early_stopping_rounds is not None:
            if not isinstance(self.early_stopping_rounds, int) or self.early_stopping_rounds <= 0:
                raise ValueError(f"early_stopping_rounds must be a positive int, got {self.early_stopping_rounds!r}.")
            if self.eval_metric is None:
                warnings.warn(
                    "early_stopping_rounds is set but eval_metric is None. "
                    "Both are required for early stopping with LightGBM/XGBoost/CatBoost."
                )
            if not is_early_stopping_compatible(self.model):
                raise ValueError(
                    "early_stopping_rounds is set but model is not compatible. "
                    "Only LightGBM, XGBoost, and CatBoost support early stopping."
                )

        use_early_stopping = self.early_stopping_rounds is not None and self.eval_metric is not None

        # --- Reproducible seed sequence ---
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = None

        # --- Validate data ---
        X_df = validate_data(X, column_names=column_names)
        if self.sort_columns:
            X_df = X_df[sorted(X_df.columns)]
        self.column_names_ = list(X_df.columns)
        self.feature_names_in_ = np.array(self.column_names_)
        y_s = validate_target(y, X_df.index)
        sw = validate_sample_weight(sample_weight, X_df.index)
        columns_to_keep = validate_columns_to_keep(columns_to_keep, self.column_names_)

        is_search = isinstance(self.model, BaseSearchCV)
        scorer = check_scoring(self.model, scoring=self.scoring)
        cv_splitter = check_cv(self.cv, y_s, classifier=is_classifier(self.model))
        groups = validate_groups(groups, X_df.index, cv_splitter)

        # Stopping floor: at least keep columns_to_keep
        n_keep = len(columns_to_keep) if columns_to_keep else 0
        stopping = max(min_features, n_keep)

        # --- Progress bar (lazy import) ---
        progress = None
        if self.verbose == 1:
            try:
                from tqdm.auto import tqdm

                # Rough estimate of total rounds
                n_rounds = self._estimate_rounds(len(self.column_names_), step, stopping)
                progress = tqdm(total=n_rounds, desc="Feature elimination", unit="round")
            except ImportError:
                pass

        # --- Main elimination loop ---
        remaining_features = list(self.column_names_)
        results: list[dict] = []
        round_number = 0

        while len(remaining_features) > stopping:
            round_number += 1

            current_features = list(remaining_features)
            # Build the working column set (features + columns_to_keep, deduplicated, order-preserved)
            working_columns = list(dict.fromkeys(current_features + (columns_to_keep or [])))
            current_X = X_df[working_columns]

            # Optional hyperparameter tuning
            if is_search:
                search_clone = clone(self.model)
                if rng is not None:
                    _seed_model(search_clone, rng.randint(0, 2**31))
                search_clone.fit(X=current_X, y=y_s, groups=groups)
                current_model = search_clone.estimator.set_params(**search_clone.best_params_)
            else:
                current_model = clone(self.model)

            # Derive per-fold seeds for deterministic parallel execution
            splits = list(cv_splitter.split(current_X, y_s, groups))
            if rng is not None:
                fold_seeds = rng.randint(0, 2**31, size=len(splits)).tolist()
            else:
                fold_seeds = [None] * len(splits)

            # CV: compute SHAP + scores per fold
            if use_early_stopping:
                fold_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fold_early_stopping)(
                        current_X,
                        y_s,
                        current_model,
                        train_idx,
                        val_idx,
                        sw,
                        scorer,
                        fold_seed,
                        **shap_kwargs,
                    )
                    for (train_idx, val_idx), fold_seed in zip(splits, fold_seeds)
                )
            else:
                fold_results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fold_standard)(
                        current_X,
                        y_s,
                        current_model,
                        train_idx,
                        val_idx,
                        sw,
                        scorer,
                        fold_seed,
                        **shap_kwargs,
                    )
                    for (train_idx, val_idx), fold_seed in zip(splits, fold_seeds)
                )

            # Aggregate SHAP values across folds
            # SHAP returns 2D (n_samples, n_features) for binary/regression/modern multiclass,
            # or 3D (n_classes, n_samples, n_features) for older multiclass SHAP versions.
            first_shap = fold_results[0][0]
            if first_shap.ndim == 2:
                shap_values = np.concatenate([r[0] for r in fold_results], axis=0)
            else:  # 3D: concat along samples axis (axis=1)
                shap_values = np.concatenate([r[0] for r in fold_results], axis=1)

            scores_train = [r[1] for r in fold_results]
            scores_val = [r[2] for r in fold_results]

            importance_df = aggregate_shap_importance(
                shap_values,
                working_columns,
                penalty_factor=penalty_factor,
            )

            # Determine features to remove
            # When columns_to_keep is active, the removable set should be able
            # to go to zero (the stopping criteria already accounts for kept columns).
            effective_min = 0 if columns_to_keep else min_features
            features_to_remove = self._pick_features_to_remove(
                importance_df,
                step,
                effective_min,
                columns_to_keep,
            )
            remove_set = set(features_to_remove)
            remaining_features = [f for f in current_features if f not in remove_set]

            # Record round
            results.append(
                {
                    "num_features": len(current_features),
                    "features_set": current_features,
                    "eliminated_features": features_to_remove,
                    "train_metric_mean": float(np.mean(scores_train)),
                    "train_metric_std": float(np.std(scores_train)),
                    "val_metric_mean": float(np.mean(scores_val)),
                    "val_metric_std": float(np.std(scores_val)),
                }
            )

            if self.verbose >= 2:
                logger.info(
                    "Round %d: %d features, val=%.4f +/- %.4f, removed %s",
                    round_number,
                    len(current_features),
                    results[-1]["val_metric_mean"],
                    results[-1]["val_metric_std"],
                    features_to_remove,
                )
            if progress is not None:
                progress.update(1)

        if progress is not None:
            progress.close()

        # --- Build report ---
        self.report_ = pd.DataFrame(results, index=range(1, len(results) + 1))

        # --- Derive sklearn-compatible attributes ---
        self._set_selection_attributes()

        return self

    # ------------------------------------------------------------------
    # Feature set selection
    # ------------------------------------------------------------------

    def get_feature_set(
        self,
        method: str | int = "best",
        threshold: float = 1.0,
    ) -> list[str]:
        """Return a feature set from the elimination report.

        Parameters
        ----------
        method : str or int
            - ``"best"``: highest validation score.
            - ``"best_coherent"``: within ``threshold`` of best score,
              pick the round with lowest validation std.
            - ``"best_parsimonious"``: within ``threshold`` of best score,
              pick the round with fewest features.
            - int: return the feature set from the round with exactly
              that many features.
        threshold : float
            Score margin for coherent / parsimonious methods.

        Returns
        -------
        list[str]
        """
        check_is_fitted(self, "report_")
        report = self.report_

        if isinstance(method, (int, np.integer)):
            method = int(method)
            match = report[report["num_features"] == method]
            if match.empty:
                valid = sorted(report["num_features"].unique())
                raise ValueError(f"No round had exactly {method} features. Available: {valid}")
            return match.iloc[0]["features_set"]

        if method == "best":
            idx = report["val_metric_mean"].idxmax()
        elif method == "best_coherent":
            best_score = report["val_metric_mean"].max()
            within = report[report["val_metric_mean"] >= best_score - threshold]
            idx = within["val_metric_std"].idxmin()
        elif method == "best_parsimonious":
            best_score = report["val_metric_mean"].max()
            within = report[report["val_metric_mean"] >= best_score - threshold]
            idx = within["num_features"].idxmin()
        else:
            raise ValueError(f"method must be 'best', 'best_coherent', 'best_parsimonious', or int; got {method!r}")

        return report.loc[idx, "features_set"]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, *, show: bool = True, **figure_kwargs):
        """Plot train/validation performance across elimination rounds.

        Requires matplotlib (optional dependency).

        Returns
        -------
        matplotlib.figure.Figure
        """
        check_is_fitted(self, "report_")
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install recursive-pietro[plot]"
            ) from e

        report = self.report_
        n_feat = report["num_features"]
        fig, ax = plt.subplots(**figure_kwargs)

        ax.plot(n_feat, report["train_metric_mean"], label="Train")
        ax.fill_between(
            n_feat,
            report["train_metric_mean"] - report["train_metric_std"],
            report["train_metric_mean"] + report["train_metric_std"],
            alpha=0.3,
        )
        ax.plot(n_feat, report["val_metric_mean"], label="Validation")
        ax.fill_between(
            n_feat,
            report["val_metric_mean"] - report["val_metric_std"],
            report["val_metric_mean"] + report["val_metric_std"],
            alpha=0.3,
        )

        ax.set_xlabel("Number of features")
        ax.set_ylabel(f"Score ({self.scoring})")
        ax.set_title("SHAP Recursive Feature Elimination")
        ax.legend(loc="lower left")
        ax.invert_xaxis()
        ax.set_xticks(list(reversed(n_feat.tolist())))

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fold_standard(
        self,
        X,
        y,
        model,
        train_idx,
        val_idx,
        sample_weight,
        scorer,
        fold_seed,
        **shap_kwargs,
    ):
        """Fit model, score, compute SHAP — standard (no early stopping) path."""
        model = clone(model)
        _seed_model(model, fold_seed)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.iloc[train_idx]

        model.fit(X_train, y_train, **fit_kwargs)

        score_train = scorer(model, X_train, y_train)
        score_val = scorer(model, X_val, y_val)

        shap_vals = compute_shap_values(
            model,
            X_val,
            approximate=self.shap_fast_mode,
            random_state=fold_seed,
            verbose=self.verbose,
            **shap_kwargs,
        )
        return shap_vals, score_train, score_val

    def _fold_early_stopping(
        self,
        X,
        y,
        model,
        train_idx,
        val_idx,
        sample_weight,
        scorer,
        fold_seed,
        **shap_kwargs,
    ):
        """Fit model with early stopping, score, compute SHAP."""
        model = clone(model)
        _seed_model(model, fold_seed)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fit_params = get_early_stopping_fit_params(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric=self.eval_metric,
            sample_weight=sample_weight,
            train_index=train_idx,
            val_index=val_idx,
            verbose=self.verbose,
        )
        model.fit(**fit_params)

        score_train = scorer(model, X_train, y_train)
        score_val = scorer(model, X_val, y_val)

        shap_vals = compute_shap_values(
            model,
            X_val,
            approximate=self.shap_fast_mode,
            random_state=fold_seed,
            verbose=self.verbose,
            **shap_kwargs,
        )
        return shap_vals, score_train, score_val

    @staticmethod
    def _pick_features_to_remove(
        importance_df: pd.DataFrame,
        step: int | float,
        min_features: int,
        columns_to_keep: list[str] | None,
    ) -> list[str]:
        """Select the least important features to remove this round."""
        # Exclude protected columns from candidates
        if columns_to_keep:
            mask = importance_df.index.isin(columns_to_keep)
            candidates = importance_df[~mask]
        else:
            candidates = importance_df

        n_candidates = len(candidates)

        if isinstance(step, int):
            n_remove = step
        else:
            n_remove = max(1, int(floor(n_candidates * step)))

        # Don't drop below min_features (among removable features)
        max_removable = n_candidates - min_features
        n_remove = max(0, min(n_remove, max_removable))

        if n_remove == 0:
            return []

        # Bottom n_remove features (lowest importance) — importance_df is sorted descending
        return candidates.iloc[-n_remove:].index.tolist()

    @staticmethod
    def _estimate_rounds(n_features: int, step: int | float, stopping: int) -> int:
        """Rough estimate of total elimination rounds (for progress bar)."""
        n = n_features
        rounds = 0
        while n > stopping:
            rounds += 1
            if isinstance(step, int):
                n -= max(1, min(step, n - stopping))
            else:
                remove = max(1, int(floor(n * step)))
                n -= max(1, min(remove, n - stopping))
        return max(rounds, 1)

    def _set_selection_attributes(self):
        """Compute support_, ranking_, selected_features_, n_features_ from report_."""
        report = self.report_
        best_idx = report["val_metric_mean"].idxmax()
        self.selected_features_ = report.loc[best_idx, "features_set"]
        self.n_features_ = len(self.selected_features_)

        selected_set = set(self.selected_features_)
        self.support_ = np.array([c in selected_set for c in self.column_names_])

        # Build ranking: 1 = selected, higher = eliminated earlier in time.
        # Walk elimination rounds in order. Features eliminated before the best
        # round are NOT selected → they get a rank based on elimination order.
        # Features in the selected set always get rank 1.
        best_num = report.loc[best_idx, "num_features"]
        eliminated_before_best = []
        for _, row in report.iterrows():
            if row["num_features"] > best_num:
                eliminated_before_best.extend(row["eliminated_features"])

        ranking = {}
        for feat in selected_set:
            ranking[feat] = 1
        # Eliminated features: first eliminated = highest rank
        # (eliminated_before_best is in chronological order, first eliminated = earliest)
        for i, feat in enumerate(reversed(eliminated_before_best)):
            ranking[feat] = 2 + i

        # Any remaining features not yet ranked (shouldn't happen, but defensive)
        for feat in self.column_names_:
            if feat not in ranking:
                ranking[feat] = 1

        self.ranking_ = np.array([ranking[c] for c in self.column_names_])
