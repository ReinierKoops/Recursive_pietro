"""recursive-pietro: SHAP-based recursive feature elimination with early stopping."""

from importlib.metadata import version

from recursive_pietro._elimination import ShapFeatureElimination

__all__ = ["ShapFeatureElimination"]
__version__ = version("recursive-pietro")
