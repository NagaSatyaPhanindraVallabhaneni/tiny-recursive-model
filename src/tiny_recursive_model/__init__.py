"""Tiny recursive reasoning model: <1M params, runs on any CPU."""

from tiny_recursive_model.baseline import BigBaseline
from tiny_recursive_model.model import TinyRecursiveModel, count_params

__all__ = ["TinyRecursiveModel", "BigBaseline", "count_params"]
__version__ = "0.1.0"
