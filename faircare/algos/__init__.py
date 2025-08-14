"""Federated learning algorithms (lightweight export for tests)."""

from faircare.algos.aggregator import Aggregator, REGISTRY, make_aggregator

__all__ = [
    "Aggregator",
    "REGISTRY",
    "make_aggregator",
]
