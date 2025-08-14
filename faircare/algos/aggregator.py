# faircare/algos/aggregator.py
"""
Aggregator base class, registry, and factory.
This module provides:
- BaseAggregator: common options + aggregation helpers.
- register_aggregator: decorator to register aggregators.
- make_aggregator: factory that builds aggregators and merges fairness_config.
It also aliases 'faircare_fl' -> 'fairfed' for backward compatibility.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Type

import torch


# ── Registry ──────────────────────────────────────────────────────────────────
REGISTRY: Dict[str, Callable[..., "BaseAggregator"]] = {}


def register_aggregator(name: str) -> Callable[[Type["BaseAggregator"]], Type["BaseAggregator"]]:
    """Class decorator to register an aggregator under a name."""
    def _wrap(cls: Type["BaseAggregator"]) -> Type["BaseAggregator"]:
        REGISTRY[name] = lambda **kwargs: cls(**kwargs)
        return cls
    return _wrap


# ── Base class ────────────────────────────────────────────────────────────────
class BaseAggregator:
    """
    Minimal base aggregator used in tests.
    Subclasses must implement `compute_weights(client_summaries)` and should
    honour `epsilon` (weight floor), `weight_clip` (max factor vs uniform), and
    `weighted` (sample-count weighting where applicable).
    """
    def __init__(
        self,
        n_clients: int,
        weighted: bool = False,
        epsilon: float = 0.0,
        weight_clip: float = 0.0,
        fairness_metric: str = "eo_gap",
        **_: Any,
    ) -> None:
        self.n_clients = n_clients
        self.weighted = weighted
        self.epsilon = float(epsilon)
        self.weight_clip = float(weight_clip)
        self.fairness_metric = fairness_metric

    # API expected by tests
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        raise NotImplementedError

    # Helper to apply floors/clipping then renormalise
    def _postprocess(self, weights: torch.Tensor) -> torch.Tensor:
        if self.epsilon > 0:
            weights = torch.maximum(weights, torch.tensor(self.epsilon, dtype=weights.dtype))
        if self.weight_clip > 0:
            # Cap as a multiple of uniform weight (1/n)
            max_w = (1.0 / len(weights)) * self.weight_clip
            weights = torch.minimum(weights, torch.tensor(max_w, dtype=weights.dtype))
        weights = weights / weights.sum()
        return weights


# ── Known modules (ensure registration side-effects) ──────────────────────────
# Import the modules that define aggregators so their decorators run.
# Failures are non-fatal: tests only need a subset.
for mod in (
    "faircare.algos.fedavg",
    # The rest are optional; ignore if absent/partial
    "faircare.algos.fedprox",
    "faircare.algos.qffl",
    "faircare.algos.afl",
    "faircare.algos.fairfate",
    "faircare.algos.fairfed",
    "faircare.algos.faircare_fl",
):
    try:
        import_module(mod)
    except Exception:
        pass


# ── Factory ───────────────────────────────────────────────────────────────────
def _to_dict(obj: Any) -> Dict[str, Any]:
    """Coerce dataclass / pydantic / mapping-like to a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore[arg-type]
    to_dict = getattr(obj, "dict", None)
    if callable(to_dict):
        return to_dict()
    return {}


def make_aggregator(name: str, *, fairness_config: Any | None = None, **kwargs: Any) -> "BaseAggregator":
    """
    Build an aggregator by name.
    - Accepts `fairness_config` as dict / dataclass / pydantic and merges into kwargs.
    - Supports alias 'faircare_fl' → 'fairfed' if the former isn't registered.
    """
    # Back-compat alias
    if name not in REGISTRY and name == "faircare_fl" and "fairfed" in REGISTRY:
        name = "fairfed"

    if name not in REGISTRY:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(REGISTRY.keys())}")

    fairness_kwargs = _to_dict(fairness_config)
    all_kwargs = {**kwargs, **fairness_kwargs}
    builder = REGISTRY[name]
    return builder(**all_kwargs)


# Explicit alias binding as well (harmless if both are present)
if "fairfed" in REGISTRY and "faircare_fl" not in REGISTRY:
    REGISTRY["faircare_fl"] = REGISTRY["fairfed"]

# Backward-compat name expected by some imports
Aggregator = BaseAggregator
