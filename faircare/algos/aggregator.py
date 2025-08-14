# faircare/algos/aggregator.py
"""
Aggregator base class, registry, and factory.
Also provides a numerically robust weight-flooring routine that preserves the
final constraint `w_i >= epsilon` after normalisation.
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
        self.weighted = bool(weighted)
        self.epsilon = float(epsilon)
        self.weight_clip = float(weight_clip)
        self.fairness_metric = fairness_metric

    # API expected by tests
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        raise NotImplementedError

    # Helper to (i) apply a LOWER FLOOR that remains after normalisation,
    # and (ii) optionally clip to an UPPER bound (multiple of the uniform weight).
    #
    # Lower-floor part uses a bounded simplex projection idea: allocate epsilon
    # to everyone, then renormalise ONLY the "slack" above epsilon so that
    # sum(weights)=1 and each weight >= epsilon. (See probability-simplex
    # projection with bound constraints for the general form.) :contentReference[oaicite:3]{index=3}
    def _postprocess(self, weights: torch.Tensor) -> torch.Tensor:
        w = weights.to(dtype=torch.float32)
        n = w.numel()
        if n == 0:
            return w

        # ── Lower bound handling (epsilon) ────────────────────────────────────
        eps = max(0.0, self.epsilon)
        if eps > 0.0:
            # If infeasible (n*eps > 1), fall back to uniform.
            if eps * n >= 1.0:
                w = torch.ones_like(w) / n
            else:
                # Allocate the mandatory floor to each weight.
                base = torch.full_like(w, eps)
                # Raw slack above the floor (negative slacks set to 0)
                raw = torch.clamp(w - eps, min=0.0)
                s = raw.sum()
                target_slack_sum = 1.0 - eps * n
                if s <= 0:
                    # Everyone exactly at the floor; distribute uniformly
                    extra = torch.zeros_like(w)
                else:
                    extra = raw * (target_slack_sum / s)
                w = base + extra
        else:
            # Just renormalise if needed
            s = w.sum()
            if s > 0:
                w = w / s
            else:
                w = torch.ones_like(w) / n

        # ── Optional upper bound: cap at c * (1/n) if weight_clip > 0 ─────────
        if self.weight_clip and self.weight_clip > 0.0:
            cap = self.weight_clip * (1.0 / n)
            # Iterative clipping: cap and then renormalise remaining mass
            # while preserving lower floors (if any).
            # This simple loop converges in at most n steps for test sizes.
            for _ in range(n):
                over = w > cap
                if not torch.any(over):
                    break
                excess = (w[over] - cap).sum()
                w[over] = cap
                # Redistribute the excess to the non-capped entries while
                # keeping their current lower floors intact.
                free = ~over
                if torch.any(free):
                    w[free] = w[free] + excess * (w[free] / w[free].sum())
            # Final normalisation (small numerical drift)
            w = w / w.sum()

        return w


# ── Known modules (ensure registration side-effects) ──────────────────────────
for mod in (
    "faircare.algos.fedavg",
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
        # Official: dataclasses.asdict → dict of field names to values. :contentReference[oaicite:4]{index=4}
        return asdict(obj)
    to_dict = getattr(obj, "to_dict", None) or getattr(obj, "dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except TypeError:
            return to_dict  # in case it's a @property returning a dict
    # Fallback
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def make_aggregator(name: str, *, fairness_config: Any | None = None, **kwargs: Any) -> "BaseAggregator":
    """
    Build an aggregator by name.
    - Accepts `fairness_config` as dict / dataclass / Pydantic-like and merges into kwargs.
    - Supports alias 'faircare_fl' → 'fairfed' if the former isn't registered.
    """
    if name not in REGISTRY and name == "faircare_fl" and "fairfed" in REGISTRY:
        name = "fairfed"

    if name not in REGISTRY:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(REGISTRY.keys())}")

    fairness_kwargs = _to_dict(fairness_config)
    all_kwargs = {**kwargs, **fairness_kwargs}
    builder = REGISTRY[name]
    return builder(**all_kwargs)


# Explicit alias binding (harmless if both are present)
if "fairfed" in REGISTRY and "faircare_fl" not in REGISTRY:
    REGISTRY["faircare_fl"] = REGISTRY["fairfed"]

# Backward-compat name expected by some imports
Aggregator = BaseAggregator
