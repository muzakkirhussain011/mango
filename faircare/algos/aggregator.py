# faircare/algos/aggregator.py
"""
Aggregator base class, registry, and factory.
Includes numerically robust weight flooring that keeps w_i >= epsilon after
normalisation, and optional upper clipping as a multiple of the uniform weight.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Type

import torch


# ── Registry ──────────────────────────────────────────────────────────────────
REGISTRY: Dict[str, Callable[..., "BaseAggregator"]] = {}


def register_aggregator(name: str) -> Callable[[Type["BaseAggregator"]], Type["BaseAggregator"]]:
    def _wrap(cls: Type["BaseAggregator"]) -> Type["BaseAggregator"]:
        REGISTRY[name] = lambda **kwargs: cls(**kwargs)
        return cls
    return _wrap


# ── Base class ────────────────────────────────────────────────────────────────
class BaseAggregator:
    """
    Minimal base aggregator used in tests.
    Subclasses must implement `compute_weights(client_summaries)` and should
    honour:
      - `epsilon` (lower weight floor),
      - `weight_clip` (upper cap as multiple of uniform),
      - `weighted` (use sample-count weighting where applicable).
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

    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        raise NotImplementedError

    def _postprocess(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Enforce lower floor (epsilon) and optional upper clip (weight_clip * 1/n),
        while preserving sum(weights) == 1.

        Lower floor approach: allocate epsilon to all, then scale only the slack
        so the total remains 1 (a bounded simplex projection idea).
        """
        w = weights.to(dtype=torch.float32)
        n = w.numel()
        if n == 0:
            return w

        eps = max(0.0, self.epsilon)
        if eps > 0.0:
            if eps * n >= 1.0:
                # Infeasible: fall back to uniform
                w = torch.ones_like(w) / n
            else:
                base = torch.full_like(w, eps)
                raw = torch.clamp(w - eps, min=0.0)
                s = raw.sum()
                target = 1.0 - eps * n
                extra = torch.zeros_like(w) if s <= 0 else raw * (target / s)
                w = base + extra
        else:
            s = w.sum()
            w = (w / s) if s > 0 else (torch.ones_like(w) / n)

        if self.weight_clip and self.weight_clip > 0.0:
            cap = self.weight_clip * (1.0 / n)
            for _ in range(n):
                over = w > cap
                if not torch.any(over):
                    break
                excess = (w[over] - cap).sum()
                w[over] = cap
                free = ~over
                if torch.any(free):
                    w[free] = w[free] + excess * (w[free] / w[free].sum())
            w = w / w.sum()

        return w


# ── Ensure aggregators register themselves on import ──────────────────────────
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
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    to_dict = getattr(obj, "to_dict", None) or getattr(obj, "dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except TypeError:
            return to_dict  # property returning a dict
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def make_aggregator(name: str, *, fairness_config: Any | None = None, **kwargs: Any) -> "BaseAggregator":
    """
    Build an aggregator by name.
    - Accepts `fairness_config` as dict/dataclass/Pydantic-like and merges it.
    - IMPORTANT: caller kwargs take precedence over fairness_config, so flags like
      `weighted=True` are never overridden by defaults inside fairness_config.
    - Alias: 'faircare_fl' → 'fairfed' if the former isn't registered.
    """
    if name not in REGISTRY and name == "faircare_fl" and "fairfed" in REGISTRY:
        name = "fairfed"

    if name not in REGISTRY:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(REGISTRY.keys())}")

    fairness_kwargs = _to_dict(fairness_config)
    all_kwargs = {**fairness_kwargs, **kwargs}  # ← kwargs win
    builder = REGISTRY[name]
    return builder(**all_kwargs)


if "fairfed" in REGISTRY and "faircare_fl" not in REGISTRY:
    REGISTRY["faircare_fl"] = REGISTRY["fairfed"]

Aggregator = BaseAggregator
