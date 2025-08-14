# faircare/algos/aggregator.py
"""
Aggregator factory & registry.

Fixes included:
- Accept fairness_config as dict / dataclass / Pydantic (coerced to dict).
- Alias 'faircare_fl' → 'fairfed' for backward compatibility.
- Defensive module loading for known aggregators.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import Callable, Dict, Any

REGISTRY: Dict[str, Callable[..., Any]] = {}


def _try_register(name: str, module_name: str) -> None:
    """
    Best-effort loader that imports faircare.algos.<module_name> and registers
    a callable/class to REGISTRY[name] if found.
    """
    if name in REGISTRY:
        return
    try:
        mod = import_module(f"faircare.algos.{module_name}")
    except Exception:
        return

    # Prefer common builder/class names
    candidates = [
        "build", "make", "get",  # builder functions
        "Aggregator",            # generic class
        "FedAvgAggregator", "FedProxAggregator", "QFFLAggregator",
        "AFLAggregator", "FairFATEAggregator", "FairFedAggregator",
        "FedAvg", "FedProx", "QFFL", "AFL", "FairFATE", "FairFed",
    ]
    for cname in candidates:
        if hasattr(mod, cname):
            REGISTRY[name] = getattr(mod, cname)
            return

    # Fallback: first public callable/class in the module
    for attr in dir(mod):
        if attr.startswith("_"):
            continue
        obj = getattr(mod, attr)
        if callable(obj):
            REGISTRY[name] = obj
            return
    # If nothing suitable found, leave unregistered.


# Pre-load known aggregators (no-op if import fails)
_try_register("fedavg", "fedavg")
_try_register("fedprox", "fedprox")
_try_register("qffl", "qffl")
_try_register("afl", "afl")
_try_register("fairfate", "fairfate")
_try_register("fairfed", "fairfed")


def make_aggregator(name: str, *, fairness_config=None, **kwargs):
    """
    Factory for aggregators.

    Parameters
    ----------
    name : str
        Aggregator key. Back-compat alias: 'faircare_fl' maps to 'fairfed'.
    fairness_config : dict | dataclass | pydantic.BaseModel | None
        Extra fairness-related settings; coerced to dict and merged into kwargs.
    **kwargs : Any
        Passed to the aggregator constructor/builder.
    """
    # Back-compat name alias
    if name not in REGISTRY and name == "faircare_fl" and "fairfed" in REGISTRY:
        name = "fairfed"

    if name not in REGISTRY:
        available = list(REGISTRY.keys())
        raise ValueError(f"Unknown aggregator: {name}. Available: {available}")

    # Normalise fairness_config → dict
    if fairness_config is None:
        fairness_kwargs = {}
    elif isinstance(fairness_config, dict):
        fairness_kwargs = dict(fairness_config)
    elif is_dataclass(fairness_config):
        fairness_kwargs = asdict(fairness_config)
    else:
        # Pydantic BaseModel or objects with .dict()
        to_dict = getattr(fairness_config, "dict", None)
        fairness_kwargs = to_dict() if callable(to_dict) else {}

    all_kwargs = {**kwargs, **fairness_kwargs}
    builder = REGISTRY[name]
    return builder(**all_kwargs)


# Also expose the alias explicitly if the base is present (harmless if unused)
if "fairfed" in REGISTRY and "faircare_fl" not in REGISTRY:
    REGISTRY["faircare_fl"] = REGISTRY["fairfed"]
