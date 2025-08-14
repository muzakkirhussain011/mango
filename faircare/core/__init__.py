# faircare/core/__init__.py
"""
Core package initialisation.

We install a small shim so dataclass configs (e.g., SecureAggConfig) expose a
`.to_dict()` method. This avoids AttributeError in trainer code that expects it.
"""
from __future__ import annotations

from dataclasses import is_dataclass, asdict

# Best-effort patching only if these classes exist.
try:
    from . import config as _cfg  # type: ignore
except Exception:
    _cfg = None  # type: ignore


def _as_dict(self):
    if is_dataclass(self):
        return asdict(self)
    d = getattr(self, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def _patch_to_dict(cls_name: str):
    if _cfg is None:
        return
    cls = getattr(_cfg, cls_name, None)
    if cls is not None and not hasattr(cls, "to_dict"):
        try:
            setattr(cls, "to_dict", _as_dict)
        except Exception:
            pass


for _name in ("SecureAggConfig", "FairnessConfig", "AlgoConfig", "TrainingConfig"):
    _patch_to_dict(_name)
