# faircare/core/__init__.py
"""
Core package initialisation.

We attach a `.to_dict()` method to any dataclass types found in configuration
modules so that code like `config.secure_agg.to_dict()` always works—even if the
original class didn't define it explicitly.
"""
from __future__ import annotations

from dataclasses import is_dataclass, asdict
import types
import sys


def _as_dict(self):
    if is_dataclass(self):
        return asdict(self)
    d = getattr(self, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {}


def _patch_module_dataclasses(mod: types.ModuleType):
    for name in dir(mod):
        cls = getattr(mod, name, None)
        # dataclass classes have __dataclass_fields__
        if isinstance(cls, type) and hasattr(cls, "__dataclass_fields__"):
            if not hasattr(cls, "to_dict"):
                try:
                    setattr(cls, "to_dict", _as_dict)
                except Exception:
                    pass


# Try plausible config modules
_candidates = [
    "faircare.core.config",
    "faircare.config",
    "faircare.secure_agg.config",
]

for modname in _candidates:
    try:
        __import__(modname)
        _patch_module_dataclasses(sys.modules[modname])
    except Exception:
        pass
