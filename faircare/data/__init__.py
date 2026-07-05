"""Data loading utilities."""

from typing import Dict, List, Optional, Tuple, Any, Union, Protocol
from faircare.data.adult import load_adult
from faircare.data.heart import load_heart
from faircare.data.synth_health import generate_synthetic_health
from faircare.data.diabetes import load_diabetes130
from faircare.data.compas import load_compas
from faircare.data.mimic_eicu import load_mimic, load_eicu


def load_dataset(
    name: str,
    sensitive_attribute: Optional[str] = None,
    **kwargs
):
    """Load a dataset by name. Returns the standard Dict contract:
    {"train","val","test","n_features","n_classes","sensitive_attribute", ...}.

    Real data: adult, heart, diabetes130, compas (downloaded + cached under ~/.faircare/data).
    Synthetic: synth_health (controlled bias). mimic/eicu are STUBS that fall back to
    synth_health — they are NOT real ICU data (credentialed access required).
    """
    if name == "adult":
        return load_adult(sensitive_attribute=sensitive_attribute, **kwargs)
    elif name == "heart":
        return load_heart(sensitive_attribute=sensitive_attribute, **kwargs)
    elif name == "diabetes130":
        return load_diabetes130(sensitive_attribute=sensitive_attribute, **kwargs)
    elif name == "compas":
        return load_compas(sensitive_attribute=sensitive_attribute, **kwargs)
    elif name == "synth_health":
        return generate_synthetic_health(**kwargs)
    elif name == "mimic":
        return load_mimic(**kwargs)
    elif name == "eicu":
        return load_eicu(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "load_dataset",
    "load_adult",
    "load_heart",
    "load_diabetes130",
    "load_compas",
    "generate_synthetic_health",
    "load_mimic",
    "load_eicu"
]
