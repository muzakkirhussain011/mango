"""Data loading utilities."""

from faircare.data.adult import load_adult
from faircare.data.heart import load_heart
from faircare.data.synth_health import generate_synthetic_health
from faircare.data.mimic_eicu import load_mimic, load_eicu


def load_dataset(
    name: str,
    sensitive_attribute: Optional[str] = None,
    **kwargs
):
    """Load dataset by name."""
    if name == "adult":
        return load_adult(sensitive_attribute=sensitive_attribute, **kwargs)
    elif name == "heart":
        return load_heart(sensitive_attribute=sensitive_attribute, **kwargs)
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
    "generate_synthetic_health",
    "load_mimic",
    "load_eicu"
]
