"""
FairCare-FL Configuration Module

Provides comprehensive configuration system for FairCare-FL enhancements.
All features are feature-flagged with backward compatibility (defaults OFF).
"""

from .faircare_fl_config import (
    FairCareFLConfig,
    MultiObjectiveConfig,
    FairSelectionConfig,
    LocalFairnessConfig,
    BiasPolicyConfig,
    DistillationConfig,
    StabilityConfig,
)
from .presets import (
    get_conservative_preset,
    get_balanced_preset,
    get_aggressive_preset,
    get_preset_by_name,
)

__all__ = [
    'FairCareFLConfig',
    'MultiObjectiveConfig',
    'FairSelectionConfig',
    'LocalFairnessConfig',
    'BiasPolicyConfig',
    'DistillationConfig',
    'StabilityConfig',
    'get_conservative_preset',
    'get_balanced_preset',
    'get_aggressive_preset',
    'get_preset_by_name',
]
