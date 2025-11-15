"""
FairCare-FL Configuration Schema

Complete configuration system for all FairCare-FL enhancements.
All features are OFF by default for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MultiObjectiveConfig:
    """Multi-objective aggregation configuration."""

    # Core enable flag
    enable: bool = False

    # Weight computation method
    weight_method: Literal['softmin', 'sample_prop', 'uniform'] = 'softmin'
    softmin_temperature: float = 0.5  # τ: lower = more aggressive fairness

    # Objective weighting
    acc_weight: float = 1.0
    fairness_weight: float = 1.0

    # Multi-objective optimization
    moo_method: Literal['mgda', 'pcgrad', 'cagrad', 'none'] = 'mgda'
    mgda_normalize: bool = True

    # Weight constraints
    clamp_weights: bool = True
    weight_min: float = 0.01
    weight_max: float = 10.0

    # FedNova normalization
    fednova_enable: bool = False

    # Server momentum
    server_momentum: float = 0.0  # 0 = disabled


@dataclass
class FairSelectionConfig:
    """Fair client selection configuration."""

    # Core enable flag
    enable: bool = False

    # Selection strategy
    strategy: Literal['random', 'fair_priority', 'loss_based'] = 'random'

    # Priority bonuses
    recency_bonus: float = 0.0  # Boost for clients not selected recently
    loss_bonus: float = 0.0     # Boost for high-loss clients
    underparticipation_bonus: float = 0.0  # Boost for under-selected groups

    # Diversity constraints
    enforce_group_diversity: bool = False
    min_group_representation: float = 0.1  # Minimum fraction per group


@dataclass
class LocalFairnessConfig:
    """Local client-side fairness enhancements."""

    # Core enable flag
    enable: bool = False

    # Fairness penalty (soft constraint)
    fairness_penalty: Literal['none', 'eo', 'dp', 'both'] = 'none'
    fairness_penalty_weight: float = 0.0

    # IRM (Invariant Risk Minimization)
    irm_enable: bool = False
    irm_lambda: float = 0.0
    irm_anneal: bool = False

    # Adversarial debiasing
    adversarial_enable: bool = False
    adversarial_lambda: float = 0.0
    adversarial_hidden_dim: int = 64

    # Data augmentation
    mixup_enable: bool = False
    mixup_alpha: float = 0.2

    counterfactual_enable: bool = False
    counterfactual_prob: float = 0.0


@dataclass
class BiasPolicyConfig:
    """Bias monitoring and dynamic policy switching."""

    # Core enable flag
    enable: bool = False

    # Bias detection
    bias_metric: Literal['eo_gap', 'sp_gap', 'worst_group_f1'] = 'eo_gap'
    bias_threshold_low: float = 0.05   # Switch to NORMAL mode
    bias_threshold_high: float = 0.15  # Switch to MITIGATION mode

    # Policy modes
    initial_mode: Literal['normal', 'mitigation'] = 'normal'

    # Mode-specific parameters
    normal_fairness_weight: float = 0.5
    mitigation_fairness_weight: float = 2.0

    # Evaluation frequency
    evaluate_every_n_rounds: int = 5


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""

    # Core enable flag
    enable: bool = False

    # Distillation parameters
    temperature: float = 3.0
    alpha: float = 0.5  # Balance between task loss and distillation loss

    # Teacher model
    use_global_as_teacher: bool = True


@dataclass
class StabilityConfig:
    """Stability and numerical safety configuration."""

    # Gradient clipping
    clip_gradients: bool = True
    max_grad_norm: float = 1.0

    # Delta clipping
    clip_deltas: bool = True
    max_delta_norm: float = 10.0

    # Numerical stability
    epsilon: float = 1e-8

    # Dual variable constraints
    clip_dual_variables: bool = True
    dual_max_value: float = 10.0


@dataclass
class FairCareFLConfig:
    """Complete FairCare-FL configuration with all enhancements."""

    # Sub-configurations
    aggregate: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    selection: FairSelectionConfig = field(default_factory=FairSelectionConfig)
    local: LocalFairnessConfig = field(default_factory=LocalFairnessConfig)
    policy: BiasPolicyConfig = field(default_factory=BiasPolicyConfig)
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    # Legacy parameters (for backward compatibility)
    legacy_mode: bool = True  # When True, all enhancements are disabled

    # Logging and telemetry
    verbose_logging: bool = False
    log_objectives: bool = False
    log_weights: bool = False
    log_policy_switches: bool = False

    def __post_init__(self):
        """Validate configuration."""
        # If legacy mode, ensure all enhancements are disabled
        if self.legacy_mode:
            self.aggregate.enable = False
            self.selection.enable = False
            self.local.enable = False
            self.policy.enable = False
            self.distill.enable = False

        # Validate temperature values
        if self.aggregate.softmin_temperature <= 0:
            raise ValueError("softmin_temperature must be > 0")
        if self.distill.temperature <= 0:
            raise ValueError("distillation temperature must be > 0")

        # Validate weight ranges
        if self.aggregate.weight_min >= self.aggregate.weight_max:
            raise ValueError("weight_min must be < weight_max")

        # Validate penalty weights
        if self.local.fairness_penalty_weight < 0:
            raise ValueError("fairness_penalty_weight must be >= 0")
        if self.local.irm_lambda < 0:
            raise ValueError("irm_lambda must be >= 0")
        if self.local.adversarial_lambda < 0:
            raise ValueError("adversarial_lambda must be >= 0")

    @classmethod
    def create_legacy(cls) -> 'FairCareFLConfig':
        """Create configuration matching current FairCare-FL behavior."""
        config = cls()
        config.legacy_mode = True
        return config

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'FairCareFLConfig':
        """Create configuration from dictionary."""
        # Extract sub-configs
        aggregate = MultiObjectiveConfig(**config_dict.get('aggregate', {}))
        selection = FairSelectionConfig(**config_dict.get('selection', {}))
        local = LocalFairnessConfig(**config_dict.get('local', {}))
        policy = BiasPolicyConfig(**config_dict.get('policy', {}))
        distill = DistillationConfig(**config_dict.get('distill', {}))
        stability = StabilityConfig(**config_dict.get('stability', {}))

        # Extract top-level params
        legacy_mode = config_dict.get('legacy_mode', True)
        verbose_logging = config_dict.get('verbose_logging', False)
        log_objectives = config_dict.get('log_objectives', False)
        log_weights = config_dict.get('log_weights', False)
        log_policy_switches = config_dict.get('log_policy_switches', False)

        return cls(
            aggregate=aggregate,
            selection=selection,
            local=local,
            policy=policy,
            distill=distill,
            stability=stability,
            legacy_mode=legacy_mode,
            verbose_logging=verbose_logging,
            log_objectives=log_objectives,
            log_weights=log_weights,
            log_policy_switches=log_policy_switches,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)
