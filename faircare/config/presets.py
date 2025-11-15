"""
FairCare-FL Configuration Presets

Three preset configurations balancing accuracy-fairness tradeoffs:
1. Conservative: Minimal regression, gentle fairness
2. Balanced: Target parity between accuracy and fairness
3. Aggressive: Maximum fairness, accept larger accuracy tradeoffs
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


def get_conservative_preset() -> FairCareFLConfig:
    """
    Conservative Preset: Minimal Risk

    Goal: Reduce accuracy regression to <1% while improving fairness
    Strategy: Gentle softmin, no aggressive features
    Expected: Small fairness gains, minimal accuracy loss
    """
    return FairCareFLConfig(
        legacy_mode=False,

        # Multi-objective aggregation: Gentle softmin only
        aggregate=MultiObjectiveConfig(
            enable=True,
            weight_method='softmin',
            softmin_temperature=2.0,  # High τ = gentle, close to uniform
            acc_weight=1.5,  # Prefer accuracy
            fairness_weight=1.0,
            moo_method='mgda',
            mgda_normalize=True,
            clamp_weights=True,
            weight_min=0.05,
            weight_max=5.0,
            fednova_enable=False,
            server_momentum=0.1,  # Light momentum
        ),

        # Fair selection: Minimal intervention
        selection=FairSelectionConfig(
            enable=True,
            strategy='fair_priority',
            recency_bonus=0.1,  # Small boost for recency
            loss_bonus=0.05,
            underparticipation_bonus=0.05,
            enforce_group_diversity=False,
            min_group_representation=0.05,
        ),

        # Local fairness: DISABLED (too aggressive)
        local=LocalFairnessConfig(
            enable=False,
        ),

        # Bias policy: DISABLED (too dynamic)
        policy=BiasPolicyConfig(
            enable=False,
        ),

        # Distillation: DISABLED
        distill=DistillationConfig(
            enable=False,
        ),

        # Stability: Standard safety
        stability=StabilityConfig(
            clip_gradients=True,
            max_grad_norm=1.0,
            clip_deltas=True,
            max_delta_norm=10.0,
            epsilon=1e-8,
            clip_dual_variables=True,
            dual_max_value=10.0,
        ),

        # Logging
        verbose_logging=True,
        log_objectives=True,
        log_weights=True,
        log_policy_switches=False,
    )


def get_balanced_preset() -> FairCareFLConfig:
    """
    Balanced Preset: Accuracy-Fairness Parity

    Goal: Achieve strong fairness with <3% accuracy regression
    Strategy: Moderate softmin, fair selection, light local debiasing
    Expected: Good fairness gains, moderate accuracy loss
    """
    return FairCareFLConfig(
        legacy_mode=False,

        # Multi-objective aggregation: Moderate softmin
        aggregate=MultiObjectiveConfig(
            enable=True,
            weight_method='softmin',
            softmin_temperature=0.8,  # Moderate fairness prioritization
            acc_weight=1.0,  # Equal weighting
            fairness_weight=1.0,
            moo_method='mgda',
            mgda_normalize=True,
            clamp_weights=True,
            weight_min=0.02,
            weight_max=8.0,
            fednova_enable=True,
            server_momentum=0.3,
        ),

        # Fair selection: Active intervention
        selection=FairSelectionConfig(
            enable=True,
            strategy='fair_priority',
            recency_bonus=0.3,
            loss_bonus=0.2,
            underparticipation_bonus=0.3,
            enforce_group_diversity=True,
            min_group_representation=0.15,
        ),

        # Local fairness: Light penalties
        local=LocalFairnessConfig(
            enable=True,
            fairness_penalty='eo',
            fairness_penalty_weight=0.1,
            irm_enable=True,
            irm_lambda=0.01,
            irm_anneal=True,
            adversarial_enable=False,  # Too complex
            mixup_enable=True,
            mixup_alpha=0.2,
            counterfactual_enable=False,
        ),

        # Bias policy: Dynamic switching
        policy=BiasPolicyConfig(
            enable=True,
            bias_metric='eo_gap',
            bias_threshold_low=0.05,
            bias_threshold_high=0.15,
            initial_mode='normal',
            normal_fairness_weight=1.0,
            mitigation_fairness_weight=2.0,
            evaluate_every_n_rounds=5,
        ),

        # Distillation: Light usage
        distill=DistillationConfig(
            enable=True,
            temperature=3.0,
            alpha=0.3,
            use_global_as_teacher=True,
        ),

        # Stability: Standard safety
        stability=StabilityConfig(
            clip_gradients=True,
            max_grad_norm=1.0,
            clip_deltas=True,
            max_delta_norm=10.0,
            epsilon=1e-8,
            clip_dual_variables=True,
            dual_max_value=10.0,
        ),

        # Logging
        verbose_logging=True,
        log_objectives=True,
        log_weights=True,
        log_policy_switches=True,
    )


def get_aggressive_preset() -> FairCareFLConfig:
    """
    Aggressive Preset: Maximum Fairness

    Goal: Maximize fairness metrics, accept accuracy tradeoffs
    Strategy: Aggressive softmin, all features enabled
    Expected: Strong fairness, potentially large accuracy loss
    """
    return FairCareFLConfig(
        legacy_mode=False,

        # Multi-objective aggregation: Aggressive softmin
        aggregate=MultiObjectiveConfig(
            enable=True,
            weight_method='softmin',
            softmin_temperature=0.3,  # Low τ = aggressive fairness
            acc_weight=0.5,  # Deprioritize accuracy
            fairness_weight=2.0,  # Prioritize fairness
            moo_method='mgda',
            mgda_normalize=True,
            clamp_weights=True,
            weight_min=0.01,
            weight_max=10.0,
            fednova_enable=True,
            server_momentum=0.5,
        ),

        # Fair selection: Maximum intervention
        selection=FairSelectionConfig(
            enable=True,
            strategy='fair_priority',
            recency_bonus=0.5,
            loss_bonus=0.4,
            underparticipation_bonus=0.5,
            enforce_group_diversity=True,
            min_group_representation=0.2,
        ),

        # Local fairness: All features enabled
        local=LocalFairnessConfig(
            enable=True,
            fairness_penalty='both',
            fairness_penalty_weight=0.3,
            irm_enable=True,
            irm_lambda=0.05,
            irm_anneal=True,
            adversarial_enable=True,
            adversarial_lambda=0.1,
            adversarial_hidden_dim=64,
            mixup_enable=True,
            mixup_alpha=0.3,
            counterfactual_enable=True,
            counterfactual_prob=0.2,
        ),

        # Bias policy: Aggressive switching
        policy=BiasPolicyConfig(
            enable=True,
            bias_metric='eo_gap',
            bias_threshold_low=0.03,  # Stricter thresholds
            bias_threshold_high=0.10,
            initial_mode='mitigation',  # Start aggressive
            normal_fairness_weight=1.5,
            mitigation_fairness_weight=3.0,
            evaluate_every_n_rounds=3,
        ),

        # Distillation: Full usage
        distill=DistillationConfig(
            enable=True,
            temperature=4.0,
            alpha=0.5,
            use_global_as_teacher=True,
        ),

        # Stability: Stricter safety
        stability=StabilityConfig(
            clip_gradients=True,
            max_grad_norm=0.5,  # Tighter clipping
            clip_deltas=True,
            max_delta_norm=5.0,  # Tighter clipping
            epsilon=1e-8,
            clip_dual_variables=True,
            dual_max_value=15.0,
        ),

        # Logging
        verbose_logging=True,
        log_objectives=True,
        log_weights=True,
        log_policy_switches=True,
    )


def get_preset_by_name(name: str) -> FairCareFLConfig:
    """Get preset configuration by name."""
    presets = {
        'conservative': get_conservative_preset,
        'balanced': get_balanced_preset,
        'aggressive': get_aggressive_preset,
        'legacy': FairCareFLConfig.create_legacy,
    }

    if name.lower() not in presets:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(presets.keys())}"
        )

    return presets[name.lower()]()
