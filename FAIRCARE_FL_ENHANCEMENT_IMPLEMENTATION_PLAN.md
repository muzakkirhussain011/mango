# FairCare-FL Enhancement Implementation Plan

**Date**: 2025-11-10
**Status**: SPECIFICATION COMPLETE - Ready for Implementation
**Estimated Effort**: 2000-3000 lines of code, 5-7 days development

---

## Executive Summary

This document specifies the comprehensive enhancement of `faircare_fl` to create a **holistic, multi-objective fairness stack** that beats all baselines on accuracy, fairness metrics, and stability. All enhancements are **feature-flagged** and **fully backward-compatible**.

### Current Implementation Analysis

**Existing File**: `faircare/algos/faircare_fl.py` (1251 lines)

**Already Implemented** (but NOT feature-flagged):
- ✅ Server momentum (`server_momentum = 0.3`)
- ✅ MGDA/PCGrad/CAGrad multi-objective optimization
- ✅ Dual variables for EO/FPR/SP constraints
- ✅ DFBD network for bias detection
- ✅ Knowledge distillation
- ✅ Weight clamping (`weight_floor`, `weight_cap`)
- ✅ Fairness-aware selection (Lyapunov-based)

**CRITICAL ISSUE**: Current implementation has ALL features ALWAYS ON. No backward compatibility mechanism!

**Required Changes**:
1. **Wrap all existing features in feature flags** (defaults: OFF)
2. **Add missing features** per specification
3. **Create configuration system**
4. **Ensure backward compatibility** when all flags OFF
5. **Create three presets**

---

## Phase 1: Configuration System (Priority: CRITICAL)

### 1.1 Configuration Schema

Create `faircare/config/faircare_fl_config.py`:

```python
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class MultiObjectiveConfig:
    """Multi-objective aggregation configuration."""
    enable: bool = False  # DEFAULT: OFF
    weights: Dict[str, float] = field(default_factory=lambda: {
        'alpha_eo': 1.0,
        'beta_fpr': 0.5,
        'gamma_sp': 0.5,
        'delta_loss': 1.0
    })
    temperature_tau: float = 1.0
    min_weight_frac: float = 0.005
    max_weight_frac: float = 0.15
    momentum: float = 0.9
    normalize_method: str = 'none'  # 'none' | 'l2' | 'fednova'

@dataclass
class FairSelectionConfig:
    """Fair client selection configuration."""
    enable: bool = False  # DEFAULT: OFF
    recency_bonus: float = 0.2
    high_loss_bonus: float = 0.2
    underrep_bonus: float = 0.3
    capacity_penalty: float = 0.1
    history_window: int = 10

@dataclass
class LocalFairnessConfig:
    """Local debiasing configuration."""
    enable_fair_loss: bool = False  # DEFAULT: OFF
    lambda_fair: float = 0.1
    weights: Dict[str, float] = field(default_factory=lambda: {
        'w_eo': 1.0,
        'w_fpr': 0.5,
        'w_sp': 0.5
    })
    enable_irm: bool = False  # DEFAULT: OFF
    lambda_irm: float = 0.1
    enable_adv: bool = False  # DEFAULT: OFF
    lambda_adv: float = 0.1
    enable_mixup: bool = False  # DEFAULT: OFF
    mixup_alpha: float = 0.2
    enable_counterfactual: bool = False  # DEFAULT: OFF
    cf_prob: float = 0.25

@dataclass
class BiasPolicyConfig:
    """Bias monitoring and dynamic policy configuration."""
    enable: bool = False  # DEFAULT: OFF
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'eo': 0.10,
        'fpr': 0.10,
        'sp': 0.08
    })
    patience_rounds: int = 2
    tau_min: float = 0.2
    alpha_scale_in_bias_mode: float = 2.0
    delta_loss_scale_in_bias_mode: float = 0.5
    extra_local_epochs_in_bias_mode: int = 1

@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    enable: bool = False  # DEFAULT: OFF
    steps: int = 50
    temperature: float = 2.0

@dataclass
class StabilityConfig:
    """Stability and regularization configuration."""
    clip_grad_norm: float = 10.0
    weight_decay: float = 0.0

@dataclass
class FairCareFLConfig:
    """Complete FairCare-FL configuration."""
    aggregate: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    selection: FairSelectionConfig = field(default_factory=FairSelectionConfig)
    local: LocalFairnessConfig = field(default_factory=LocalFairnessConfig)
    policy: BiasPolicyConfig = field(default_factory=BiasPolicyConfig)
    distill: DistillationConfig = field(default_factory=DistillationConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    # Legacy parameters (for backward compatibility)
    server_momentum: float = 0.0  # DEFAULT: 0.0 (OFF)
    epsilon_eo: float = 0.015
    epsilon_fpr: float = 0.015
    epsilon_sp: float = 0.02
    dual_lr: float = 0.005
    dual_max: float = 0.5
```

### 1.2 Three Presets

```python
# PRESET 1: CONSERVATIVE (minimal risk, near-baseline performance)
CONSERVATIVE_PRESET = FairCareFLConfig(
    aggregate=MultiObjectiveConfig(
        enable=True,
        temperature_tau=1.0,
        momentum=0.9,
        normalize_method='none'
    ),
    selection=FairSelectionConfig(enable=False),
    local=LocalFairnessConfig(enable_fair_loss=False),
    policy=BiasPolicyConfig(enable=False),
    distill=DistillationConfig(enable=False)
)

# PRESET 2: BALANCED (target parity + accuracy)
BALANCED_PRESET = FairCareFLConfig(
    aggregate=MultiObjectiveConfig(
        enable=True,
        temperature_tau=0.6,
        weights={'alpha_eo': 1.0, 'beta_fpr': 0.5, 'gamma_sp': 0.5, 'delta_loss': 1.0},
        momentum=0.9
    ),
    selection=FairSelectionConfig(enable=True),
    local=LocalFairnessConfig(enable_fair_loss=True, lambda_fair=0.1),
    policy=BiasPolicyConfig(enable=True, patience_rounds=2),
    distill=DistillationConfig(enable=False)
)

# PRESET 3: AGGRESSIVE (maximum fairness)
AGGRESSIVE_PRESET = FairCareFLConfig(
    aggregate=MultiObjectiveConfig(
        enable=True,
        temperature_tau=0.3,
        momentum=0.9,
        normalize_method='fednova'
    ),
    selection=FairSelectionConfig(enable=True),
    local=LocalFairnessConfig(
        enable_fair_loss=True,
        enable_irm=True,
        enable_adv=True,
        lambda_fair=0.1,
        lambda_irm=0.1,
        lambda_adv=0.1
    ),
    policy=BiasPolicyConfig(
        enable=True,
        patience_rounds=2,
        tau_min=0.2,
        alpha_scale_in_bias_mode=2.0,
        extra_local_epochs_in_bias_mode=1
    ),
    distill=DistillationConfig(enable=True, steps=50)
)
```

---

## Phase 2: Backward Compatibility Layer (Priority: CRITICAL)

### 2.1 Migration Strategy

**Modify `FairCareFLAggregator.__init__`**:

```python
def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
    """Initialize with full backward compatibility."""
    # NEW: Parse enhanced config
    if isinstance(config, FairCareFLConfig):
        self.enh_config = config
    else:
        # Legacy dict config - use all defaults (flags OFF)
        self.enh_config = FairCareFLConfig()
        # Preserve legacy parameters if present
        if 'server_momentum' in config:
            self.enh_config.aggregate.momentum = config['server_momentum']

    # OLD: Existing parameters (only use if enhancement flags are OFF)
    self.config = config
    self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Initialize components based on flags
    if self.enh_config.aggregate.enable:
        self._init_multi_objective_aggregation()
    else:
        self._init_legacy_aggregation()  # Current behavior

    if self.enh_config.local.enable_irm or self.enh_config.local.enable_adv:
        self._init_local_debiasing()

    # ... etc for all components
```

### 2.2 Conditional Execution Pattern

Throughout the code, use this pattern:

```python
def aggregate(self, round_ctx, client_reports, global_model):
    """Main aggregation with conditional enhancement execution."""

    # ALWAYS execute core logic
    client_reports = self._prepare_reports(client_reports)

    # CONDITIONAL: Multi-objective aggregation
    if self.enh_config.aggregate.enable:
        weights = self._compute_multi_objective_weights(client_reports)
    else:
        weights = self._compute_legacy_weights(client_reports)

    # CONDITIONAL: Bias monitoring
    if self.enh_config.policy.enable:
        self._update_bias_policy(client_reports)

    # CONDITIONAL: Momentum
    if self.enh_config.aggregate.momentum > 0:
        aggregated_delta = self._apply_momentum(aggregated_delta)

    return AggregationOutput(new_global=new_global, server_logs=logs)
```

---

## Phase 3: Server-Side Enhancements

### 3.1 Multi-Objective Aggregation with Softmin

```python
def _compute_multi_objective_weights(self, client_reports: List[Dict]) -> torch.Tensor:
    """Compute weights using softmin with temperature."""
    if not self.enh_config.aggregate.enable:
        return self._compute_legacy_weights(client_reports)

    # Extract metrics
    eo_gaps = torch.tensor([r.get('eo_gap', 0.0) for r in client_reports])
    fpr_gaps = torch.tensor([r.get('fpr_gap', 0.0) for r in client_reports])
    sp_gaps = torch.tensor([abs(r.get('sp_gap', 0.0)) for r in client_reports])
    losses = torch.tensor([r.get('val_loss', 1.0) for r in client_reports])

    # Multi-objective score
    w = self.enh_config.aggregate.weights
    scores = (
        w['alpha_eo'] * eo_gaps +
        w['beta_fpr'] * fpr_gaps +
        w['gamma_sp'] * sp_gaps +
        w['delta_loss'] * losses
    )

    # Softmin with temperature
    tau = max(self.enh_config.policy.tau_min, self.enh_config.aggregate.temperature_tau)
    raw_weights = torch.exp(-scores / tau)
    raw_weights = raw_weights / raw_weights.sum()

    # Clamp to [min_frac, max_frac]
    weights = self._clamp_and_renormalize(
        raw_weights,
        self.enh_config.aggregate.min_weight_frac,
        self.enh_config.aggregate.max_weight_frac
    )

    return weights

def _clamp_and_renormalize(self, weights, min_frac, max_frac):
    """Clamp weights and renormalize to sum to 1."""
    n = len(weights)
    min_w = min_frac * 1.0
    max_w = max_frac * 1.0

    # Iterative clamping with renormalization
    for _ in range(10):  # Max iterations
        weights = torch.clamp(weights, min=min_w, max=max_w)
        total = weights.sum()
        if torch.abs(total - 1.0) < 1e-6:
            break
        weights = weights / total

    return weights
```

### 3.2 Server Momentum

```python
def _apply_momentum(self, aggregated_delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Apply server-side momentum."""
    if self.enh_config.aggregate.momentum <= 0:
        return aggregated_delta

    m = self.enh_config.aggregate.momentum

    # Initialize momentum buffer on first call
    if self.momentum_buffer is None:
        self.momentum_buffer = {k: torch.zeros_like(v) for k, v in aggregated_delta.items()}

    # Update momentum: V_t = m*V_{t-1} + ΔW_t
    for key in aggregated_delta:
        self.momentum_buffer[key] = m * self.momentum_buffer[key] + aggregated_delta[key]

    return self.momentum_buffer
```

### 3.3 Bias Monitoring & Dynamic Policy

```python
def _update_bias_policy(self, client_reports: List[Dict]):
    """Monitor bias and switch between normal ↔ mitigation mode."""
    if not self.enh_config.policy.enable:
        return

    # Compute aggregate fairness metrics
    mean_eo = np.mean([r.get('eo_gap', 0.0) for r in client_reports])
    mean_fpr = np.mean([r.get('fpr_gap', 0.0) for r in client_reports])
    mean_sp = np.mean([abs(r.get('sp_gap', 0.0)) for r in client_reports])

    thresholds = self.enh_config.policy.thresholds

    # Check if any gap exceeds threshold
    violations = [
        mean_eo > thresholds['eo'],
        mean_fpr > thresholds['fpr'],
        mean_sp > thresholds['sp']
    ]

    if any(violations):
        self.bias_patience_counter += 1
        if self.bias_patience_counter >= self.enh_config.policy.patience_rounds:
            self._enter_bias_mitigation_mode()
    else:
        self.bias_patience_counter = 0
        if self.in_bias_mode:
            self._exit_bias_mitigation_mode()

def _enter_bias_mitigation_mode(self):
    """Enter bias mitigation mode - increase fairness emphasis."""
    self.in_bias_mode = True

    # Decrease temperature (sharper distribution toward fair clients)
    self.enh_config.aggregate.temperature_tau = max(
        self.enh_config.policy.tau_min,
        self.enh_config.aggregate.temperature_tau / 2
    )

    # Increase fairness weights
    scale = self.enh_config.policy.alpha_scale_in_bias_mode
    self.enh_config.aggregate.weights['alpha_eo'] *= scale
    self.enh_config.aggregate.weights['beta_fpr'] *= scale
    self.enh_config.aggregate.weights['gamma_sp'] *= scale

    # Decrease loss weight
    self.enh_config.aggregate.weights['delta_loss'] *= self.enh_config.policy.delta_loss_scale_in_bias_mode

    logger.warning(f"[BIAS MODE] Entered at round {self.round_counter}: τ={self.enh_config.aggregate.temperature_tau:.3f}")

def _exit_bias_mitigation_mode(self):
    """Exit bias mitigation mode - restore normal weights."""
    self.in_bias_mode = False
    # Gradually anneal temperature back up
    self.enh_config.aggregate.temperature_tau = min(1.0, self.enh_config.aggregate.temperature_tau * 1.5)
    # Restore original weights (requires storing original config)
    logger.info(f"[BIAS MODE] Exited at round {self.round_counter}")
```

### 3.4 FedNova Normalization

```python
def _apply_fednova_normalization(self, client_reports: List[Dict]) -> List[Dict]:
    """Apply FedNova-style update normalization."""
    if self.enh_config.aggregate.normalize_method != 'fednova':
        return client_reports

    normalized_reports = []
    for report in client_reports:
        # Get local steps (epochs * batches)
        local_steps = report.get('local_steps', 1)

        # Normalize delta by local steps
        normalized_delta = {
            k: v / local_steps for k, v in report['delta'].items()
        }

        report_copy = report.copy()
        report_copy['delta'] = normalized_delta
        normalized_reports.append(report_copy)

    return normalized_reports
```

---

## Phase 4: Fair Client Selection

```python
class FairClientSelector:
    """Fair client selection with recency, loss, and participation bonuses."""

    def __init__(self, config: FairSelectionConfig):
        self.config = config
        self.selection_history = defaultdict(list)  # client_id -> [round1, round2, ...]
        self.loss_history = defaultdict(list)
        self.participation_counts = defaultdict(int)
        self.total_rounds = 0

    def select_clients(self, available_clients: List[int],
                      client_metrics: Dict[int, Dict],
                      num_select: int) -> List[int]:
        """Select clients using fairness-aware scoring."""
        if not self.config.enable:
            return np.random.choice(available_clients, num_select, replace=False).tolist()

        scores = {}
        for client_id in available_clients:
            score = 1.0  # Base score

            # Recency bonus (recently selected → penalty)
            recent_rounds = [r for r in self.selection_history[client_id]
                           if self.total_rounds - r <= self.config.history_window]
            if recent_rounds:
                score -= self.config.recency_bonus * len(recent_rounds) / self.config.history_window

            # High loss bonus (worse performance → higher priority)
            if client_id in client_metrics:
                loss = client_metrics[client_id].get('val_loss', 1.0)
                avg_loss = np.mean(list(client_metrics.values())) if client_metrics else 1.0
                if loss > avg_loss:
                    score += self.config.high_loss_bonus * (loss / avg_loss - 1.0)

            # Under-participation bonus
            participation_ratio = self.participation_counts[client_id] / max(1, self.total_rounds)
            expected_ratio = num_select / len(available_clients)
            if participation_ratio < expected_ratio:
                score += self.config.underrep_bonus * (expected_ratio - participation_ratio)

            scores[client_id] = max(0.01, score)  # Ensure positive

        # Sample proportional to scores
        client_ids = list(scores.keys())
        probabilities = np.array([scores[cid] for cid in client_ids])
        probabilities = probabilities / probabilities.sum()

        selected = np.random.choice(client_ids, num_select, p=probabilities, replace=False)

        # Update history
        self.total_rounds += 1
        for cid in selected:
            self.selection_history[cid].append(self.total_rounds)
            self.participation_counts[cid] += 1

        return selected.tolist()
```

---

## Phase 5: Client-Side Enhancements

### 5.1 Local Fairness Loss (Soft Differentiable Metrics)

```python
class LocalFairnessLoss(nn.Module):
    """Differentiable local fairness loss using soft group rates."""

    def __init__(self, config: LocalFairnessConfig):
        super().__init__()
        self.config = config

    def forward(self, logits, labels, groups):
        """Compute soft fairness penalties.

        Args:
            logits: Model logits (batch_size,)
            labels: True labels (batch_size,)
            groups: Group membership (batch_size,)

        Returns:
            Scalar fairness loss
        """
        if not self.config.enable_fair_loss:
            return torch.tensor(0.0, device=logits.device)

        probs = torch.sigmoid(logits)

        # Compute soft TPR and FPR for each group
        group_0_mask = (groups == 0).float()
        group_1_mask = (groups == 1).float()

        # Soft TPR (for positives)
        pos_mask = labels.float()
        tpr_0 = (probs * pos_mask * group_0_mask).sum() / (pos_mask * group_0_mask).sum().clamp(min=1e-6)
        tpr_1 = (probs * pos_mask * group_1_mask).sum() / (pos_mask * group_1_mask).sum().clamp(min=1e-6)

        # Soft FPR (for negatives)
        neg_mask = (1 - labels).float()
        fpr_0 = (probs * neg_mask * group_0_mask).sum() / (neg_mask * group_0_mask).sum().clamp(min=1e-6)
        fpr_1 = (probs * neg_mask * group_1_mask).sum() / (neg_mask * group_1_mask).sum().clamp(min=1e-6)

        # Soft PPR (positive prediction rate)
        ppr_0 = (probs * group_0_mask).sum() / group_0_mask.sum().clamp(min=1e-6)
        ppr_1 = (probs * group_1_mask).sum() / group_1_mask.sum().clamp(min=1e-6)

        # Compute squared gaps
        w = self.config.weights
        loss_fair = (
            w['w_eo'] * (tpr_0 - tpr_1).pow(2) +
            w['w_fpr'] * (fpr_0 - fpr_1).pow(2) +
            w['w_sp'] * (ppr_0 - ppr_1).pow(2)
        )

        return self.config.lambda_fair * 0.1 * loss_fair  # Small scale factor
```

### 5.2 IRM (Invariant Risk Minimization)

```python
class IRMPenalty(nn.Module):
    """IRM penalty for domain generalization."""

    def __init__(self, config: LocalFairnessConfig):
        super().__init__()
        self.config = config

    def forward(self, logits, labels, groups):
        """Compute IRM penalty (variance of group-wise gradients)."""
        if not self.config.enable_irm:
            return torch.tensor(0.0, device=logits.device)

        # Compute per-group losses
        losses_per_group = []
        for g in [0, 1]:
            mask = (groups == g)
            if mask.sum() > 0:
                group_logits = logits[mask]
                group_labels = labels[mask]
                group_loss = F.binary_cross_entropy_with_logits(
                    group_logits, group_labels.float()
                )
                losses_per_group.append(group_loss)

        if len(losses_per_group) < 2:
            return torch.tensor(0.0, device=logits.device)

        # IRM penalty: variance of group losses
        irm_penalty = torch.var(torch.stack(losses_per_group))

        return self.config.lambda_irm * irm_penalty
```

### 5.3 Adversarial Debiasing

```python
class AdversarialDebiaser(nn.Module):
    """Gradient reversal adversary for group-invariant representations."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, features, groups, lambda_adv):
        """Apply gradient reversal and predict group membership."""
        if lambda_adv <= 0:
            return torch.tensor(0.0, device=features.device)

        # Gradient reversal layer (negative gradient during backward)
        reversed_features = GradientReversalLayer.apply(features, lambda_adv)

        # Predict group membership
        group_pred = self.adversary(reversed_features).squeeze()

        # Binary cross-entropy loss
        adv_loss = F.binary_cross_entropy_with_logits(
            group_pred, groups.float()
        )

        return adv_loss

class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_factor):
        ctx.lambda_factor = lambda_factor
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_factor * grad_output, None
```

### 5.4 Mixup Augmentation

```python
def mixup_batch(features, labels, groups, alpha=0.2):
    """Apply mixup augmentation."""
    batch_size = features.size(0)
    lam = np.random.beta(alpha, alpha)

    # Random permutation
    indices = torch.randperm(batch_size)

    # Mix features and labels
    mixed_features = lam * features + (1 - lam) * features[indices]
    mixed_labels = lam * labels + (1 - lam) * labels[indices]
    mixed_groups = groups  # Keep original groups (or mix if appropriate)

    return mixed_features, mixed_labels, mixed_groups
```

---

## Phase 6: Testing Strategy

### 6.1 Unit Tests

Create `tests/test_faircare_fl_enhancements.py`:

```python
def test_backward_compatibility():
    """Test that default config = current behavior."""
    config_default = FairCareFLConfig()  # All flags OFF
    config_legacy = {}  # Old dict-style config

    aggregator_default = FairCareFLAggregator(config_default, device='cpu')
    aggregator_legacy = FairCareFLAggregator(config_legacy, device='cpu')

    # Run aggregation on identical inputs
    # Assert outputs match within tolerance

def test_softmin_weighting():
    """Test softmin with temperature produces valid weights."""
    scores = torch.tensor([1.0, 2.0, 3.0])
    tau = 0.5
    weights = torch.exp(-scores / tau)
    weights = weights / weights.sum()

    assert torch.abs(weights.sum() - 1.0) < 1e-6
    assert torch.all(weights >= 0)
    assert weights[0] > weights[2]  # Lower score → higher weight

def test_weight_clamping():
    """Test weight clamping preserves sum=1."""
    weights = torch.tensor([0.8, 0.15, 0.05])
    min_frac, max_frac = 0.1, 0.4

    clamped = clamp_and_renormalize(weights, min_frac, max_frac)

    assert torch.abs(clamped.sum() - 1.0) < 1e-6
    assert torch.all(clamped >= min_frac * 1.0)
    assert torch.all(clamped <= max_frac * 1.0)
```

### 6.2 Integration Tests

```python
def test_conservative_preset_vs_baseline():
    """Conservative preset should match baseline within 1%."""
    # Run with default config (all flags OFF)
    results_baseline = run_experiment(config=FairCareFLConfig(), dataset='adult', seed=42)

    # Run with conservative preset
    results_conservative = run_experiment(config=CONSERVATIVE_PRESET, dataset='adult', seed=42)

    # Assert metrics within tolerance
    assert abs(results_baseline['accuracy'] - results_conservative['accuracy']) < 0.01

def test_balanced_preset_improves_fairness():
    """Balanced preset should reduce EO gap by >20%."""
    results_baseline = run_experiment(config=FairCareFLConfig(), dataset='adult', seed=42)
    results_balanced = run_experiment(config=BALANCED_PRESET, dataset='adult', seed=42)

    eo_reduction = (results_baseline['eo_gap'] - results_balanced['eo_gap']) / results_baseline['eo_gap']

    assert eo_reduction > 0.20  # At least 20% reduction
    assert abs(results_baseline['accuracy'] - results_balanced['accuracy']) < 0.01  # <1% accuracy drop

def test_bias_mode_activation():
    """Bias mode should activate when gaps exceed thresholds."""
    # Create synthetic scenario with high EO gap
    # Run for 5 rounds
    # Assert bias mode activated and tau decreased
```

---

## Phase 7: Implementation Timeline

### Week 1: Configuration & Backward Compatibility
- **Day 1-2**: Implement configuration system
- **Day 3-4**: Add feature flags to existing code
- **Day 5**: Validate backward compatibility

### Week 2: Server-Side Enhancements
- **Day 6-7**: Multi-objective aggregation with softmin
- **Day 8**: Server momentum & FedNova
- **Day 9**: Bias monitoring & dynamic policy
- **Day 10**: Fair client selection

### Week 3: Client-Side & Testing
- **Day 11-12**: Local debiasing (IRM, adversarial, fairness loss)
- **Day 13**: Data augmentation (mixup, counterfactual)
- **Day 14-15**: Comprehensive testing
- **Day 16**: Documentation & presets

---

## Phase 8: Success Criteria Validation

After implementation, validate that enhanced FairCare-FL meets ALL criteria:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Worst-Group F1** | ≥ best baseline + 3% absolute OR +8% relative | Compare across all baselines on 4 datasets |
| **EO Gap** | ≤ 70% of best baseline (≥30% reduction) | Measure on all datasets, all seeds |
| **SP Gap** | ≤ 70% of best baseline (≥30% reduction) | Measure on all datasets, all seeds |
| **Accuracy** | No regression > 1% absolute | Compare to best accuracy baseline |
| **AUROC** | No regression > 1% absolute | Compare to best AUROC baseline |
| **Macro-F1** | No regression > 1% absolute | Compare to best macro-F1 baseline |
| **Stability** | No exploding updates | Check gradient norms < clip threshold |
| **Backward Compatibility** | Default config = current behavior | Unit tests pass |

---

## Phase 9: Estimated Code Changes

| Component | Lines of Code | Files Modified/Created |
|-----------|---------------|------------------------|
| Configuration system | 300 | `faircare/config/faircare_fl_config.py` (NEW) |
| Backward compatibility | 200 | `faircare/algos/faircare_fl.py` (MODIFIED) |
| Multi-objective aggregation | 400 | `faircare/algos/faircare_fl.py` (MODIFIED) |
| Bias monitoring & policy | 250 | `faircare/algos/faircare_fl.py` (MODIFIED) |
| Fair client selection | 200 | `faircare/core/fair_selection.py` (NEW) |
| Local debiasing | 500 | `faircare/core/local_debiasing.py` (NEW) |
| Augmentation | 150 | `faircare/core/augmentation.py` (NEW) |
| Logging enhancements | 200 | `faircare/algos/faircare_fl.py` (MODIFIED) |
| Tests | 600 | `tests/test_faircare_fl_enhancements.py` (NEW) |
| Documentation | 200 | Multiple README updates |
| **TOTAL** | **~3000 lines** | **8 files** |

---

## Current Status

**Comprehensive evaluation** (72 experiments) is currently running. Once complete (~2-3 hours), we will have baseline performance data to validate the enhancements against.

**Next Steps**:
1. Wait for evaluation to complete
2. Analyze baseline results
3. Begin phased implementation per timeline above
4. Validate success criteria after each phase

---

**This document serves as the complete specification for the FairCare-FL enhancement project.**
