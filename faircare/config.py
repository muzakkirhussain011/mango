"""Configuration management with FairCare-FL++ support."""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: str = "mlp"
    input_dim: int = 30
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 1
    dropout: float = 0.2
    activation: str = "relu"


@dataclass
class DataConfig:
    """Data configuration."""
    dataset: str = "adult"
    sensitive_attribute: Optional[str] = "sex"
    n_clients: int = 10
    partition: str = "dirichlet"
    alpha: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    batch_size: int = 32
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training configuration."""
    algo: str = "faircare_fl"
    rounds: int = 20
    local_epochs: int = 1
    lr: float = 0.01
    weight_decay: float = 0.0
    momentum: float = 0.0
    server_lr: float = 1.0
    eval_every: int = 1
    checkpoint_every: int = 5
    early_stopping_rounds: Optional[int] = None
    device: str = "cpu"


@dataclass
class FairnessConfig:
    """Enhanced fairness configuration for FairCare-FL++."""
    # Fairness metric weights
    alpha: float = 1.0          # EO gap weight
    beta: float = 0.5           # FPR gap weight
    gamma: float = 0.5          # SP gap weight
    delta: float = 0.2          # Accuracy weight
    delta_init: float = 0.2     # Initial delta value
    delta_min: float = 0.01     # Minimum delta in bias mode
    
    # Temperature parameters
    tau: float = 1.0            # Initial temperature
    tau_init: float = 1.0       # Initial temperature (alias)
    tau_min: float = 0.1        # Minimum temperature
    tau_anneal: bool = True     # Enable temperature annealing
    tau_anneal_rate: float = 0.95  # Temperature annealing rate
    
    # Momentum parameters
    mu: float = 0.9             # Client-side momentum (mu_client)
    mu_client: float = 0.9      # Client-side momentum
    theta_server: float = 0.8   # Server-side momentum
    
    # Fairness penalty for local training
    lambda_fair: float = 0.1    # Current fairness penalty weight
    lambda_fair_init: float = 0.1    # Initial fairness penalty weight
    lambda_fair_min: float = 0.01     # Minimum lambda
    lambda_fair_max: float = 2.0     # Maximum lambda
    lambda_adapt_rate: float = 1.2  # Lambda adjustment rate
    
    # Bias detection thresholds
    bias_threshold_eo: float = 0.15   # EO gap threshold
    bias_threshold_fpr: float = 0.15  # FPR gap threshold
    bias_threshold_sp: float = 0.10    # SP gap threshold
    thr_eo: float = 0.15           # EO gap threshold (alias)
    thr_fpr: float = 0.15          # FPR gap threshold (alias)
    thr_sp: float = 0.10           # SP gap threshold (alias)
    
    # Client fairness loss weights
    w_eo: float = 1.0              # EO loss weight
    w_fpr: float = 0.5             # FPR loss weight
    w_sp: float = 0.5              # SP loss weight
    
    # Weight constraints
    epsilon: float = 0.01       # Weight floor
    weight_clip: float = 10.0   # Weight ceiling multiplier
    
    # Advanced features
    enable_bias_detection: bool = True
    enable_server_momentum: bool = True
    enable_multi_metric: bool = True
    variance_penalty: float = 0.1
    improvement_bonus: float = 0.1
    participation_boost: float = 0.15
    fairness_loss_type: str = "eo_sp_combined"  # Type of fairness loss for local training


@dataclass
class SecureAggConfig:
    """Secure aggregation configuration."""
    enabled: bool = False
    protocol: str = "additive_masking"
    precision: int = 16
    modulus: int = 2**32


@dataclass
class AlgoConfig:
    """Algorithm-specific configuration."""
    # FedProx
    fedprox_mu: float = 0.01
    
    # q-FFL
    q: float = 2.0
    q_eps: float = 1e-4
    
    # AFL
    afl_lambda: float = 0.1
    afl_smoothing: float = 0.01
    
    # FairCare-FL++ specific
    faircare_momentum: float = 0.9
    faircare_anneal_rounds: int = 5
    convergence_threshold: float = 0.01
    
    # Bias mitigation
    bias_mitigation_extra_epochs: int = 1
    bias_mitigation_lr_multiplier: float = 1.2


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with FairCare-FL++ support."""
    name: str = "faircare_fl_plus_plus"
    seed: int = 42
    logdir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)
    secure_agg: SecureAggConfig = field(default_factory=SecureAggConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        config = cls()
        
        # Update nested configs
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "fairness" in data:
            config.fairness = FairnessConfig(**data["fairness"])
        if "secure_agg" in data:
            config.secure_agg = SecureAggConfig(**data["secure_agg"])
        if "algo" in data:
            config.algo = AlgoConfig(**data["algo"])
        
        # Update top-level fields
        for key in ["name", "seed", "logdir"]:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def update_from_args(self, args: Any) -> None:
        """Update config from command-line arguments."""
        # Map CLI args to config fields
        arg_mapping = {
            "algo": ("training", "algo"),
            "dataset": ("data", "dataset"),
            "sensitive": ("data", "sensitive_attribute"),
            "clients": ("data", "n_clients"),
            "rounds": ("training", "rounds"),
            "local_epochs": ("training", "local_epochs"),
            "lr": ("training", "lr"),
            "weight_decay": ("training", "weight_decay"),
            "seed": (None, "seed"),
            "logdir": (None, "logdir"),
            "device": ("training", "device"),
        }
        
        for arg_name, (config_section, config_field) in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                if config_section is None:
                    # Handle top-level fields
                    field_name = config_field if config_field else arg_name
                    setattr(self, field_name, getattr(args, arg_name))
                else:
                    section = getattr(self, config_section)
                    if config_field:
                        setattr(section, config_field, getattr(args, arg_name))
                    else:
                        setattr(section, arg_name, getattr(args, arg_name))
    
    def get_fairness_config_for_client(self) -> Dict[str, Any]:
        """Get fairness configuration to pass to clients."""
        return {
            'lambda_fair': self.fairness.lambda_fair,
            'lambda_min': self.fairness.lambda_fair_min,
            'lambda_max': self.fairness.lambda_fair_max,
            'fairness_loss_type': self.fairness.fairness_loss_type,
            'bias_thresholds': {
                'eo': self.fairness.bias_threshold_eo,
                'fpr': self.fairness.bias_threshold_fpr,
                'sp': self.fairness.bias_threshold_sp
            }
        }
