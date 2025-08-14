"""Configuration management with dataclasses and YAML support."""
from typing import Optional, Dict, Any, List

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
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
    algo: str = "fedavg"
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
    """Fairness configuration."""
    alpha: float = 1.0  # EO gap weight
    beta: float = 0.5   # FPR gap weight
    gamma: float = 0.5  # SP gap weight
    delta: float = 0.1  # val loss weight
    tau: float = 1.0    # temperature
    mu: float = 0.9     # momentum
    epsilon: float = 0.01  # weight floor
    tau_anneal: bool = False
    weight_clip: float = 10.0


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
    
    # FairCare-FL
    faircare_momentum: float = 0.9
    faircare_anneal_rounds: int = 5


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "experiment"
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
            "seed": ("seed", None),
            "logdir": ("logdir", None),
        }
        
        for arg_name, (config_section, config_field) in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                if config_section is None:
                    setattr(self, arg_name, getattr(args, arg_name))
                else:
                    section = getattr(self, config_section)
                    if config_field:
                        setattr(section, config_field, getattr(args, arg_name))
                    else:
                        setattr(section, arg_name, getattr(args, arg_name))
