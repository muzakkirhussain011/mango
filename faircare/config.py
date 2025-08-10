from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

@dataclass
class AlgoConfig:
    name: str = "faircare_fl"
    lambda_g: float = 0.0
    lambda_c: float = 0.0
    beta_momentum: float = 0.8
    fairness_mode: str = "penalty"  # "penalty" or "adversarial"
    qffl_q: float = 0.5
    fedprox_mu: float = 0.0
    afl_lr: float = 0.05  # step for dual weights
    max_rounds: int = 100
    local_epochs: int = 2
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    target_fair_gap: float = 0.05
    adapt_every: int = 2
    adapt_scale: float = 1.15
    seed: int = 1337
    device: str = "auto"

@dataclass
class DataConfig:
    name: str = "adult"
    n_clients: int = 10
    dirichlet_alpha: float = 0.5
    sensitive_feature: str = "sex"  # or "race", dataset-specific
    cache_dir: str = "data"

@dataclass
class ExperimentConfig:
    algorithm: AlgoConfig = field(default_factory=AlgoConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "reports"
    log_every: int = 1
    eval_every: int = 1

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_config(path: str | Path) -> ExperimentConfig:
    cfg = load_yaml(path)
    algo = AlgoConfig(**cfg.get("algorithm", {}))
    data = DataConfig(**cfg.get("data", {}))
    return ExperimentConfig(algorithm=algo, data=data,
                            output_dir=cfg.get("output_dir", "reports"),
                            log_every=cfg.get("log_every", 1),
                            eval_every=cfg.get("eval_every", 1))
