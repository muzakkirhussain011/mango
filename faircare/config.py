# faircare/config.py
from dataclasses import dataclass

@dataclass
class Defaults:
    seed: int = 42
    num_clients: int = 5
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    beta_momentum: float = 0.9   # FAIR-FATE
    lambdaG: float = 2.0         # group fairness
    lambdaC: float = 0.5         # client fairness proxy
    lambdaA: float = 0.5         # agnostic/worst-case penalty strength
    q: float = 0.5               # q-FFL exponent
    dirichlet_alpha: float = 0.5
    sensitive_attr: str = "sex"
