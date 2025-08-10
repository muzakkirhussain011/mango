# faircare/algos/faircare_f1.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FairCareF1Config:
    name: str = "faircare_f1"
    local_epochs: int = 1
    lr: float = 1e-2
    weight_decay: float = 0.0
    beta_momentum: float = 0.7
    # fairness constraints
    target_eo: float = 0.03
    target_sp: float = 0.03
    dual_step: float = 0.05  # step for dual ascent
    max_dual: float = 10.0
    # server-side reweighting / sampling
    client_reweight_gamma: float = 0.2
    client_resample_frac: float = 1.0  # cross-silo: often 1.0
    # post-processing thresholds
    postprocess_equalized_odds: bool = True
    # stability
    max_weight_scale: float = 2.0
    min_weight_scale: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
