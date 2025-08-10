from __future__ import annotations
from typing import Dict, Any, List
import torch
from torch import nn

class BaseAggregator:
    def __init__(self, sens_present: bool):
        self.sens_present = sens_present

    def client_weights_signal(self) -> List[float]:
        # Default uniform factor for client-local loss scaling
        return self._uniform(10)  # will be overridden via compute_weights call

    def _uniform(self, n: int) -> List[float]:
        return [1.0 for _ in range(n)]

    def compute_weights(self, local_meta: List[Dict[str, Any]]) -> List[float]:
        # Default FedAvg (uniform)
        n = len(local_meta)
        return [1.0 / n for _ in range(n)]

class FairMomentumAggregator:
    """
    Keeps a momentum vector in parameter space and projects aggregated update
    back onto fairness-improving direction if provided.
    """
    def __init__(self, beta: float, model: nn.Module, device: torch.device):
        self.beta = beta
        self.device = device
        self.shapes = [p.shape for p in model.state_dict().values()]
        self.m = torch.zeros(sum(p.numel() for p in model.state_dict().values()), device=device)

    def apply_update(self, base_state: Dict[str, torch.Tensor], agg_vec: torch.Tensor):
        self.m = self.beta * self.m + (1 - self.beta) * agg_vec
        vec = self.m
        out, ptr = {}, 0
        for (k, v) in base_state.items():
            num = v.numel()
            upd = vec[ptr:ptr+num].view_as(v).to(v.dtype).to(v.device)
            out[k] = v + upd
            ptr += num
        return out
