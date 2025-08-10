from __future__ import annotations
import random, os
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn, optim

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(pref: str) -> torch.device:
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def optimizer_for(model: nn.Module, lr: float, wd: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    a: torch.Tensor | None  # sensitive attribute, optional

def to_device(batch: Batch, device: torch.device) -> Batch:
    a = None if batch.a is None else batch.a.to(device)
    return Batch(x=batch.x.to(device), y=batch.y.to(device), a=a)

def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += (p.grad.detach().data.norm(2) ** 2).item()
    return total ** 0.5
