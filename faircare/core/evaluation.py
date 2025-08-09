# faircare/core/evaluation.py
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch import nn
from .utils import to_device
from faircare.fairness.metrics import compute_metrics, threshold_sweep

@torch.no_grad()
def evaluate_union(model: nn.Module, X: np.ndarray, y: np.ndarray, s: np.ndarray,
                   threshold: float = 0.5, device: str = "cpu") -> Dict[str, float]:
    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t).cpu().numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return compute_metrics(y_true=y, y_prob=probs, s=s, threshold=threshold)

@torch.no_grad()
def evaluate_with_sweep(model: nn.Module, X: np.ndarray, y: np.ndarray, s: np.ndarray,
                        device: str = "cpu") -> Dict[str, Any]:
    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t).cpu().numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    best = threshold_sweep(y_true=y, y_prob=probs, s=s)
    return best
