# faircare/core/evaluation.py
from typing import Dict, Any
import numpy as np
import torch
from torch import nn
from faircare.fairness.metrics import compute_metrics, threshold_sweep

@torch.no_grad()
def evaluate_union(model: nn.Module,
                   X: np.ndarray, y: np.ndarray, s: np.ndarray,
                   threshold: float = 0.5,
                   device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate the *global* model on the union validation split and compute
    global Accuracy, AUROC, DP/EO/FPR gaps, and ECE at a fixed threshold.
    """
    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t)                           # [N, 2]
    probs = torch.softmax(logits, dim=1)[:, 1]   # P(y=1)
    y_prob = probs.detach().cpu().numpy()
    return compute_metrics(y_true=y, y_prob=y_prob, s=s, threshold=threshold)

@torch.no_grad()
def evaluate_with_sweep(model: nn.Module,
                        X: np.ndarray, y: np.ndarray, s: np.ndarray,
                        device: str = "cpu") -> Dict[str, Any]:
    """
    Sweep thresholds in [0.05, 0.95] and return:
      - best_acc: highest accuracy config
      - best_eo : minimal equal-opportunity gap config
      - best_dp : minimal demographic parity gap config
    """
    model.eval().to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(X_t)                           # [N, 2]
    probs = torch.softmax(logits, dim=1)[:, 1]
    y_prob = probs.detach().cpu().numpy()
    return threshold_sweep(y_true=y, y_prob=y_prob, s=s)
