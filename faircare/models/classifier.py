from __future__ import annotations
import torch
from torch import nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def make_classifier(input_dim: int, n_classes: int) -> nn.Module:
    return MLPClassifier(input_dim, n_classes)
