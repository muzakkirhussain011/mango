from __future__ import annotations
import torch
from torch import nn
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class Adversary(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64, n_groups: int = 2):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_groups)
        )

    def forward(self, rep):
        return self.clf(rep)

class AdversaryLoss(nn.Module):
    """
    Uses gradient reversal on logits to make sensitive attribute prediction hard.
    """
    def __init__(self, lambd: float = 1.0, n_groups: int = 2):
        super().__init__()
        self.lambd = lambd
        self.adv = Adversary(in_dim=2, n_groups=n_groups)  # if logits are 2-class
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, a):
        rev = GradReverse.apply(logits, self.lambd)
        pred = self.adv(rev)
        return self.ce(pred, a)
