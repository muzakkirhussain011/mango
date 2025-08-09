# faircare/algos/aggregator.py
import math
from typing import Dict, List, Tuple, Sequence
import numpy as np

class FedAvgAggregator:
    """Baseline FedAvg aggregator kept for compatibility."""
    def __call__(self, weights: List[Tuple[np.ndarray, float]]):
        # weights: list of (delta, client_weight)
        total = sum(w for _, w in weights) + 1e-12
        agg = sum(delta * (w / total) for delta, w in weights)
        return agg


def _extract_reports(payloads: Sequence[Dict]) -> List[Dict]:
    """Normalize a variety of legacy payload formats.

    Returns a list of dictionaries with ``loss``, ``gap`` and ``num_samples``
    keys populated (missing values default to ``0``).
    """

    reports: List[Dict] = []
    for pl in payloads:
        if isinstance(pl, dict):
            rep = pl.get("report", pl)
        elif isinstance(pl, (list, tuple)) and pl and isinstance(pl[0], dict):
            rep = pl[0]
        else:
            raise TypeError("Unsupported payload format for weighting")

        loss = rep.get("loss", rep.get("val_loss", rep.get("train_loss", 0.0)))
        gap = rep.get("gap")
        if gap is None:
            summary = rep.get("summary", {})
            if isinstance(summary, dict):
                gap = summary.get("dp_gap", summary.get("eo_gap", 0.0))
            else:
                gap = 0.0
        num = rep.get("num_samples", rep.get("n", rep.get("num", 0)))
        reports.append({"loss": float(loss), "gap": float(gap), "num_samples": int(num)})
    return reports


def weights_fedavg(payloads: Sequence[Dict]) -> np.ndarray:
    """Standard FedAvg weighting based on client sample counts."""

    reps = _extract_reports(payloads)
    ns = np.array([r.get("num_samples", 0) for r in reps], dtype=np.float64)
    total = ns.sum() + 1e-12
    return ns / total


def weights_fairfed(payloads: Sequence[Dict], eps: float = 1e-6) -> np.ndarray:
    """FairFed-style weights favouring clients with larger fairness gaps."""

    reps = _extract_reports(payloads)
    gaps = np.array([r.get("gap", 0.0) for r in reps], dtype=np.float64)
    inv = 1.0 / (np.maximum(gaps, eps))
    inv = inv / (inv.sum() + 1e-12)
    return inv


def weights_qffl(payloads: Sequence[Dict], q: float) -> np.ndarray:
    """q-FFL weighting – emphasises high-loss clients."""

    reps = _extract_reports(payloads)
    losses = np.array([max(r.get("loss", 1e-12), 1e-12) for r in reps], dtype=np.float64)
    w = losses ** q
    return w / (w.sum() + 1e-12)


def weights_afl(payloads: Sequence[Dict], boost: float = 5.0) -> np.ndarray:
    """AFL weighting using an exponential boost on client losses."""

    reps = _extract_reports(payloads)
    losses = np.array([r.get("loss", 0.0) for r in reps], dtype=np.float64)
    w = np.exp(boost * losses)
    return w / (w.sum() + 1e-12)

def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize a list of weights so they sum to one."""
    w = np.asarray(list(weights), dtype=np.float64)
    return w / (w.sum() + 1e-12)

def weights_faircare(
    payloads: Sequence[Dict],
    q: float = 0.5,
    gamma: float = 1.0,
    eps: float = 1e-3,
    clip: float = 5.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Legacy FairCare weighting helper."""
    reps = _extract_reports(payloads)
    fw = FairnessAwareWeights(q=q, gamma=gamma, eps=eps, clip=clip, temperature=temperature)
    return fw(reps)

class FairnessAwareWeights:
    """
    Compute client aggregation weights with fairness-awareness and temperature.
    w_i ∝ (loss_i)^q * 1/(gap_i + eps)^(gamma)
    Then normalized; supports clipping to avoid domination.
    """
    def __init__(self, q: float = 0.5, gamma: float = 1.0, eps: float = 1e-3,
                 clip: float = 5.0, temperature: float = 1.0):
        self.q = q
        self.gamma = gamma
        self.eps = eps
        self.clip = clip
        self.temperature = temperature

    def __call__(self, client_reports: List[Dict]) -> np.ndarray:
        # Each report should contain: {"loss": float, "gap": float, "num_samples": int}
        losses = np.array([max(r.get("loss", 1e-6), 1e-6) for r in client_reports], dtype=np.float64)
        gaps = np.array([max(r.get("gap", 0.0), 0.0) for r in client_reports], dtype=np.float64)

        raw = (losses ** self.q) * (1.0 / np.maximum(gaps, self.eps) ** self.gamma)
        raw = np.clip(raw, 1e-8, self.clip)
        # temperature softmax for stability
        logits = np.log(raw + 1e-12)
        w = np.exp(logits / max(self.temperature, 1e-6))
        w = w / (w.sum() + 1e-12)
        return w

class FedAdamAggregator:
    """
    FedOpt / FedAdam server optimizer (Reddi et al., ICLR'21).
    Keeps first/second moments of the *aggregated* delta; supports fairness-aware momentum.
    """
    def __init__(self, dim: int, lr: float = 1.0, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, agg_delta: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * agg_delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (agg_delta ** 2)
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * (mhat / (np.sqrt(vhat) + self.eps))

def flatten_params(param_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([p.ravel() for p in param_list]).astype(np.float64, copy=False)

def unflatten_params(flat: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
    out, idx = [], 0
    for t in template:
        size = t.size
        out.append(flat[idx:idx+size].reshape(t.shape))
        idx += size
    return out
