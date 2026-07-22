"""FedGMA - Federated Gated Mixture-of-Aggregators with No-Regret Fair Blending.

Casts fair-FL aggregation-rule selection as full-information online convex optimization.
Each round a bank of M *correctly implemented* aggregation rules (experts) maps the client
reports to a per-client simplex weight vector w^(m) in Delta_{K-1}. A server-side convex
surrogate

    L_t(w) = sum_k w_k * loss_k  +  sum_c lambda_c * | sum_k w_k * dr^(c)_k |

(linear utility term + |linear| group-gap terms => convex in w) is evaluable at every expert's
w^(m) from client reports alone, without deploying any candidate. A Hedge / exponentiated-weights
update over the M experts then blends them:

    w_t = sum_m p_{t,m} w^(m)_t ,   p_{t+1,m} ∝ p_{t,m} exp(-eta_g * L_t(w^(m)_t))

Because a convex combination of simplex vectors is a simplex vector, w_t is a valid aggregator by
construction, and by convexity + Jensen + Hedge the blend has regret O(sqrt(T ln M)) against the
best single rule in hindsight -- i.e. it is asymptotically no worse than the best baseline in the
bank, self-tuning per dataset with a certificate. Slow-timescale dual ascent on the group-fairness
gaps couples in as a primal-dual saddle point (eta_lambda << eta_g).

This module is the "spine" (GMA-Hedge): fixed expert bank + convex surrogate + Hedge + slow duals
+ weight post-processing. The learned gate, online DFBD training, and distillation hull-escape are
deferred ablations. See research/FEDGMA_DESIGN.md for the full spec, theory, and citations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import torch

from faircare.algos.aggregator import BaseAggregator, register_aggregator


# ── per-client feature extractors (permissive to key naming) ──────────────────
def _num_samples(s: Dict[str, Any]) -> float:
    for k in ("n_samples", "num_samples", "num_examples", "samples", "dataset_size", "size", "n"):
        v = s.get(k)
        if v:
            try:
                return float(v)
            except Exception:
                pass
    return 1.0


def _loss(s: Dict[str, Any]) -> float:
    for k in ("val_loss", "train_loss", "loss"):
        v = s.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return 1.0


def _group_rates(gc: Any) -> Dict[Any, Tuple[float, float, float, float]]:
    """{group_id: {TP,FP,TN,FN}} -> {group_id: (tpr, fpr, ppr, err)}. Robust to key case."""
    out: Dict[Any, Tuple[float, float, float, float]] = {}
    if not isinstance(gc, dict):
        return out
    for gid, st in gc.items():
        if not isinstance(st, dict):
            continue

        def g(*names: str) -> float:
            for nm in names:
                if nm in st:
                    try:
                        return float(st[nm])
                    except Exception:
                        return 0.0
            return 0.0

        tp, fp, tn, fn = g("TP", "tp"), g("FP", "fp"), g("TN", "tn"), g("FN", "fn")
        n = tp + fp + tn + fn
        if n <= 0:
            continue
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        ppr = (tp + fp) / n
        err = (fp + fn) / n
        out[gid] = (tpr, fpr, ppr, err)
    return out


def _client_gap_deltas(gc: Any) -> Tuple[float, float, float, bool]:
    """Signed per-client group-gap contributions (rate[first group] - rate[last group]) for
    EO (tpr), FPR, SP (ppr). Consistent ordering (sorted group ids) makes the signs comparable
    across clients so cancellation in sum_k w_k dr_k is meaningful. Returns (d_eo,d_fpr,d_sp,ok)."""
    rates = _group_rates(gc)
    gids = sorted(rates.keys(), key=lambda x: (str(type(x)), x))
    if len(gids) < 2:
        return 0.0, 0.0, 0.0, False
    a, b = rates[gids[0]], rates[gids[-1]]
    return (a[0] - b[0]), (a[1] - b[1]), (a[2] - b[2]), True


def _client_worst_group_risk(gc: Any, wg_f1_fallback: float) -> float:
    rates = _group_rates(gc)
    if rates:
        return max(r[3] for r in rates.values())  # worst per-group error rate
    try:
        return max(0.0, 1.0 - float(wg_f1_fallback))
    except Exception:
        return 0.0


def _simplex(v: torch.Tensor) -> torch.Tensor:
    v = torch.clamp(v.to(torch.float32), min=0.0)
    s = float(v.sum())
    if s > 0:
        return v / s
    return torch.ones_like(v) / max(1, v.numel())


@register_aggregator("fedgma")
class FedGMAAggregator(BaseAggregator):
    """No-regret mixture of fair-FL aggregators (see module docstring / FEDGMA_DESIGN.md)."""

    def __init__(
        self,
        n_clients: int,
        total_rounds: int = 40,
        q_values: Tuple[float, ...] = (0.5, 2.0),
        afl_tau: float = 1.0,
        dro_tau: float = 0.5,
        eps_eo: float = 0.015,
        eps_fpr: float = 0.015,
        eps_sp: float = 0.02,
        dual_lr: float = 0.02,
        dual_max: float = 5.0,
        fairfed_eps: float = 1e-3,
        epsilon: float = 0.0,
        weight_clip: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_clients=n_clients, epsilon=epsilon, weight_clip=weight_clip)
        self.total_rounds = int(total_rounds) if total_rounds else 40
        self.q_values = list(q_values)
        self.afl_tau = float(afl_tau)
        self.dro_tau = float(dro_tau)
        self.eps = {"eo": float(eps_eo), "fpr": float(eps_fpr), "sp": float(eps_sp)}
        self.dual_lr = float(dual_lr)   # eta_lambda (slow timescale, << eta_g)
        self.dual_max = float(dual_max)
        self.fairfed_eps = float(fairfed_eps)

        # Expert bank: FedAvg, real q-FFL (one per q), real AFL/minimax, FairFed, group-DRO.
        self.expert_names = (
            ["fedavg"]
            + ["qffl_q%s" % q for q in self.q_values]
            + ["afl", "fairfed", "dro"]
        )
        self.M = len(self.expert_names)
        # Hedge state (log-domain for stability); uniform prior.
        self.log_p = torch.zeros(self.M, dtype=torch.float32)
        self.eta_g = math.sqrt(8.0 * math.log(max(self.M, 2)) / max(self.total_rounds, 1))
        self.lam = {"eo": 0.0, "fpr": 0.0, "sp": 0.0}
        self.round = 0
        self.last_logs: Dict[str, Any] = {}

    # ── expert bank ──────────────────────────────────────────────────────────
    def _build_experts(self, sizes, losses, local_gap, wgr) -> torch.Tensor:
        p_k = _simplex(sizes)
        experts = [
            _simplex(sizes),                                                  # FedAvg (sample-prop)
        ]
        for q in self.q_values:                                              # real q-FFL: p_k * loss^q
            experts.append(_simplex(p_k * torch.clamp(losses, min=0.0).pow(q)))
        # real AFL / minimax: up-weight the worst client (stabilized softmax of loss)
        experts.append(_simplex(torch.softmax((losses - losses.max()) / self.afl_tau, dim=0)))
        # FairFed: up-weight locally-fair clients (inverse local group gap)
        experts.append(_simplex(1.0 / (local_gap + self.fairfed_eps)))
        # group-DRO / FedMinMax flavour: up-weight clients with high worst-group risk
        experts.append(_simplex(torch.softmax(wgr / self.dro_tau, dim=0)))
        return torch.stack(experts, dim=0)  # [M, K]

    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        n = len(client_summaries)
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        if n == 1:
            return torch.ones(1, dtype=torch.float32)

        sizes = torch.tensor([_num_samples(s) for s in client_summaries], dtype=torch.float32)
        losses = torch.tensor([_loss(s) for s in client_summaries], dtype=torch.float32)
        d_eo = torch.zeros(n); d_fpr = torch.zeros(n); d_sp = torch.zeros(n)
        wgr = torch.zeros(n); local_gap = torch.zeros(n)
        for i, s in enumerate(client_summaries):
            gc = s.get("group_counts", {})
            de, df, ds, _ok = _client_gap_deltas(gc)
            d_eo[i], d_fpr[i], d_sp[i] = de, df, ds
            local_gap[i] = abs(de) + abs(df) + abs(ds)
            wgr[i] = _client_worst_group_risk(gc, s.get("wg_f1", s.get("worst_group_f1", 1.0)))

        W = self._build_experts(sizes, losses, local_gap, wgr)   # [M, K], each row a simplex

        # Current Hedge distribution over experts, then blend (use p_t to form w_t).
        p = torch.softmax(self.log_p, dim=0)                     # [M]
        w = _simplex((p.unsqueeze(1) * W).sum(dim=0))            # [K]

        # Convex surrogate evaluated at every expert w^(m): L_t(w^(m)).
        util = W @ losses                                        # [M] linear utility
        eo_t = (W @ d_eo).abs(); fpr_t = (W @ d_fpr).abs(); sp_t = (W @ d_sp).abs()
        Lm = util + self.lam["eo"] * eo_t + self.lam["fpr"] * fpr_t + self.lam["sp"] * sp_t

        # Hedge update on the surrogate, normalized to [0,1] each round for a stable step.
        lo, hi = float(Lm.min()), float(Lm.max())
        Ln = (Lm - lo) / (hi - lo + 1e-8)
        self.log_p = self.log_p - self.eta_g * Ln
        self.log_p = self.log_p - float(self.log_p.max())        # log-domain stabilization

        # Slow dual ascent on the blended round gaps (eta_lambda = dual_lr << eta_g).
        gaps = {
            "eo": float((w @ d_eo).abs()),
            "fpr": float((w @ d_fpr).abs()),
            "sp": float((w @ d_sp).abs()),
        }
        for c in self.lam:
            self.lam[c] = float(min(self.dual_max, max(0.0, self.lam[c] + self.dual_lr * (gaps[c] - self.eps[c]))))

        self.round += 1
        self.last_logs = {
            "fedgma/round": self.round,
            "fedgma/eta_g": self.eta_g,
            "fedgma/gap_eo": gaps["eo"], "fedgma/gap_fpr": gaps["fpr"], "fedgma/gap_sp": gaps["sp"],
            "fedgma/lambda_eo": self.lam["eo"], "fedgma/lambda_fpr": self.lam["fpr"],
            "fedgma/lambda_sp": self.lam["sp"],
            "fedgma/alpha": {name: float(a) for name, a in zip(self.expert_names, p.tolist())},
            "fedgma/surrogate_per_expert": {name: float(l) for name, l in zip(self.expert_names, Lm.tolist())},
        }
        return self._postprocess(w)

    # Duck-typed hook some server paths look for.
    def get_statistics(self) -> Dict[str, Any]:
        return dict(self.last_logs)
