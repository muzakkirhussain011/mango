# faircare/algos/aggregator_f1.py
from typing import Dict, List, Tuple
import copy
import math
import torch

def _state_add(dst, src, alpha=1.0):
    for k in dst.keys():
        dst[k] = dst[k] + alpha * src[k]
    return dst

def _state_sub(a, b):
    out = {}
    for k in a.keys():
        out[k] = a[k] - b[k]
    return out

def _state_scale(a, s):
    out = {}
    for k, v in a.items():
        out[k] = v * s
    return out

def _zeros_like(state):
    return {k: torch.zeros_like(v) for k, v in state.items()}

def _norm(state):
    return math.sqrt(sum((v.float()**2).sum().item() for v in state.values()))

def _project_to_not_increase_gap(delta, fairness_grad, max_shrink=0.5):
    """
    Simple safeguard: if <delta, fairness_grad> > 0 (would increase gap),
    shrink the component along fairness_grad.
    """
    # compute inner product
    num = 0.0
    den = 0.0
    for k in delta.keys():
        d = delta[k].float()
        g = fairness_grad[k].float()
        num += (d * g).sum().item()
        den += (g * g).sum().item()
    if den <= 1e-12:
        return delta
    if num > 0:  # moving in the wrong direction
        # subtract tau * g where tau = min(max_shrink, num/den)
        tau = min(max_shrink, num / den)
        out = {}
        for k in delta.keys():
            out[k] = delta[k] - tau * fairness_grad[k]
        return out
    return delta

class FairCareF1Aggregator:
    """
    Fairness-aware aggregator with dual control and momentum projection.
    Expects:
      - per-client model deltas (state_dict-like tensors)
      - client_sizes
      - aggregated global stats to compute EO/SP gaps, and a fairness "gradient" proxy
    """
    def __init__(self, beta=0.7, target_eo=0.03, target_sp=0.03, dual_step=0.05, max_dual=10.0,
                 max_weight_scale=2.0, min_weight_scale=0.25):
        self.beta = beta
        self.target_eo = target_eo
        self.target_sp = target_sp
        self.dual_step = dual_step
        self.max_dual = max_dual
        self.lambda_eo = 0.0
        self.lambda_sp = 0.0
        self.momentum = None
        self.max_weight_scale = max_weight_scale
        self.min_weight_scale = min_weight_scale

    def client_weights_signal(self, n_clients: int = None):
        # length-safe baseline signal: uniform (server will shape final weights)
        if n_clients is None:
            return None
        import numpy as np
        w = np.ones(n_clients, dtype=float)
        w /= w.sum()
        return w

    def _compute_group_gaps(self, global_stats: Dict[str, float]) -> Tuple[float, float]:
        eo_gap = abs(global_stats.get("EO_gap", 0.0))
        sp_gap = abs(global_stats.get("SP_gap", 0.0))
        return eo_gap, sp_gap

    def _update_duals(self, eo_gap, sp_gap):
        # primal-dual (ascent on duals when constraint violated)
        if eo_gap > self.target_eo:
            self.lambda_eo = min(self.max_dual, self.lambda_eo + self.dual_step * (eo_gap - self.target_eo))
        else:
            self.lambda_eo = max(0.0, self.lambda_eo - self.dual_step * (self.target_eo - eo_gap))
        if sp_gap > self.target_sp:
            self.lambda_sp = min(self.max_dual, self.lambda_sp + self.dual_step * (sp_gap - self.target_sp))
        else:
            self.lambda_sp = max(0.0, self.lambda_sp - self.dual_step * (self.target_sp - sp_gap))

    def _fairness_gradient_proxy(self, deltas: List[Dict[str, torch.Tensor]],
                                 client_group_focus: List[float]):
        """
        Build a proxy fairness gradient as a weighted sum of client deltas emphasizing
        clients with more presence of disadvantaged groups (client_group_focus given
        by server using current worst-group coverage).
        """
        agg = None
        for d, w in zip(deltas, client_group_focus):
            if agg is None:
                agg = _state_scale(d, w)
            else:
                agg = _state_add(agg, d, alpha=w)
        if agg is None:
            return None
        # normalize direction
        n = _norm(agg)
        if n > 1e-12:
            return _state_scale(agg, 1.0 / n)
        return agg

    def aggregate(self,
                  global_state: Dict[str, torch.Tensor],
                  client_states: List[Dict[str, torch.Tensor]],
                  client_bases: List[Dict[str, torch.Tensor]],
                  client_sizes: List[int],
                  global_stats: Dict[str, float],
                  client_group_focus: List[float]):
        """
        Args:
          global_state: current W_t
          client_states: list of client new params W_i
          client_bases: list of copies of W_t for each client to compute deltas
          client_sizes: |D_i|
          global_stats: dict with EO_gap, SP_gap, etc.
          client_group_focus: emphasis weight per client in favor of worst-off group (>=0)
        Returns:
          new_global_state
        """
        # compute deltas
        deltas = []
        for W_i, W0 in zip(client_states, client_bases):
            deltas.append(_state_sub(W_i, W0))

        # update duals
        eo_gap, sp_gap = self._compute_group_gaps(global_stats)
        self._update_duals(eo_gap, sp_gap)

        # base weights proportional to data size
        total = float(sum(client_sizes)) + 1e-12
        base_w = [s / total for s in client_sizes]

        # fairness scaling: (1 + λ_eo + λ_sp) * focus  (bounded)
        scales = []
        fairness_amp = 1.0 + self.lambda_eo + self.lambda_sp
        for b, f in zip(base_w, client_group_focus):
            scale = b * (1.0 + fairness_amp * f)
            scales.append(scale)
        ssum = sum(scales) + 1e-12
        scales = [max(self.min_weight_scale * b, min(self.max_weight_scale * b, s)) for s, b in zip(scales, base_w)]
        # renormalize
        ssum = sum(scales) + 1e-12
        scales = [s / ssum for s in scales]

        # aggregate preliminary delta
        prelim = None
        for d, w in zip(deltas, scales):
            if prelim is None:
                prelim = _state_scale(d, w)
            else:
                prelim = _state_add(prelim, d, alpha=w)

        # fairness gradient proxy for projection
        fgrad = self._fairness_gradient_proxy(deltas, client_group_focus)
        if fgrad is not None:
            prelim = _project_to_not_increase_gap(prelim, fgrad, max_shrink=0.5)

        # momentum
        if self.momentum is None:
            self.momentum = _zeros_like(global_state)
        update = copy.deepcopy(prelim)
        update = _state_add(update, self.momentum, alpha=self.beta)
        self.momentum = copy.deepcopy(update)

        # apply
        new_global = copy.deepcopy(global_state)
        new_global = _state_add(new_global, update, alpha=1.0)
        return new_global
