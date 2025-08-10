from __future__ import annotations
from typing import Dict, Any, List, Tuple
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
from .secure_agg import SecureAggregator
from ..algos.aggregator import BaseAggregator, FairMomentumAggregator
from ..fairness.global_stats import GlobalStats
from ..core.evaluation import evaluate

class Server:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        aggregator: BaseAggregator,
        momentum_aggregator: FairMomentumAggregator | None,
        secure_agg: SecureAggregator,
        target_gap: float,
        adapt_every: int,
        adapt_scale: float,
    ):
        self.model = model
        self.device = device
        self.aggregator = aggregator
        self.momentum = momentum_aggregator
        self.secure_agg = secure_agg
        self.stats = GlobalStats()
        self.target_gap = target_gap
        self.adapt_every = adapt_every
        self.adapt_scale = adapt_scale
        self.round = 0

    def broadcast(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def set_model_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)

    def aggregate(self, client_states: List[Dict[str, torch.Tensor]], weights: List[float]) -> None:
        # Compute weighted delta and apply momentum+secure agg
        stacked = []
        base_state = self.broadcast()
        for cs, w in zip(client_states, weights):
            delta = {k: cs[k] - base_state[k] for k in base_state.keys()}
            # Flatten to one big vector for secure-agg simulation
            vec = torch.concat([delta[k].flatten() for k in delta.keys()]).to(self.device) * w
            stacked.append(vec)

        agg_vec = self.secure_agg.aggregate(stacked)  # average masked sum
        # Reconstruct parameter dictionary shape
        # (since we didn't store shapes, we will apply through momentum helper)
        new_state = self.momentum.apply_update(base_state, agg_vec) if self.momentum else self._apply_vec(base_state, agg_vec)
        self.set_model_state(new_state)

    def _apply_vec(self, state: Dict[str, torch.Tensor], vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, ptr = {}, 0
        for k, v in state.items():
            num = v.numel()
            upd = vec[ptr:ptr+num].view_as(v).to(v.dtype).to(v.device)
            out[k] = v + upd
            ptr += num
        return out

    def adapt_hyperparams(self, fairness_summary: Dict[str, Any], algo):
        if (self.round + 1) % self.adapt_every != 0:
            return
        gap = fairness_summary.get("max_group_gap", 0.0)
        if gap > self.target_gap:
            algo.lambda_g *= self.adapt_scale
            algo.lambda_c *= self.adapt_scale
        else:
            algo.lambda_g /= self.adapt_scale
            algo.lambda_c /= self.adapt_scale

    def run_round(
        self,
        clients,
        train_loaders: List[DataLoader],
        val_loader: DataLoader,
        sens_present: bool,
        algo,
    ) -> Dict[str, Any]:
        self.round += 1
        # 1) Broadcast model
        global_state = self.broadcast()
        for c in clients:
            c.model.load_state_dict(global_state)
            c.reference_model = copy.deepcopy(c.model)  # for FedProx and client penalty

        # 2) Local train + collect fairness metadata
        updates, local_meta = [], []
        client_weights = self.aggregator.client_weights_signal()  # fairness-driven importance factors
        for c, loader, wf in zip(clients, train_loaders, client_weights):
            out = c.local_train(loader, algo.local_epochs, algo.lr, algo.weight_decay,
                                sens_present=sens_present, client_weight_factor=wf)
            updates.append(out["weights"])
            local_meta.append({"cid": c.cid, "factor": wf})

        # 3) Compute server weights (a_i) using aggregator on latest fairness stats
        weights = self.aggregator.compute_weights(local_meta)

        # 4) Aggregate with secure agg + momentum projection
        self.aggregate(updates, weights)

        # 5) Evaluate + adapt
        fair_report = evaluate(self.model, val_loader, self.device, sens_present)
        fairness_summary = self.stats.update_and_summarize(fair_report)
        self.adapt_hyperparams(fairness_summary, algo)

        return {"round": self.round, "metrics": fairness_summary}
