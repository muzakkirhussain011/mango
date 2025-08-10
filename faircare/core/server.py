# faircare/core/server.py
from typing import List, Dict, Any, Tuple
import torch
import copy
from .secure_agg import secure_sum_dicts
from ..fairness.global_stats import compute_global_fairness_from_counts, worst_group_focus_per_client
from ..algos.faircare_f1 import FairCareF1Config
from ..algos.aggregator_f1 import FairCareF1Aggregator

class Server:
    def __init__(self, model_ctor, device="cpu"):
        self.model_ctor = model_ctor
        self.device = device
        self.global_model = self.model_ctor().to(self.device)
        self.round = 0
        # Late-init aggregator to allow algo-specific params
        self.aggregator = None
        self.algo_name = None

    def _ensure_aggregator(self, algo_cfg: Any):
        if isinstance(algo_cfg, dict):
            name = algo_cfg.get("name", "")
        else:
            name = getattr(algo_cfg, "name", "")
        if self.aggregator is not None and self.algo_name == name:
            return
        if name == "faircare_f1":
            if isinstance(algo_cfg, dict):
                cfg = FairCareF1Config(**algo_cfg)
            else:
                cfg = algo_cfg
            self.aggregator = FairCareF1Aggregator(
                beta=cfg.beta_momentum,
                target_eo=cfg.target_eo,
                target_sp=cfg.target_sp,
                dual_step=cfg.dual_step,
                max_dual=cfg.max_dual,
                max_weight_scale=cfg.max_weight_scale,
                min_weight_scale=cfg.min_weight_scale,
            )
            self.algo_cfg = cfg
            self.algo_name = name
        else:
            raise ValueError(f"Unknown algorithm: {name}")

    def broadcast_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    @torch.no_grad()
    def run_round(
        self,
        clients,
        train_loaders: List[Any],
        val_loader: Any,
        sens_present: bool,
        algo_cfg: Any,
    ) -> Dict[str, float]:
        self._ensure_aggregator(algo_cfg)
        self.global_model.eval()
        W_t = self.broadcast_state()

        # Local training
        client_states = []
        client_bases = []
        client_sizes = []
        per_client_counts = []  # for fairness aggregation
        for c, loader in zip(clients, train_loaders):
            base = copy.deepcopy(W_t)
            out = c.local_train(
                loader=loader,
                epochs=self.algo_cfg.local_epochs,
                lr=self.algo_cfg.lr,
                weight_decay=self.algo_cfg.weight_decay,
                fairness_weights={"lambda_eo": getattr(self.aggregator, "lambda_eo", 0.0),
                                  "lambda_sp": getattr(self.aggregator, "lambda_sp", 0.0)},
                sens_present=sens_present
            )
            client_states.append(out["state_dict"])
            client_bases.append(base)
            client_sizes.append(out["num_samples"])
            per_client_counts.append(out.get("fairness_counts", {}))

        # Aggregate fairness counts to compute global EO/SP
        global_counts = secure_sum_dicts(per_client_counts)
        global_stats = compute_global_fairness_from_counts(global_counts)

        # Identify worst-off groups and compute a per-client "focus" score
        client_focus = worst_group_focus_per_client(per_client_counts, global_stats)

        # Aggregate models with fairness-aware aggregator
        new_W = self.aggregator.aggregate(
            global_state=W_t,
            client_states=client_states,
            client_bases=client_bases,
            client_sizes=client_sizes,
            global_stats=global_stats,
            client_group_focus=client_focus,
        )
        self.global_model.load_state_dict(new_W)
        self.round += 1

        # Server-side evaluation (optionally with post-processing)
        from .evaluation import evaluate_global
        eval_out = evaluate_global(
            model=self.global_model,
            loader=val_loader,
            device=self.device,
            sens_present=sens_present,
            apply_postprocess=getattr(self.algo_cfg, "postprocess_equalized_odds", True)
        )
        # merge with fairness stats for logging
        eval_out.update({
            "EO_gap": global_stats.get("EO_gap", 0.0),
            "SP_gap": global_stats.get("SP_gap", 0.0),
        })
        return eval_out
