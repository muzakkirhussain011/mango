# faircare/core/trainer.py
import os, csv, numpy as np, torch
from .utils import ensure_dir, save_json
from ..fairness.global_stats import aggregate_group_stats, eo_gap_from_stats, dp_gap_from_stats

class Trainer:
    def __init__(self, server, clients, outdir, sensitive_attr="sex"):
        self.server = server
        self.clients = clients
        self.outdir = outdir
        self.sensitive_attr = sensitive_attr
        ensure_dir(outdir)
        self.csv_path = os.path.join(outdir, "metrics.csv")
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "round","global_accuracy","global_auroc","global_dp_gap","global_eo_gap"
            ])
            w.writeheader()

    def run_round(self, algo):
        self.server.broadcast(self.clients)
        payloads = [c.train_local(algo.client_cfg()) for c in self.clients]
        weights = algo.compute_weights(payloads)
        self.server.aggregate(payloads, weights, use_momentum=algo.use_momentum())
        # global fairness from aggregated stats (FedGFT-style summaries)
        group_stats = [pl["summary"]["conf_by_group"] for pl in payloads]
        agg = aggregate_group_stats(group_stats)
        dp = dp_gap_from_stats(agg)
        eo = eo_gap_from_stats(agg)
        acc = float(np.mean([pl["val_acc"] for pl in payloads]))
        return {"global_accuracy": acc, "global_auroc": float("nan"),
                "global_dp_gap": dp, "global_eo_gap": eo, "weights": weights, "agg_stats": agg}

    def train(self, rounds, algo):
        history = []
        for r in range(1, rounds+1):
            out = self.run_round(algo)
            history.append({"round": r, **{k:v for k,v in out.items() if k.startswith("global_")}})
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["round","global_accuracy","global_auroc","global_dp_gap","global_eo_gap"])
                w.writerow({"round": r, **{k:v for k,v in out.items() if k.startswith("global_")}})
        save_json(history, os.path.join(self.outdir, "history.json"))
        return history
