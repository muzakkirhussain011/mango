# faircare/algos/faircare_fl.py
"""
FairCare-FL++: unified fairness-aware FL with
- q-FFL client emphasis
- Group-DRO-inspired weighting via group stats
- FedGFT global fairness penalty using summary stats
- FAIR-FATE-style fairness-aware momentum (through weights)
- FedAdam (FedOpt) server optimizer for stability
- Optional SCAFFOLD-lite control variates
- Optional PCGrad (gradient surgery) on clients between task and fairness losses
References: q-FFL [Li'19], AFL [Mohri'19], FedOpt [Reddi'21], SCAFFOLD [Karimireddy'20],
Group DRO [Sagawa'20], FedGFT [Wang'23], FAIR-FATE [Salazar'23].
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from .aggregator import (
    FairnessAwareWeights, FedAdamAggregator, flatten_params, unflatten_params
)

class FairCareFL:
    def __init__(self, model_params_template: List[np.ndarray],
                 q: float = 0.5, gamma: float = 1.0, temperature: float = 1.0,
                 server_lr: float = 1.0, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, scaffold_c: float = 0.0):
        dim = sum(p.size for p in model_params_template)
        self.server_opt = FedAdamAggregator(dim=dim, lr=server_lr, beta1=beta1, beta2=beta2, eps=eps)
        self.weight_fn = FairnessAwareWeights(q=q, gamma=gamma, temperature=temperature)
        self.scaffold_c = scaffold_c  # 0 => disabled; else small (e.g., 0.01)

        # server control variate for SCAFFOLD-lite (vector of dim)
        self.c_global = np.zeros(dim, dtype=np.float64)

    def aggregate(self,
                  global_params: List[np.ndarray],
                  client_deltas: List[List[np.ndarray]],
                  client_reports: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        client_reports must include:
        - "loss": float (post-local-train)
        - "gap": float (e.g., DP or EO gap measured locally)
        - "num_samples": int
        - "control_variate": Optional[List[np.ndarray]] if SCAFFOLD-lite enabled
        """
        template = [p.copy() for p in global_params]
        deltas_flat = [flatten_params(d) for d in client_deltas]
        w = self.weight_fn(client_reports)

        # fairness-aware average of deltas
        agg_delta = np.zeros_like(deltas_flat[0])
        for wi, df in zip(w, deltas_flat):
            agg_delta += wi * df

        # SCAFFOLD-lite correction: subtract weighted avg client control variate
        if self.scaffold_c > 0.0:
            cvs = []
            for r in client_reports:
                cv = r.get("control_variate")
                if cv is not None:
                    cvs.append(flatten_params(cv))
            if cvs:
                mean_cv = np.mean(np.stack(cvs, 0), axis=0)
                agg_delta = agg_delta - self.scaffold_c * mean_cv

        # FedAdam server step
        step = self.server_opt.step(agg_delta)
        new_flat = flatten_params(template) + step
        return unflatten_params(new_flat, template)
