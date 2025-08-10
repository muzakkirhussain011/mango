from __future__ import annotations
from copy import deepcopy
from ..config import load_config
from ..core.trainer import run_experiment

def sweep(base_cfg_path: str):
    base = load_config(base_cfg_path)
    for algo in ["fedavg","afl","qffl","fairfed","fedgft","fairfate","faircare_fl"]:
        cfg = deepcopy(base)
        cfg.algorithm.name = algo
        run_experiment(cfg)

if __name__ == "__main__":
    sweep("faircare/experiments/configs/default.yaml")
