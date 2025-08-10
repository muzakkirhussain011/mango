from __future__ import annotations
import argparse
from ..config import load_config
from ..core.trainer import run_experiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)

if __name__ == "__main__":
    main()
