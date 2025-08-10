# faircare/experiments/run_experiments.py
import argparse, yaml, os
from ..core.trainer import run_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="faircare_f1")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--config_root", type=str, default=os.path.join(os.path.dirname(__file__), "configs"))
    args = parser.parse_args()

    with open(os.path.join(args.config_root, "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.config_root, "algos.yaml")) as f:
        algos = yaml.safe_load(f)
    with open(os.path.join(args.config_root, "datasets.yaml")) as f:
        dsets = yaml.safe_load(f)

    cfg["algorithm"] = algos[args.algo]
    cfg["dataset"] = dsets[args.dataset]
    run_experiment(cfg)

if __name__ == "__main__":
    main()
