"""Main experiment runner CLI."""
from typing import Dict, Any

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any

from faircare.config import ExperimentConfig
from faircare.core.trainer import run_experiment
from faircare.core.utils import set_seed, Logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run federated learning experiment")
    
    # Algorithm and dataset
    parser.add_argument(
        "--algo",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "qffl", "afl", "fairfate", "faircare_fl", "fairfed"],
        help="Algorithm to use"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "heart", "synth_health", "mimic", "eicu"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--sensitive",
        type=str,
        default="sex",
        help="Sensitive attribute (sex, race, age, etc.)"
    )
    
    # Training parameters
    parser.add_argument("--clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--rounds", type=int, default=20, help="Number of rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N rounds")
    parser.add_argument("--checkpoint_every", type=int, default=5, help="Checkpoint every N rounds")
    
    # Experiment settings
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--logdir", type=str, default="runs", help="Log directory")
    parser.add_argument("--config", type=str, help="Config file (overrides other args)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    # Algorithm-specific parameters
    parser.add_argument("--fedprox_mu", type=float, default=0.01, help="FedProx mu parameter")
    parser.add_argument("--q", type=float, default=2.0, help="q-FFL q parameter")
    parser.add_argument("--afl_lambda", type=float, default=0.1, help="AFL lambda")
    parser.add_argument("--faircare_momentum", type=float, default=0.9, help="FairCare momentum")
    
    # Fairness parameters
    parser.add_argument("--alpha", type=float, default=1.0, help="EO gap weight")
    parser.add_argument("--beta", type=float, default=0.5, help="FPR gap weight")
    parser.add_argument("--gamma", type=float, default=0.5, help="SP gap weight")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature")
    
    # Data partitioning
    parser.add_argument(
        "--partition",
        type=str,
        default="dirichlet",
        choices=["iid", "dirichlet", "label_skew"],
        help="Data partition strategy"
    )
    parser.add_argument("--partition_alpha", type=float, default=0.5, help="Dirichlet alpha")
    
    return parser.parse_args()


def get_dataset_dimensions(dataset_name: str, sensitive_attr: str = None) -> int:
    """Get the actual input dimensions for a dataset."""
    from faircare.data import load_dataset
    
    # Load a small sample to get dimensions
    if dataset_name == "adult":
        # Load just to get dimensions
        data = load_dataset("adult", sensitive_attribute=sensitive_attr)
        return data["n_features"]
    elif dataset_name == "heart":
        return 13
    elif dataset_name == "synth_health":
        return 20
    elif dataset_name in ["mimic", "eicu"]:
        return 50
    else:
        return 30  # default


def main():
    """Main experiment entry point."""
    args = parse_args()
    
    # Load config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        # Override with CLI args
        config.update_from_args(args)
    else:
        # Create config from args
        config = ExperimentConfig()
        config.update_from_args(args)
    
    # Update specific fields
    config.training.algo = args.algo
    config.data.dataset = args.dataset
    config.data.sensitive_attribute = args.sensitive
    config.data.n_clients = args.clients
    config.training.rounds = args.rounds
    config.training.local_epochs = args.local_epochs
    config.training.lr = args.lr
    config.training.weight_decay = args.weight_decay
    config.data.batch_size = args.batch_size
    config.training.device = args.device
    config.seed = args.seed
    
    # Set partition parameters
    config.data.partition = args.partition
    config.data.alpha = args.partition_alpha
    
    # Algorithm-specific parameters
    config.algo.fedprox_mu = args.fedprox_mu
    config.algo.q = args.q
    config.algo.afl_lambda = args.afl_lambda
    config.algo.faircare_momentum = args.faircare_momentum
    
    # Fairness parameters
    config.fairness.alpha = args.alpha
    config.fairness.beta = args.beta
    config.fairness.gamma = args.gamma
    config.fairness.tau = args.tau
    
    # Set experiment name and logdir
    config.name = f"{args.algo}_{args.dataset}_seed{args.seed}"
    config.logdir = Path(args.logdir) / args.algo / args.dataset / f"seed{args.seed}"
    
    # Get actual dataset dimensions
    print(f"Getting dataset dimensions for {args.dataset}...")
    actual_dim = get_dataset_dimensions(args.dataset, args.sensitive)
    config.model.input_dim = actual_dim
    print(f"Dataset has {actual_dim} features")
    
    # Run experiment
    print(f"Starting experiment: {config.name}")
    print(f"Algorithm: {config.training.algo}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Clients: {config.data.n_clients}")
    print(f"Rounds: {config.training.rounds}")
    print(f"Seed: {config.seed}")
    print(f"Log directory: {config.logdir}")
    print(f"Model input dimension: {config.model.input_dim}")
    
    results = run_experiment(config)
    
    # Print final results
    if "final_metrics" in results:
        print("\nFinal Results:")
        for key, value in results["final_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    print(f"\nResults saved to: {config.logdir}")
    
    return results


if __name__ == "__main__":
    main()
