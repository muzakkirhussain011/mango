#!/usr/bin/env python3
"""
FairCare-FL++ Demo Script

Demonstrates the next-generation fairness-aware federated learning framework with:
- Priority-mixture client selection (FedFair³)
- Dual variable ascent for fairness constraints
- Demographics-Free Bias Detection (DFBD)
- Multi-objective optimization (MGDA/PCGrad/CAGrad)
- Server momentum and optional distillation
- Full SA/DP compatibility
"""

from pathlib import Path
import torch
from faircare.config import ExperimentConfig
from faircare.core.trainer import run_experiment
from faircare.core.utils import set_seed


def run_faircare_fl_demo():
    """Run a quick demo of FairCare-FL++ on synthetic health data."""

    print("=" * 70)
    print("FairCare-FL++ Framework Demo")
    print("=" * 70)
    print()
    print("Features enabled:")
    print("  [*] Priority-mixture client selection (FedFair3)")
    print("  [*] Dual variable ascent (lambda_eo, lambda_fpr, lambda_sp)")
    print("  [*] Demographics-Free Bias Detection (DFBD)")
    print("  [*] Multi-objective optimization")
    print("  [*] Server momentum (0.9)")
    print("  [*] SA/DP compatible (aggregate stats only)")
    print()
    print("=" * 70)
    print()

    # Set seed for reproducibility
    set_seed(42)

    # Create experiment configuration using from_dict
    config = ExperimentConfig.from_dict({
        "name": "faircare_fl_demo",
        "seed": 42,
        "logdir": "results/demo",

        # Data configuration
        "data": {
            "dataset": "synth_health",
            "sensitive_attribute": "sex",
            "n_clients": 10,
            "partition": "dirichlet",
            "alpha": 0.5,  # Non-IID
            "batch_size": 32,
            "val_ratio": 0.15,
            "test_ratio": 0.2,
            "seed": 42
        },

        # Model configuration
        "model": {
            "model_type": "mlp",
            "input_dim": 10,  # Will be updated based on dataset
            "hidden_dims": [64, 32],
            "output_dim": 1,  # Binary classification
            "dropout": 0.1
        },

        # Training configuration
        "training": {
            "algo": "faircare_fl",
            "rounds": 25,
            "local_epochs": 1,
            "lr": 0.01,
            "weight_decay": 0.0001,
            "server_lr": 1.0,
            "eval_every": 1,
            "checkpoint_every": 10,
            "device": "cpu"
        },

        # FairCare-FL++ specific configuration
        "fairness": {
            # Fairness weights (alpha, beta, gamma)
            "alpha": 1.0,      # Equal Opportunity weight
            "beta": 1.0,       # False Positive Rate weight
            "gamma": 0.8,      # Statistical Parity weight

            # Fairness penalty (adaptive via dual ascent)
            "lambda_fair": 0.1,
            "lambda_fair_min": 0.01,
            "lambda_fair_max": 2.0,

            # Temperature for soft aggregation
            "tau": 1.0,
            "tau_min": 0.3,

            # Bias detection thresholds
            "bias_threshold_eo": 0.15,
            "bias_threshold_fpr": 0.15,
            "bias_threshold_sp": 0.10,
            "enable_bias_detection": True,

            # Multi-metric fairness
            "enable_multi_metric": True,

            # Client and server momentum
            "mu_client": 0.0,      # FedProx mu (optional)
            "theta_server": 0.9,   # Server momentum

            # Weight constraints
            "epsilon": 0.005,      # Lower weight floor
            "weight_clip": 3.0,    # Upper weight cap (as multiple of uniform)
        },

        # Algorithm-specific parameters
        "algo": {
        },

        # Secure aggregation (optional)
        "secure_agg": {
            "enabled": False,  # Set to True for SA
        }
    })

    print("Starting FairCare-FL++ training...")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Clients: {config.data.n_clients}")
    print(f"  Rounds: {config.training.rounds}")
    print(f"  Partition: {config.data.partition} (alpha={config.data.alpha})")
    print(f"  Priority selection: {config.enable_priority_selection if hasattr(config, 'enable_priority_selection') else 'True'}")
    print()

    # Run experiment
    results = run_experiment(config)

    # Print final results
    print()
    print("=" * 70)
    print("Final Results:")
    print("=" * 70)

    final_metrics = results.get("final_metrics", {})

    print(f"  Accuracy:        {final_metrics.get('final_accuracy', 0):.4f}")
    print(f"  Worst Group F1:  {final_metrics.get('final_worst_group_F1', 0):.4f}")
    print()
    print("Fairness Gaps:")
    print(f"  EO Gap:          {final_metrics.get('final_EO_gap', 0):.4f}")
    print(f"  FPR Gap:         {final_metrics.get('final_FPR_gap', 0):.4f}")
    print(f"  SP Gap:          {final_metrics.get('final_SP_gap', 0):.4f}")
    print(f"  Max Gap:         {final_metrics.get('final_max_group_gap', 0):.4f}")
    print()

    # Aggregator statistics
    if "aggregator_statistics" in results:
        agg_stats = results["aggregator_statistics"]
        print("Aggregator Statistics:")
        print(f"  Final round:     {agg_stats.get('round', 0)}")
        print(f"  Bias mode:       {agg_stats.get('bias_mitigation_mode', False)}")
        print(f"  λ_fair:          {agg_stats.get('lambda_fair', 0):.4f}")
        print()

    # Bias mitigation summary
    if "bias_mitigation_summary" in results:
        bias_summary = results["bias_mitigation_summary"]
        print("Bias Mitigation:")
        print(f"  Rounds in bias mode: {bias_summary['total_rounds_in_bias_mode']}")
        print(f"  Percentage:          {bias_summary['percentage_rounds_in_bias_mode']:.1f}%")
        print()

    print("=" * 70)
    print(f"Results saved to: {config.logdir}")
    print("=" * 70)

    return results


def run_baseline_comparison():
    """Run a quick comparison: FedAvg vs FairCare-FL++"""

    print("\n" + "=" * 70)
    print("Baseline Comparison: FedAvg vs FairCare-FL++")
    print("=" * 70)
    print()

    results = {}

    for algo in ["fedavg", "faircare_fl"]:
        print(f"Running {algo.upper()}...")

        fairness_dict = {
            "alpha": 1.0 if algo == "faircare_fl" else 0.0,
            "beta": 1.0 if algo == "faircare_fl" else 0.0,
            "gamma": 0.8 if algo == "faircare_fl" else 0.0,
            "lambda_fair": 0.1 if algo == "faircare_fl" else 0.0,
        }

        config = ExperimentConfig.from_dict({
            "name": f"{algo}_comparison",
            "seed": 42,
            "logdir": f"results/comparison/{algo}",
            "data": {
                "dataset": "synth_health",
                "sensitive_attribute": "sex",
                "n_clients": 10,
                "partition": "dirichlet",
                "alpha": 0.5,
                "batch_size": 32,
                "seed": 42
            },
            "model": {
                "model_type": "mlp",
                "input_dim": 10,
                "hidden_dims": [64, 32],
                "output_dim": 1,
                "dropout": 0.1
            },
            "training": {
                "algo": algo,
                "rounds": 20,
                "local_epochs": 1,
                "lr": 0.01,
                "device": "cpu"
            },
            "fairness": fairness_dict,
            "algo": {}
        })

        results[algo] = run_experiment(config)
        print(f"{algo.upper()} complete.\n")

    # Compare results
    print("\n" + "=" * 70)
    print("Comparison Results:")
    print("=" * 70)
    print(f"{'Metric':<20} {'FedAvg':>12} {'FairCare-FL++':>15} {'Improvement':>12}")
    print("-" * 70)

    metrics_to_compare = [
        ("Accuracy", "final_accuracy", False),
        ("Worst Group F1", "final_worst_group_F1", False),
        ("EO Gap", "final_EO_gap", True),
        ("FPR Gap", "final_FPR_gap", True),
        ("SP Gap", "final_SP_gap", True),
    ]

    for metric_name, key, lower_is_better in metrics_to_compare:
        fedavg_val = results["fedavg"]["final_metrics"].get(key, 0)
        faircare_val = results["faircare_fl"]["final_metrics"].get(key, 0)

        if lower_is_better:
            improvement = ((fedavg_val - faircare_val) / (fedavg_val + 1e-8)) * 100
            arrow = "[-]" if improvement > 0 else "[+]"
        else:
            improvement = ((faircare_val - fedavg_val) / (fedavg_val + 1e-8)) * 100
            arrow = "[+]" if improvement > 0 else "[-]"

        print(f"{metric_name:<20} {fedavg_val:>12.4f} {faircare_val:>15.4f} {arrow}{abs(improvement):>10.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_baseline_comparison()
    else:
        run_faircare_fl_demo()
