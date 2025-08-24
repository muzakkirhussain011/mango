# faircare/experiments/run_sweep.py
"""Comprehensive sweep runner for all algorithms across all datasets."""
from typing import Dict, List, Any, Optional
import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, List
import traceback
import time
import sys

from faircare.config import ExperimentConfig
from faircare.core.trainer import run_experiment
from faircare.core.evaluation import Evaluator
from faircare.fairness.summarize import create_summary_table, export_latex_table
from faircare.core.utils import set_seed


def load_sweep_config(config_path: str) -> Dict:
    """Load sweep configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_dimensions(dataset_name: str, sensitive_attr: str = None) -> int:
    """Get the actual input dimensions for a dataset."""
    # Use cached dimensions to avoid loading datasets multiple times
    dimensions_map = {
        "adult": 14,  # From the logs we see it's 14 features
        "heart": 13,
        "synth_health": 20,
        "mimic": 50,
        "eicu": 50
    }
    return dimensions_map.get(dataset_name, 30)


def create_config_dict(base_config: Dict, param_values: Dict, seed: int) -> Dict:
    """Create a serializable config dictionary."""
    config_dict = {
        "name": "",
        "seed": seed,
        "logdir": "",
        "model": {
            "model_type": "mlp",
            "input_dim": 30,
            "hidden_dims": [64, 32],
            "output_dim": 1,
            "dropout": 0.2,
            "activation": "relu"
        },
        "data": {
            "dataset": "adult",
            "sensitive_attribute": "sex",
            "n_clients": 10,
            "partition": "dirichlet",
            "alpha": 0.5,
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "batch_size": 32,
            "seed": seed
        },
        "training": {
            "algo": "fedavg",
            "rounds": 20,
            "local_epochs": 5,
            "lr": 0.01,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "server_lr": 1.0,
            "eval_every": 1,
            "checkpoint_every": 5,
            "early_stopping_rounds": None,
            "device": "cpu"
        },
        "fairness": {
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 0.5,
            "delta": 0.1,
            "delta_init": 0.2,
            "delta_min": 0.01,
            "tau": 1.0,
            "tau_init": 1.0,
            "tau_min": 0.1,
            "tau_anneal": True,
            "tau_anneal_rate": 0.95,
            "mu": 0.9,
            "mu_client": 0.9,
            "theta_server": 0.8,
            "lambda_fair": 0.1,
            "lambda_fair_init": 0.1,
            "lambda_fair_min": 0.01,
            "lambda_fair_max": 2.0,
            "lambda_adapt_rate": 1.2,
            "bias_threshold_eo": 0.15,
            "bias_threshold_fpr": 0.15,
            "bias_threshold_sp": 0.10,
            "thr_eo": 0.15,
            "thr_fpr": 0.15,
            "thr_sp": 0.10,
            "w_eo": 1.0,
            "w_fpr": 0.5,
            "w_sp": 0.5,
            "epsilon": 0.01,
            "weight_clip": 10.0,
            "enable_bias_detection": True,
            "enable_server_momentum": True,
            "enable_multi_metric": True,
            "variance_penalty": 0.1,
            "improvement_bonus": 0.1,
            "participation_boost": 0.15,
            "fairness_loss_type": "eo_sp_combined"
        },
        "secure_agg": {
            "enabled": False,
            "protocol": "additive_masking",
            "precision": 16,
            "modulus": 2**32
        },
        "algo": {
            "fedprox_mu": 0.01,
            "q": 2.0,
            "q_eps": 1e-4,
            "afl_lambda": 0.1,
            "afl_smoothing": 0.01,
            "faircare_momentum": 0.9,
            "faircare_anneal_rounds": 5,
            "convergence_threshold": 0.01,
            "bias_mitigation_extra_epochs": 1,
            "bias_mitigation_lr_multiplier": 1.2
        }
    }
    
    # Update with base config
    for key, value in base_config.items():
        if isinstance(value, dict) and key in config_dict:
            config_dict[key].update(value)
        else:
            config_dict[key] = value
    
    # Update with parameter values
    for key, value in param_values.items():
        if "." in key:
            parts = key.split(".")
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config_dict[key] = value
    
    # Set specific values
    algo = config_dict["training"]["algo"]
    dataset = config_dict["data"]["dataset"]
    config_dict["name"] = f"{algo}_{dataset}_seed{seed}"
    config_dict["logdir"] = f"runs/sweep/{algo}/{dataset}/seed{seed}"
    config_dict["data"]["seed"] = seed
    
    # Get actual dataset dimensions
    actual_dim = get_dataset_dimensions(dataset, config_dict["data"].get("sensitive_attribute"))
    config_dict["model"]["input_dim"] = actual_dim
    
    return config_dict


def run_single_experiment_wrapper(config_dict: Dict) -> Dict:
    """Wrapper to run a single experiment from a config dictionary."""
    try:
        # Create ExperimentConfig from dictionary
        config = ExperimentConfig.from_dict(config_dict)
        
        print(f"Starting: {config.name}")
        print(f"  Algorithm: {config.training.algo}")
        print(f"  Dataset: {config.data.dataset}")
        print(f"  Rounds: {config.training.rounds}")
        print(f"  Local epochs: {config.training.local_epochs}")
        print(f"  Seed: {config.seed}")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Run experiment
        results = run_experiment(config)
        
        print(f"Completed: {config.name}")
        
        # Add config info to results
        results["experiment_name"] = config.name
        results["algorithm"] = config.training.algo
        results["dataset"] = config.data.dataset
        results["seed"] = config.seed
        
        return results
        
    except Exception as e:
        print(f"Failed: {config_dict.get('name', 'unknown')}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        
        return {
            "error": str(e),
            "experiment_name": config_dict.get("name", "unknown"),
            "algorithm": config_dict.get("training", {}).get("algo", "unknown"),
            "dataset": config_dict.get("data", {}).get("dataset", "unknown"),
            "seed": config_dict.get("seed", -1),
            "config": config_dict,
            "final_metrics": {}
        }


def generate_config_dicts_from_sweep(sweep_config: Dict) -> List[Dict]:
    """Generate serializable config dictionaries from sweep specification."""
    base_config = sweep_config.get("base", {})
    param_grid = sweep_config.get("param_grid", {})
    seeds = sweep_config.get("seeds", [0, 1, 2, 3, 4])
    
    config_dicts = []
    
    if param_grid:
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Generate all combinations
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            
            # Create config for each seed
            for seed in seeds:
                config_dict = create_config_dict(base_config, param_dict, seed)
                config_dicts.append(config_dict)
    
    return config_dicts


def aggregate_results(all_results: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Aggregate results by algorithm and dataset."""
    aggregated = {}
    
    for result in all_results:
        if "error" in result and result.get("final_metrics", {}) == {}:
            print(f"Skipping failed experiment: {result.get('experiment_name', 'unknown')}")
            continue
        
        algo = result.get("algorithm", "unknown")
        dataset = result.get("dataset", "unknown")
        
        # Create nested structure
        if algo not in aggregated:
            aggregated[algo] = {}
        if dataset not in aggregated[algo]:
            aggregated[algo][dataset] = []
        
        # Extract key metrics
        final_metrics = result.get("final_metrics", {})
        
        # Store result
        aggregated[algo][dataset].append({
            "seed": result.get("seed", -1),
            "final_accuracy": final_metrics.get("final_accuracy", 0),
            "final_macro_f1": final_metrics.get("final_macro_F1", 0),
            "final_worst_group_F1": final_metrics.get("final_worst_group_F1", 0),
            "final_EO_gap": final_metrics.get("final_EO_gap", 0),
            "final_FPR_gap": final_metrics.get("final_FPR_gap", 0),
            "final_SP_gap": final_metrics.get("final_SP_gap", 0),
            "final_max_group_gap": final_metrics.get("final_max_group_gap", 0),
        })
    
    return aggregated


def print_summary_table(aggregated: Dict[str, Dict[str, List[Dict]]]):
    """Print a formatted summary table of results."""
    print("\n" + "="*120)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*120)
    
    # Algorithms and datasets
    algorithms = sorted(aggregated.keys())
    datasets = sorted(set(d for algo_data in aggregated.values() for d in algo_data.keys()))
    
    # Print results by dataset
    for dataset in datasets:
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-"*100)
        print(f"{'Algorithm':<15} {'Accuracy':<12} {'Worst F1':<12} {'EO Gap':<12} {'FPR Gap':<12} {'SP Gap':<12} {'Max Gap':<12}")
        print("-"*100)
        
        for algo in algorithms:
            if dataset in aggregated.get(algo, {}):
                results = aggregated[algo][dataset]
                
                if results:
                    # Compute means and stds
                    acc_mean = np.mean([r["final_accuracy"] for r in results])
                    acc_std = np.std([r["final_accuracy"] for r in results])
                    
                    wf1_mean = np.mean([r["final_worst_group_F1"] for r in results])
                    wf1_std = np.std([r["final_worst_group_F1"] for r in results])
                    
                    eo_mean = np.mean([r["final_EO_gap"] for r in results])
                    eo_std = np.std([r["final_EO_gap"] for r in results])
                    
                    fpr_mean = np.mean([r["final_FPR_gap"] for r in results])
                    fpr_std = np.std([r["final_FPR_gap"] for r in results])
                    
                    sp_mean = np.mean([r["final_SP_gap"] for r in results])
                    sp_std = np.std([r["final_SP_gap"] for r in results])
                    
                    max_gap_mean = np.mean([r["final_max_group_gap"] for r in results])
                    max_gap_std = np.std([r["final_max_group_gap"] for r in results])
                    
                    print(f"{algo:<15} "
                          f"{acc_mean:.3f}±{acc_std:.3f}  "
                          f"{wf1_mean:.3f}±{wf1_std:.3f}  "
                          f"{eo_mean:.3f}±{eo_std:.3f}  "
                          f"{fpr_mean:.3f}±{fpr_std:.3f}  "
                          f"{sp_mean:.3f}±{sp_std:.3f}  "
                          f"{max_gap_mean:.3f}±{max_gap_std:.3f}")
    
    # Overall best performers
    print("\n" + "="*120)
    print("🏆 BEST PERFORMERS")
    print("="*120)
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        
        # Find best accuracy
        best_acc = 0
        best_acc_algo = ""
        for algo in algorithms:
            if dataset in aggregated.get(algo, {}):
                results = aggregated[algo][dataset]
                if results:
                    acc = np.mean([r["final_accuracy"] for r in results])
                    if acc > best_acc:
                        best_acc = acc
                        best_acc_algo = algo
        
        # Find best worst-group F1
        best_wf1 = 0
        best_wf1_algo = ""
        for algo in algorithms:
            if dataset in aggregated.get(algo, {}):
                results = aggregated[algo][dataset]
                if results:
                    wf1 = np.mean([r["final_worst_group_F1"] for r in results])
                    if wf1 > best_wf1:
                        best_wf1 = wf1
                        best_wf1_algo = algo
        
        # Find lowest EO gap
        best_eo = float('inf')
        best_eo_algo = ""
        for algo in algorithms:
            if dataset in aggregated.get(algo, {}):
                results = aggregated[algo][dataset]
                if results:
                    eo = np.mean([r["final_EO_gap"] for r in results])
                    if eo < best_eo:
                        best_eo = eo
                        best_eo_algo = algo
        
        print(f"  Best Accuracy: {best_acc_algo} ({best_acc:.3f})")
        print(f"  Best Worst-Group F1: {best_wf1_algo} ({best_wf1:.3f})")
        print(f"  Lowest EO Gap: {best_eo_algo} ({best_eo:.3f})")


def main():
    """Main sweep runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive sweep across all algorithms and datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="faircare/experiments/configs/all_algos_sweep.yaml",
        help="Sweep configuration file"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers (use 1 for sequential to avoid pickling issues)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/comprehensive_sweep",
        help="Output directory for results"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Specific algorithms to test (default: all)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to test (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load sweep config
    print(f"Loading configuration from: {args.config}")
    sweep_config = load_sweep_config(args.config)
    
    # Override algorithms/datasets if specified
    if args.algorithms:
        sweep_config["param_grid"]["training.algo"] = args.algorithms
        print(f"Testing algorithms: {args.algorithms}")
    
    if args.datasets:
        sweep_config["param_grid"]["data.dataset"] = args.datasets
        print(f"Testing datasets: {args.datasets}")
    
    # Generate experiment config dictionaries
    config_dicts = generate_config_dicts_from_sweep(sweep_config)
    
    total_experiments = len(config_dicts)
    print(f"\n📋 Generated {total_experiments} experiment configurations")
    print(f"   Algorithms: {sweep_config['param_grid'].get('training.algo', [])}")
    print(f"   Datasets: {sweep_config['param_grid'].get('data.dataset', [])}")
    print(f"   Seeds: {sweep_config.get('seeds', [])}")
    print(f"   Rounds: {sweep_config['base']['training']['rounds']}")
    print(f"   Local epochs: {sweep_config['base']['training']['local_epochs']}")
    
    # Start timer
    start_time = time.time()
    
    # Run experiments
    print(f"\n🚀 Starting {total_experiments} experiments...")
    
    if args.n_workers > 1:
        print(f"Using {args.n_workers} parallel workers")
        # Use spawn method to avoid pickling issues
        mp.set_start_method('spawn', force=True)
        
        with mp.Pool(args.n_workers) as pool:
            all_results = pool.map(run_single_experiment_wrapper, config_dicts)
    else:
        print("Running sequentially (recommended to avoid multiprocessing issues)")
        all_results = []
        for i, config_dict in enumerate(config_dicts, 1):
            print(f"\n[{i}/{total_experiments}] ", end="")
            result = run_single_experiment_wrapper(config_dict)
            all_results.append(result)
            
            # Print progress
            if i % 5 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (total_experiments - i) / rate if rate > 0 else 0
                print(f"\n⏱️  Progress: {i}/{total_experiments} completed. "
                      f"Est. remaining: {int(remaining//60)}m {int(remaining%60)}s")
    
    # Aggregate results
    print("\n📈 Aggregating results...")
    aggregated = aggregate_results(all_results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    raw_results_path = output_dir / "raw_results.json"
    with open(raw_results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save aggregated results
    aggregated_path = output_dir / "aggregated_results.json"
    with open(aggregated_path, "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    # Print summary
    print_summary_table(aggregated)
    
    # Create LaTeX tables for each dataset
    for dataset in set(d for algo_data in aggregated.values() for d in algo_data.keys()):
        dataset_results = {}
        for algo in aggregated:
            if dataset in aggregated[algo]:
                dataset_results[algo] = aggregated[algo][dataset]
        
        if dataset_results:
            summary_df = create_summary_table(dataset_results)
            
            # Save as CSV
            csv_path = output_dir / f"summary_{dataset}.csv"
            summary_df.to_csv(csv_path, index=False)
            
            # Save as LaTeX
            tex_path = output_dir / f"summary_{dataset}.tex"
            export_latex_table(
                summary_df,
                str(tex_path),
                caption=f"Results on {dataset.upper()} dataset",
                label=f"tab:{dataset}_results"
            )
    
    # Print completion info
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print("\n" + "="*120)
    print("✅ SWEEP COMPLETED")
    print("="*120)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len([r for r in all_results if 'error' not in r or r.get('final_metrics')])}")
    print(f"Failed: {len([r for r in all_results if 'error' in r and not r.get('final_metrics')])}")
    print(f"Time elapsed: {hours}h {minutes}m {seconds}s")
    print(f"Results saved to: {output_dir}")
    print("="*120)


if __name__ == "__main__":
    main()
