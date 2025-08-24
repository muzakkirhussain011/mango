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


def generate_configs_from_sweep(sweep_config: Dict) -> List[ExperimentConfig]:
    """Generate experiment configs from sweep specification."""
    base_config = sweep_config.get("base", {})
    param_grid = sweep_config.get("param_grid", {})
    seeds = sweep_config.get("seeds", [0, 1, 2, 3, 4])
    
    configs = []
    
    if param_grid:
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Generate all combinations
        for values in itertools.product(*param_values):
            # Create config dict
            config_dict = base_config.copy()
            
            for name, value in zip(param_names, values):
                # Handle nested parameters
                if "." in name:
                    parts = name.split(".")
                    current = config_dict
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    config_dict[name] = value
            
            # Create config object
            config = ExperimentConfig()
            
            # Update model config
            if "model" in config_dict:
                for key, val in config_dict["model"].items():
                    setattr(config.model, key, val)
            
            # Update data config
            if "data" in config_dict:
                for key, val in config_dict["data"].items():
                    setattr(config.data, key, val)
            
            # Update training config
            if "training" in config_dict:
                for key, val in config_dict["training"].items():
                    setattr(config.training, key, val)
            
            # Update fairness config
            if "fairness" in config_dict:
                for key, val in config_dict["fairness"].items():
                    setattr(config.fairness, key, val)
            
            # Update algo-specific config
            if "algo" in config_dict:
                for key, val in config_dict["algo"].items():
                    setattr(config.algo, key, val)
            
            # Update secure aggregation config
            if "secure_agg" in config_dict:
                for key, val in config_dict["secure_agg"].items():
                    setattr(config.secure_agg, key, val)
            
            # Get actual dataset dimensions
            dataset = config.data.dataset
            sensitive = config.data.sensitive_attribute
            actual_dim = get_dataset_dimensions(dataset, sensitive)
            config.model.input_dim = actual_dim
            
            # Add each seed as a separate config
            for seed in seeds:
                seed_config = ExperimentConfig()
                
                # Deep copy all attributes
                seed_config.model.input_dim = config.model.input_dim
                seed_config.model.hidden_dims = config.model.hidden_dims.copy() if hasattr(config.model, 'hidden_dims') else [64, 32]
                seed_config.model.output_dim = config.model.output_dim
                seed_config.model.dropout = config.model.dropout
                
                seed_config.data.dataset = config.data.dataset
                seed_config.data.sensitive_attribute = config.data.sensitive_attribute
                seed_config.data.n_clients = config.data.n_clients
                seed_config.data.partition = config.data.partition
                seed_config.data.alpha = config.data.alpha
                seed_config.data.batch_size = config.data.batch_size
                
                seed_config.training.algo = config.training.algo
                seed_config.training.rounds = config.training.rounds
                seed_config.training.local_epochs = config.training.local_epochs
                seed_config.training.lr = config.training.lr
                seed_config.training.weight_decay = config.training.weight_decay
                seed_config.training.eval_every = config.training.eval_every
                seed_config.training.checkpoint_every = config.training.checkpoint_every
                seed_config.training.device = config.training.device
                
                # Copy fairness parameters
                for attr in dir(config.fairness):
                    if not attr.startswith('_'):
                        setattr(seed_config.fairness, attr, getattr(config.fairness, attr))
                
                # Copy algo parameters
                for attr in dir(config.algo):
                    if not attr.startswith('_'):
                        setattr(seed_config.algo, attr, getattr(config.algo, attr))
                
                # Set seed and naming
                seed_config.seed = seed
                seed_config.data.seed = seed
                algo = seed_config.training.algo
                dataset = seed_config.data.dataset
                seed_config.name = f"{algo}_{dataset}_seed{seed}"
                seed_config.logdir = f"runs/sweep/{algo}/{dataset}/seed{seed}"
                
                configs.append(seed_config)
    
    return configs


def run_single_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment with error handling."""
    try:
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
        print(f"Failed: {config.name}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        
        return {
            "error": str(e),
            "experiment_name": config.name,
            "algorithm": config.training.algo,
            "dataset": config.data.dataset,
            "seed": config.seed,
            "config": config.to_dict(),
            "final_metrics": {}  # Empty metrics for failed experiments
        }


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
        help="Number of parallel workers (1 for sequential)"
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
    
    # Generate experiment configs
    configs = generate_configs_from_sweep(sweep_config)
    
    total_experiments = len(configs)
    print(f"\n📋 Generated {total_experiments} experiment configurations")
    print(f"   Algorithms: {sweep_config['param_grid'].get('training.algo', [])}")
    print(f"   Datasets: {sweep_config['param_grid'].get('data.dataset', [])}")
    print(f"   Seeds: {sweep_config.get('seeds', [])}")
    print(f"   Rounds: {sweep_config['base']['training']['rounds']}")
    print(f"   Local epochs: {sweep_config['base']['training']['local_epochs']}")
    
    # Confirm before running
    print(f"\nThis will run {total_experiments} experiments.")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Start timer
    start_time = time.time()
    
    # Run experiments
    print(f"\n🚀 Starting experiments with {args.n_workers} worker(s)...")
    
    if args.n_workers > 1:
        # Parallel execution
        with mp.Pool(args.n_workers) as pool:
            all_results = pool.map(run_single_experiment, configs)
    else:
        # Sequential execution
        all_results = []
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{total_experiments}] ", end="")
            result = run_single_experiment(config)
            all_results.append(result)
    
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
