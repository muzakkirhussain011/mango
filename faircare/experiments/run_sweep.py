"""Hyperparameter sweep runner."""

import argparse
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import yaml
from typing import Dict, List, Any

from faircare.config import ExperimentConfig
from faircare.core.trainer import run_experiment
from faircare.core.evaluation import Evaluator
from faircare.fairness.summarize import create_summary_table, export_latex_table


def load_sweep_config(config_path: str) -> Dict:
    """Load sweep configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_configs(sweep_config: Dict) -> List[ExperimentConfig]:
    """Generate experiment configs from sweep specification."""
    base_config = sweep_config.get("base", {})
    param_grid = sweep_config.get("param_grid", {})
    random_search = sweep_config.get("random_search", {})
    
    configs = []
    
    if param_grid:
        # Grid search
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for values in itertools.product(*param_values):
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
            
            configs.append(ExperimentConfig.from_dict(config_dict))
    
    elif random_search:
        # Random search
        n_trials = random_search.get("n_trials", 10)
        param_distributions = random_search.get("distributions", {})
        
        for trial in range(n_trials):
            config_dict = base_config.copy()
            
            for name, dist_spec in param_distributions.items():
                if dist_spec["type"] == "uniform":
                    value = np.random.uniform(dist_spec["min"], dist_spec["max"])
                elif dist_spec["type"] == "loguniform":
                    value = np.exp(np.random.uniform(
                        np.log(dist_spec["min"]),
                        np.log(dist_spec["max"])
                    ))
                elif dist_spec["type"] == "choice":
                    value = np.random.choice(dist_spec["values"])
                else:
                    value = dist_spec.get("default", 0)
                
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
            
            configs.append(ExperimentConfig.from_dict(config_dict))
    
    # Add seeds
    seeds = sweep_config.get("seeds", [0, 1, 2])
    final_configs = []
    
    for config in configs:
        for seed in seeds:
            seed_config = ExperimentConfig.from_dict(config.to_dict())
            seed_config.seed = seed
            seed_config.data.seed = seed
            
            # Update name and logdir
            algo = seed_config.training.algo
            dataset = seed_config.data.dataset
            seed_config.name = f"{algo}_{dataset}_seed{seed}"
            seed_config.logdir = f"runs/sweep/{algo}/{dataset}/seed{seed}"
            
            final_configs.append(seed_config)
    
    return final_configs


def run_single_experiment(config: ExperimentConfig) -> Dict:
    """Run a single experiment (for multiprocessing)."""
    try:
        print(f"Starting: {config.name}")
        results = run_experiment(config)
        print(f"Completed: {config.name}")
        return results
    except Exception as e:
        print(f"Failed: {config.name} - {e}")
        return {"error": str(e)}


def aggregate_results(all_results: List[Dict]) -> Dict[str, List[Dict]]:
    """Aggregate results by algorithm."""
    aggregated = {}
    
    for result in all_results:
        if "error" in result:
            continue
        
        config = result.get("config", {})
        algo = config.get("training", {}).get("algo", "unknown")
        
        if algo not in aggregated:
            aggregated[algo] = []
        
        # Extract key metrics
        final_metrics = result.get("final_metrics", {})
        aggregated[algo].append(final_metrics)
    
    return aggregated


def main():
    """Main sweep runner."""
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="faircare/experiments/configs/search.yaml",
        help="Sweep configuration file"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load sweep config
    sweep_config = load_sweep_config(args.config)
    
    # Generate experiment configs
    configs = generate_configs(sweep_config)
    print(f"Generated {len(configs)} experiment configurations")
    
    # Run experiments
    if args.n_workers > 1:
        # Parallel execution
        with mp.Pool(args.n_workers) as pool:
            all_results = pool.map(run_single_experiment, configs)
    else:
        # Sequential execution
        all_results = []
        for config in configs:
            result = run_single_experiment(config)
            all_results.append(result)
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated results
    with open(output_dir / "aggregated_results.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    # Create summary table
    if aggregated:
        summary_df = create_summary_table(aggregated)
        
        # Save as CSV
        summary_df.to_csv(output_dir / "summary.csv", index=False)
        
        # Save as LaTeX
        export_latex_table(
            summary_df,
            output_dir / "summary.tex",
            caption="Experimental Results",
            label="tab:results"
        )
        
        # Print summary
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print(summary_df.to_string())
        
        # Statistical comparisons
        evaluator = Evaluator()
        comparisons = evaluator.compare_algorithms(
            aggregated,
            metric="worst_group_F1"
        )
        
        print("\n" + "="*80)
        print("STATISTICAL COMPARISONS (vs FedAvg)")
        print("="*80)
        for algo, stats in comparisons.items():
            print(f"\n{algo}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  p-value: {stats['p_value']:.4f}")
            print(f"  Significant: {stats['significant']}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
