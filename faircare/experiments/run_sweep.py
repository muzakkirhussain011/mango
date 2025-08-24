"""
Run comprehensive federated learning experiments across all algorithms and datasets.
Uses optimal hyperparameters for fair comparison and evaluation.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from run_experiments import FederatedExperiment


# Optimal hyperparameters for each algorithm
OPTIMAL_CONFIGS = {
    'faircare_fl': {
        'version': '2.0.0',
        'prox_mu': 0.001,
        'lambda_irm': 0.5,
        'lambda_adv': 0.2,
        'use_mixup': True,
        'use_cia': True,
        'server_momentum': 0.9,
        'mgda': {'normalize_grads': True, 'solver': 'qp', 'step_size': 0.8},
        'pcgrad': {'enabled': True},
        'cagrad': {'enabled': True, 'rho': 0.7},
        'fairness_duals': {'enabled': True, 'epsilon_eo': 0.015, 'epsilon_fpr': 0.015, 'lr': 0.15},
        'arl': {'enabled': True, 'eta': 2.0, 'width': 128, 'depth': 3},
        'selector': {'enabled': True, 'mode': 'lyapunov', 'tau': 0.015, 'kappa': 0.6},
        'distill': {'enabled': True, 'temperature': 3.0, 'steps': 300, 'batch_size': 128},
        'w_eo': 1.2,
        'w_fpr': 1.2,
        'w_sp': 0.8,
        'weight_floor': 0.005,
        'weight_capx': 0.15,
        'tau': 0.5
    },
    'fedavg': {
        'weighted_aggregation': True
    },
    'fedprox': {
        'mu': 0.001,
        'weighted_aggregation': True
    },
    'afl': {
        'lambda_param': 0.1,
        'weighted_aggregation': True
    },
    'qffl': {
        'q': 2.0,
        'weighted_aggregation': True
    },
    'fairfed': {
        'beta': 0.5,
        'weighted_aggregation': True
    }
}


class ComprehensiveSweep:
    """Orchestrates comprehensive experimental sweep across algorithms and datasets."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """Initialize sweep runner.
        
        Args:
            config_path: Path to sweep configuration file
            **kwargs: Override configuration parameters
        """
        # Load base configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self.get_default_config()
        
        # Override with kwargs
        self.config.update(kwargs)
        
        # Setup paths
        self.setup_paths()
        
        # Setup logging
        self.setup_logging()
        
        # Prepare sweep configurations
        self.sweep_configs = self.prepare_sweep_configs()
        
        self.logger.info(f"Sweep initialized with {len(self.sweep_configs)} configurations")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default sweep configuration."""
        return {
            'experiment_name': 'comprehensive_sweep',
            'save_dir': 'results/sweeps',
            'seeds': [13, 42, 123],
            
            'datasets': [
                {
                    'name': 'mimic',
                    'task': 'mortality',
                    'sensitive_attr': 'race',
                    'preprocessing': {'normalize': True, 'impute': 'median'},
                    'split': {'train_val_test': [0.8, 0.1, 0.1]},
                    'heterogeneity': {'dirichlet_alpha': 0.3, 'num_clients': 40}
                },
                {
                    'name': 'eicu',
                    'task': 'mortality',
                    'sensitive_attr': 'sex',
                    'preprocessing': {'normalize': True, 'impute': 'median'},
                    'split': {'train_val_test': [0.8, 0.1, 0.1]},
                    'heterogeneity': {'dirichlet_alpha': 0.3, 'num_clients': 40}
                },
                {
                    'name': 'adult',
                    'task': 'income',
                    'sensitive_attr': 'sex',
                    'preprocessing': {'normalize': True},
                    'split': {'train_val_test': [0.8, 0.1, 0.1]},
                    'heterogeneity': {'dirichlet_alpha': 0.5, 'num_clients': 30}
                },
                {
                    'name': 'compas',
                    'task': 'recidivism',
                    'sensitive_attr': 'race',
                    'preprocessing': {'normalize': True},
                    'split': {'train_val_test': [0.8, 0.1, 0.1]},
                    'heterogeneity': {'dirichlet_alpha': 0.5, 'num_clients': 20}
                }
            ],
            
            'algorithms': ['faircare_fl', 'fedavg', 'fedprox', 'afl', 'qffl', 'fairfed'],
            
            'trainer': {
                'rounds': 200,
                'local_epochs': 2,
                'batch_size': 128,
                'optimizer': 'adam',
                'lr': 0.001,
                'weight_decay': 0.0,
                'scheduler': 'cosine',
                'grad_clip': 5.0,
                'early_stopping': {
                    'enabled': True,
                    'monitor': 'val/worst_group_f1',
                    'mode': 'max',
                    'patience_rounds': 20
                }
            },
            
            'client_participation': {
                'scheme': 'fraction',
                'client_fraction': 0.3,
                'min_clients_per_round': 10
            },
            
            'model': {
                'type': 'mlp',
                'hidden_dims': [256, 128]
            },
            
            'evaluation': {
                'metrics': ['accuracy', 'macro_f1', 'worst_group_f1', 'eo_gap', 'fpr_gap', 'sp_gap', 'auroc'],
                'on_server_val': True,
                'log_every': 1
            },
            
            'sweep': {
                'n_workers': 3,
                'use_gpu': True,
                'gpu_per_worker': 1,
                'timeout_hours': 24
            }
        }
    
    def setup_paths(self):
        """Setup sweep directories."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config.get('experiment_name', 'sweep')
        self.sweep_dir = Path(self.config.get('save_dir', 'results')) / f"{exp_name}_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.sweep_dir / 'experiments').mkdir(exist_ok=True)
        (self.sweep_dir / 'analysis').mkdir(exist_ok=True)
        (self.sweep_dir / 'figures').mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.sweep_dir / 'sweep_config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def setup_logging(self):
        """Setup logging for sweep."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('sweep')
        
        # File handler
        fh = logging.FileHandler(self.sweep_dir / 'sweep.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def prepare_sweep_configs(self) -> List[Dict[str, Any]]:
        """Prepare all experiment configurations for the sweep."""
        configs = []
        
        algorithms = self.config['algorithms']
        datasets = self.config['datasets']
        seeds = self.config['seeds']
        
        # Generate all combinations
        for algo, dataset_config, seed in product(algorithms, datasets, seeds):
            # Base configuration
            exp_config = {
                'algorithm': algo,
                'dataset': dataset_config['name'],
                'task': dataset_config['task'],
                'sensitive_attr': dataset_config['sensitive_attr'],
                'seed': seed,
                'preprocessing': dataset_config.get('preprocessing', {}),
                'num_clients': dataset_config['heterogeneity']['num_clients'],
                'dirichlet_alpha': dataset_config['heterogeneity']['dirichlet_alpha'],
                **self.config['trainer'],
                **self.config['client_participation'],
                'model': self.config['model'],
                'save_dir': self.sweep_dir / 'experiments',
                'experiment_name': f"{algo}_{dataset_config['name']}_seed{seed}"
            }
            
            # Add optimal algorithm-specific parameters
            if algo in OPTIMAL_CONFIGS:
                exp_config['algo_params'] = OPTIMAL_CONFIGS[algo]
            
            configs.append(exp_config)
        
        return configs
    
    def run_single_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment results
        """
        try:
            experiment = FederatedExperiment(config)
            results = experiment.run()
            
            # Add metadata
            results['status'] = 'completed'
            results['error'] = None
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {config['experiment_name']}")
            self.logger.error(str(e))
            
            return {
                'algorithm': config['algorithm'],
                'dataset': config['dataset'],
                'seed': config['seed'],
                'status': 'failed',
                'error': str(e),
                'final_metrics': {}
            }
    
    def run_parallel(self):
        """Run experiments in parallel."""
        n_workers = self.config['sweep'].get('n_workers', 3)
        self.logger.info(f"Starting parallel sweep with {n_workers} workers")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all experiments
            futures = []
            for config in self.sweep_configs:
                future = executor.submit(self.run_single_experiment, config)
                futures.append((future, config))
            
            # Collect results with progress bar
            for future, config in tqdm(futures, desc="Running experiments"):
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout
                    results.append(result)
                    
                    # Log progress
                    if result['status'] == 'completed':
                        self.logger.info(
                            f"Completed: {config['algorithm']} on {config['dataset']} "
                            f"(seed {config['seed']})"
                        )
                    
                except Exception as e:
                    self.logger.error(f"Experiment timeout or error: {config['experiment_name']}")
                    results.append({
                        'algorithm': config['algorithm'],
                        'dataset': config['dataset'],
                        'seed': config['seed'],
                        'status': 'timeout',
                        'error': str(e),
                        'final_metrics': {}
                    })
        
        return results
    
    def run_sequential(self):
        """Run experiments sequentially (for debugging)."""
        self.logger.info("Starting sequential sweep")
        
        results = []
        
        for config in tqdm(self.sweep_configs, desc="Running experiments"):
            result = self.run_single_experiment(config)
            results.append(result)
            
            if result['status'] == 'completed':
                self.logger.info(
                    f"Completed: {config['algorithm']} on {config['dataset']} "
                    f"(seed {config['seed']})"
                )
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]):
        """Analyze and aggregate results across experiments.
        
        Args:
            results: List of experiment results
        """
        self.logger.info("Analyzing results")
        
        # Convert to DataFrame for easier analysis
        records = []
        for result in results:
            if result['status'] == 'completed':
                record = {
                    'algorithm': result['algorithm'],
                    'dataset': result['dataset'],
                    'seed': result.get('seed', 0),
                    'total_rounds': result.get('total_rounds', 0),
                    **{k: v for k, v in result.get('final_metrics', {}).items() if isinstance(v, (int, float))}
                }
                records.append(record)
        
        if not records:
            self.logger.warning("No completed experiments to analyze")
            return
        
        df = pd.DataFrame(records)
        
        # Aggregate by algorithm and dataset
        aggregated = df.groupby(['algorithm', 'dataset']).agg({
            col: ['mean', 'std'] for col in df.columns 
            if col not in ['algorithm', 'dataset', 'seed']
        })
        
        # Save aggregated results
        aggregated.to_csv(self.sweep_dir / 'analysis' / 'aggregated_results.csv')
        
        # Create comparison tables
        self.create_comparison_tables(df)
        
        # Create visualizations
        self.create_comprehensive_visualizations(df)
        
        # Statistical tests
        self.run_statistical_tests(df)
        
        # Generate final report
        self.generate_report(df, aggregated)
    
    def create_comparison_tables(self, df: pd.DataFrame):
        """Create comparison tables for key metrics.
        
        Args:
            df: Results DataFrame
        """
        metrics = ['test/accuracy', 'test/worst_group_f1', 'test/eo_gap', 'test/fpr_gap', 'test/sp_gap']
        
        for metric in metrics:
            if metric in df.columns:
                # Pivot table for mean values
                pivot_mean = df.pivot_table(
                    values=metric,
                    index='dataset',
                    columns='algorithm',
                    aggfunc='mean'
                )
                
                # Pivot table for std values
                pivot_std = df.pivot_table(
                    values=metric,
                    index='dataset',
                    columns='algorithm',
                    aggfunc='std'
                )
                
                # Combined table with mean ± std
                combined = pivot_mean.round(4).astype(str) + ' ± ' + pivot_std.round(4).astype(str)
                
                # Highlight best values
                if 'gap' in metric:
                    # Lower is better for gaps
                    best_algo = pivot_mean.idxmin(axis=1)
                else:
                    # Higher is better for accuracy/F1
                    best_algo = pivot_mean.idxmax(axis=1)
                
                # Save table
                combined.to_csv(self.sweep_dir / 'analysis' / f'comparison_{metric.replace("/", "_")}.csv')
    
    def create_comprehensive_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualization plots.
        
        Args:
            df: Results DataFrame
        """
        # Set style
        sns.set_style('whitegrid')
        sns.set_palette('husl')
        
        # 1. Algorithm comparison across datasets
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = [
            ('test/accuracy', 'Accuracy', axes[0, 0]),
            ('test/worst_group_f1', 'Worst-Group F1', axes[0, 1]),
            ('test/eo_gap', 'Equal Opportunity Gap', axes[0, 2]),
            ('test/fpr_gap', 'False Positive Rate Gap', axes[1, 0]),
            ('test/sp_gap', 'Statistical Parity Gap', axes[1, 1]),
            ('test/macro_f1', 'Macro F1', axes[1, 2])
        ]
        
        for metric, title, ax in metrics:
            if metric in df.columns:
                sns.barplot(data=df, x='dataset', y=metric, hue='algorithm', ax=ax)
                ax.set_title(title)
                ax.set_xlabel('Dataset')
                ax.set_ylabel(title)
                ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add significance markers
                if 'gap' not in metric:
                    ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(self.sweep_dir / 'figures' / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Fairness-Accuracy Trade-off
        fig, axes = plt.subplots(1, len(df['dataset'].unique()), figsize=(20, 5))
        
        if len(df['dataset'].unique()) == 1:
            axes = [axes]
        
        for idx, dataset in enumerate(df['dataset'].unique()):
            dataset_df = df[df['dataset'] == dataset]
            
            if 'test/accuracy' in df.columns and 'test/eo_gap' in df.columns:
                for algo in dataset_df['algorithm'].unique():
                    algo_df = dataset_df[dataset_df['algorithm'] == algo]
                    axes[idx].scatter(
                        algo_df['test/eo_gap'],
                        algo_df['test/accuracy'],
                        label=algo,
                        s=100,
                        alpha=0.7
                    )
                
                axes[idx].set_xlabel('Equal Opportunity Gap')
                axes[idx].set_ylabel('Accuracy')
                axes[idx].set_title(f'Fairness-Accuracy Trade-off ({dataset})')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.sweep_dir / 'figures' / 'fairness_accuracy_tradeoff.png', dpi=150)
        plt.close()
        
        # 3. Worst-Group Performance Comparison
        if 'test/worst_group_f1' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            pivot = df.pivot_table(
                values='test/worst_group_f1',
                index='algorithm',
                columns='dataset',
                aggfunc='mean'
            )
            
            pivot.plot(kind='bar', ax=ax)
            ax.set_title('Worst-Group F1 Score Comparison')
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Worst-Group F1')
            ax.legend(title='Dataset')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.sweep_dir / 'figures' / 'worst_group_comparison.png', dpi=150)
            plt.close()
        
        # 4. Convergence Analysis (if available)
        self.plot_convergence_analysis()
        
        # 5. Heatmap of all metrics
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data for heatmap
        metrics_for_heatmap = ['test/accuracy', 'test/worst_group_f1', 'test/macro_f1',
                               'test/eo_gap', 'test/fpr_gap', 'test/sp_gap']
        
        heatmap_data = []
        for algo in df['algorithm'].unique():
            row = []
            for metric in metrics_for_heatmap:
                if metric in df.columns:
                    mean_val = df[df['algorithm'] == algo][metric].mean()
                    # Normalize gaps (lower is better) and metrics (higher is better)
                    if 'gap' in metric:
                        # Invert gaps so higher is better
                        normalized = 1 - (mean_val / df[metric].max())
                    else:
                        normalized = mean_val / df[metric].max()
                    row.append(normalized)
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        sns.heatmap(
            heatmap_data,
            xticklabels=[m.replace('test/', '') for m in metrics_for_heatmap if m in df.columns],
            yticklabels=df['algorithm'].unique(),
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_title('Normalized Performance Heatmap (Higher is Better)')
        plt.tight_layout()
        plt.savefig(self.sweep_dir / 'figures' / 'performance_heatmap.png', dpi=150)
        plt.close()
    
    def plot_convergence_analysis(self):
        """Plot convergence analysis if metrics history is available."""
        # This would require loading the metrics history from individual experiments
        # For now, create a placeholder
        pass
    
    def run_statistical_tests(self, df: pd.DataFrame):
        """Run statistical significance tests.
        
        Args:
            df: Results DataFrame
        """
        self.logger.info("Running statistical tests")
        
        test_results = []
        
        # Metrics to test
        metrics = ['test/accuracy', 'test/worst_group_f1', 'test/eo_gap']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            # Compare FairCare-FL against each baseline
            faircare_data = df[df['algorithm'] == 'faircare_fl'][metric].values
            
            for baseline in df['algorithm'].unique():
                if baseline == 'faircare_fl':
                    continue
                
                baseline_data = df[df['algorithm'] == baseline][metric].values
                
                # Perform t-test
                if len(faircare_data) > 1 and len(baseline_data) > 1:
                    t_stat, p_value = stats.ttest_ind(faircare_data, baseline_data)
                    
                    # Cohen's d effect size
                    pooled_std = np.sqrt((np.var(faircare_data) + np.var(baseline_data)) / 2)
                    cohens_d = (np.mean(faircare_data) - np.mean(baseline_data)) / pooled_std
                    
                    test_results.append({
                        'metric': metric,
                        'comparison': f'faircare_fl vs {baseline}',
                        'faircare_mean': np.mean(faircare_data),
                        'baseline_mean': np.mean(baseline_data),
                        'difference': np.mean(faircare_data) - np.mean(baseline_data),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    })
        
        # Save test results
        if test_results:
            test_df = pd.DataFrame(test_results)
            test_df.to_csv(self.sweep_dir / 'analysis' / 'statistical_tests.csv', index=False)
            
            # Print summary
            self.logger.info("\nStatistical Test Summary:")
            for _, row in test_df[test_df['significant']].iterrows():
                self.logger.info(
                    f"{row['metric']}: {row['comparison']} - "
                    f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}"
                )
    
    def generate_report(self, df: pd.DataFrame, aggregated: pd.DataFrame):
        """Generate comprehensive report.
        
        Args:
            df: Results DataFrame
            aggregated: Aggregated results
        """
        report_path = self.sweep_dir / 'analysis' / 'report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Federated Learning Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- Total experiments: {len(self.sweep_configs)}\n")
            f.write(f"- Completed experiments: {len(df)}\n")
            f.write(f"- Algorithms tested: {', '.join(df['algorithm'].unique())}\n")
            f.write(f"- Datasets: {', '.join(df['dataset'].unique())}\n")
            f.write(f"- Seeds: {', '.join(map(str, df['seed'].unique()))}\n\n")
            
            # Best performing algorithm per dataset
            f.write("## Best Performing Algorithms\n\n")
            
            for dataset in df['dataset'].unique():
                f.write(f"### {dataset}\n\n")
                dataset_df = df[df['dataset'] == dataset]
                
                # Best for each metric
                metrics_summary = {
                    'Accuracy': 'test/accuracy',
                    'Worst-Group F1': 'test/worst_group_f1',
                    'EO Gap': 'test/eo_gap',
                    'FPR Gap': 'test/fpr_gap'
                }
                
                for name, metric in metrics_summary.items():
                    if metric in dataset_df.columns:
                        if 'gap' in metric.lower():
                            best_algo = dataset_df.groupby('algorithm')[metric].mean().idxmin()
                            best_value = dataset_df.groupby('algorithm')[metric].mean().min()
                        else:
                            best_algo = dataset_df.groupby('algorithm')[metric].mean().idxmax()
                            best_value = dataset_df.groupby('algorithm')[metric].mean().max()
                        
                        f.write(f"- Best {name}: **{best_algo}** ({best_value:.4f})\n")
                
                f.write("\n")
            
            # FairCare-FL Performance
            f.write("## FairCare-FL Performance\n\n")
            
            faircare_df = df[df['algorithm'] == 'faircare_fl']
            if not faircare_df.empty:
                f.write("### Average Metrics Across All Datasets\n\n")
                
                key_metrics = ['test/accuracy', 'test/worst_group_f1', 'test/eo_gap', 'test/fpr_gap', 'test/sp_gap']
                
                for metric in key_metrics:
                    if metric in faircare_df.columns:
                        mean_val = faircare_df[metric].mean()
                        std_val = faircare_df[metric].std()
                        f.write(f"- {metric}: {mean_val:.4f} ± {std_val:.4f}\n")
                
                f.write("\n### Improvements Over Baselines\n\n")
                
                # Calculate improvements
                for baseline in ['fedavg', 'fedprox']:
                    if baseline in df['algorithm'].unique():
                        baseline_df = df[df['algorithm'] == baseline]
                        
                        if 'test/worst_group_f1' in df.columns:
                            faircare_wg = faircare_df['test/worst_group_f1'].mean()
                            baseline_wg = baseline_df['test/worst_group_f1'].mean()
                            improvement = ((faircare_wg - baseline_wg) / baseline_wg) * 100
                            
                            f.write(f"- Worst-Group F1 vs {baseline.upper()}: "
                                   f"**{improvement:+.1f}%**\n")
                        
                        if 'test/eo_gap' in df.columns:
                            faircare_gap = faircare_df['test/eo_gap'].mean()
                            baseline_gap = baseline_df['test/eo_gap'].mean()
                            reduction = ((baseline_gap - faircare_gap) / baseline_gap) * 100
                            
                            f.write(f"- EO Gap Reduction vs {baseline.upper()}: "
                                   f"**{reduction:.1f}%**\n")
                
                f.write("\n")
            
            # Statistical Significance
            f.write("## Statistical Significance\n\n")
            
            stat_tests_path = self.sweep_dir / 'analysis' / 'statistical_tests.csv'
            if stat_tests_path.exists():
                stat_df = pd.read_csv(stat_tests_path)
                significant_tests = stat_df[stat_df['significant']]
                
                if not significant_tests.empty:
                    f.write("Significant differences (p < 0.05):\n\n")
                    
                    for _, row in significant_tests.iterrows():
                        f.write(f"- {row['comparison']} on {row['metric']}: "
                               f"p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.3f}\n")
                else:
                    f.write("No statistically significant differences found.\n")
            
            f.write("\n## Conclusion\n\n")
            
            # Automated conclusion based on results
            if 'faircare_fl' in df['algorithm'].unique():
                faircare_rank = self.compute_algorithm_ranking(df)
                
                if faircare_rank.get('faircare_fl', 999) == 1:
                    f.write("**FairCare-FL achieved the best overall performance**, ")
                    f.write("demonstrating superior fairness-utility trade-offs across datasets.\n")
                else:
                    f.write(f"FairCare-FL ranked #{faircare_rank.get('faircare_fl', 'N/A')} overall. ")
                    f.write("Further tuning may improve performance.\n")
        
        self.logger.info(f"Report generated: {report_path}")
    
    def compute_algorithm_ranking(self, df: pd.DataFrame) -> Dict[str, int]:
        """Compute overall algorithm ranking.
        
        Args:
            df: Results DataFrame
            
        Returns:
            Dictionary of algorithm rankings
        """
        # Weighted scoring based on multiple metrics
        weights = {
            'test/accuracy': 0.3,
            'test/worst_group_f1': 0.4,  # Higher weight for worst-group
            'test/eo_gap': -0.2,  # Negative weight for gaps
            'test/fpr_gap': -0.1
        }
        
        scores = {}
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            score = 0
            
            for metric, weight in weights.items():
                if metric in algo_df.columns:
                    # Normalize metric
                    if weight < 0:  # Gap metric (lower is better)
                        normalized = 1 - (algo_df[metric].mean() / df[metric].max())
                    else:  # Performance metric (higher is better)
                        normalized = algo_df[metric].mean() / df[metric].max()
                    
                    score += abs(weight) * normalized
            
            scores[algo] = score
        
        # Rank algorithms
        sorted_algos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {algo: rank+1 for rank, (algo, _) in enumerate(sorted_algos)}
        
        return rankings
    
    def run(self, parallel: bool = True):
        """Run the complete sweep.
        
        Args:
            parallel: Whether to run experiments in parallel
            
        Returns:
            Sweep results
        """
        self.logger.info("Starting comprehensive sweep")
        start_time = time.time()
        
        # Run experiments
        if parallel:
            results = self.run_parallel()
        else:
            results = self.run_sequential()
        
        # Save raw results
        with open(self.sweep_dir / 'raw_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Analyze results
        self.analyze_results(results)
        
        total_time = time.time() - start_time
        self.logger.info(f"Sweep completed in {total_time/3600:.2f} hours")
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print sweep summary to console.
        
        Args:
            results: List of experiment results
        """
        print("\n" + "="*80)
        print("SWEEP COMPLETED")
        print("="*80)
        
        completed = [r for r in results if r['status'] == 'completed']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"Total experiments: {len(results)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        
        if completed:
            # Best results
            print("\nBEST RESULTS:")
            print("-"*40)
            
            metrics_to_show = ['test/accuracy', 'test/worst_group_f1', 'test/eo_gap']
            
            for metric in metrics_to_show:
                best_result = None
                best_value = None
                
                for result in completed:
                    if metric in result.get('final_metrics', {}):
                        value = result['final_metrics'][metric]
                        
                        if best_value is None:
                            best_value = value
                            best_result = result
                        elif 'gap' in metric:
                            # Lower is better for gaps
                            if value < best_value:
                                best_value = value
                                best_result = result
                        else:
                            # Higher is better for accuracy/F1
                            if value > best_value:
                                best_value = value
                                best_result = result
                
                if best_result:
                    print(f"{metric}: {best_value:.4f} "
                         f"({best_result['algorithm']} on {best_result['dataset']})")
        
        print("\nResults saved to:", self.sweep_dir)
        print("="*80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run comprehensive federated learning sweep')
    
    parser.add_argument('--config', type=str, 
                       default='faircare/experiments/configs/comprehensive_test.yaml',
                       help='Path to sweep configuration file')
    
    parser.add_argument('--n_workers', type=int, default=3,
                       help='Number of parallel workers')
    
    parser.add_argument('--output_dir', type=str, default='results/comprehensive_test',
                       help='Output directory for results')
    
    parser.add_argument('--sequential', action='store_true',
                       help='Run experiments sequentially instead of parallel')
    
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['faircare_fl', 'fedavg', 'fedprox', 'afl', 'qffl', 'fairfed'],
                       help='Algorithms to test')
    
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['mimic', 'eicu', 'adult', 'compas'],
                       help='Datasets to use')
    
    parser.add_argument('--seeds', type=int, nargs='+', default=[13, 42, 123],
                       help='Random seeds')
    
    parser.add_argument('--rounds', type=int, default=200,
                       help='Number of federated rounds')
    
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced rounds')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command-line arguments
    if args.n_workers:
        config.setdefault('sweep', {})['n_workers'] = args.n_workers
    
    if args.output_dir:
        config['save_dir'] = args.output_dir
    
    if args.algorithms:
        config['algorithms'] = args.algorithms
    
    if args.datasets:
        # Filter datasets
        all_datasets = config.get('datasets', [])
        config['datasets'] = [d for d in all_datasets if d['name'] in args.datasets]
    
    if args.seeds:
        config['seeds'] = args.seeds
    
    if args.rounds:
        config.setdefault('trainer', {})['rounds'] = args.rounds
    
    if args.quick_test:
        # Quick test configuration
        config['trainer']['rounds'] = 10
        config['seeds'] = [42]
        config['algorithms'] = ['faircare_fl', 'fedavg']
        if config.get('datasets'):
            config['datasets'] = config['datasets'][:1]
    
    # Run sweep
    sweep = ComprehensiveSweep(config_path=None, **config)
    results = sweep.run(parallel=not args.sequential)
    
    print("\nSweep completed successfully!")


if __name__ == '__main__':
    main()
