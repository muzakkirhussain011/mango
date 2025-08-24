"""
Run single federated learning experiments with flexible configuration.
Supports any algorithm-dataset combination with custom hyperparameters.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import algorithm implementations
from faircare.algos.faircare_fl import FairCareFLAggregator
from faircare.core.client import client_update
from faircare.data.datasets import load_dataset, create_federated_splits
from faircare.models.networks import create_model
from faircare.utils.metrics import compute_fairness_metrics, compute_worst_group_metrics
from faircare.utils.logging import setup_logger, MetricsLogger


class FederatedExperiment:
    """Main experiment runner for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize experiment with configuration.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup paths
        self.setup_paths()
        
        # Setup logging
        self.logger = setup_logger(
            name='experiment',
            log_file=self.experiment_dir / 'experiment.log'
        )
        self.metrics_logger = MetricsLogger(self.experiment_dir / 'metrics.csv')
        
        # Log configuration
        self.logger.info(f"Experiment configuration:\n{yaml.dump(config, default_flow_style=False)}")
        self.logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self.set_seeds(config.get('seed', 42))
        
        # Load algorithm configuration
        self.load_algorithm_config()
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.aggregator = None
        self.client_data = {}
        self.test_loader = None
        self.val_loader = None
        
    def setup_paths(self):
        """Setup experiment directories."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = self.config.get('experiment_name', 'experiment')
        self.experiment_dir = Path(self.config.get('save_dir', 'results')) / f"{exp_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'figures').mkdir(exist_ok=True)
        
        # Save configuration
        with open(self.experiment_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_algorithm_config(self):
        """Load algorithm-specific configuration."""
        algo_config_path = Path(__file__).parent / 'configs' / 'algos.yaml'
        with open(algo_config_path, 'r') as f:
            algos_config = yaml.safe_load(f)
        
        algo_name = self.config['algorithm']
        if algo_name in algos_config['algorithms']:
            self.algo_config = algos_config['algorithms'][algo_name]
            # Merge with any custom parameters
            if 'algo_params' in self.config:
                self.algo_config.update(self.config['algo_params'])
        else:
            self.logger.warning(f"Algorithm {algo_name} not found in config, using defaults")
            self.algo_config = {}
    
    def prepare_data(self):
        """Load and prepare dataset for federated learning."""
        self.logger.info(f"Loading dataset: {self.config['dataset']}")
        
        # Load dataset
        dataset_config = {
            'name': self.config['dataset'],
            'task': self.config.get('task', 'classification'),
            'sensitive_attr': self.config.get('sensitive_attr', 'sex'),
            'preprocessing': self.config.get('preprocessing', {'normalize': True})
        }
        
        train_data, val_data, test_data = load_dataset(**dataset_config)
        
        # Create federated splits
        num_clients = self.config.get('num_clients', 40)
        alpha = self.config.get('dirichlet_alpha', 0.3)
        
        self.client_data = create_federated_splits(
            train_data,
            num_clients=num_clients,
            alpha=alpha,
            seed=self.config.get('seed', 42)
        )
        
        # Create validation and test loaders
        batch_size = self.config.get('batch_size', 128)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # Get data statistics
        input_dim = train_data[0][0].shape[0] if hasattr(train_data[0][0], 'shape') else 100
        num_classes = len(torch.unique(torch.tensor([y for _, y, _ in train_data])))
        num_groups = len(torch.unique(torch.tensor([a for _, _, a in train_data])))
        
        self.data_info = {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'num_groups': num_groups,
            'num_clients': num_clients,
            'total_samples': len(train_data)
        }
        
        self.logger.info(f"Data prepared: {self.data_info}")
    
    def initialize_model(self):
        """Initialize the model architecture."""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'mlp')
        
        if model_type == 'mlp':
            hidden_dims = model_config.get('hidden_dims', [256, 128])
            self.model = create_model(
                'mlp',
                input_dim=self.data_info['input_dim'],
                hidden_dims=hidden_dims,
                output_dim=self.data_info['num_classes']
            )
        elif model_type == 'cnn':
            self.model = create_model(
                'cnn',
                num_classes=self.data_info['num_classes']
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        self.logger.info(f"Model initialized: {model_type}")
    
    def initialize_aggregator(self):
        """Initialize the aggregator based on algorithm."""
        algo_name = self.config['algorithm']
        
        if algo_name == 'faircare_fl':
            self.aggregator = FairCareFLAggregator(self.algo_config, self.device)
        elif algo_name == 'fedavg':
            from faircare.algos.fedavg import FedAvgAggregator
            self.aggregator = FedAvgAggregator(self.algo_config, self.device)
        elif algo_name == 'fedprox':
            from faircare.algos.fedprox import FedProxAggregator
            self.aggregator = FedProxAggregator(self.algo_config, self.device)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        self.logger.info(f"Aggregator initialized: {algo_name}")
    
    def select_clients(self, round_num: int) -> List[int]:
        """Select clients for participation in current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected client indices
        """
        num_clients = self.data_info['num_clients']
        client_fraction = self.config.get('client_fraction', 0.3)
        min_clients = self.config.get('min_clients_per_round', 10)
        
        num_selected = max(min_clients, int(client_fraction * num_clients))
        
        # For FairCare-FL, use fairness-aware selection if available
        if self.config['algorithm'] == 'faircare_fl' and hasattr(self.aggregator, 'fairness_debt_scores'):
            # Select based on fairness debt scores
            scores = self.aggregator.fairness_debt_scores
            if scores:
                # Probability proportional to debt
                client_ids = list(range(num_clients))
                probs = np.array([scores.get(i, 1.0) for i in client_ids])
                probs = probs / probs.sum()
                selected = np.random.choice(client_ids, size=num_selected, replace=False, p=probs)
            else:
                selected = np.random.choice(num_clients, size=num_selected, replace=False)
        else:
            # Random selection
            selected = np.random.choice(num_clients, size=num_selected, replace=False)
        
        return selected.tolist()
    
    def train_client(self, client_id: int, global_weights: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Train a single client.
        
        Args:
            client_id: Client identifier
            global_weights: Current global model weights
            
        Returns:
            Client report with updates and metrics
        """
        # Get client data
        client_dataset = self.client_data[client_id]
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 128)
        train_loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        # Split for validation (last 20%)
        val_size = int(0.2 * len(client_dataset))
        val_indices = list(range(len(client_dataset) - val_size, len(client_dataset)))
        val_subset = Subset(client_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Create model copy
        client_model = create_model(
            self.config.get('model', {}).get('type', 'mlp'),
            input_dim=self.data_info['input_dim'],
            hidden_dims=self.config.get('model', {}).get('hidden_dims', [256, 128]),
            output_dim=self.data_info['num_classes']
        ).to(self.device)
        
        # Perform client update
        local_epochs = self.config.get('local_epochs', 2)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        report = client_update(
            algorithm=self.config['algorithm'],
            client_id=client_id,
            model=client_model,
            config=self.algo_config,
            global_weights=global_weights,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            device=str(self.device)
        )
        
        return report
    
    def aggregate_updates(self, client_reports: List[Dict[str, Any]], 
                         round_ctx: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates.
        
        Args:
            client_reports: List of client reports
            round_ctx: Round context information
            
        Returns:
            Updated global model weights
        """
        global_weights = self.model.state_dict()
        
        if self.config['algorithm'] == 'faircare_fl':
            # Use FairCare-FL aggregator
            result = self.aggregator.aggregate(round_ctx, client_reports, global_weights)
            new_weights = result.new_global
            self.round_logs = result.server_logs
        else:
            # Simple weighted averaging for other algorithms
            new_weights = self.weighted_average(client_reports, global_weights)
            self.round_logs = {}
        
        return new_weights
    
    def weighted_average(self, client_reports: List[Dict[str, Any]], 
                        global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform simple weighted averaging of client updates."""
        total_samples = sum(r['n_samples'] for r in client_reports)
        
        averaged_weights = {}
        for key in global_weights:
            weighted_sum = torch.zeros_like(global_weights[key])
            
            for report in client_reports:
                weight = report['n_samples'] / total_samples
                weighted_sum += weight * (global_weights[key] + report['delta'][key])
            
            averaged_weights[key] = weighted_sum
        
        return averaged_weights
    
    def evaluate_global_model(self, data_loader: DataLoader, prefix: str = 'val') -> Dict[str, float]:
        """Evaluate global model on given data.
        
        Args:
            data_loader: Data loader for evaluation
            prefix: Metric prefix
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_groups = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target, sensitive_attr in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                sensitive_attr = sensitive_attr.to(self.device)
                
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, target)
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_groups.extend(sensitive_attr.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_groups = np.array(all_groups)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = {}
        
        # Basic metrics
        metrics[f'{prefix}/loss'] = total_loss / len(data_loader)
        metrics[f'{prefix}/accuracy'] = accuracy_score(all_targets, all_preds)
        metrics[f'{prefix}/macro_f1'] = f1_score(all_targets, all_preds, average='macro')
        
        # AUROC (for binary classification)
        if self.data_info['num_classes'] == 2:
            metrics[f'{prefix}/auroc'] = roc_auc_score(all_targets, all_probs[:, 1])
        
        # Fairness metrics
        fairness_metrics = compute_fairness_metrics(
            all_preds, all_targets, all_groups
        )
        metrics.update({f'{prefix}/{k}': v for k, v in fairness_metrics.items()})
        
        # Worst-group metrics
        wg_metrics = compute_worst_group_metrics(
            all_preds, all_targets, all_groups
        )
        metrics.update({f'{prefix}/{k}': v for k, v in wg_metrics.items()})
        
        return metrics
    
    def run(self) -> Dict[str, Any]:
        """Run the complete experiment.
        
        Returns:
            Dictionary of final results
        """
        self.logger.info("Starting experiment")
        start_time = time.time()
        
        # Prepare components
        self.prepare_data()
        self.initialize_model()
        self.initialize_aggregator()
        
        # Training parameters
        num_rounds = self.config.get('rounds', 200)
        early_stopping = self.config.get('early_stopping', {})
        use_early_stopping = early_stopping.get('enabled', True)
        patience = early_stopping.get('patience', 20)
        monitor_metric = early_stopping.get('monitor', 'val/worst_group_f1')
        
        # Initialize tracking
        best_metric = -float('inf') if 'f1' in monitor_metric else float('inf')
        best_round = 0
        patience_counter = 0
        
        # Training loop
        for round_num in range(num_rounds):
            round_start = time.time()
            
            # Select participating clients
            selected_clients = self.select_clients(round_num)
            
            # Train selected clients
            client_reports = []
            for client_id in tqdm(selected_clients, desc=f"Round {round_num+1}/{num_rounds}"):
                report = self.train_client(client_id, self.model.state_dict())
                client_reports.append(report)
            
            # Aggregate updates
            round_ctx = {
                'round': round_num,
                'selected_clients': selected_clients,
                'num_clients': self.data_info['num_clients']
            }
            
            new_weights = self.aggregate_updates(client_reports, round_ctx)
            self.model.load_state_dict(new_weights)
            
            # Evaluate global model
            val_metrics = self.evaluate_global_model(self.val_loader, 'val')
            
            # Log metrics
            round_metrics = {
                'round': round_num,
                'time': time.time() - round_start,
                **val_metrics,
                **self.round_logs
            }
            
            self.metrics_logger.log(round_metrics)
            
            # Log to console
            self.logger.info(
                f"Round {round_num+1}: "
                f"val_acc={val_metrics['val/accuracy']:.4f}, "
                f"val_wg_f1={val_metrics.get('val/worst_group_f1', 0):.4f}, "
                f"val_eo_gap={val_metrics.get('val/eo_gap', 0):.4f}"
            )
            
            # Check early stopping
            if use_early_stopping:
                current_metric = val_metrics.get(monitor_metric, 0)
                
                if 'f1' in monitor_metric or 'acc' in monitor_metric:
                    is_better = current_metric > best_metric
                else:
                    is_better = current_metric < best_metric
                
                if is_better:
                    best_metric = current_metric
                    best_round = round_num
                    patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'round': round_num,
                        'model_state_dict': self.model.state_dict(),
                        'metrics': val_metrics
                    }, self.experiment_dir / 'checkpoints' / 'best_model.pt')
                else:
                    patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered at round {round_num+1}")
                        break
            
            # Save periodic checkpoint
            if (round_num + 1) % 10 == 0:
                torch.save({
                    'round': round_num,
                    'model_state_dict': self.model.state_dict(),
                    'metrics': val_metrics
                }, self.experiment_dir / 'checkpoints' / f'checkpoint_round_{round_num+1}.pt')
        
        # Load best model for final evaluation
        if use_early_stopping and (self.experiment_dir / 'checkpoints' / 'best_model.pt').exists():
            checkpoint = torch.load(self.experiment_dir / 'checkpoints' / 'best_model.pt')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from round {checkpoint['round']+1}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate_global_model(self.test_loader, 'test')
        
        # Prepare final results
        total_time = time.time() - start_time
        final_results = {
            'algorithm': self.config['algorithm'],
            'dataset': self.config['dataset'],
            'total_rounds': round_num + 1,
            'best_round': best_round + 1,
            'total_time': total_time,
            'final_metrics': test_metrics,
            'config': self.config
        }
        
        # Save final results
        with open(self.experiment_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Create visualizations
        self.create_visualizations()
        
        self.logger.info(f"Experiment completed in {total_time:.2f} seconds")
        self.logger.info(f"Final test accuracy: {test_metrics['test/accuracy']:.4f}")
        self.logger.info(f"Final test worst-group F1: {test_metrics.get('test/worst_group_f1', 0):.4f}")
        
        return final_results
    
    def create_visualizations(self):
        """Create and save visualization plots."""
        # Load metrics history
        metrics_df = pd.read_csv(self.experiment_dir / 'metrics.csv')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot 1: Accuracy over rounds
        axes[0, 0].plot(metrics_df['round'], metrics_df['val/accuracy'])
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Validation Accuracy')
        axes[0, 0].set_title('Accuracy Progress')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Worst-group F1 over rounds
        if 'val/worst_group_f1' in metrics_df.columns:
            axes[0, 1].plot(metrics_df['round'], metrics_df['val/worst_group_f1'])
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Worst-Group F1')
            axes[0, 1].set_title('Worst-Group Performance')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Fairness gaps over rounds
        if 'val/eo_gap' in metrics_df.columns:
            axes[0, 2].plot(metrics_df['round'], metrics_df['val/eo_gap'], label='EO Gap')
            axes[0, 2].plot(metrics_df['round'], metrics_df.get('val/fpr_gap', 0), label='FPR Gap')
            axes[0, 2].plot(metrics_df['round'], metrics_df.get('val/sp_gap', 0), label='SP Gap')
            axes[0, 2].set_xlabel('Round')
            axes[0, 2].set_ylabel('Fairness Gap')
            axes[0, 2].set_title('Fairness Metrics')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Loss over rounds
        axes[1, 0].plot(metrics_df['round'], metrics_df['val/loss'])
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Loss Progress')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: MGDA weights (for FairCare-FL)
        if 'mgda/alphas' in metrics_df.columns:
            # Parse MGDA alphas (stored as string)
            mgda_alphas = metrics_df['mgda/alphas'].apply(eval)
            alphas_df = pd.DataFrame(mgda_alphas.tolist(), columns=['Accuracy', 'Worst-Group', 'Fairness'])
            
            axes[1, 1].plot(metrics_df['round'], alphas_df['Accuracy'], label='Accuracy')
            axes[1, 1].plot(metrics_df['round'], alphas_df['Worst-Group'], label='Worst-Group')
            axes[1, 1].plot(metrics_df['round'], alphas_df['Fairness'], label='Fairness')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('MGDA Weight')
            axes[1, 1].set_title('Multi-Objective Weights')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Client participation (histogram)
        if 'participating_clients' in metrics_df.columns:
            axes[1, 2].hist(metrics_df['participating_clients'], bins=20, edgecolor='black')
            axes[1, 2].set_xlabel('Number of Clients')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Client Participation Distribution')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'figures' / 'training_progress.png', dpi=150)
        plt.close()
        
        self.logger.info("Visualizations saved")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run federated learning experiment')
    
    # Core parameters
    parser.add_argument('--algorithm', type=str, default='faircare_fl',
                       choices=['faircare_fl', 'fedavg', 'fedprox', 'afl', 'qffl', 'fairfed'],
                       help='Federated learning algorithm')
    parser.add_argument('--dataset', type=str, default='mimic',
                       choices=['mimic', 'eicu', 'adult', 'compas', 'synthetic'],
                       help='Dataset name')
    parser.add_argument('--task', type=str, default='mortality',
                       help='Prediction task')
    parser.add_argument('--sensitive_attr', type=str, default='race',
                       choices=['race', 'sex', 'age', 'ethnicity'],
                       help='Sensitive attribute for fairness')
    
    # Training parameters
    parser.add_argument('--rounds', type=int, default=200,
                       help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=2,
                       help='Number of local epochs per round')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--client_fraction', type=float, default=0.3,
                       help='Fraction of clients to sample per round')
    
    # Federated setup
    parser.add_argument('--num_clients', type=int, default=40,
                       help='Number of federated clients')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3,
                       help='Dirichlet alpha for non-IID split')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='mlp',
                       choices=['mlp', 'cnn', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                       help='Hidden dimensions for MLP')
    
    # Algorithm-specific parameters (for FairCare-FL)
    parser.add_argument('--prox_mu', type=float, default=None,
                       help='FedProx regularization strength')
    parser.add_argument('--lambda_irm', type=float, default=None,
                       help='IRM penalty weight')
    parser.add_argument('--lambda_adv', type=float, default=None,
                       help='Adversarial debiasing weight')
    parser.add_argument('--w_eo', type=float, default=None,
                       help='Equal Opportunity weight')
    parser.add_argument('--w_fpr', type=float, default=None,
                       help='False Positive Rate weight')
    parser.add_argument('--w_sp', type=float, default=None,
                       help='Statistical Parity weight')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results/experiments',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--config_file', type=str, default=None,
                       help='Path to configuration file (overrides other arguments)')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--monitor_metric', type=str, default='val/worst_group_f1',
                       help='Metric to monitor for early stopping')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration from file if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Build configuration from arguments
        config = {
            'algorithm': args.algorithm,
            'dataset': args.dataset,
            'task': args.task,
            'sensitive_attr': args.sensitive_attr,
            'rounds': args.rounds,
            'local_epochs': args.local_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'client_fraction': args.client_fraction,
            'num_clients': args.num_clients,
            'dirichlet_alpha': args.dirichlet_alpha,
            'model': {
                'type': args.model_type,
                'hidden_dims': args.hidden_dims
            },
            'seed': args.seed,
            'save_dir': args.save_dir,
            'experiment_name': args.experiment_name or f"{args.algorithm}_{args.dataset}",
            'early_stopping': {
                'enabled': args.early_stopping,
                'patience': args.patience,
                'monitor': args.monitor_metric
            }
        }
        
        # Add algorithm-specific parameters if provided
        algo_params = {}
        if args.prox_mu is not None:
            algo_params['prox_mu'] = args.prox_mu
        if args.lambda_irm is not None:
            algo_params['lambda_irm'] = args.lambda_irm
        if args.lambda_adv is not None:
            algo_params['lambda_adv'] = args.lambda_adv
        if args.w_eo is not None:
            algo_params['w_eo'] = args.w_eo
        if args.w_fpr is not None:
            algo_params['w_fpr'] = args.w_fpr
        if args.w_sp is not None:
            algo_params['w_sp'] = args.w_sp
        
        if algo_params:
            config['algo_params'] = algo_params
    
    # Run experiment
    experiment = FederatedExperiment(config)
    results = experiment.run()
    
    # Print final results
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Algorithm: {results['algorithm']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Total Rounds: {results['total_rounds']}")
    print(f"Best Round: {results['best_round']}")
    print(f"Total Time: {results['total_time']:.2f} seconds")
    print("\nFinal Test Metrics:")
    for metric, value in results['final_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
