"""Federated learning server implementation with bias detection."""
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import copy
from faircare.core.utils import (
    sample_clients,
    weighted_average_weights,
    apply_model_delta,
    Logger
)
from faircare.core.client import Client
from faircare.core.secure_agg import SecureAggregator
from faircare.algos.aggregator import Aggregator, make_aggregator
from faircare.fairness.metrics import fairness_report


class Server:
    """Federated learning server with bias detection and adaptive response."""
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[Client],
        aggregator: Aggregator,
        val_data: Optional[Tuple] = None,
        test_data: Optional[Tuple] = None,
        secure_agg_config: Optional[Dict] = None,
        logger: Optional[Logger] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.clients = clients
        self.aggregator = aggregator
        self.val_data = val_data
        self.test_data = test_data
        self.device = torch.device(device)
        self.logger = logger
        
        # Initialize secure aggregation
        if secure_agg_config and secure_agg_config.get("enabled", False):
            self.secure_agg = SecureAggregator(
                n_clients=len(clients),
                **secure_agg_config
            )
        else:
            self.secure_agg = None
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize global weights
        self.global_weights = copy.deepcopy(self.model.state_dict())
        
        # Server-level momentum buffer (for FairCare-FL++)
        self.momentum_buffer = None
        
        # History tracking with enhanced metrics
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "test_acc": [],
            "fairness_metrics": [],
            "eo_gaps": [],
            "fpr_gaps": [],
            "sp_gaps": [],
            "worst_group_f1": [],
            "bias_mitigation_rounds": []
        }
        
        # Bias detection state
        self.consecutive_high_bias_rounds = 0
        self.bias_alert_triggered = False
    
    def train_round(
        self,
        round_idx: int,
        n_clients: int,
        local_epochs: int,
        lr: float,
        weight_decay: float = 0.0,
        proximal_mu: float = 0.0,
        server_lr: float = 1.0
    ) -> Dict[str, Any]:
        """Execute one training round with fairness-aware local training."""
        # Sample clients
        selected_clients = sample_clients(
            n_total=len(self.clients),
            n_sample=n_clients,
            seed=round_idx
        )
        
        if self.logger:
            self.logger.info(f"Round {round_idx}: Selected clients {selected_clients}")
        
        # Get current fairness configuration from aggregator
        fairness_config = None
        if hasattr(self.aggregator, 'get_fairness_config'):
            fairness_config = self.aggregator.get_fairness_config()
            if self.logger and fairness_config.get('bias_mitigation_mode', False):
                self.logger.info(f"Round {round_idx}: Bias mitigation mode active")
        
        # Collect client updates with fairness penalty
        client_updates = []
        client_weights = []
        client_stats = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Client local training with fairness penalty
            delta, n_samples, stats = client.train(
                global_weights=self.global_weights,
                epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                proximal_mu=proximal_mu,
                server_val_data=self.val_data,
                fairness_config=fairness_config  # Pass fairness config to client
            )
            
            # Add client ID to stats
            stats['client_id'] = client_id
            stats['n_samples'] = n_samples
            
            client_updates.append(delta)
            client_weights.append(n_samples)
            client_stats.append(stats)
        
        # Compute fairness-aware aggregation weights
        agg_weights = self.aggregator.compute_weights(client_stats)
        
        # Log client weights distribution
        if self.logger:
            weight_stats = {
                'min': float(agg_weights.min()),
                'max': float(agg_weights.max()),
                'std': float(agg_weights.std())
            }
            self.logger.info(f"Round {round_idx} weights: {weight_stats}")
        
        # Secure aggregation if enabled
        if self.secure_agg:
            aggregated_delta = self.secure_agg.aggregate(
                client_updates,
                weights=agg_weights
            )
        else:
            # Standard weighted aggregation
            aggregated_delta = weighted_average_weights(
                client_updates,
                agg_weights
            )
        
        # Apply server-side momentum if using FairCare-FL++
        if hasattr(self.aggregator, 'apply_server_momentum'):
            aggregated_delta = self.aggregator.apply_server_momentum(aggregated_delta)
        elif hasattr(self.aggregator, "enable_server_momentum") and self.aggregator.enable_server_momentum:
            # Manual server momentum
            if self.momentum_buffer is None:
                self.momentum_buffer = aggregated_delta
            else:
                theta = getattr(self.aggregator, 'theta_server', 0.8)
                for key in aggregated_delta.keys():
                    aggregated_delta[key] = (
                        theta * self.momentum_buffer[key] +
                        (1 - theta) * aggregated_delta[key]
                    )
                self.momentum_buffer = aggregated_delta
        
        # Update global model
        self.global_weights = apply_model_delta(
            self.global_weights,
            aggregated_delta,
            lr=server_lr
        )
        self.model.load_state_dict(self.global_weights)
        
        # Compute round statistics with fairness metrics
        round_stats = self._compute_round_stats(
            round_idx, n_clients, selected_clients, 
            agg_weights, client_stats
        )
        
        # Perform bias detection and response
        self._detect_and_respond_to_bias(round_stats)
        
        return round_stats
    
    def _compute_round_stats(
        self,
        round_idx: int,
        n_clients: int,
        selected_clients: List[int],
        agg_weights: torch.Tensor,
        client_stats: List[Dict]
    ) -> Dict[str, Any]:
        """Compute comprehensive round statistics."""
        round_stats = {
            "round": round_idx,
            "n_clients": n_clients,
            "selected_clients": selected_clients,
            "client_weights": agg_weights.tolist(),
            "avg_train_loss": sum(s.get("train_loss", 0) for s in client_stats) / len(client_stats)
        }
        
        # Add average fairness metrics
        fairness_metrics = ['eo_gap', 'fpr_gap', 'sp_gap', 'worst_group_F1']
        for metric in fairness_metrics:
            values = [s.get(metric, 0) for s in client_stats if metric in s]
            if values:
                round_stats[f"avg_{metric}"] = sum(values) / len(values)
        
        # Add fairness penalty info
        lambda_values = [s.get('lambda_fair', 0) for s in client_stats]
        if lambda_values:
            round_stats['avg_lambda_fair'] = sum(lambda_values) / len(lambda_values)
        
        # Check if in bias mitigation mode
        if hasattr(self.aggregator, 'bias_mitigation_mode'):
            round_stats['bias_mitigation_mode'] = self.aggregator.bias_mitigation_mode
        
        return round_stats
    
    def _detect_and_respond_to_bias(self, round_stats: Dict[str, Any]):
        """Detect bias and trigger adaptive response."""
        # Extract fairness gaps
        eo_gap = round_stats.get('avg_eo_gap', 0)
        fpr_gap = round_stats.get('avg_fpr_gap', 0)
        sp_gap = round_stats.get('avg_sp_gap', 0)
        
        # Define thresholds (can be configurable)
        eo_threshold = 0.15
        fpr_threshold = 0.15
        sp_threshold = 0.2
        
        # Check if bias is high
        high_bias = (
            eo_gap > eo_threshold or
            fpr_gap > fpr_threshold or
            sp_gap > sp_threshold
        )
        
        if high_bias:
            self.consecutive_high_bias_rounds += 1
            
            if self.consecutive_high_bias_rounds >= 2 and not self.bias_alert_triggered:
                # Trigger bias alert after 2 consecutive rounds of high bias
                self.bias_alert_triggered = True
                if self.logger:
                    self.logger.info(
                        f"⚠️ BIAS ALERT: High bias detected for {self.consecutive_high_bias_rounds} rounds. "
                        f"EO gap: {eo_gap:.3f}, FPR gap: {fpr_gap:.3f}, SP gap: {sp_gap:.3f}"
                    )
                
                # Notify aggregator to enter bias mitigation mode
                if hasattr(self.aggregator, '_detect_and_respond_to_bias'):
                    # FairCare-FL++ has built-in bias response
                    pass
                else:
                    # Manual adjustment for other aggregators
                    if hasattr(self.aggregator, 'tau'):
                        self.aggregator.tau *= 0.7  # Sharpen temperature
                    if hasattr(self.aggregator, 'epsilon'):
                        self.aggregator.epsilon = min(0.1, self.aggregator.epsilon * 1.5)
        else:
            # Bias is under control
            if self.consecutive_high_bias_rounds > 0:
                if self.logger:
                    self.logger.info(
                        f"✓ Bias reduced. EO gap: {eo_gap:.3f}, FPR gap: {fpr_gap:.3f}, SP gap: {sp_gap:.3f}"
                    )
            
            self.consecutive_high_bias_rounds = 0
            self.bias_alert_triggered = False
        
        # Record bias mitigation status
        self.history['bias_mitigation_rounds'].append(self.bias_alert_triggered)
    
    def evaluate(
        self,
        data: Optional[Tuple] = None,
        prefix: str = "val"
    ) -> Dict[str, float]:
        """Evaluate global model with comprehensive fairness metrics."""
        if data is None:
            data = self.val_data if prefix == "val" else self.test_data
        
        if data is None:
            return {}
        
        X, y, a = data
        X = X.to(self.device)
        
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            outputs = self.model(X).squeeze()
            loss = criterion(outputs, y.to(self.device).float()).item()
            y_pred = (torch.sigmoid(outputs) > 0.5).int().cpu()
        
        # Compute metrics
        metrics = {
            f"{prefix}_loss": loss,
            f"{prefix}_accuracy": (y_pred == y).float().mean().item()
        }
        
        # Add comprehensive fairness metrics
        if a is not None:
            fair_report = fairness_report(y_pred, y, a)
            
            # Add all fairness metrics
            for key, value in fair_report.items():
                if key != "group_stats":  # Skip raw counts
                    metrics[f"{prefix}_{key}"] = value
            
            # Track key fairness metrics in history
            if prefix == "val":
                self.history['eo_gaps'].append(fair_report['EO_gap'])
                self.history['fpr_gaps'].append(fair_report['FPR_gap'])
                self.history['sp_gaps'].append(fair_report['SP_gap'])
                self.history['worst_group_f1'].append(fair_report.get('worst_group_F1', 0))
        
        return metrics
    
    def run(
        self,
        rounds: int,
        n_clients_per_round: int,
        local_epochs: int,
        lr: float,
        weight_decay: float = 0.0,
        server_lr: float = 1.0,
        eval_every: int = 1,
        checkpoint_every: int = 10
    ) -> Dict[str, List]:
        """Run federated training with bias detection and mitigation."""
        if self.logger:
            self.logger.info("Starting FairCare-FL++ federated training")
            self.logger.info(f"Total rounds: {rounds}")
            self.logger.info(f"Clients per round: {n_clients_per_round}")
            
            # Log aggregator configuration
            if hasattr(self.aggregator, 'get_statistics'):
                agg_stats = self.aggregator.get_statistics()
                self.logger.info(f"Aggregator config: {agg_stats}")
        
        # Initial evaluation
        if eval_every > 0:
            val_metrics = self.evaluate(self.val_data, prefix="val")
            test_metrics = self.evaluate(self.test_data, prefix="test")
            
            if self.logger:
                self.logger.info(f"Initial - Val: {val_metrics}")
                self.logger.info(f"Initial - Test: {test_metrics}")
                self.logger.log_metrics(
                    {**val_metrics, **test_metrics},
                    step=0
                )
        
        # Training rounds
        best_worst_group_f1 = 0.0
        best_round = 0
        
        for round_idx in range(1, rounds + 1):
            # Train round
            round_stats = self.train_round(
                round_idx=round_idx,
                n_clients=n_clients_per_round,
                local_epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                server_lr=server_lr
            )
            
            # Evaluation
            if eval_every > 0 and round_idx % eval_every == 0:
                val_metrics = self.evaluate(self.val_data, prefix="val")
                test_metrics = self.evaluate(self.test_data, prefix="test")
                
                round_stats.update(val_metrics)
                round_stats.update(test_metrics)
                
                # Track best model based on worst group F1
                current_worst_f1 = val_metrics.get('val_worst_group_F1', 0)
                if current_worst_f1 > best_worst_group_f1:
                    best_worst_group_f1 = current_worst_f1
                    best_round = round_idx
                    # Save best model
                    self.best_model_weights = copy.deepcopy(self.global_weights)
                
                if self.logger:
                    # Enhanced logging for fairness metrics
                    log_msg = (
                        f"Round {round_idx} - "
                        f"Loss: {round_stats['avg_train_loss']:.4f}, "
                        f"Acc: {val_metrics.get('val_accuracy', 0):.4f}, "
                        f"EO Gap: {val_metrics.get('val_EO_gap', 0):.4f}, "
                        f"Worst F1: {current_worst_f1:.4f}"
                    )
                    
                    if round_stats.get('bias_mitigation_mode', False):
                        log_msg += " [BIAS MITIGATION]"
                    
                    self.logger.info(log_msg)
                    self.logger.log_metrics(round_stats, step=round_idx)
            
            # Save checkpoint
            if checkpoint_every > 0 and round_idx % checkpoint_every == 0:
                self.save_checkpoint(round_idx)
            
            # Update history
            self.history["train_loss"].append(round_stats["avg_train_loss"])
            if "val_accuracy" in round_stats:
                self.history["val_acc"].append(round_stats["val_accuracy"])
            if "test_accuracy" in round_stats:
                self.history["test_acc"].append(round_stats["test_accuracy"])
        
        # Final evaluation
        final_val = self.evaluate(self.val_data, prefix="val")
        final_test = self.evaluate(self.test_data, prefix="test")
        
        if self.logger:
            self.logger.info("="*60)
            self.logger.info("Training completed")
            self.logger.info(f"Best round: {best_round} (Worst Group F1: {best_worst_group_f1:.4f})")
            self.logger.info(f"Final Val: {final_val}")
            self.logger.info(f"Final Test: {final_test}")
            
            # Log aggregator statistics
            if hasattr(self.aggregator, 'get_statistics'):
                final_stats = self.aggregator.get_statistics()
                self.logger.info(f"Final aggregator stats: {final_stats}")
        
        return self.history
    
    def save_checkpoint(self, round_idx: int) -> None:
        """Save model checkpoint with fairness metrics."""
        if self.logger:
            checkpoint_path = self.logger.logdir / f"checkpoint_round_{round_idx}.pt"
            
            # Include fairness metrics in checkpoint
            checkpoint = {
                "round": round_idx,
                "model_state_dict": self.global_weights,
                "history": self.history,
                "fairness_metrics": {
                    "eo_gap": self.history['eo_gaps'][-1] if self.history['eo_gaps'] else None,
                    "fpr_gap": self.history['fpr_gaps'][-1] if self.history['fpr_gaps'] else None,
                    "sp_gap": self.history['sp_gaps'][-1] if self.history['sp_gaps'] else None,
                    "worst_group_f1": self.history['worst_group_f1'][-1] if self.history['worst_group_f1'] else None
                }
            }
            
            # Add aggregator state if available
            if hasattr(self.aggregator, 'get_statistics'):
                checkpoint['aggregator_state'] = self.aggregator.get_statistics()
            
            torch.save(checkpoint, checkpoint_path)
