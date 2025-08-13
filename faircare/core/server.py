"""Federated learning server implementation."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
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
    """Federated learning server."""
    
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
        
        # Momentum for FairCare-FL
        self.momentum_buffer = None
        
        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "test_acc": [],
            "fairness_metrics": []
        }
    
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
        """Execute one training round."""
        # Sample clients
        selected_clients = sample_clients(
            n_total=len(self.clients),
            n_sample=n_clients,
            seed=round_idx
        )
        
        if self.logger:
            self.logger.info(f"Round {round_idx}: Selected clients {selected_clients}")
        
        # Collect client updates
        client_updates = []
        client_weights = []
        client_stats = []
        
        for client_id in selected_clients:
            client = self.clients[client_id]
            
            # Client local training
            delta, n_samples, stats = client.train(
                global_weights=self.global_weights,
                epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                proximal_mu=proximal_mu,
                server_val_data=self.val_data
            )
            
            client_updates.append(delta)
            client_weights.append(n_samples)
            client_stats.append(stats)
        
        # Compute aggregation weights
        agg_weights = self.aggregator.compute_weights(client_stats)
        
        # Secure aggregation if enabled
        if self.secure_agg:
            aggregated_delta = self.secure_agg.aggregate(
                [delta for delta in client_updates],
                weights=agg_weights
            )
        else:
            # Standard weighted aggregation
            aggregated_delta = weighted_average_weights(
                client_updates,
                agg_weights
            )
        
        # Apply momentum if using FairCare-FL
        if hasattr(self.aggregator, "use_momentum") and self.aggregator.use_momentum:
            if self.momentum_buffer is None:
                self.momentum_buffer = aggregated_delta
            else:
                momentum = self.aggregator.momentum
                for key in self.momentum_buffer.keys():
                    self.momentum_buffer[key] = (
                        momentum * self.momentum_buffer[key] +
                        (1 - momentum) * aggregated_delta[key]
                    )
                aggregated_delta = self.momentum_buffer
        
        # Update global model
        self.global_weights = apply_model_delta(
            self.global_weights,
            aggregated_delta,
            lr=server_lr
        )
        self.model.load_state_dict(self.global_weights)
        
        # Compute round statistics
        round_stats = {
            "round": round_idx,
            "n_clients": n_clients,
            "selected_clients": selected_clients,
            "client_weights": agg_weights.tolist(),
            "avg_train_loss": sum(s["train_loss"] for s in client_stats) / len(client_stats)
        }
        
        # Add fairness stats if available
        if "eo_gap" in client_stats[0]:
            round_stats.update({
                "avg_eo_gap": sum(s["eo_gap"] for s in client_stats) / len(client_stats),
                "avg_fpr_gap": sum(s["fpr_gap"] for s in client_stats) / len(client_stats),
                "avg_sp_gap": sum(s["sp_gap"] for s in client_stats) / len(client_stats)
            })
        
        return round_stats
    
    def evaluate(
        self,
        data: Optional[Tuple] = None,
        prefix: str = "val"
    ) -> Dict[str, float]:
        """Evaluate global model."""
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
        
        # Add fairness metrics
        if a is not None:
            fair_report = fairness_report(y_pred, y, a)
            for key, value in fair_report.items():
                metrics[f"{prefix}_{key}"] = value
        
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
        """Run federated training."""
        if self.logger:
            self.logger.info("Starting federated training")
            self.logger.info(f"Total rounds: {rounds}")
            self.logger.info(f"Clients per round: {n_clients_per_round}")
        
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
                
                if self.logger:
                    self.logger.info(
                        f"Round {round_idx} - "
                        f"Train Loss: {round_stats['avg_train_loss']:.4f}, "
                        f"Val Acc: {val_metrics.get('val_accuracy', 0):.4f}, "
                        f"Val EO Gap: {val_metrics.get('val_EO_gap', 0):.4f}"
                    )
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
            self.logger.info("Training completed")
            self.logger.info(f"Final Val: {final_val}")
            self.logger.info(f"Final Test: {final_test}")
        
        return self.history
    
    def save_checkpoint(self, round_idx: int) -> None:
        """Save model checkpoint."""
        if self.logger:
            checkpoint_path = self.logger.logdir / f"checkpoint_round_{round_idx}.pt"
            torch.save({
                "round": round_idx,
                "model_state_dict": self.global_weights,
                "history": self.history
            }, checkpoint_path)
