"""Federated learning client implementation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, Any
import copy
from faircare.core.utils import compute_model_delta, create_optimizer
from faircare.fairness.metrics import fairness_report


class Client:
    """Federated learning client."""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        else:
            self.val_loader = None
    
    def train(
        self,
        global_weights: Dict[str, torch.Tensor],
        epochs: int,
        lr: float,
        weight_decay: float = 0.0,
        proximal_mu: float = 0.0,
        server_val_data: Optional[Tuple] = None
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
        """
        Train local model.
        
        Returns:
            - Model delta (update)
            - Number of samples
            - Training statistics including fairness metrics
        """
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        self.model.train()
        
        # Store initial weights for proximal term
        if proximal_mu > 0:
            global_model = copy.deepcopy(self.model)
            global_model.eval()
        
        # Setup optimizer
        optimizer = create_optimizer(
            self.model,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Training loop
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        n_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Handle both with and without sensitive attributes
                if len(batch) == 3:
                    X, y, a = batch
                    a = a.to(self.device) if a is not None else None
                else:
                    X, y = batch
                    a = None
                
                X = X.to(self.device)
                y = y.to(self.device).float()
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X).squeeze()
                loss = criterion(outputs, y)
                
                # Add proximal term if using FedProx
                if proximal_mu > 0:
                    proximal_term = 0.0
                    for w, w_global in zip(
                        self.model.parameters(),
                        global_model.parameters()
                    ):
                        proximal_term += torch.norm(w - w_global) ** 2
                    loss += (proximal_mu / 2) * proximal_term
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
            
            total_loss += epoch_loss
        
        # Compute average loss
        avg_loss = total_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        
        # Compute model delta
        delta = compute_model_delta(
            self.model.state_dict(),
            global_weights
        )
        
        # Compute fairness metrics on server validation if provided
        stats = {"train_loss": avg_loss}
        
        if server_val_data is not None:
            X_val, y_val, a_val = server_val_data
            X_val = X_val.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_val).squeeze()
                y_pred = (torch.sigmoid(outputs) > 0.5).int()
            
            # Compute fairness metrics
            fair_report = fairness_report(
                y_pred.cpu(),
                y_val,
                a_val if a_val is not None else None
            )
            
            stats.update({
                "val_loss": criterion(outputs, y_val.to(self.device).float()).item(),
                "eo_gap": fair_report["EO_gap"],
                "fpr_gap": fair_report["FPR_gap"],
                "sp_gap": fair_report["SP_gap"],
                "val_acc": fair_report["accuracy"]
            })
        
        return delta, len(self.train_dataset), stats
    
    def evaluate(
        self,
        weights: Dict[str, torch.Tensor],
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        """Evaluate model on test data."""
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()
        
        if test_loader is None:
            test_loader = self.val_loader
            if test_loader is None:
                return {}
        
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_sensitive = []
        n_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    X, y, a = batch
                    all_sensitive.append(a)
                else:
                    X, y = batch
                    a = None
                
                X = X.to(self.device)
                y = y.to(self.device).float()
                
                outputs = self.model(X).squeeze()
                loss = criterion(outputs, y)
                
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu().int())
        
        # Aggregate predictions
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        if all_sensitive:
            all_sensitive = torch.cat(all_sensitive)
        else:
            all_sensitive = None
        
        # Compute metrics
        avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
        accuracy = (all_preds == all_labels).float().mean().item()
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "n_samples": n_samples
        }
        
        # Add fairness metrics if sensitive attributes available
        if all_sensitive is not None:
            fair_report = fairness_report(all_preds, all_labels, all_sensitive)
            metrics.update(fair_report)
        
        return metrics
