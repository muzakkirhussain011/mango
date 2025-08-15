"""Federated learning client implementation with fairness penalty."""
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
from faircare.core.utils import compute_model_delta, create_optimizer
from faircare.fairness.metrics import fairness_report, compute_fairness_loss


class Client:
    """Federated learning client with fairness-aware local training."""
    
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
        server_val_data: Optional[Tuple] = None,
        fairness_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
        """
        Train local model with optional fairness penalty.
        
        Args:
            global_weights: Global model weights
            epochs: Number of local epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            proximal_mu: FedProx proximal term coefficient
            server_val_data: Server validation data for evaluation
            fairness_config: Fairness configuration from server
        
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
        
        # Extract fairness penalty weight
        lambda_fair = 0.0
        bias_mitigation_mode = False
        if fairness_config is not None:
            lambda_fair = fairness_config.get('lambda_fair', 0.0)
            bias_mitigation_mode = fairness_config.get('bias_mitigation_mode', False)
            
            # Increase lambda if in bias mitigation mode
            if bias_mitigation_mode:
                lambda_fair *= 1.5
        
        # Setup optimizer
        optimizer = create_optimizer(
            self.model,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Training loop with fairness penalty
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_fair_loss = 0.0
        n_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_pred_loss = 0.0
            epoch_fair_loss = 0.0
            
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
                outputs = self.model(X)
                
                # Handle shape for BCEWithLogitsLoss
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                elif outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                
                # Prediction loss
                pred_loss = criterion(outputs, y)
                
                # Fairness penalty loss (only if sensitive attribute available)
                fair_loss = torch.tensor(0.0, device=self.device)
                if lambda_fair > 0 and a is not None:
                    fair_loss = compute_fairness_loss(
                        outputs=outputs,
                        labels=y,
                        sensitive=a,
                        loss_type='eo_sp_combined',  # Combined EO and SP loss
                        device=self.device
                    )
                
                # Total loss with fairness penalty
                loss = pred_loss + lambda_fair * fair_loss
                
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
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track losses
                batch_size = X.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_pred_loss += pred_loss.item() * batch_size
                epoch_fair_loss += fair_loss.item() * batch_size
                n_samples += batch_size
            
            total_loss += epoch_loss
            total_pred_loss += epoch_pred_loss
            total_fair_loss += epoch_fair_loss
            
            # Adaptive training: extra epochs if high bias detected
            if bias_mitigation_mode and epoch == epochs - 1:
                # Check if we need extra training
                with torch.no_grad():
                    # Quick bias check on a sample
                    if a is not None and epoch_fair_loss / n_samples > 0.1:
                        # Add one more epoch with higher fairness weight
                        epochs += 1
                        lambda_fair *= 1.2
        
        # Compute average losses
        avg_loss = total_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_pred_loss = total_pred_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_fair_loss = total_fair_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        
        # Compute model delta
        delta = compute_model_delta(
            self.model.state_dict(),
            global_weights
        )
        
        # Compute comprehensive evaluation metrics
        stats = {
            "train_loss": avg_loss,
            "train_pred_loss": avg_pred_loss,
            "train_fair_loss": avg_fair_loss,
            "lambda_fair": lambda_fair,
            "client_id": self.client_id
        }
        
        # Evaluate on server validation data if provided
        if server_val_data is not None:
            val_stats = self._evaluate_fairness(server_val_data)
            stats.update(val_stats)
        
        # Add local validation if available
        if self.val_loader is not None:
            local_val_stats = self._evaluate_local()
            for key, value in local_val_stats.items():
                stats[f"local_{key}"] = value
        
        return delta, len(self.train_dataset), stats
    
    def _evaluate_fairness(self, val_data: Tuple) -> Dict[str, float]:
        """Evaluate model on validation data with fairness metrics."""
        X_val, y_val, a_val = val_data
        X_val = X_val.to(self.device)
        
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            outputs = self.model(X_val)
            
            # Handle shape
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            elif outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            
            # Compute loss
            val_loss = criterion(outputs, y_val.to(self.device).float()).item()
            
            # Get predictions
            y_pred = (torch.sigmoid(outputs) > 0.5).int().cpu()
        
        # Compute fairness metrics
        fair_report = fairness_report(
            y_pred,
            y_val,
            a_val if a_val is not None else None
        )
        
        # Extract key metrics
        return {
            "val_loss": val_loss,
            "val_acc": fair_report["accuracy"],
            "val_accuracy": fair_report["accuracy"],
            "eo_gap": fair_report["EO_gap"],
            "fpr_gap": fair_report["FPR_gap"],
            "sp_gap": fair_report["SP_gap"],
            "val_EO_gap": fair_report["EO_gap"],
            "val_FPR_gap": fair_report["FPR_gap"],
            "val_SP_gap": fair_report["SP_gap"],
            "worst_group_F1": fair_report.get("worst_group_F1", 0.5),
            "max_group_gap": fair_report.get("max_group_gap", 0.0),
            "macro_F1": fair_report.get("macro_F1", 0.5)
        }
    
    def _evaluate_local(self) -> Dict[str, float]:
        """Evaluate on local validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_sensitive = []
        n_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    X, y, a = batch
                    all_sensitive.append(a)
                else:
                    X, y = batch
                    a = None
                
                X = X.to(self.device)
                y = y.to(self.device).float()
                
                outputs = self.model(X)
                
                # Handle shape
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                elif outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                
                loss = criterion(outputs, y)
                total_loss += loss.item() * X.size(0)
                n_samples += X.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu().int())
        
        # Aggregate results
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
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }
        
        # Add fairness metrics if sensitive attributes available
        if all_sensitive is not None:
            fair_report = fairness_report(all_preds, all_labels, all_sensitive)
            metrics.update({
                "eo_gap": fair_report["EO_gap"],
                "fpr_gap": fair_report["FPR_gap"],
                "sp_gap": fair_report["SP_gap"],
            })
        
        return metrics
    
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
                
                outputs = self.model(X)
                
                # Handle shape
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                elif outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                if y.dim() == 0:
                    y = y.unsqueeze(0)
                    
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
