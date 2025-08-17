# faircare/core/client.py
"""Federated learning client with FedBLE enhancements."""
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import copy
from faircare.core.utils import compute_model_delta, create_optimizer
from faircare.fairness.metrics import fairness_report
from faircare.fairness.losses import (
    combined_fairness_loss,
    AdversaryNetwork,
    gradient_reversal
)


class Client:
    """Federated learning client with fairness-aware training and adversarial debiasing."""
    
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
        
        # Initialize adversary (will be created on demand)
        self.adversary = None
        self.adversary_optimizer = None
    
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
        Train local model with FedBLE enhancements.
        
        Includes fairness-aware loss, optional adversarial debiasing,
        and FedProx proximal term support.
        
        Returns:
            - Model delta (update)
            - Number of samples
            - Training statistics including fairness metrics
        """
        # Extract fairness configuration
        if fairness_config is None:
            fairness_config = {}
        
        lambda_fair = fairness_config.get('lambda_fair', 0.0)
        bias_mitigation_mode = fairness_config.get('bias_mitigation_mode', False)
        use_adversary = fairness_config.get('use_adversary', False)
        adv_weight = fairness_config.get('adv_weight', 0.1)
        w_eo = fairness_config.get('w_eo', 1.0)
        w_fpr = fairness_config.get('w_fpr', 0.5)
        w_sp = fairness_config.get('w_sp', 0.5)
        prox_mu = fairness_config.get('prox_mu', proximal_mu)
        extra_epoch = fairness_config.get('extra_epoch', False)
        
        # Adjust for bias mitigation mode
        if bias_mitigation_mode:
            if lambda_fair > 0:
                lambda_fair = lambda_fair * 1.5  # Increase penalty
            if extra_epoch:
                epochs += 1  # Add extra epoch
        
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        self.model.train()
        
        # Store initial weights for proximal term
        global_model = None
        if prox_mu > 0:
            global_model = copy.deepcopy(self.model)
            global_model.eval()
        
        # Setup optimizer
        optimizer = create_optimizer(
            self.model,
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Setup adversary if needed
        if use_adversary and self.adversary is None:
            # Get output dimension from model
            with torch.no_grad():
                dummy_input = torch.randn(1, next(self.model.parameters()).shape[1])
                dummy_output = self.model(dummy_input.to(self.device))
                output_dim = dummy_output.shape[-1] if dummy_output.dim() > 1 else 1
            
            self.adversary = AdversaryNetwork(
                input_dim=output_dim,
                hidden_dims=[32, 16],
                n_sensitive_classes=2
            ).to(self.device)
            
            self.adversary_optimizer = torch.optim.Adam(
                self.adversary.parameters(),
                lr=lr * 2  # Higher LR for adversary
            )
        
        # Training loop
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_fairness_loss = 0.0
        total_adv_loss = 0.0
        n_samples = 0
        n_batches = 0
        
        # Track norms for drift detection
        delta_norms = []
        grad_norms = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_fairness_loss = 0.0
            epoch_adv_loss = 0.0
            
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
                
                # Fairness loss
                fair_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                if a is not None and lambda_fair > 0:
                    fair_loss = combined_fairness_loss(
                        outputs.unsqueeze(1) if outputs.dim() == 1 else outputs,
                        y,
                        a,
                        w_eo=w_eo,
                        w_fpr=w_fpr,
                        w_sp=w_sp
                    )
                
                # Adversarial loss
                adv_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                if use_adversary and a is not None and self.adversary is not None:
                    # Train adversary to predict sensitive attribute
                    adv_input = outputs.detach()
                    if adv_input.dim() == 1:
                        adv_input = adv_input.unsqueeze(1)
                    
                    adv_pred = self.adversary(adv_input)
                    
                    # Adversary loss (cross-entropy for classification)
                    if adv_pred.shape[1] > 1:
                        adv_criterion = nn.CrossEntropyLoss()
                        adv_loss_value = adv_criterion(adv_pred, a.long())
                    else:
                        adv_criterion = nn.BCEWithLogitsLoss()
                        adv_loss_value = adv_criterion(adv_pred.squeeze(), a.float())
                    
                    # Update adversary
                    self.adversary_optimizer.zero_grad()
                    adv_loss_value.backward()
                    self.adversary_optimizer.step()
                    
                    # Now compute adversarial loss for main model
                    # Use gradient reversal to fool adversary
                    adv_input_rev = gradient_reversal(
                        outputs.unsqueeze(1) if outputs.dim() == 1 else outputs,
                        alpha=1.0
                    )
                    adv_pred_rev = self.adversary(adv_input_rev)
                    
                    if adv_pred_rev.shape[1] > 1:
                        adv_loss = adv_criterion(adv_pred_rev, a.long())
                    else:
                        adv_loss = adv_criterion(adv_pred_rev.squeeze(), a.float())
                
                # Total loss
                total_loss_batch = pred_loss
                
                if lambda_fair > 0:
                    total_loss_batch = total_loss_batch + lambda_fair * fair_loss
                
                if use_adversary and adv_weight > 0:
                    total_loss_batch = total_loss_batch + adv_weight * adv_loss
                
                # Add proximal term if using FedProx
                if prox_mu > 0 and global_model is not None:
                    proximal_term = 0.0
                    for w, w_global in zip(
                        self.model.parameters(),
                        global_model.parameters()
                    ):
                        proximal_term += torch.norm(w - w_global) ** 2
                    total_loss_batch = total_loss_batch + (prox_mu / 2) * proximal_term
                
                # Backward pass
                total_loss_batch.backward()
                
                # Track gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)
                
                optimizer.step()
                
                # Track losses
                epoch_loss += pred_loss.item() * X.size(0)
                if lambda_fair > 0:
                    epoch_fairness_loss += fair_loss.item() * X.size(0)
                if use_adversary:
                    epoch_adv_loss += adv_loss.item() * X.size(0)
                
                n_samples += X.size(0)
                n_batches += 1
            
            total_loss += epoch_loss
            total_fairness_loss += epoch_fairness_loss
            total_adv_loss += epoch_adv_loss
        
        # Compute average losses
        avg_loss = total_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_fair_loss = total_fairness_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_adv_loss = total_adv_loss / (n_samples * epochs) if n_samples > 0 and use_adversary else 0.0
        
        # Compute model delta
        delta = compute_model_delta(
            self.model.state_dict(),
            global_weights
        )
        
        # Compute delta norm
        delta_norm = 0.0
        for key in delta:
            delta_norm += torch.norm(delta[key]) ** 2
        delta_norm = (delta_norm ** 0.5).item()
        delta_norms.append(delta_norm)
        
        # Compute fairness metrics on server validation if provided
        stats = {
            "train_loss": avg_loss,
            "fairness_loss": avg_fair_loss,
            "adversary_loss": avg_adv_loss,
            "n_samples": len(self.train_dataset),
            "client_id": self.client_id,
            "delta_norm": delta_norm,
            "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        }
        
        if server_val_data is not None:
            X_val, y_val, a_val = server_val_data
            X_val = X_val.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_val)
                
                # Handle shape
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                elif outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                    
                y_pred = (torch.sigmoid(outputs) > 0.5).int()
                val_loss = criterion(outputs, y_val.to(self.device).float()).item()
            
            # Compute fairness metrics
            fair_report = fairness_report(
                y_pred.cpu(),
                y_val,
                a_val if a_val is not None else None
            )
            
            stats.update({
                "val_loss": val_loss,
                "eo_gap": fair_report.get("EO_gap", 0.0),
                "fpr_gap": fair_report.get("FPR_gap", 0.0),
                "sp_gap": fair_report.get("SP_gap", 0.0),
                "val_acc": fair_report.get("accuracy", 0.0),
                "val_accuracy": fair_report.get("accuracy", 0.0),
                "worst_group_F1": fair_report.get("worst_group_F1", 0.0),
                "worst_group_f1": fair_report.get("worst_group_F1", 0.0)  # Alias
            })
        
        # Log if in bias mitigation mode
        if bias_mitigation_mode:
            print(f"[Client {self.client_id}] Bias mitigation active - "
                  f"λ_fair: {lambda_fair:.3f}, adversary: {use_adversary}, "
                  f"extra epochs: {1 if extra_epoch else 0}")
        
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
