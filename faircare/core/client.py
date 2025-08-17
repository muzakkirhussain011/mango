"""Federated learning client with fairness-aware training and CALT hooks."""
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import copy
import numpy as np
from faircare.core.utils import compute_model_delta, create_optimizer
from faircare.fairness.metrics import fairness_report


class Client:
    """Federated learning client with enhanced fairness-aware training."""
    
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
        Train local model with optional CALT enhancements for faircare_fl.
        
        Returns:
            - Model delta (update)
            - Number of samples
            - Training statistics including fairness metrics
        """
        # Extract fairness configuration
        if fairness_config is None:
            fairness_config = {}
        
        # CALT parameters (only active for faircare_fl v2+)
        version = fairness_config.get('version', '1.0.0')
        use_calt = version >= '2.0.0'  # Enable CALT for v2+
        
        lambda_irm = fairness_config.get('lambda_irm', 0.0) if use_calt else 0.0
        lambda_adv = fairness_config.get('lambda_adv', 0.0) if use_calt else 0.0
        use_mixup = fairness_config.get('use_mixup', False) if use_calt else False
        use_cia = fairness_config.get('use_cia', False) if use_calt else False
        prox_mu = fairness_config.get('prox_mu', proximal_mu)
        
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
        
        # Training loop
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_irm_loss = 0.0
        total_adv_loss = 0.0
        n_samples = 0
        n_batches = 0
        
        # Track statistics
        delta_norms = []
        grad_norms = []
        group_confusion_counts = {}
        
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
                
                # Apply mixup augmentation if enabled
                if use_mixup and np.random.random() < 0.5:
                    X, y, a = self._apply_mixup(X, y, a)
                
                # Apply CIA augmentation if enabled
                if use_cia and a is not None and np.random.random() < 0.5:
                    X_cia, y_cia, a_cia = self._apply_cia(X, y, a)
                    # Concatenate with original batch
                    X = torch.cat([X, X_cia], dim=0)
                    y = torch.cat([y, y_cia], dim=0)
                    a = torch.cat([a, a_cia], dim=0)
                
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
                
                # Base prediction loss
                pred_loss = criterion(outputs, y)
                
                # Total loss
                total_loss_batch = pred_loss
                
                # Add IRM penalty if enabled
                if lambda_irm > 0 and a is not None:
                    irm_loss = self._compute_irm_penalty(outputs, y, a)
                    total_loss_batch = total_loss_batch + lambda_irm * irm_loss
                    total_irm_loss += irm_loss.item() * X.size(0)
                
                # Add adversarial loss if enabled
                if lambda_adv > 0 and a is not None:
                    adv_loss = self._compute_adversarial_loss(outputs, a)
                    total_loss_batch = total_loss_batch - lambda_adv * adv_loss  # Negative for GRL
                    total_adv_loss += adv_loss.item() * X.size(0)
                
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
                n_samples += X.size(0)
                n_batches += 1
                
                # Update group confusion counts
                if a is not None:
                    self._update_confusion_counts(outputs, y, a, group_confusion_counts)
            
            total_loss += epoch_loss
        
        # Compute average losses
        avg_loss = total_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_irm_loss = total_irm_loss / (n_samples * epochs) if n_samples > 0 and lambda_irm > 0 else 0.0
        avg_adv_loss = total_adv_loss / (n_samples * epochs) if n_samples > 0 and lambda_adv > 0 else 0.0
        
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
        
        # Compute fairness metrics
        stats = {
            "train_loss": avg_loss,
            "irm_loss": avg_irm_loss,
            "adv_loss": avg_adv_loss,
            "n_samples": len(self.train_dataset),
            "client_id": self.client_id,
            "delta_norm": delta_norm,
            "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            "group_stats": group_confusion_counts,
            "calt_enabled": use_calt
        }
        
        # Compute validation metrics if data provided
        if server_val_data is not None:
            val_stats = self._compute_validation_metrics(server_val_data)
            stats.update(val_stats)
        
        # Compute calibration error proxy
        stats['calibration_error'] = self._compute_calibration_error()
        
        return delta, len(self.train_dataset), stats
    
    def _apply_mixup(self, X: torch.Tensor, y: torch.Tensor, a: Optional[torch.Tensor], alpha: float = 0.2):
        """Apply mixup augmentation."""
        batch_size = X.size(0)
        
        # Sample lambda from Beta distribution
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        # Random permutation
        index = torch.randperm(batch_size).to(self.device)
        
        # Mix inputs
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        # For sensitive attribute, use majority voting
        if a is not None:
            mixed_a = a  # Keep original for simplicity
        else:
            mixed_a = None
        
        return mixed_X, mixed_y, mixed_a
    
    def _apply_cia(self, X: torch.Tensor, y: torch.Tensor, a: torch.Tensor):
        """Apply Counterfactual Interpolation Augmentation."""
        batch_size = X.size(0)
        
        # Find samples from different groups
        group_0_mask = (a == 0)
        group_1_mask = (a == 1)
        
        if group_0_mask.sum() == 0 or group_1_mask.sum() == 0:
            return X[:0], y[:0], a[:0]  # Return empty tensors
        
        # Sample pairs from different groups
        n_pairs = min(group_0_mask.sum(), group_1_mask.sum(), batch_size // 4)
        
        if n_pairs == 0:
            return X[:0], y[:0], a[:0]
        
        group_0_indices = torch.where(group_0_mask)[0][:n_pairs]
        group_1_indices = torch.where(group_1_mask)[0][:n_pairs]
        
        # Interpolate between different groups
        alpha = torch.rand(n_pairs, 1).to(self.device)
        
        X_0 = X[group_0_indices]
        X_1 = X[group_1_indices]
        y_0 = y[group_0_indices]
        y_1 = y[group_1_indices]
        
        X_cia = alpha * X_0 + (1 - alpha) * X_1
        y_cia = (alpha.squeeze() * y_0 + (1 - alpha.squeeze()) * y_1)
        
        # Assign mixed sensitive attribute (could be probabilistic)
        a_cia = torch.where(alpha.squeeze() > 0.5, a[group_0_indices], a[group_1_indices])
        
        return X_cia, y_cia, a_cia
    
    def _compute_irm_penalty(self, logits: torch.Tensor, labels: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        """Compute IRM penalty for environment invariance."""
        # Split by sensitive attribute as environments
        unique_envs = torch.unique(sensitive)
        
        if len(unique_envs) < 2:
            return torch.tensor(0.0, device=self.device)
        
        penalties = []
        
        for env in unique_envs:
            env_mask = (sensitive == env)
            if env_mask.sum() < 2:
                continue
            
            env_logits = logits[env_mask]
            env_labels = labels[env_mask]
            
            # Compute environment-specific loss
            env_loss = F.binary_cross_entropy_with_logits(env_logits, env_labels, reduction='mean')
            
            # Compute gradient norm w.r.t. a dummy scale parameter
            scale = torch.tensor(1.0, requires_grad=True, device=self.device)
            scaled_loss = env_loss * scale
            
            grad = torch.autograd.grad(scaled_loss, [scale], create_graph=True)[0]
            penalty = torch.sum(grad ** 2)
            
            penalties.append(penalty)
        
        if penalties:
            return torch.stack(penalties).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_adversarial_loss(self, features: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        """Compute adversarial loss for domain-invariant features (simplified)."""
        # Simple linear adversary (to avoid adding complex networks)
        # In production, this would be a proper adversarial network
        
        # Use features as input to predict sensitive attribute
        if features.dim() == 1:
            features = features.unsqueeze(1)
        
        # Simple projection to predict sensitive attribute
        adv_weight = torch.randn(features.shape[1], 1, device=self.device) * 0.01
        adv_logits = features.detach() @ adv_weight
        
        # Adversarial loss (binary cross-entropy)
        adv_loss = F.binary_cross_entropy_with_logits(
            adv_logits.squeeze(),
            sensitive.float(),
            reduction='mean'
        )
        
        return adv_loss
    
    def _update_confusion_counts(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor, 
        sensitive: torch.Tensor,
        counts_dict: Dict
    ):
        """Update per-group confusion counts."""
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).int()
            
            for group_val in torch.unique(sensitive):
                group_mask = (sensitive == group_val)
                group_name = f"group_{group_val.item()}"
                
                if group_name not in counts_dict:
                    counts_dict[group_name] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                
                group_preds = preds[group_mask]
                group_labels = labels[group_mask].int()
                
                counts_dict[group_name]["TP"] += ((group_preds == 1) & (group_labels == 1)).sum().item()
                counts_dict[group_name]["FP"] += ((group_preds == 1) & (group_labels == 0)).sum().item()
                counts_dict[group_name]["FN"] += ((group_preds == 0) & (group_labels == 1)).sum().item()
                counts_dict[group_name]["TN"] += ((group_preds == 0) & (group_labels == 0)).sum().item()
    
    def _compute_validation_metrics(self, server_val_data: Tuple) -> Dict[str, float]:
        """Compute validation metrics including fairness."""
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
            val_loss = F.binary_cross_entropy_with_logits(
                outputs, y_val.to(self.device).float()
            ).item()
        
        # Compute fairness metrics
        fair_report = fairness_report(
            y_pred.cpu(),
            y_val,
            a_val if a_val is not None else None
        )
        
        return {
            "val_loss": val_loss,
            "eo_gap": fair_report.get("EO_gap", 0.0),
            "fpr_gap": fair_report.get("FPR_gap", 0.0),
            "sp_gap": fair_report.get("SP_gap", 0.0),
            "val_acc": fair_report.get("accuracy", 0.0),
            "val_accuracy": fair_report.get("accuracy", 0.0),
            "worst_group_F1": fair_report.get("worst_group_F1", 0.0),
            "worst_group_f1": fair_report.get("worst_group_F1", 0.0)
        }
    
    def _compute_calibration_error(self) -> float:
        """Compute expected calibration error proxy."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        total_ece = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    X, y, _ = batch
                else:
                    X, y = batch
                
                X = X.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(X)
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                probs = torch.sigmoid(outputs)
                
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (probs > bin_lower) & (probs <= bin_upper)
                    prop_in_bin = in_bin.float().mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = y[in_bin].float().mean()
                        avg_confidence_in_bin = probs[in_bin].mean()
                        ece = torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                        total_ece += ece.item()
                
                total_samples += len(y)
        
        return total_ece
    
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