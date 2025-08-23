# faircare/core/client.py
"""Federated learning client with CALT (Causal-Aware Local Training) enhancements."""
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
    """Federated learning client with fairness-aware training and CALT hooks."""
    
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
        
        # Initialize adversary for domain invariance (created on demand)
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
        Train local model with CALT enhancements.
        
        CALT includes:
        - FedProx proximal term
        - IRM penalty for invariance
        - Domain adversarial invariance
        - Fairness-aware augmentation (mixup/CIA)
        
        Returns:
            - Model delta (update)
            - Number of samples
            - Training statistics including fairness metrics
        """
        # Extract fairness configuration
        if fairness_config is None:
            fairness_config = {}
        
        # Check if CALT is enabled (only for faircare_fl v2.0)
        calt_enabled = fairness_config.get('version', '1.0') == '2.0.0'
        
        # CALT parameters
        prox_mu = fairness_config.get('prox_mu', proximal_mu)
        lambda_irm = fairness_config.get('lambda_irm', 0.0) if calt_enabled else 0.0
        lambda_adv = fairness_config.get('lambda_adv', 0.0) if calt_enabled else 0.0
        lambda_fair = fairness_config.get('lambda_fair', 0.0)
        use_mixup = fairness_config.get('use_mixup', False) and calt_enabled
        use_cia = fairness_config.get('use_cia', False) and calt_enabled
        
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
        
        # Setup adversary if needed for domain invariance
        if lambda_adv > 0 and self.adversary is None and calt_enabled:
            self.adversary = self._create_adversary()
            self.adversary_optimizer = torch.optim.Adam(
                self.adversary.parameters(),
                lr=lr * 2
            )
        
        # Training loop
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        total_irm_loss = 0.0
        total_adv_loss = 0.0
        total_fairness_loss = 0.0
        n_samples = 0
        n_batches = 0
        
        # Track confusion counts for fairness metrics
        g0_tp, g0_fp, g0_fn, g0_tn = 0, 0, 0, 0
        g1_tp, g1_fp, g1_fn, g1_tn = 0, 0, 0, 0
        
        for epoch in range(epochs):
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
                
                # Data augmentation if enabled
                if use_mixup and np.random.random() < 0.5:
                    X, y = self._mixup(X, y)
                
                if use_cia and a is not None and np.random.random() < 0.3:
                    X = self._counterfactual_interpolation(X, a)
                
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
                
                # IRM penalty if enabled
                irm_loss = torch.tensor(0.0, device=self.device)
                if lambda_irm > 0 and calt_enabled:
                    irm_loss = self._compute_irm_penalty(X, y, outputs)
                
                # Adversarial loss if enabled
                adv_loss = torch.tensor(0.0, device=self.device)
                if lambda_adv > 0 and a is not None and self.adversary is not None:
                    adv_loss = self._compute_adversarial_loss(outputs, a)
                
                # Basic fairness loss (for compatibility)
                fairness_loss = torch.tensor(0.0, device=self.device)
                if lambda_fair > 0 and a is not None:
                    fairness_loss = self._compute_simple_fairness_loss(outputs, y, a)
                
                # Total loss
                total_loss_batch = pred_loss
                
                if lambda_irm > 0:
                    total_loss_batch = total_loss_batch + lambda_irm * irm_loss
                
                if lambda_adv > 0:
                    total_loss_batch = total_loss_batch - lambda_adv * adv_loss
                
                if lambda_fair > 0:
                    total_loss_batch = total_loss_batch + lambda_fair * fairness_loss
                
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
                optimizer.step()
                
                # Update adversary if used
                if self.adversary is not None and lambda_adv > 0:
                    self._update_adversary(outputs.detach(), a)
                
                # Track losses
                total_loss += pred_loss.item() * X.size(0)
                total_irm_loss += irm_loss.item() * X.size(0) if lambda_irm > 0 else 0
                total_adv_loss += adv_loss.item() * X.size(0) if lambda_adv > 0 else 0
                total_fairness_loss += fairness_loss.item() * X.size(0) if lambda_fair > 0 else 0
                
                # Track confusion counts
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    if a is not None:
                        for i in range(len(preds)):
                            if a[i] == 0:  # Group 0
                                if y[i] == 1 and preds[i] == 1:
                                    g0_tp += 1
                                elif y[i] == 0 and preds[i] == 1:
                                    g0_fp += 1
                                elif y[i] == 1 and preds[i] == 0:
                                    g0_fn += 1
                                else:
                                    g0_tn += 1
                            else:  # Group 1
                                if y[i] == 1 and preds[i] == 1:
                                    g1_tp += 1
                                elif y[i] == 0 and preds[i] == 1:
                                    g1_fp += 1
                                elif y[i] == 1 and preds[i] == 0:
                                    g1_fn += 1
                                else:
                                    g1_tn += 1
                
                n_samples += X.size(0)
                n_batches += 1
        
        # Compute average losses
        avg_loss = total_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_irm_loss = total_irm_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_adv_loss = total_adv_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        avg_fairness_loss = total_fairness_loss / (n_samples * epochs) if n_samples > 0 else 0.0
        
        # Compute model delta
        delta = compute_model_delta(
            self.model.state_dict(),
            global_weights
        )
        
        # Compute delta norm and gradient norm
        delta_norm = 0.0
        grad_norm = 0.0
        for key in delta:
            delta_norm += torch.norm(delta[key]) ** 2
        delta_norm = (delta_norm ** 0.5).item()
        
        # Compute calibration error proxy
        calibration_error = self._compute_calibration_error()
        
        # Compute drift
        drift = delta_norm / (n_samples + 1)
        
        # Build stats
        stats = {
            "train_loss": avg_loss,
            "irm_loss": avg_irm_loss,
            "adv_loss": avg_adv_loss,
            "adversary_loss": avg_adv_loss,  # Alias for test compatibility
            "fairness_loss": avg_fairness_loss,  # For test compatibility
            "n_samples": len(self.train_dataset),
            "client_id": self.client_id,
            "delta_norm": delta_norm,
            "grad_norm": grad_norm,
            "calibration_error": calibration_error,
            "drift": drift,
            "calt_enabled": calt_enabled,
            # Group confusion counts
            "g0_tp": g0_tp,
            "g0_fp": g0_fp,
            "g0_fn": g0_fn,
            "g0_tn": g0_tn,
            "g1_tp": g1_tp,
            "g1_fp": g1_fp,
            "g1_fn": g1_fn,
            "g1_tn": g1_tn
        }
        
        # Compute fairness metrics on server validation if provided
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
                "val_acc": fair_report.get("accuracy", 0.0),
                "eo_gap": fair_report.get("EO_gap", 0.0),
                "fpr_gap": fair_report.get("FPR_gap", 0.0),
                "sp_gap": fair_report.get("SP_gap", 0.0),
                "worst_group_f1": fair_report.get("worst_group_F1", 0.0)
            })
        
        return delta, len(self.train_dataset), stats
    
    def _create_adversary(self) -> nn.Module:
        """Create adversary network for domain invariance."""
        # Get model output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, next(self.model.parameters()).shape[1])
            dummy_output = self.model(dummy_input.to(self.device))
            output_dim = dummy_output.shape[-1] if dummy_output.dim() > 1 else 1
        
        adversary = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary sensitive attribute
        )
        
        return adversary.to(self.device)
    
    def _compute_irm_penalty(self, X: torch.Tensor, y: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute IRM (Invariant Risk Minimization) penalty.
        """
        # Create pseudo-environments by splitting batch
        n = len(X)
        if n < 4:
            return torch.tensor(0.0, device=self.device)
        
        # Split into two environments
        env1_idx = torch.randperm(n)[:n//2]
        env2_idx = torch.randperm(n)[n//2:]
        
        # Compute gradients per environment
        scale1 = torch.tensor(1.0, requires_grad=True, device=self.device)
        loss1 = F.binary_cross_entropy_with_logits(outputs[env1_idx] * scale1, y[env1_idx])
        grad1 = torch.autograd.grad(loss1, [scale1], create_graph=True)[0]
        
        scale2 = torch.tensor(1.0, requires_grad=True, device=self.device)
        loss2 = F.binary_cross_entropy_with_logits(outputs[env2_idx] * scale2, y[env2_idx])
        grad2 = torch.autograd.grad(loss2, [scale2], create_graph=True)[0]
        
        # IRM penalty: variance of gradients across environments
        penalty = torch.var(torch.stack([grad1, grad2]))
        
        return penalty
    
    def _compute_adversarial_loss(self, outputs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial loss for domain invariance.
        """
        if self.adversary is None:
            return torch.tensor(0.0, device=self.device)
        
        # Gradient reversal: fool adversary
        adv_input = outputs.unsqueeze(1) if outputs.dim() == 1 else outputs
        adv_pred = self.adversary(adv_input.detach())
        
        # Cross-entropy loss for adversary
        adv_loss = F.cross_entropy(adv_pred, a.long())
        
        return adv_loss
    
    def _update_adversary(self, outputs: torch.Tensor, a: torch.Tensor):
        """
        Update adversary to predict sensitive attribute.
        """
        if self.adversary is None or self.adversary_optimizer is None:
            return
        
        self.adversary_optimizer.zero_grad()
        
        adv_input = outputs.unsqueeze(1) if outputs.dim() == 1 else outputs
        adv_pred = self.adversary(adv_input)
        
        loss = F.cross_entropy(adv_pred, a.long())
        loss.backward()
        
        self.adversary_optimizer.step()
    
    def _compute_simple_fairness_loss(self, outputs: torch.Tensor, y: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Simple fairness loss for compatibility.
        """
        probs = torch.sigmoid(outputs)
        
        # Group-wise statistics
        g0_mask = (a == 0)
        g1_mask = (a == 1)
        
        if g0_mask.sum() > 0 and g1_mask.sum() > 0:
            # Demographic parity loss
            g0_pos_rate = probs[g0_mask].mean()
            g1_pos_rate = probs[g1_mask].mean()
            dp_loss = (g0_pos_rate - g1_pos_rate) ** 2
            
            return dp_loss
        
        return torch.tensor(0.0, device=self.device)
    
    # faircare/core/client.py (key excerpt - fairness loss implementation)
    # Add this method to the Client class to compute proper fairness loss

    def _compute_fairness_loss(
        self,
        outputs: torch.Tensor,
        y: torch.Tensor,
        a: Optional[torch.Tensor],
        config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute fairness loss for local training.
        
        This is a differentiable approximation of fairness gaps.
        """
        if a is None:
            return torch.tensor(0.0, device=self.device)
        
        # Get weights from config
        w_eo = config.get('w_eo', 1.0)
        w_fpr = config.get('w_fpr', 0.5)
        w_sp = config.get('w_sp', 0.5)
        
        # Convert to probabilities
        probs = torch.sigmoid(outputs.squeeze())
        
        # Separate groups
        g0_mask = (a == 0)
        g1_mask = (a == 1)
        
        # Need at least 2 samples per group
        if g0_mask.sum() < 2 or g1_mask.sum() < 2:
            return torch.tensor(0.0, device=self.device)
        
        eps = 1e-7
        
        # Equal Opportunity loss: |TPR_0 - TPR_1|
        # Soft TPR for group 0
        g0_pos_mask = g0_mask & (y == 1)
        if g0_pos_mask.sum() > 0:
            tpr_0 = probs[g0_pos_mask].mean()
        else:
            tpr_0 = torch.tensor(0.5, device=self.device)
        
        # Soft TPR for group 1
        g1_pos_mask = g1_mask & (y == 1)
        if g1_pos_mask.sum() > 0:
            tpr_1 = probs[g1_pos_mask].mean()
        else:
            tpr_1 = torch.tensor(0.5, device=self.device)
        
        eo_loss = (tpr_0 - tpr_1) ** 2
        
        # False Positive Rate loss: |FPR_0 - FPR_1|
        # Soft FPR for group 0
        g0_neg_mask = g0_mask & (y == 0)
        if g0_neg_mask.sum() > 0:
            fpr_0 = probs[g0_neg_mask].mean()
        else:
            fpr_0 = torch.tensor(0.5, device=self.device)
        
        # Soft FPR for group 1
        g1_neg_mask = g1_mask & (y == 0)
        if g1_neg_mask.sum() > 0:
            fpr_1 = probs[g1_neg_mask].mean()
        else:
            fpr_1 = torch.tensor(0.5, device=self.device)
        
        fpr_loss = (fpr_0 - fpr_1) ** 2
        
        # Statistical Parity loss: |PPR_0 - PPR_1|
        ppr_0 = probs[g0_mask].mean()
        ppr_1 = probs[g1_mask].mean()
        sp_loss = (ppr_0 - ppr_1) ** 2
        
        # Combine losses
        total_fairness_loss = w_eo * eo_loss + w_fpr * fpr_loss + w_sp * sp_loss
        
        # Scale by lambda_fair from config
        lambda_fair = config.get('lambda_fair', 0.1)
        
        # Reduce strength to prevent collapse
        return lambda_fair * 0.1 * total_fairness_loss  # Scale down by 0.1


    def _mixup(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup augmentation.
        """
        batch_size = X.size(0)
        if batch_size < 2:
            return X, y
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(self.device)
        
        # Mix inputs and targets
        mixed_X = lam * X + (1 - lam) * X[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_X, mixed_y
    
    def _counterfactual_interpolation(self, X: torch.Tensor, a: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
        """
        Counterfactual interpolation augmentation (CIA).
        """
        g0_mask = (a == 0)
        g1_mask = (a == 1)
        
        if g0_mask.sum() > 0 and g1_mask.sum() > 0:
            # Sample from opposite groups
            g0_samples = X[g0_mask]
            g1_samples = X[g1_mask]
            
            # Interpolate
            lam = np.random.beta(alpha, alpha)
            
            # Create counterfactuals
            X_cf = X.clone()
            for i in range(len(X)):
                if a[i] == 0 and len(g1_samples) > 0:
                    # Interpolate with random sample from group 1
                    idx = np.random.randint(0, len(g1_samples))
                    X_cf[i] = lam * X[i] + (1 - lam) * g1_samples[idx]
                elif a[i] == 1 and len(g0_samples) > 0:
                    # Interpolate with random sample from group 0
                    idx = np.random.randint(0, len(g0_samples))
                    X_cf[i] = lam * X[i] + (1 - lam) * g0_samples[idx]
            
            return X_cf
        
        return X
    
    def _compute_calibration_error(self) -> float:
        """
        Compute calibration error proxy.
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    X, y, _ = batch
                else:
                    X, y = batch
                
                X = X.to(self.device)
                outputs = self.model(X)
                
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                
                probs = torch.sigmoid(outputs).cpu()
                all_probs.extend(probs.tolist())
                all_labels.extend(y.tolist())
        
        if not all_probs:
            return 0.0
        
        # Simple calibration: mean predicted prob vs actual positive rate
        mean_prob = np.mean(all_probs)
        pos_rate = np.mean(all_labels)
        calibration_error = abs(mean_prob - pos_rate)
        
        return calibration_error
    
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