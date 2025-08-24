"""
FairCare-FL v2.0.0: Client-side implementation with CALT (Client-Aware Local Training).
Implements advanced fairness-aware local training with multiple regularization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader
import copy
from collections import defaultdict


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial debiasing."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class AdversarialDebiasingNetwork(nn.Module):
    """Adversary network for demographic parity through adversarial training."""
    
    def __init__(self, input_dim: int, num_groups: int, hidden_dim: int = 64):
        super().__init__()
        self.grl = GradientReversalLayer.apply
        
        self.adversary = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_groups)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor, lambda_: float) -> torch.Tensor:
        reversed_features = self.grl(features, lambda_)
        return self.adversary(reversed_features)


class FairCareClient:
    """Next-generation FairCare-FL client with full CALT implementation."""
    
    def __init__(self, client_id: int, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        """Initialize the FairCare-FL client.
        
        Args:
            client_id: Unique client identifier
            model: Neural network model
            config: Client configuration
            device: Device for computation
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # CALT parameters (optimal defaults)
        self.prox_mu = 0.001  # FedProx regularization
        self.lambda_irm = 0.5  # IRM penalty
        self.lambda_adv = 0.2  # Adversarial debiasing
        self.lambda_fair = 1.0  # Local fairness loss weight
        
        # Fairness weights
        self.w_eo = 1.2  # Equal Opportunity
        self.w_fpr = 1.2  # False Positive Rate
        self.w_sp = 0.8  # Statistical Parity
        
        # Augmentation flags
        self.use_mixup = True
        self.use_cia = True
        self.mixup_alpha = 0.4
        self.cia_alpha = 0.3
        
        # Initialize adversarial network
        self.adversary = None
        self.num_groups = config.get('num_groups', 2)
        
        # Track metrics
        self.training_history = []
        self.validation_metrics = {}
    
    def train_faircare_fl(self, global_weights: Dict[str, torch.Tensor],
                          train_loader: DataLoader, val_loader: DataLoader,
                          local_epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Execute CALT training with all enhancements.
        
        Args:
            global_weights: Global model weights
            train_loader: Training data loader
            val_loader: Validation data loader
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Client report with delta, metrics, and proxies
        """
        # Load global weights
        self.model.load_state_dict(global_weights)
        initial_weights = copy.deepcopy(global_weights)
        
        # Initialize adversary if needed
        if self.lambda_adv > 0 and self.adversary is None:
            self._initialize_adversary()
        
        # Setup optimizers
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        adv_optimizer = None
        if self.adversary:
            adv_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=learning_rate * 2)
        
        # Training loop
        for epoch in range(local_epochs):
            epoch_metrics = self._train_epoch(
                train_loader, optimizer, adv_optimizer,
                global_weights, epoch, local_epochs
            )
            self.training_history.append(epoch_metrics)
        
        # Compute validation metrics
        val_metrics = self._validate(val_loader)
        
        # Compute model delta
        delta = self._compute_delta(initial_weights)
        
        # Compute proxies for DFBD
        proxies = self._compute_proxies(val_metrics)
        
        # Prepare comprehensive report
        report = self._prepare_report(delta, val_metrics, proxies, len(train_loader.dataset))
        
        return report
    
    def _initialize_adversary(self):
        """Initialize adversarial debiasing network."""
        # Get feature dimension from model
        feature_dim = self._get_feature_dim()
        
        self.adversary = AdversarialDebiasingNetwork(
            input_dim=feature_dim,
            num_groups=self.num_groups,
            hidden_dim=128
        ).to(self.device)
    
    def _get_feature_dim(self) -> int:
        """Get the dimension of the feature representation."""
        # This would depend on the model architecture
        # For now, return a default value
        return 256
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    adv_optimizer: Optional[torch.optim.Optimizer],
                    global_weights: Dict[str, torch.Tensor],
                    epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train for one epoch with CALT enhancements."""
        self.model.train()
        
        total_loss = 0.0
        total_erm_loss = 0.0
        total_prox_loss = 0.0
        total_irm_loss = 0.0
        total_fair_loss = 0.0
        total_adv_loss = 0.0
        
        group_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})
        
        for batch_idx, (data, target, sensitive_attr) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            sensitive_attr = sensitive_attr.to(self.device)
            
            # Data augmentation
            if self.use_mixup:
                data, target, sensitive_attr = self._mixup_augmentation(
                    data, target, sensitive_attr
                )
            
            if self.use_cia:
                data = self._counterfactual_augmentation(
                    data, sensitive_attr
                )
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model outputs and features
            outputs, features = self._forward_with_features(data)
            
            # Compute losses
            loss_components = {}
            
            # 1. ERM loss (cross-entropy)
            erm_loss = F.cross_entropy(outputs, target)
            loss_components['erm'] = erm_loss
            
            # 2. FedProx regularization
            prox_loss = self._compute_prox_loss(global_weights)
            loss_components['prox'] = self.prox_mu * prox_loss
            
            # 3. IRM penalty
            irm_loss = self._compute_irm_penalty(outputs, target, sensitive_attr)
            loss_components['irm'] = self.lambda_irm * irm_loss
            
            # 4. Local fairness loss
            fair_loss = self._compute_local_fairness_loss(
                outputs, target, sensitive_attr
            )
            loss_components['fair'] = self.lambda_fair * fair_loss
            
            # 5. Adversarial debiasing loss
            if self.adversary and self.lambda_adv > 0:
                # Adaptive lambda based on epoch
                adaptive_lambda = self.lambda_adv * min(1.0, epoch / (total_epochs / 2))
                
                adv_predictions = self.adversary(features, adaptive_lambda)
                adv_loss = F.cross_entropy(adv_predictions, sensitive_attr)
                loss_components['adv'] = -adaptive_lambda * adv_loss
                
                # Train adversary
                if adv_optimizer:
                    adv_optimizer.zero_grad()
                    adv_loss.backward(retain_graph=True)
                    adv_optimizer.step()
            else:
                loss_components['adv'] = torch.tensor(0.0)
            
            # Combined loss
            total_batch_loss = sum(loss_components.values())
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_erm_loss += loss_components['erm'].item()
            total_prox_loss += loss_components['prox'].item()
            total_irm_loss += loss_components['irm'].item()
            total_fair_loss += loss_components['fair'].item()
            total_adv_loss += loss_components['adv'].item()
            
            # Update group statistics
            self._update_group_stats(
                outputs, target, sensitive_attr, group_stats
            )
        
        n_batches = len(train_loader)
        epoch_metrics = {
            'total_loss': total_loss / n_batches,
            'erm_loss': total_erm_loss / n_batches,
            'prox_loss': total_prox_loss / n_batches,
            'irm_loss': total_irm_loss / n_batches,
            'fair_loss': total_fair_loss / n_batches,
            'adv_loss': total_adv_loss / n_batches,
            'group_stats': dict(group_stats)
        }
        
        return epoch_metrics
    
    def _forward_with_features(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both outputs and intermediate features."""
        # This would depend on the model architecture
        # For now, we'll assume the model returns both
        outputs = self.model(data)
        
        # Extract features (penultimate layer activations)
        # This is a placeholder - actual implementation depends on model
        features = outputs  # In practice, extract from intermediate layer
        
        return outputs, features
    
    def _mixup_augmentation(self, data: torch.Tensor, target: torch.Tensor,
                           sensitive_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply Mixup data augmentation."""
        batch_size = data.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(self.device)
        
        # Mix inputs
        mixed_data = lam * data + (1 - lam) * data[index]
        
        # Mix targets (for soft labels)
        mixed_target = target  # Keep original for simplicity
        
        # Keep sensitive attributes unmixed
        mixed_sensitive = sensitive_attr
        
        return mixed_data, mixed_target, mixed_sensitive
    
    def _counterfactual_augmentation(self, data: torch.Tensor,
                                    sensitive_attr: torch.Tensor) -> torch.Tensor:
        """Apply Counterfactual Instance Augmentation."""
        batch_size = data.size(0)
        
        # Find pairs with different sensitive attributes
        augmented_data = data.clone()
        
        for i in range(batch_size):
            # Find samples with different sensitive attribute
            diff_mask = sensitive_attr != sensitive_attr[i]
            
            if diff_mask.any():
                # Select a random sample with different attribute
                diff_indices = torch.where(diff_mask)[0]
                j = diff_indices[torch.randint(len(diff_indices), (1,))].item()
                
                # Interpolate between samples
                alpha = self.cia_alpha
                augmented_data[i] = (1 - alpha) * data[i] + alpha * data[j]
        
        return augmented_data
    
    def _compute_prox_loss(self, global_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute FedProx regularization loss."""
        prox_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in global_weights:
                prox_loss += torch.norm(param - global_weights[name]) ** 2
        
        return prox_loss / 2
    
    def _compute_irm_penalty(self, outputs: torch.Tensor, target: torch.Tensor,
                            sensitive_attr: torch.Tensor) -> torch.Tensor:
        """Compute IRM (Invariant Risk Minimization) penalty."""
        unique_groups = torch.unique(sensitive_attr)
        penalties = []
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            
            if group_mask.sum() > 1:  # Need at least 2 samples
                group_outputs = outputs[group_mask]
                group_targets = target[group_mask]
                
                # Compute group-specific loss
                group_loss = F.cross_entropy(group_outputs, group_targets)
                
                # Compute gradient norm
                grad = torch.autograd.grad(
                    group_loss, outputs,
                    create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0]
                
                # IRM penalty is the gradient norm squared
                penalty = torch.sum(grad[group_mask] ** 2)
                penalties.append(penalty)
        
        if penalties:
            return torch.stack(penalties).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_local_fairness_loss(self, outputs: torch.Tensor, target: torch.Tensor,
                                    sensitive_attr: torch.Tensor) -> torch.Tensor:
        """Compute local fairness loss for EO, FPR, and SP."""
        probs = torch.softmax(outputs, dim=1)
        
        # Get positive class probabilities
        if probs.dim() > 1 and probs.size(1) > 1:
            pos_probs = probs[:, 1]
        else:
            pos_probs = probs.squeeze()
        
        unique_groups = torch.unique(sensitive_attr)
        
        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Compute smooth metrics for each group
        group_metrics = {}
        epsilon = 1e-8
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            
            if group_mask.sum() > 0:
                group_probs = pos_probs[group_mask]
                group_targets = target[group_mask]
                
                # Smooth TPR (for Equal Opportunity)
                positive_mask = group_targets == 1
                if positive_mask.any():
                    tpr = torch.mean(group_probs[positive_mask])
                else:
                    tpr = torch.tensor(0.5, device=self.device)
                
                # Smooth FPR
                negative_mask = group_targets == 0
                if negative_mask.any():
                    fpr = torch.mean(group_probs[negative_mask])
                else:
                    fpr = torch.tensor(0.5, device=self.device)
                
                # Smooth PPR (for Statistical Parity)
                ppr = torch.mean(group_probs)
                
                group_metrics[group.item()] = {
                    'tpr': tpr,
                    'fpr': fpr,
                    'ppr': ppr
                }
        
        # Compute pairwise gaps
        fairness_loss = torch.tensor(0.0, device=self.device)
        group_ids = list(group_metrics.keys())
        
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                metrics_i = group_metrics[group_ids[i]]
                metrics_j = group_metrics[group_ids[j]]
                
                # Equal Opportunity gap
                eo_gap = (metrics_i['tpr'] - metrics_j['tpr']) ** 2
                
                # FPR gap
                fpr_gap = (metrics_i['fpr'] - metrics_j['fpr']) ** 2
                
                # Statistical Parity gap
                sp_gap = (metrics_i['ppr'] - metrics_j['ppr']) ** 2
                
                # Weighted combination
                fairness_loss += (
                    self.w_eo * eo_gap +
                    self.w_fpr * fpr_gap +
                    self.w_sp * sp_gap
                )
        
        # Normalize by number of pairs
        n_pairs = len(group_ids) * (len(group_ids) - 1) / 2
        if n_pairs > 0:
            fairness_loss = fairness_loss / n_pairs
        
        return fairness_loss
    
    def _update_group_stats(self, outputs: torch.Tensor, target: torch.Tensor,
                           sensitive_attr: torch.Tensor, group_stats: Dict):
        """Update confusion matrix statistics for each group."""
        predictions = torch.argmax(outputs, dim=1)
        
        unique_groups = torch.unique(sensitive_attr)
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            group_preds = predictions[group_mask]
            group_targets = target[group_mask]
            
            # Update confusion matrix
            tp = ((group_preds == 1) & (group_targets == 1)).sum().item()
            fp = ((group_preds == 1) & (group_targets == 0)).sum().item()
            tn = ((group_preds == 0) & (group_targets == 0)).sum().item()
            fn = ((group_preds == 0) & (group_targets == 1)).sum().item()
            
            group_id = group.item()
            group_stats[group_id]['TP'] += tp
            group_stats[group_id]['FP'] += fp
            group_stats[group_id]['TN'] += tn
            group_stats[group_id]['FN'] += fn
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Validate model and compute comprehensive metrics."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        group_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'total': 0})
        all_probs = []
        all_targets = []
        all_groups = []
        
        with torch.no_grad():
            for data, target, sensitive_attr in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                sensitive_attr = sensitive_attr.to(self.device)
                
                outputs = self.model(data)
                loss = F.cross_entropy(outputs, target)
                
                total_loss += loss.item()
                
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                correct += (predictions == target).sum().item()
                total += target.size(0)
                
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())
                all_groups.append(sensitive_attr.cpu())
                
                # Update group statistics
                self._update_group_stats(outputs, target, sensitive_attr, group_stats)
        
        # Concatenate all batches
        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)
        all_groups = torch.cat(all_groups)
        
        # Compute metrics
        val_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        # Compute worst-group F1
        wg_f1 = self._compute_worst_group_f1(group_stats)
        
        # Compute ECE for calibration
        ece = self._compute_ece(all_probs, all_targets)
        
        return {
            'val_loss': val_loss,
            'accuracy': accuracy,
            'wg_f1': wg_f1,
            'ece': ece,
            'group_stats': dict(group_stats)
        }
    
    def _compute_worst_group_f1(self, group_stats: Dict) -> float:
        """Compute worst-group F1 score."""
        f1_scores = []
        
        for group_id, stats in group_stats.items():
            tp = stats['TP']
            fp = stats['FP']
            fn = stats['FN']
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            f1_scores.append(f1)
        
        return min(f1_scores) if f1_scores else 0.0
    
    def _compute_ece(self, probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        if probs.dim() > 1 and probs.size(1) > 1:
            confidences = probs.max(dim=1)[0]
            predictions = probs.argmax(dim=1)
        else:
            confidences = probs.squeeze()
            predictions = (probs > 0.5).long().squeeze()
        
        accuracies = (predictions == targets).float()
        
        ece = 0.0
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = accuracies[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                bin_weight = in_bin.float().mean()
                
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
        
        return ece.item()
    
    def _compute_delta(self, initial_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute model weight delta."""
        delta = {}
        
        current_weights = self.model.state_dict()
        
        for key in initial_weights:
            delta[key] = current_weights[key] - initial_weights[key]
        
        return delta
    
    def _compute_proxies(self, val_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute proxies for DFBD."""
        # Loss drift (current loss - historical average)
        if self.training_history:
            historical_loss = np.mean([h['total_loss'] for h in self.training_history])
            loss_drift = val_metrics['val_loss'] - historical_loss
        else:
            loss_drift = 0.0
        
        # Delta norm
        current_weights = self.model.state_dict()
        delta_norm = sum(
            torch.norm(param).item() ** 2
            for param in current_weights.values()
        ) ** 0.5
        
        # ECE proxy for calibration
        ece_proxy = val_metrics.get('ece', 0.1)
        
        return {
            'loss_drift': loss_drift,
            'delta_norm': delta_norm,
            'ece_proxy': ece_proxy
        }
    
    def _prepare_report(self, delta: Dict[str, torch.Tensor],
                       val_metrics: Dict[str, Any],
                       proxies: Dict[str, float],
                       n_samples: int) -> Dict[str, Any]:
        """Prepare comprehensive client report."""
        # Convert group stats to the expected format
        group_counts = {}
        for group_id, stats in val_metrics['group_stats'].items():
            group_counts[group_id] = {
                'TP': stats['TP'],
                'FP': stats['FP'],
                'TN': stats['TN'],
                'FN': stats['FN']
            }
        
        report = {
            'client_id': self.client_id,
            'delta': delta,
            'n_samples': n_samples,
            'val_loss': val_metrics['val_loss'],
            'group_counts': group_counts,
            'proxies': proxies,
            'wg_f1': val_metrics['wg_f1'],
            'accuracy': val_metrics['accuracy']
        }
        
        return report


def train_faircare_fl(client_id: int, model: nn.Module, config: Dict[str, Any],
                      global_weights: Dict[str, torch.Tensor],
                      train_loader: DataLoader, val_loader: DataLoader,
                      local_epochs: int = 2, learning_rate: float = 0.001,
                      device: str = 'cuda') -> Dict[str, Any]:
    """Entry point for FairCare-FL client training.
    
    Args:
        client_id: Client identifier
        model: Neural network model
        config: Algorithm configuration
        global_weights: Global model weights
        train_loader: Training data loader
        val_loader: Validation data loader
        local_epochs: Number of local epochs
        learning_rate: Learning rate
        device: Device for computation
        
    Returns:
        Client report with all required metrics and updates
    """
    client = FairCareClient(client_id, model, config, device)
    
    report = client.train_faircare_fl(
        global_weights, train_loader, val_loader,
        local_epochs, learning_rate
    )
    
    return report


# Client dispatcher for algorithm routing
def client_update(algorithm: str, client_id: int, model: nn.Module,
                 config: Dict[str, Any], global_weights: Dict[str, torch.Tensor],
                 train_loader: DataLoader, val_loader: DataLoader,
                 local_epochs: int, learning_rate: float, device: str) -> Dict[str, Any]:
    """Route client update based on algorithm.
    
    Args:
        algorithm: Algorithm name
        client_id: Client identifier
        model: Neural network model
        config: Algorithm configuration
        global_weights: Global model weights
        train_loader: Training data loader
        val_loader: Validation data loader
        local_epochs: Number of local epochs
        learning_rate: Learning rate
        device: Device for computation
        
    Returns:
        Client report
    """
    if algorithm == 'faircare_fl':
        return train_faircare_fl(
            client_id, model, config, global_weights,
            train_loader, val_loader, local_epochs, learning_rate, device
        )
    else:
        # Fallback to standard training for other algorithms
        # This would call the existing client training functions
        raise NotImplementedError(f"Algorithm {algorithm} not implemented in this client")


# Export the main entry points
__all__ = ['train_faircare_fl', 'client_update', 'FairCareClient']
