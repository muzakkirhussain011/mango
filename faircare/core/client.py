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

    def __init__(self, client_id: int, model: nn.Module,
                 train_dataset: Any = None, val_dataset: Any = None,
                 batch_size: int = 32, config: Optional[Dict[str, Any]] = None,
                 device: str = 'cuda'):
        """Initialize the FairCare-FL client.

        Args:
            client_id: Unique client identifier
            model: Neural network model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size for data loaders
            config: Client configuration
            device: Device for computation
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.config = config if config is not None else {}
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
        self.num_groups = self.config.get('num_groups', 2)
        
        # Track metrics
        self.training_history = []
        self.validation_metrics = {}

    def train(self, global_weights: Dict[str, torch.Tensor],
              epochs: int, lr: float,
              weight_decay: float = 0.0, proximal_mu: float = 0.0,
              server_val_data: Optional[Any] = None,
              fairness_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
        """Train client model with federated learning.

        Args:
            global_weights: Global model weights
            epochs: Number of local training epochs
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            proximal_mu: FedProx proximal term coefficient
            server_val_data: Server validation data (optional)
            fairness_config: Fairness configuration from server

        Returns:
            Tuple of (delta, n_samples, stats)
        """
        # Override proximal mu if provided
        if proximal_mu > 0:
            self.prox_mu = proximal_mu

        # Apply fairness config if provided
        if fairness_config:
            self.lambda_fair = fairness_config.get('lambda_fair', self.lambda_fair)
            self.lambda_adv = fairness_config.get('lambda_adv', self.lambda_adv)
            if fairness_config.get('use_adversary', False):
                self.lambda_adv = max(self.lambda_adv, 0.2)

        # Create data loaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

        if self.val_dataset is not None:
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            # Use a portion of training data for validation
            val_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

        # Call the main training method
        report = self.train_faircare_fl(
            global_weights=global_weights,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=epochs,
            learning_rate=lr
        )

        # Extract delta, n_samples, and stats for backward compatibility
        delta = report['delta']
        n_samples = report['n_samples']

        stats = {
            'train_loss': report.get('val_loss', 0.0),  # Use val_loss as proxy for train_loss
            'val_loss': report.get('val_loss', 0.0),
            'accuracy': report.get('accuracy', 0.0),
            'wg_f1': report.get('wg_f1', 0.0),
            'worst_group_F1': report.get('wg_f1', 0.0),
            'fairness_loss': report.get('fairness_loss', 0.0),
            'adversary_loss': report.get('adversary_loss', 0.0),
        }

        # Add fairness metrics if group_counts are available
        if 'group_counts' in report and len(report['group_counts']) >= 2:
            group_ids = sorted(report['group_counts'].keys())
            if len(group_ids) >= 2:
                # Compute fairness gaps
                tpr_list = []
                fpr_list = []
                ppr_list = []

                for gid in group_ids:
                    gc = report['group_counts'][gid]
                    tp, fp, tn, fn = gc['TP'], gc['FP'], gc['TN'], gc['FN']

                    tpr = tp / (tp + fn + 1e-8)
                    fpr = fp / (fp + tn + 1e-8)
                    ppr = (tp + fp) / (tp + fp + tn + fn + 1e-8)

                    tpr_list.append(tpr)
                    fpr_list.append(fpr)
                    ppr_list.append(ppr)

                stats['eo_gap'] = max(tpr_list) - min(tpr_list)
                stats['fpr_gap'] = max(fpr_list) - min(fpr_list)
                stats['sp_gap'] = max(ppr_list) - min(ppr_list)

        return delta, n_samples, stats

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
        last_epoch_metrics = {}
        for epoch in range(local_epochs):
            epoch_metrics = self._train_epoch(
                train_loader, optimizer, adv_optimizer,
                global_weights, epoch, local_epochs
            )
            self.training_history.append(epoch_metrics)
            last_epoch_metrics = epoch_metrics

        # Compute validation metrics
        val_metrics = self._validate(val_loader)

        # Compute model delta
        delta = self._compute_delta(initial_weights)

        # Compute proxies for DFBD
        proxies = self._compute_proxies(val_metrics)

        # Prepare comprehensive report
        report = self._prepare_report(delta, val_metrics, proxies, len(train_loader.dataset), last_epoch_metrics)

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
        # Get the actual output dimension from the model by checking the last layer
        try:
            # Find the last linear layer output dimension
            last_layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    last_layer = module

            if last_layer is not None:
                feature_dim = last_layer.out_features
            else:
                # Fallback: infer from a dummy forward pass
                first_param = next(self.model.parameters())
                input_dim = first_param.shape[1] if len(first_param.shape) > 1 else 64
                dummy_input = torch.randn(2, input_dim, device=self.device)
                with torch.no_grad():
                    dummy_output = self.model(dummy_input)
                feature_dim = dummy_output.shape[-1] if dummy_output.dim() > 1 else 1
        except:
            # Fallback to reasonable default
            feature_dim = 1

        return max(feature_dim, 1)  # At least 1 for the adversary
    
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

            # 1. ERM loss (cross-entropy or BCE depending on output shape)
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                # Binary classification with single output
                outputs_flat = outputs.squeeze() if outputs.dim() == 2 else outputs
                erm_loss = F.binary_cross_entropy_with_logits(outputs_flat, target.float())
            else:
                # Multi-class classification
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
            adv_loss_value = 0.0
            if self.adversary and self.lambda_adv > 0:
                # Adaptive lambda based on epoch
                adaptive_lambda = self.lambda_adv * min(1.0, epoch / (total_epochs / 2))

                adv_predictions = self.adversary(features.detach(), adaptive_lambda)
                adv_loss = F.cross_entropy(adv_predictions, sensitive_attr)

                # Train adversary separately (don't include in main backward pass)
                if adv_optimizer:
                    adv_optimizer.zero_grad()
                    adv_loss.backward()
                    adv_optimizer.step()

                adv_loss_value = adv_loss.item()
                # For the main model, we want to fool the adversary
                # Re-compute with non-detached features for gradient flow to main model
                adv_predictions_main = self.adversary(features, adaptive_lambda)
                loss_components['adv'] = -adaptive_lambda * F.cross_entropy(adv_predictions_main, sensitive_attr)
            else:
                loss_components['adv'] = torch.tensor(0.0, device=self.device)

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
        # Get model outputs
        outputs = self.model(data)

        # For the adversary, we need features that match its expected input dim
        # Flatten outputs to create feature vector
        if outputs.dim() == 1:
            features = outputs.unsqueeze(1)  # Make it 2D: [batch, 1]
        elif outputs.dim() == 2 and outputs.size(1) == 1:
            features = outputs  # Already [batch, 1]
        else:
            # Multi-class: use logits as features
            features = outputs

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

                # Compute group-specific loss based on output shape
                if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                    # Binary classification
                    group_outputs_flat = group_outputs.squeeze() if group_outputs.dim() == 2 else group_outputs
                    group_loss = F.binary_cross_entropy_with_logits(group_outputs_flat, group_targets.float())
                else:
                    # Multi-class classification
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
        # Handle both single-output (binary) and multi-class outputs
        if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
            outputs_flat = outputs.squeeze() if outputs.dim() == 2 else outputs
            predictions = (torch.sigmoid(outputs_flat) > 0.5).long()
        else:
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

                # Compute loss based on output shape
                if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                    outputs_flat = outputs.squeeze() if outputs.dim() == 2 else outputs
                    loss = F.binary_cross_entropy_with_logits(outputs_flat, target.float())
                    probs_flat = torch.sigmoid(outputs_flat)
                    # Convert to 2D probs for consistency
                    probs = torch.stack([1 - probs_flat, probs_flat], dim=1)
                    predictions = (probs_flat > 0.5).long()
                else:
                    loss = F.cross_entropy(outputs, target)
                    probs = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)

                total_loss += loss.item()
                
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
                       n_samples: int,
                       last_epoch_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
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

        # Add training metrics from last epoch if available
        if last_epoch_metrics:
            report['fairness_loss'] = last_epoch_metrics.get('fair_loss', 0.0)
            report['adversary_loss'] = last_epoch_metrics.get('adv_loss', 0.0)

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
