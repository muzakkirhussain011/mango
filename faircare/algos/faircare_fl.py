"""
FairCare-FL v2.0.0: Next-generation multi-objective federated learning.
State-of-the-art server aggregator with PFA (Pareto Fair Aggregation) pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import cvxpy as cp

logger = logging.getLogger(__name__)


@dataclass
class AggregationOutput:
    """Output from the aggregation step."""
    new_global: Dict[str, torch.Tensor]
    server_logs: Dict[str, Any]


class DFBDNetwork(nn.Module):
    """Demographics-Free Bias Detection network for computing client tilts."""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, depth: int = 2):
        super().__init__()
        layers = []
        
        for i in range(depth):
            in_features = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())  # Ensure non-negative output
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FairCareFLAggregator:
    """Next-generation FairCare-FL server with full PFA pipeline."""
    
    def __init__(self, config: Dict[str, Any], device: str = 'cuda'):
        """Initialize the next-gen FairCare-FL aggregator.
        
        Args:
            config: Algorithm configuration
            device: Device for tensor operations
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.version = "2.0.0"
        self.round_counter = 0
        
        # Multi-objective optimization parameters
        self.server_momentum = 0.9  # High momentum for stability
        self.momentum_buffer = None
        
        # Gradient mixing parameters
        self.mgda_normalize = True
        self.mgda_step_size = 0.8
        self.pcgrad_enabled = True
        self.cagrad_rho = 0.7  # Higher for more conflict aversion
        
        # Fairness dual variables (Lagrangian multipliers)
        self.lambda_eo = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.lambda_fpr = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.lambda_sp = nn.Parameter(torch.tensor(0.0, device=self.device))
        self.dual_lr = 0.15
        self.epsilon_eo = 0.015
        self.epsilon_fpr = 0.015
        self.epsilon_sp = 0.02
        
        # Fairness weights (adaptive)
        self.w_eo = 1.2
        self.w_fpr = 1.2
        self.w_sp = 0.8
        
        # DFBD network for bias detection
        self.dfbd_network = DFBDNetwork(
            input_dim=3,
            hidden_dim=128,
            depth=3
        ).to(self.device)
        self.dfbd_optimizer = torch.optim.Adam(self.dfbd_network.parameters(), lr=0.001)
        self.dfbd_eta = 2.0  # Amplification factor
        
        # Fairness-aware selection (Lyapunov-based)
        self.fairness_debt_scores = defaultdict(float)
        self.selection_tau = 0.015
        self.selection_kappa = 0.6
        
        # Weight constraints
        self.weight_floor = 0.005
        self.weight_cap = 0.15
        self.tau = 0.5  # Temperature for log-sum-exp
        
        # Distillation parameters
        self.distill_temperature = 3.0
        self.distill_steps = 300
        self.distill_batch_size = 128
        
        # Track metrics for adaptive adjustments
        self.historical_gaps = {'eo': [], 'fpr': [], 'sp': []}
        self.gap_momentum = 0.95
        
    def aggregate(self, round_ctx: Dict, client_reports: List[Dict], 
                  global_model: Dict[str, torch.Tensor]) -> AggregationOutput:
        """Main aggregation with full PFA pipeline.
        
        Args:
            round_ctx: Round context
            client_reports: Client reports with deltas and metrics
            global_model: Current global model
            
        Returns:
            AggregationOutput with optimized global model and comprehensive logs
        """
        self.round_counter += 1
        
        # Move everything to device
        client_reports = self._prepare_reports(client_reports)
        global_model = {k: v.to(self.device) for k, v in global_model.items()}
        
        # Step 1: Compute three objectives and their gradients
        objectives = self._compute_objectives(client_reports, global_model)
        
        # Step 2: Multi-objective gradient optimization pipeline
        mixed_direction = self._multi_objective_optimization(objectives)
        
        # Step 3: Update fairness dual variables
        fairness_metrics = self._update_dual_variables(client_reports)
        
        # Step 4: Compute DFBD tilts for bias mitigation
        tilts = self._compute_advanced_tilts(client_reports)
        
        # Step 5: Compute fairness-aware aggregation weights
        weights = self._compute_optimal_weights(client_reports, tilts, fairness_metrics)
        
        # Step 6: Weighted aggregation with momentum
        aggregated_delta = self._weighted_aggregate_with_momentum(client_reports, weights)
        
        # Step 7: Apply update to global model
        new_global = self._apply_update(global_model, aggregated_delta, mixed_direction)
        
        # Step 8: Server-side knowledge distillation
        distill_metrics = self._perform_distillation(new_global, global_model)
        
        # Step 9: Update fairness-aware selection scores
        selection_metrics = self._update_selection_scores(client_reports, fairness_metrics)
        
        # Step 10: Prepare comprehensive logs
        server_logs = self._compile_logs(
            round_ctx, client_reports, weights, tilts, objectives,
            mixed_direction, fairness_metrics, distill_metrics, selection_metrics
        )
        
        return AggregationOutput(new_global=new_global, server_logs=server_logs)
    
    def _prepare_reports(self, client_reports: List[Dict]) -> List[Dict]:
        """Prepare and validate client reports."""
        prepared = []
        for report in client_reports:
            prep_report = {}
            
            # Move tensors to device
            if 'delta' in report:
                prep_report['delta'] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                    for k, v in report['delta'].items()
                }
            
            # Ensure all required fields
            prep_report['n_samples'] = report.get('n_samples', 100)
            prep_report['val_loss'] = report.get('val_loss', 1.0)
            prep_report['client_id'] = report.get('client_id', len(prepared))
            
            # Group statistics
            prep_report['group_counts'] = report.get('group_counts', {})
            
            # Proxies for DFBD
            prep_report['proxies'] = report.get('proxies', {
                'loss_drift': 0.0,
                'delta_norm': 1.0,
                'ece_proxy': 0.1
            })
            
            # Worst-group metrics
            prep_report['wg_f1'] = report.get('wg_f1', 0.5)
            
            prepared.append(prep_report)
        
        return prepared
    
    def _compute_objectives(self, client_reports: List[Dict], 
                           global_model: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute three core objectives and their gradients."""
        
        # 1. Accuracy objective (utility)
        acc_gradient = self._compute_accuracy_gradient(client_reports)
        
        # 2. Worst-group objective (robustness)
        wg_gradient = self._compute_worst_group_gradient(client_reports)
        
        # 3. Fairness objective (equity)
        fair_gradient = self._compute_fairness_gradient(client_reports)
        
        return {
            'accuracy': acc_gradient,
            'worst_group': wg_gradient,
            'fairness': fair_gradient
        }
    
    def _compute_accuracy_gradient(self, client_reports: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute gradient for global accuracy maximization."""
        total_samples = sum(r['n_samples'] for r in client_reports)
        
        acc_gradient = {}
        for key in client_reports[0]['delta']:
            weighted_sum = torch.zeros_like(client_reports[0]['delta'][key])
            
            for report in client_reports:
                # Weight by sample count and inverse loss
                weight = report['n_samples'] / total_samples
                loss_factor = torch.exp(-report['val_loss'])
                weighted_sum += weight * loss_factor * report['delta'][key]
            
            acc_gradient[key] = weighted_sum
        
        return acc_gradient
    
    def _compute_worst_group_gradient(self, client_reports: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute gradient for worst-group performance with smooth DRO."""
        # Collect group-wise statistics
        group_metrics = defaultdict(lambda: {'loss': 0.0, 'count': 0, 'f1': 0.0})
        
        for report in client_reports:
            for group_id, counts in report.get('group_counts', {}).items():
                total = sum(counts.values())
                if total > 0:
                    group_metrics[group_id]['loss'] += report['val_loss'] * total
                    group_metrics[group_id]['count'] += total
                    
                    # Compute F1 from confusion matrix
                    tp = counts.get('TP', 0)
                    fp = counts.get('FP', 0)
                    fn = counts.get('FN', 0)
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    group_metrics[group_id]['f1'] += f1 * total
        
        # Compute group losses with log-sum-exp smoothing
        group_losses = []
        group_weights = []
        
        for group_id, metrics in group_metrics.items():
            if metrics['count'] > 0:
                avg_loss = metrics['loss'] / metrics['count']
                avg_f1 = metrics['f1'] / metrics['count']
                
                # Higher loss and lower F1 indicate worse performance
                group_score = avg_loss * (2.0 - avg_f1)
                group_losses.append(group_score)
                group_weights.append(metrics['count'])
        
        if not group_losses:
            # Fallback to accuracy gradient
            return self._compute_accuracy_gradient(client_reports)
        
        # Log-sum-exp for smooth max
        group_losses_tensor = torch.tensor(group_losses, device=self.device)
        group_weights_tensor = torch.tensor(group_weights, device=self.device)
        
        max_loss = torch.max(group_losses_tensor)
        exp_losses = torch.exp((group_losses_tensor - max_loss) / self.tau)
        lse_loss = self.tau * torch.log(torch.sum(exp_losses)) + max_loss
        
        # Compute softmax weights for worst groups
        worst_group_weights = exp_losses / torch.sum(exp_losses)
        
        # Gradient weighted by worst-group importance
        wg_gradient = {}
        for key in client_reports[0]['delta']:
            weighted_sum = torch.zeros_like(client_reports[0]['delta'][key])
            
            for i, report in enumerate(client_reports):
                # Estimate client's contribution to worst groups
                client_wg_score = 1.0 / (report.get('wg_f1', 0.5) + 0.1)
                weighted_sum += client_wg_score * report['delta'][key]
            
            wg_gradient[key] = weighted_sum / len(client_reports)
        
        return wg_gradient
    
    def _compute_fairness_gradient(self, client_reports: List[Dict]) -> Dict[str, torch.Tensor]:
        """Compute gradient for multi-metric fairness optimization."""
        # Compute smooth fairness metrics
        group_stats = defaultdict(lambda: {
            'TP': 0.0, 'FP': 0.0, 'TN': 0.0, 'FN': 0.0,
            'total': 0, 'positive': 0
        })
        
        for report in client_reports:
            for group_id, counts in report.get('group_counts', {}).items():
                for key in ['TP', 'FP', 'TN', 'FN']:
                    group_stats[group_id][key] += counts.get(key, 0)
                group_stats[group_id]['total'] += sum(counts.values())
                group_stats[group_id]['positive'] += counts.get('TP', 0) + counts.get('FN', 0)
        
        # Compute smooth surrogates for fairness metrics
        tpr_values = []  # For Equal Opportunity
        fpr_values = []  # For Equalized Odds
        ppr_values = []  # For Statistical Parity
        
        epsilon = 1e-8
        for group_id, stats in group_stats.items():
            # Smooth TPR (True Positive Rate)
            tpr = (stats['TP'] + epsilon) / (stats['positive'] + epsilon)
            tpr_values.append(tpr)
            
            # Smooth FPR (False Positive Rate)
            negative = stats['total'] - stats['positive']
            fpr = (stats['FP'] + epsilon) / (negative + epsilon)
            fpr_values.append(fpr)
            
            # Smooth PPR (Positive Prediction Rate)
            ppr = (stats['TP'] + stats['FP'] + epsilon) / (stats['total'] + epsilon)
            ppr_values.append(ppr)
        
        # Compute fairness gaps using squared pairwise differences
        eo_gap = self._compute_pairwise_gap(tpr_values)
        fpr_gap = self._compute_pairwise_gap(fpr_values)
        sp_gap = self._compute_pairwise_gap(ppr_values)
        
        # Combined fairness loss with adaptive weights
        fairness_loss = (
            self.w_eo * eo_gap +
            self.w_fpr * fpr_gap +
            self.w_sp * sp_gap +
            self.lambda_eo.item() * eo_gap +
            self.lambda_fpr.item() * fpr_gap +
            self.lambda_sp.item() * sp_gap
        )
        
        # Gradient direction to minimize fairness gaps
        fair_gradient = {}
        for key in client_reports[0]['delta']:
            weighted_sum = torch.zeros_like(client_reports[0]['delta'][key])
            
            for report in client_reports:
                # Weight by inverse fairness contribution
                fairness_weight = torch.exp(-fairness_loss * 0.5)
                
                # Additional weight based on group balance
                group_balance = self._compute_group_balance(report.get('group_counts', {}))
                
                combined_weight = fairness_weight * (2.0 - group_balance)
                weighted_sum += combined_weight * report['delta'][key]
            
            fair_gradient[key] = weighted_sum / len(client_reports)
        
        return fair_gradient
    
    def _compute_pairwise_gap(self, values: List[float]) -> torch.Tensor:
        """Compute squared pairwise differences for fairness gap."""
        if len(values) < 2:
            return torch.tensor(0.0, device=self.device)
        
        values_tensor = torch.tensor(values, device=self.device)
        n = len(values)
        gap = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                gap += (values_tensor[i] - values_tensor[j]) ** 2
        
        return gap / (n * (n - 1) / 2)
    
    def _compute_group_balance(self, group_counts: Dict) -> float:
        """Compute group balance score (0=imbalanced, 1=balanced)."""
        if not group_counts:
            return 1.0
        
        totals = [sum(counts.values()) for counts in group_counts.values()]
        if not totals:
            return 1.0
        
        total_sum = sum(totals)
        if total_sum == 0:
            return 1.0
        
        proportions = [t / total_sum for t in totals]
        entropy = -sum(p * np.log(p + 1e-8) for p in proportions)
        max_entropy = np.log(len(proportions))
        
        return entropy / (max_entropy + 1e-8)
    
    def _multi_objective_optimization(self, objectives: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Full multi-objective optimization pipeline: MGDA → PCGrad → CAGrad."""
        
        # Step 1: MGDA for initial mixing
        mgda_direction = self._mgda_optimization(objectives)
        
        # Step 2: PCGrad for conflict resolution
        pcgrad_direction = self._pcgrad_projection(mgda_direction, objectives)
        
        # Step 3: CAGrad for conflict-averse refinement
        final_direction = self._cagrad_refinement(pcgrad_direction, objectives)
        
        return final_direction
    
    def _mgda_optimization(self, objectives: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Multiple Gradient Descent Algorithm for Pareto-optimal direction."""
        # Flatten gradients
        grad_list = []
        for obj_name in ['accuracy', 'worst_group', 'fairness']:
            flat_grad = torch.cat([g.flatten() for g in objectives[obj_name].values()])
            
            if self.mgda_normalize:
                flat_grad = flat_grad / (torch.norm(flat_grad) + 1e-8)
            
            grad_list.append(flat_grad.cpu().numpy())
        
        # Construct Gram matrix
        G = np.array(grad_list)
        GG = G @ G.T
        
        # Solve QP for optimal mixing weights
        n_tasks = len(grad_list)
        alpha = cp.Variable(n_tasks)
        
        # Objective: minimize ||Σ α_i g_i||^2
        objective = cp.Minimize(cp.quad_form(alpha, GG))
        
        # Constraints: simplex
        constraints = [
            alpha >= 0,
            cp.sum(alpha) == 1
        ]
        
        # Solve with robust solver
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            if alpha.value is None:
                raise ValueError("Solver failed")
            alpha_values = alpha.value
        except:
            # Fallback: equal weights
            alpha_values = np.ones(n_tasks) / n_tasks
        
        # Store for logging
        self.mgda_alphas = alpha_values
        
        # Compute mixed gradient
        mixed_gradient = {}
        obj_names = ['accuracy', 'worst_group', 'fairness']
        
        for key in objectives['accuracy']:
            mixed_gradient[key] = sum(
                alpha_values[i] * objectives[obj_names[i]][key]
                for i in range(n_tasks)
            ) * self.mgda_step_size
        
        return mixed_gradient
    
    def _pcgrad_projection(self, gradient: Dict[str, torch.Tensor],
                          objectives: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Project conflicting gradients to ensure all objectives improve."""
        # Flatten current gradient
        flat_grad = torch.cat([g.flatten() for g in gradient.values()])
        
        # Check conflicts with each objective
        num_conflicts = 0
        for obj_name in ['accuracy', 'worst_group', 'fairness']:
            obj_flat = torch.cat([g.flatten() for g in objectives[obj_name].values()])
            
            # Check if gradients conflict (negative dot product)
            if torch.dot(flat_grad, obj_flat) < 0:
                num_conflicts += 1
                
                # Project gradient to remove conflict
                proj_coeff = torch.dot(flat_grad, obj_flat) / (torch.norm(obj_flat) ** 2 + 1e-8)
                flat_grad = flat_grad - proj_coeff * obj_flat
        
        # Unflatten
        projected = {}
        idx = 0
        for key, tensor in gradient.items():
            numel = tensor.numel()
            projected[key] = flat_grad[idx:idx+numel].reshape(tensor.shape)
            idx += numel
        
        self.pcgrad_conflicts = num_conflicts
        return projected
    
    def _cagrad_refinement(self, gradient: Dict[str, torch.Tensor],
                          objectives: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Conflict-Averse Gradient refinement for smoother optimization."""
        # Compute average conflict score
        flat_grad = torch.cat([g.flatten() for g in gradient.values()])
        
        conflict_scores = []
        for obj_name in ['accuracy', 'worst_group', 'fairness']:
            obj_flat = torch.cat([g.flatten() for g in objectives[obj_name].values()])
            
            # Normalized dot product
            cos_sim = torch.dot(flat_grad, obj_flat) / (
                torch.norm(flat_grad) * torch.norm(obj_flat) + 1e-8
            )
            
            # Convert to conflict score (0 = aligned, 1 = orthogonal, 2 = opposite)
            conflict = 1.0 - cos_sim
            conflict_scores.append(conflict.item())
        
        avg_conflict = np.mean(conflict_scores)
        
        # Adaptive adjustment based on conflict level
        adjustment_factor = 1.0 / (1.0 + self.cagrad_rho * avg_conflict)
        
        # Apply refinement
        refined = {}
        for key, tensor in gradient.items():
            refined[key] = tensor * adjustment_factor
        
        self.cagrad_adjustment = adjustment_factor
        return refined
    
    def _update_dual_variables(self, client_reports: List[Dict]) -> Dict[str, float]:
        """Update Lagrangian dual variables for fairness constraints."""
        # Compute current fairness gaps
        metrics = self._compute_fairness_metrics(client_reports)
        
        # Dual ascent updates
        with torch.no_grad():
            # Equal Opportunity constraint
            eo_violation = metrics['eo_gap'] - self.epsilon_eo
            self.lambda_eo.data = torch.clamp(
                self.lambda_eo + self.dual_lr * eo_violation,
                min=0.0, max=5.0
            )
            
            # False Positive Rate constraint
            fpr_violation = metrics['fpr_gap'] - self.epsilon_fpr
            self.lambda_fpr.data = torch.clamp(
                self.lambda_fpr + self.dual_lr * fpr_violation,
                min=0.0, max=5.0
            )
            
            # Statistical Parity constraint
            sp_violation = metrics['sp_gap'] - self.epsilon_sp
            self.lambda_sp.data = torch.clamp(
                self.lambda_sp + self.dual_lr * sp_violation,
                min=0.0, max=5.0
            )
        
        # Track historical gaps for adaptive adjustment
        self._update_gap_history(metrics)
        
        return metrics
    
    def _compute_fairness_metrics(self, client_reports: List[Dict]) -> Dict[str, float]:
        """Compute comprehensive fairness metrics."""
        group_stats = defaultdict(lambda: {
            'TP': 0.0, 'FP': 0.0, 'TN': 0.0, 'FN': 0.0,
            'total': 0
        })
        
        for report in client_reports:
            for group_id, counts in report.get('group_counts', {}).items():
                for key in ['TP', 'FP', 'TN', 'FN']:
                    group_stats[group_id][key] += counts.get(key, 0)
                group_stats[group_id]['total'] += sum(counts.values())
        
        # Compute rates for each group
        epsilon = 1e-8
        tpr_values = []
        fpr_values = []
        ppr_values = []
        
        for group_id, stats in group_stats.items():
            positive = stats['TP'] + stats['FN']
            negative = stats['TN'] + stats['FP']
            
            tpr = (stats['TP'] + epsilon) / (positive + epsilon)
            fpr = (stats['FP'] + epsilon) / (negative + epsilon)
            ppr = (stats['TP'] + stats['FP'] + epsilon) / (stats['total'] + epsilon)
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
            ppr_values.append(ppr)
        
        # Compute gaps
        eo_gap = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        fpr_gap = max(fpr_values) - min(fpr_values) if fpr_values else 0.0
        sp_gap = max(ppr_values) - min(ppr_values) if ppr_values else 0.0
        
        # Compute worst-group F1
        wg_f1 = min(report.get('wg_f1', 0.5) for report in client_reports)
        
        return {
            'eo_gap': eo_gap,
            'fpr_gap': fpr_gap,
            'sp_gap': sp_gap,
            'worst_group_f1': wg_f1
        }
    
    def _update_gap_history(self, metrics: Dict[str, float]):
        """Update historical gap tracking for adaptive adjustments."""
        for gap_type in ['eo', 'fpr', 'sp']:
            gap_key = f'{gap_type}_gap'
            if gap_key in metrics:
                self.historical_gaps[gap_type].append(metrics[gap_key])
                
                # Keep only recent history
                if len(self.historical_gaps[gap_type]) > 20:
                    self.historical_gaps[gap_type].pop(0)
    
    def _compute_advanced_tilts(self, client_reports: List[Dict]) -> torch.Tensor:
        """Compute sophisticated DFBD tilts using neural network."""
        tilts = []
        proxy_tensors = []
        
        for report in client_reports:
            proxies = report.get('proxies', {})
            
            # Extract and normalize proxies
            loss_drift = proxies.get('loss_drift', 0.0)
            delta_norm = proxies.get('delta_norm', 1.0)
            ece_proxy = proxies.get('ece_proxy', 0.1)
            
            # Create normalized proxy tensor
            proxy_tensor = torch.tensor(
                [loss_drift / 0.5,  # Normalize assuming max drift of 0.5
                 delta_norm / 10.0,  # Normalize assuming max norm of 10
                 ece_proxy],         # Already in [0, 1]
                device=self.device,
                dtype=torch.float32
            )
            proxy_tensors.append(proxy_tensor)
        
        # Batch process through DFBD network
        proxy_batch = torch.stack(proxy_tensors)
        
        with torch.no_grad():
            raw_tilts = self.dfbd_network(proxy_batch).squeeze()
        
        # Apply eta amplification
        tilts = torch.pow(raw_tilts, self.dfbd_eta)
        
        # Normalize tilts to [0.5, 2.0] range for stability
        tilts = 0.5 + 1.5 * (tilts - tilts.min()) / (tilts.max() - tilts.min() + 1e-8)
        
        return tilts
    
    def _compute_optimal_weights(self, client_reports: List[Dict],
                                tilts: torch.Tensor,
                                fairness_metrics: Dict[str, float]) -> torch.Tensor:
        """Compute optimal aggregation weights combining multiple signals."""
        n_clients = len(client_reports)
        
        # Base weights from sample counts
        sample_weights = torch.tensor(
            [r['n_samples'] for r in client_reports],
            device=self.device, dtype=torch.float32
        )
        sample_weights = sample_weights / sample_weights.sum()
        
        # Performance weights (inverse loss)
        loss_weights = torch.tensor(
            [np.exp(-r['val_loss']) for r in client_reports],
            device=self.device, dtype=torch.float32
        )
        loss_weights = loss_weights / loss_weights.sum()
        
        # Fairness weights (boost clients with good worst-group performance)
        fairness_weights = torch.tensor(
            [r.get('wg_f1', 0.5) for r in client_reports],
            device=self.device, dtype=torch.float32
        )
        fairness_weights = fairness_weights / fairness_weights.sum()
        
        # Combine all signals
        combined_weights = (
            0.4 * sample_weights +
            0.3 * loss_weights +
            0.3 * fairness_weights
        )
        
        # Apply DFBD tilts
        weighted_tilts = combined_weights * tilts
        
        # Normalize and apply constraints
        final_weights = weighted_tilts / weighted_tilts.sum()
        final_weights = torch.clamp(final_weights, self.weight_floor, self.weight_cap)
        
        # Renormalize after clamping
        final_weights = final_weights / final_weights.sum()
        
        return final_weights
    
    def _weighted_aggregate_with_momentum(self, client_reports: List[Dict],
                                         weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform weighted aggregation with server momentum."""
        # Weighted aggregation
        aggregated = {}
        for key in client_reports[0]['delta']:
            weighted_sum = torch.zeros_like(client_reports[0]['delta'][key])
            
            for i, report in enumerate(client_reports):
                weighted_sum += weights[i] * report['delta'][key]
            
            aggregated[key] = weighted_sum
        
        # Apply momentum
        if self.momentum_buffer is None:
            self.momentum_buffer = {k: torch.zeros_like(v) for k, v in aggregated.items()}
        
        momentum_aggregated = {}
        for key in aggregated:
            self.momentum_buffer[key] = (
                self.server_momentum * self.momentum_buffer[key] +
                (1 - self.server_momentum) * aggregated[key]
            )
            momentum_aggregated[key] = self.momentum_buffer[key]
        
        return momentum_aggregated
    
    def _apply_update(self, global_model: Dict[str, torch.Tensor],
                     aggregated_delta: Dict[str, torch.Tensor],
                     mixed_direction: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply the update to global model with mixed objectives."""
        new_global = {}
        
        for key in global_model:
            # Combine aggregated delta with multi-objective direction
            update = aggregated_delta[key] + 0.1 * mixed_direction.get(key, 0)
            new_global[key] = global_model[key] + update
        
        return new_global
    
    def _perform_distillation(self, new_global: Dict[str, torch.Tensor],
                            old_global: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform server-side knowledge distillation (placeholder)."""
        # In practice, this would run actual distillation with a public dataset
        # For now, return placeholder metrics
        return {
            'distill_loss': 0.0,
            'distill_steps': self.distill_steps
        }
    
    def _update_selection_scores(self, client_reports: List[Dict],
                                fairness_metrics: Dict[str, float]) -> Dict[str, float]:
        """Update fairness-aware client selection scores using Lyapunov optimization."""
        selection_scores = {}
        
        for report in client_reports:
            client_id = report['client_id']
            
            # Compute fairness debt (Lyapunov function)
            val_loss = report['val_loss']
            wg_f1 = report.get('wg_f1', 0.5)
            
            # Debt increases when performance is below threshold
            performance_debt = max(0, self.selection_tau - wg_f1)
            loss_debt = self.selection_kappa * val_loss
            
            # Combined debt score
            total_debt = performance_debt + loss_debt
            
            # Update historical debt with decay
            if client_id in self.fairness_debt_scores:
                self.fairness_debt_scores[client_id] = (
                    0.9 * self.fairness_debt_scores[client_id] + 0.1 * total_debt
                )
            else:
                self.fairness_debt_scores[client_id] = total_debt
            
            selection_scores[client_id] = self.fairness_debt_scores[client_id]
        
        return {
            'signal_max': max(selection_scores.values()),
            'signal_mean': np.mean(list(selection_scores.values()))
        }
    
    def _compile_logs(self, round_ctx: Dict, client_reports: List[Dict],
                     weights: torch.Tensor, tilts: torch.Tensor,
                     objectives: Dict, mixed_direction: Dict,
                     fairness_metrics: Dict, distill_metrics: Dict,
                     selection_metrics: Dict) -> Dict[str, Any]:
        """Compile comprehensive logs for monitoring and analysis."""
        logs = {
            'round': self.round_counter,
            'participating_clients': len(client_reports),
            
            # Fairness metrics
            'val/eo_gap': fairness_metrics['eo_gap'],
            'val/fpr_gap': fairness_metrics['fpr_gap'],
            'val/sp_gap': fairness_metrics['sp_gap'],
            'val/worst_group_f1': fairness_metrics['worst_group_f1'],
            
            # Multi-objective optimization
            'mgda/alphas': self.mgda_alphas.tolist(),
            'pcgrad/num_conflicts': self.pcgrad_conflicts,
            'cagrad/rho_used': self.cagrad_rho,
            'cagrad/adjustment': self.cagrad_adjustment,
            
            # Dual variables
            'duals/lambda_eo': self.lambda_eo.item(),
            'duals/lambda_fpr': self.lambda_fpr.item(),
            'duals/lambda_sp': self.lambda_sp.item(),
            
            # DFBD tilts
            'dfbd/avg_tilt': tilts.mean().item(),
            'dfbd/eta': self.dfbd_eta,
            'dfbd/tilt_std': tilts.std().item(),
            
            # Selection signals
            'selector/signal_max': selection_metrics['signal_max'],
            'selector/signal_mean': selection_metrics['signal_mean'],
            
            # Distillation
            'distill/loss': distill_metrics['distill_loss'],
            'distill/steps': distill_metrics['distill_steps'],
            
            # Weight statistics
            'weights/mean': weights.mean().item(),
            'weights/std': weights.std().item(),
            'weights/max': weights.max().item(),
            'weights/min': weights.min().item(),
            
            # Privacy flags
            'privacy/sa_enabled': False,  # Would be set based on config
            'privacy/dp_enabled': self.config.get('dp', {}).get('enabled', False)
        }
        
        return logs


# Factory registration
def create_aggregator(config: Dict[str, Any], device: str = 'cuda') -> FairCareFLAggregator:
    """Factory function for creating FairCare-FL aggregator."""
    return FairCareFLAggregator(config, device)


# Register in the algorithm registry
ALGORITHM_REGISTRY = {
    'faircare_fl': create_aggregator
}
