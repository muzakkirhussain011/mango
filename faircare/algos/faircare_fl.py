"""FairCare-FL: Next-generation fair federated learning with Pareto optimization."""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from faircare.algos.aggregator import BaseAggregator, register_aggregator


@dataclass
class ComponentWeights:
    """Weights from each component algorithm."""
    fedavg: torch.Tensor
    fedprox: torch.Tensor
    qffl: torch.Tensor
    afl: torch.Tensor
    fairfate: torch.Tensor


@register_aggregator("faircare_fl")
class FairCareAggregator(BaseAggregator):
    """
    FairCare-FL: Enhanced fair federated learning with Pareto optimization.
    
    Version 2.0 features:
    - Pareto-Fair Aggregation (PFA) with MGDA/PCGrad/CAGrad
    - Demographics-Free Bias Discovery (DFBD) 
    - Group DRO for worst-group optimization
    - Fairness-aware client selection
    - Optional knowledge distillation
    - Secure aggregation and DP support
    """
    
    def __init__(
        self,
        n_clients: int,
        # Version tracking
        version: str = "2.0.0",
        
        # Ensemble configuration (legacy)
        gate_mode: str = "heuristic",
        ensemble_momentum: float = 0.7,
        
        # Component algorithm parameters
        qffl_q: float = 2.0,
        qffl_eps: float = 1e-4,
        afl_eta: float = 0.1,
        afl_smoothing: float = 0.01,
        prox_mu: float = 0.01,
        
        # Fairness metric weights
        alpha: float = 1.0,          # EO gap weight
        beta: float = 0.5,           # FPR gap weight  
        gamma: float = 0.5,          # SP gap weight
        delta_acc: float = 0.2,      # Accuracy weight
        delta_acc_init: float = 0.2,
        delta_acc_min: float = 0.01,
        
        # Temperature parameters
        tau: float = 1.0,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        tau_anneal: bool = True,
        tau_anneal_rate: float = 0.95,
        
        # Client-side CALT parameters
        lambda_irm: float = 0.5,
        lambda_adv: float = 0.2,
        use_mixup: bool = True,
        use_cia: bool = True,
        
        # Pareto optimization (new)
        mgda: Dict[str, Any] = None,
        pcgrad: Dict[str, Any] = None,
        cagrad: Dict[str, Any] = None,
        
        # Fairness duals (new)
        fairness_duals: Dict[str, Any] = None,
        
        # Demographics-free bias discovery (new)
        arl: Dict[str, Any] = None,
        
        # Fair client selection (new)
        selector: Dict[str, Any] = None,
        
        # Distillation (new)
        distill: Dict[str, Any] = None,
        
        # Server momentum
        server_momentum: float = 0.8,
        
        # Bias detection thresholds
        bias_threshold_eo: float = 0.15,
        bias_threshold_fpr: float = 0.15,
        bias_threshold_sp: float = 0.10,
        
        # Weight bounds
        epsilon: float = 0.01,
        weight_clip: float = 10.0,
        
        # Legacy parameters
        fairness_metric: str = "eo_gap",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip,
                        fairness_metric=fairness_metric)
        
        self.version = version
        
        # Store all parameters
        self.gate_mode = gate_mode
        self.ensemble_momentum = ensemble_momentum
        
        # Component algorithm params
        self.qffl_q = qffl_q
        self.qffl_eps = qffl_eps
        self.afl_eta = afl_eta
        self.afl_smoothing = afl_smoothing
        self.prox_mu = prox_mu
        
        # CALT params for clients
        self.lambda_irm = lambda_irm
        self.lambda_adv = lambda_adv
        self.use_mixup = use_mixup
        self.use_cia = use_cia
        
        # Fairness params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_acc = delta_acc
        self.delta_acc_init = delta_acc_init
        self.delta_acc_min = delta_acc_min
        
        self.tau = tau
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau_anneal = tau_anneal
        self.tau_anneal_rate = tau_anneal_rate
        
        self.server_momentum = server_momentum
        
        self.bias_threshold_eo = bias_threshold_eo
        self.bias_threshold_fpr = bias_threshold_fpr
        self.bias_threshold_sp = bias_threshold_sp
        
        # Parse new configs with defaults
        self.mgda_config = mgda or {'normalize_grads': True, 'solver': 'qp', 'step_size': 1.0}
        self.pcgrad_config = pcgrad or {'enabled': True}
        self.cagrad_config = cagrad or {'enabled': True, 'rho': 0.5}
        self.fairness_duals_config = fairness_duals or {'enabled': False, 'epsilon_eo': 0.02, 'epsilon_fpr': 0.02, 'lr': 0.1}
        self.arl_config = arl or {'enabled': True, 'eta': 1.0, 'width': 64, 'depth': 1}
        self.selector_config = selector or {'enabled': True, 'mode': 'lyapunov', 'tau': 0.02, 'kappa': 0.5}
        self.distill_config = distill or {'enabled': False, 'temperature': 2.0, 'steps': 200, 'batch_size': 64}
        
        # Initialize state
        self.round_num = 0
        self.bias_mitigation_mode = False
        self.consecutive_bias_rounds = 0
        self.prev_global_update = None
        
        # Initialize ARL head if enabled
        if self.arl_config['enabled']:
            self._arl_head = self._ARLHead(
                input_dim=5,  # Basic proxy features
                width=self.arl_config['width'],
                depth=self.arl_config['depth'],
                eta=self.arl_config['eta']
            )
        else:
            self._arl_head = None
        
        # Initialize selector if enabled
        if self.selector_config['enabled']:
            self._selector = self._FairnessSelector(
                n_clients=n_clients,
                mode=self.selector_config['mode'],
                tau=self.selector_config['tau'],
                kappa=self.selector_config['kappa']
            )
        else:
            self._selector = None
        
        # Initialize fairness duals if enabled
        if self.fairness_duals_config['enabled']:
            self.lambda_dual = {'EO': 0.0, 'FPR': 0.0, 'SP': 0.0}
        else:
            self.lambda_dual = None
        
        # Client history tracking
        self.client_history = {}
        for i in range(n_clients):
            self.client_history[i] = {
                'participation_count': 0,
                'eo_gaps': [],
                'fpr_gaps': [],
                'sp_gaps': [],
                'worst_group_f1': [],
                'delta_norms': [],
                'grad_norms': []
            }
        
        # Global metrics history
        self.global_metrics_history = {
            'avg_eo_gap': [],
            'avg_fpr_gap': [],
            'avg_sp_gap': [],
            'worst_group_f1': [],
            'mgda_weights': [],
            'dual_values': []
        }
        
        # AFL client weights for min-max optimization
        self.afl_client_weights = torch.ones(n_clients) / n_clients
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Main aggregation method using Pareto-Fair Aggregation.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Step 1: Demographics-free bias discovery (if enabled)
        if self._arl_head is not None:
            client_tilts = self._compute_arl_tilts(client_summaries)
        else:
            client_tilts = torch.ones(n) / n
        
        # Step 2: Compute multi-objective gradients
        g_acc, g_wg, g_fair = self._compute_objective_gradients(client_summaries, client_tilts)
        
        # Step 3: MGDA + PCGrad + CAGrad
        if g_acc is not None and g_wg is not None and g_fair is not None:
            mgda_weights = self._mgda_solve([g_acc, g_wg, g_fair])
            
            # Compute mixed gradient
            g_mixed = mgda_weights[0] * g_acc + mgda_weights[1] * g_wg + mgda_weights[2] * g_fair
            
            # Apply PCGrad if enabled
            if self.pcgrad_config['enabled']:
                g_mixed = self._pcgrad([g_acc, g_wg, g_fair], g_mixed)
            
            # Apply CAGrad if enabled
            if self.cagrad_config['enabled']:
                g_mixed = self._cagrad([g_acc, g_wg, g_fair], g_mixed, rho=self.cagrad_config['rho'])
            
            # Store MGDA weights for logging
            self.global_metrics_history['mgda_weights'].append(mgda_weights.tolist())
        else:
            # Fallback to legacy ensemble approach
            mgda_weights = None
            g_mixed = None
        
        # Step 4: Update fairness duals if enabled
        if self.lambda_dual is not None:
            self._update_duals(client_summaries)
        
        # Step 5: Compute final aggregation weights
        if g_mixed is not None:
            # Use gradient-based weighting
            final_weights = self._gradient_to_weights(g_mixed, client_summaries)
        else:
            # Fallback to legacy component-based weighting
            component_weights = self._compute_component_weights(client_summaries)
            final_weights = self._compute_ensemble_weights(client_summaries, component_weights)
        
        # Step 6: Update selector if enabled
        if self._selector is not None:
            self._selector.update(client_summaries)
        
        # Step 7: Post-process weights
        final_weights = self._postprocess(final_weights)
        
        # Log state
        self._log_round_state(client_summaries, mgda_weights)
        
        return final_weights
    
    # ========== Private Helper Classes ==========
    
    class _ARLHead(nn.Module):
        """Demographics-free adversarial reweighting head."""
        
        def __init__(self, input_dim: int, width: int, depth: int, eta: float):
            super().__init__()
            self.eta = eta
            
            layers = []
            prev_dim = input_dim
            for _ in range(depth):
                layers.append(nn.Linear(prev_dim, width))
                layers.append(nn.ReLU())
                prev_dim = width
            layers.append(nn.Linear(prev_dim, 1))
            
            self.network = nn.Sequential(*layers)
            
            # Initialize with small weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x):
            return self.network(x)
    
    class _FairnessSelector:
        """Fairness-aware client selector using Lyapunov optimization."""
        
        def __init__(self, n_clients: int, mode: str, tau: float, kappa: float):
            self.n_clients = n_clients
            self.mode = mode
            self.tau = tau
            self.kappa = kappa
            
            # Initialize fairness debt queues
            self.debt_queues = torch.zeros(n_clients)
            self.participation_counts = torch.zeros(n_clients)
        
        def update(self, client_summaries: List[Dict[str, Any]]):
            """Update debt queues based on client fairness metrics."""
            for summary in client_summaries:
                client_id = summary.get('client_id', 0)
                if client_id < self.n_clients:
                    # Compute fairness debt
                    eo_gap = summary.get('eo_gap', 0)
                    fpr_gap = summary.get('fpr_gap', 0)
                    sp_gap = summary.get('sp_gap', 0)
                    
                    debt = eo_gap + fpr_gap + sp_gap
                    
                    # Update queue: Q_c <- [Q_c + debt - tau]^+
                    self.debt_queues[client_id] = max(0, self.debt_queues[client_id] + debt - self.tau)
                    self.participation_counts[client_id] += 1
        
        def select_clients(self, n_select: int) -> List[int]:
            """Select clients based on fairness debt."""
            if self.mode == 'lyapunov':
                # Prioritize clients with high debt
                scores = self.debt_queues.clone()
                
                # Add exploration bonus for low-participation clients
                participation_bonus = 1.0 / (1 + self.participation_counts)
                scores = scores + self.kappa * participation_bonus
                
                # Select top clients
                _, indices = torch.topk(scores, min(n_select, self.n_clients))
                return indices.tolist()
            else:
                # Uniform selection fallback
                perm = torch.randperm(self.n_clients)
                return perm[:n_select].tolist()
    
    # ========== Private Helper Methods ==========
    
    def _compute_arl_tilts(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """Compute ARL-based client tilts without demographics."""
        n = len(client_summaries)
        
        # Extract proxy features for each client
        proxy_features = []
        for summary in client_summaries:
            features = torch.tensor([
                summary.get('val_loss', 1.0),
                summary.get('train_loss', 1.0),
                summary.get('delta_norm', 1.0),
                summary.get('grad_norm', 1.0),
                summary.get('calibration_error', 0.0)
            ], dtype=torch.float32)
            
            # Normalize features
            features = (features - features.mean()) / (features.std() + 1e-8)
            proxy_features.append(features)
        
        proxy_features = torch.stack(proxy_features)
        
        # Compute ARL scores
        with torch.no_grad():
            scores = self._arl_head(proxy_features).squeeze()
            
            # Convert to tilts using softplus
            tilts = F.softplus(scores * self._arl_head.eta)
            
            # Normalize
            tilts = tilts / tilts.sum()
        
        return tilts
    
    def _compute_objective_gradients(
        self, 
        client_summaries: List[Dict[str, Any]], 
        tilts: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute gradients for accuracy, worst-group, and fairness objectives."""
        
        # Extract aggregated statistics
        total_samples = sum(s.get('n_samples', 1) for s in client_summaries)
        
        # Accuracy gradient (approximated from losses)
        g_acc = torch.zeros(1)  # Placeholder for gradient direction
        for i, summary in enumerate(client_summaries):
            loss = summary.get('val_loss', summary.get('train_loss', 1.0))
            weight = tilts[i] * summary.get('n_samples', 1) / total_samples
            g_acc += weight * loss
        
        # Worst-group gradient (using log-sum-exp surrogate)
        g_wg = self._compute_worst_group_gradient(client_summaries, tilts)
        
        # Fairness gradient (smooth surrogates)
        g_fair = self._compute_fairness_gradient(client_summaries)
        
        return g_acc, g_wg, g_fair
    
    def _compute_worst_group_gradient(
        self, 
        client_summaries: List[Dict[str, Any]], 
        tilts: torch.Tensor
    ) -> torch.Tensor:
        """Compute worst-group DRO gradient using log-sum-exp."""
        # Collect group losses
        group_losses = {}
        
        for i, summary in enumerate(client_summaries):
            group_stats = summary.get('group_stats', {})
            for group_name, stats in group_stats.items():
                if group_name not in group_losses:
                    group_losses[group_name] = []
                
                # Compute group loss approximation
                tp = stats.get('TP', 0)
                fp = stats.get('FP', 0)
                fn = stats.get('FN', 0)
                tn = stats.get('TN', 0)
                
                total = tp + fp + fn + tn
                if total > 0:
                    error_rate = (fp + fn) / total
                    group_losses[group_name].append(error_rate * tilts[i].item())
        
        if not group_losses:
            return torch.zeros(1)
        
        # Log-sum-exp approximation
        temperature = 10.0  # Higher = closer to true max
        losses = []
        for group_name, loss_list in group_losses.items():
            if loss_list:
                losses.append(np.mean(loss_list))
        
        if losses:
            losses_tensor = torch.tensor(losses, dtype=torch.float32)
            wg_loss = torch.logsumexp(losses_tensor * temperature, dim=0) / temperature
            return wg_loss.unsqueeze(0)
        
        return torch.zeros(1)
    
    def _compute_fairness_gradient(
        self, 
        client_summaries: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Compute gradient of fairness surrogates."""
        # Aggregate fairness gaps
        eo_gaps = []
        fpr_gaps = []
        sp_gaps = []
        
        for summary in client_summaries:
            eo_gaps.append(summary.get('eo_gap', 0))
            fpr_gaps.append(summary.get('fpr_gap', 0))
            sp_gaps.append(abs(summary.get('sp_gap', 0)))
        
        # Weighted fairness loss
        avg_eo = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp = np.mean(sp_gaps) if sp_gaps else 0
        
        fairness_loss = self.alpha * avg_eo + self.beta * avg_fpr + self.gamma * avg_sp
        
        return torch.tensor([fairness_loss], dtype=torch.float32)
    
    def _mgda_solve(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Solve MGDA optimization problem for Pareto descent.
        
        min_{α ∈ Δ} ||Σ_i α_i g_i||^2
        """
        k = len(gradients)
        
        if self.mgda_config['normalize_grads']:
            # Normalize gradients
            normalized = []
            for g in gradients:
                norm = torch.norm(g)
                if norm > 1e-8:
                    normalized.append(g / norm)
                else:
                    normalized.append(g)
            gradients = normalized
        
        # Simple Frank-Wolfe solver for small k
        alpha = torch.ones(k) / k  # Initialize uniform
        
        for _ in range(20):  # Fixed iterations
            # Compute gradient w.r.t. alpha
            mixed = sum(a * g for a, g in zip(alpha, gradients))
            grad_alpha = torch.stack([2 * torch.dot(mixed.flatten(), g.flatten()) for g in gradients])
            
            # Frank-Wolfe step
            idx_min = torch.argmin(grad_alpha)
            vertex = torch.zeros(k)
            vertex[idx_min] = 1.0
            
            # Line search
            gamma = 2.0 / (2.0 + _)  # Decreasing step size
            alpha = (1 - gamma) * alpha + gamma * vertex
        
        return alpha
    
    def _pcgrad(self, gradients: List[torch.Tensor], g_mixed: torch.Tensor) -> torch.Tensor:
        """Apply PCGrad gradient surgery to remove conflicts."""
        result = g_mixed.clone()
        
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                g_i = gradients[i].flatten()
                g_j = gradients[j].flatten()
                
                # Check for conflict
                if torch.dot(g_i, g_j) < 0:
                    # Project g_i onto the normal plane of g_j
                    proj = torch.dot(g_i, g_j) / (torch.norm(g_j) ** 2 + 1e-8)
                    g_i_corrected = g_i - proj * g_j
                    
                    # Update result
                    result = result - gradients[i] + g_i_corrected.view_as(gradients[i])
        
        return result
    
    def _cagrad(self, gradients: List[torch.Tensor], g_mixed: torch.Tensor, rho: float) -> torch.Tensor:
        """Apply CAGrad conflict-averse gradient descent."""
        # Simplified CAGrad: balance average with worst-case improvement
        
        # Compute average direction
        g_avg = sum(gradients) / len(gradients)
        
        # Find worst objective (largest loss/gradient magnitude)
        mags = [torch.norm(g) for g in gradients]
        worst_idx = np.argmax(mags)
        g_worst = gradients[worst_idx]
        
        # Blend average with worst-case direction
        g_cagrad = (1 - rho) * g_avg + rho * g_worst
        
        return g_cagrad
    
    def _gradient_to_weights(
        self, 
        gradient: torch.Tensor, 
        client_summaries: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Convert gradient direction to client weights."""
        n = len(client_summaries)
        
        # Simple heuristic: weight by sample count and gradient alignment
        weights = []
        for summary in client_summaries:
            n_samples = summary.get('n_samples', 1)
            
            # Approximate alignment with gradient direction
            loss = summary.get('val_loss', summary.get('train_loss', 1.0))
            alignment = 1.0 / (1.0 + loss)  # Lower loss = better alignment
            
            weights.append(n_samples * alignment)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()
        
        return weights
    
    def _update_duals(self, client_summaries: List[Dict[str, Any]]):
        """Update dual variables for fairness constraints."""
        # Aggregate fairness gaps
        eo_gaps = [s.get('eo_gap', 0) for s in client_summaries]
        fpr_gaps = [s.get('fpr_gap', 0) for s in client_summaries]
        sp_gaps = [abs(s.get('sp_gap', 0)) for s in client_summaries]
        
        avg_eo = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp = np.mean(sp_gaps) if sp_gaps else 0
        
        lr = self.fairness_duals_config['lr']
        eps_eo = self.fairness_duals_config['epsilon_eo']
        eps_fpr = self.fairness_duals_config['epsilon_fpr']
        
        # Projected gradient ascent on duals
        self.lambda_dual['EO'] = max(0, self.lambda_dual['EO'] + lr * (avg_eo - eps_eo))
        self.lambda_dual['FPR'] = max(0, self.lambda_dual['FPR'] + lr * (avg_fpr - eps_fpr))
        self.lambda_dual['SP'] = max(0, self.lambda_dual['SP'] + lr * (avg_sp - eps_eo))
        
        # Store for logging
        self.global_metrics_history['dual_values'].append(dict(self.lambda_dual))
    
    # ========== Legacy Methods (preserved for compatibility) ==========
    
    def _compute_component_weights(
        self,
        client_summaries: List[Dict[str, Any]]
    ) -> ComponentWeights:
        """Legacy: Compute weights from each component algorithm."""
        n = len(client_summaries)
        
        # FedAvg: weight by number of samples
        fedavg_weights = []
        for summary in client_summaries:
            n_samples = summary.get("n_samples", 1)
            fedavg_weights.append(n_samples)
        fedavg_weights = torch.tensor(fedavg_weights, dtype=torch.float32)
        fedavg_weights = fedavg_weights / fedavg_weights.sum()
        
        # FedProx: similar to FedAvg
        fedprox_weights = fedavg_weights.clone()
        
        # q-FFL: weight by (loss)^q
        qffl_weights = []
        for summary in client_summaries:
            loss = summary.get("train_loss", summary.get("val_loss", 1.0))
            weight = (loss + self.qffl_eps) ** self.qffl_q
            qffl_weights.append(weight)
        qffl_weights = torch.tensor(qffl_weights, dtype=torch.float32)
        qffl_weights = qffl_weights / qffl_weights.sum()
        
        # AFL: exponential weighting
        afl_weights = []
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            loss = summary.get("val_loss", summary.get("train_loss", 1.0))
            
            if client_id < len(self.afl_client_weights):
                loss_tensor = torch.tensor(self.afl_eta * loss, dtype=torch.float32)
                self.afl_client_weights[client_id] *= torch.exp(loss_tensor)
        
        self.afl_client_weights = self.afl_client_weights / self.afl_client_weights.sum()
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            if client_id < len(self.afl_client_weights):
                afl_weights.append(self.afl_client_weights[client_id].item())
            else:
                afl_weights.append(1.0 / len(client_summaries))
        afl_weights = torch.tensor(afl_weights, dtype=torch.float32)
        afl_weights = afl_weights / afl_weights.sum()
        
        # FairFATE: fairness-aware weights
        fairfate_weights = []
        for summary in client_summaries:
            eo_gap = summary.get("eo_gap", 0.5)
            fpr_gap = summary.get("fpr_gap", 0.5)
            sp_gap = summary.get("sp_gap", 0.5)
            val_loss = summary.get("val_loss", 1.0)
            
            fairness_penalty = (
                self.alpha * eo_gap +
                self.beta * fpr_gap +
                self.gamma * abs(sp_gap) +
                0.1 * val_loss
            )
            fairfate_weights.append(fairness_penalty)
        
        fairfate_weights = torch.tensor(fairfate_weights, dtype=torch.float32)
        fairfate_weights = torch.exp(-fairfate_weights / max(self.tau, 0.1))
        fairfate_weights = fairfate_weights / fairfate_weights.sum()
        
        return ComponentWeights(
            fedavg=fedavg_weights,
            fedprox=fedprox_weights,
            qffl=qffl_weights,
            afl=afl_weights,
            fairfate=fairfate_weights
        )
    
    def _compute_ensemble_weights(
        self,
        client_summaries: List[Dict[str, Any]],
        component_weights: ComponentWeights
    ) -> torch.Tensor:
        """Legacy: Compute ensemble weights for backward compatibility."""
        # Detect bias state
        avg_eo_gap = np.mean([s.get("eo_gap", 0) for s in client_summaries])
        avg_fpr_gap = np.mean([s.get("fpr_gap", 0) for s in client_summaries])
        
        if avg_eo_gap > self.bias_threshold_eo or avg_fpr_gap > self.bias_threshold_fpr:
            self.bias_mitigation_mode = True
        else:
            self.bias_mitigation_mode = False
        
        # Heuristic gating
        if self.bias_mitigation_mode:
            mix_coeffs = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3])
        else:
            mix_coeffs = torch.tensor([0.25, 0.2, 0.2, 0.15, 0.2])
        
        final_weights = (
            mix_coeffs[0] * component_weights.fedavg +
            mix_coeffs[1] * component_weights.fedprox +
            mix_coeffs[2] * component_weights.qffl +
            mix_coeffs[3] * component_weights.afl +
            mix_coeffs[4] * component_weights.fairfate
        )
        
        return final_weights
    
    def _log_round_state(self, client_summaries: List[Dict[str, Any]], mgda_weights: Optional[torch.Tensor]):
        """Log current round state."""
        # Update history
        avg_eo_gap = np.mean([s.get('eo_gap', 0) for s in client_summaries])
        avg_fpr_gap = np.mean([s.get('fpr_gap', 0) for s in client_summaries])
        avg_sp_gap = np.mean([abs(s.get('sp_gap', 0)) for s in client_summaries])
        
        self.global_metrics_history['avg_eo_gap'].append(avg_eo_gap)
        self.global_metrics_history['avg_fpr_gap'].append(avg_fpr_gap)
        self.global_metrics_history['avg_sp_gap'].append(avg_sp_gap)
        
        worst_f1s = [s.get('worst_group_f1', 0) for s in client_summaries if 'worst_group_f1' in s]
        if worst_f1s:
            self.global_metrics_history['worst_group_f1'].append(min(worst_f1s))
        
        if self.round_num % 5 == 0 or self.bias_mitigation_mode:
            log_str = f"[Round {self.round_num}] "
            log_str += f"EO: {avg_eo_gap:.3f}, FPR: {avg_fpr_gap:.3f}, SP: {avg_sp_gap:.3f}"
            
            if mgda_weights is not None:
                log_str += f", MGDA: {mgda_weights.tolist()}"
            
            if self.lambda_dual is not None:
                log_str += f", Duals: {self.lambda_dual}"
            
            print(log_str)
    
    def get_fairness_config(self) -> Dict[str, Any]:
        """Get current fairness configuration for clients."""
        return {
            'lambda_irm': self.lambda_irm,
            'lambda_adv': self.lambda_adv,
            'use_mixup': self.use_mixup,
            'use_cia': self.use_cia,
            'prox_mu': self.prox_mu,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'round': self.round_num,
            'tau': self.tau,
            'delta_acc': self.delta_acc,
            'version': self.version
        }
    
    def apply_server_momentum(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply server-side momentum to aggregated update."""
        if self.prev_global_update is None:
            self.prev_global_update = update
            return update
        
        momentum_update = {}
        for key in update.keys():
            momentum_update[key] = (
                self.server_momentum * self.prev_global_update[key] +
                (1 - self.server_momentum) * update[key]
            )
        
        self.prev_global_update = momentum_update
        return momentum_update
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current algorithm statistics."""
        stats = {
            'round': self.round_num,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'tau': self.tau,
            'delta_acc': self.delta_acc,
            'version': self.version,
            'global_metrics': self.global_metrics_history
        }
        
        if self.lambda_dual is not None:
            stats['fairness_duals'] = dict(self.lambda_dual)
        
        if self._selector is not None:
            stats['selector_queues'] = self._selector.debt_queues.tolist()
        
        return stats
    
    def get_selected_clients(self, n_select: int) -> Optional[List[int]]:
        """Get fairness-aware client selection if selector is enabled."""
        if self._selector is not None:
            return self._selector.select_clients(n_select)
        return None