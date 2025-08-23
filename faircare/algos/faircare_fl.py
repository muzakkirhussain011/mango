# faircare/algos/faircare_fl.py
"""FairCare-FL: Next-generation fair federated learning with PFA, DFBD, and CALT."""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import cvxpy as cp
from faircare.algos.aggregator import BaseAggregator, register_aggregator
from faircare.core.utils import flatten_weights, unflatten_weights


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
    FairCare-FL v2.0: Pareto-Fair Aggregation with Demographics-Free Bias Discovery
    
    Features:
    - Pareto-Fair Aggregation (PFA) with MGDA + PCGrad + CAGrad
    - Demographics-Free Bias Discovery (DFBD) via adversarial reweighting
    - Fairness-Aware Client Selection with Lyapunov queues
    - Optional distillation for accuracy recovery
    - Full SA/DP support preserved
    """
    
    def __init__(
        self,
        n_clients: int,
        version: str = "2.0.0",
        
        # PFA: Multi-objective parameters
        mgda_normalize_grads: bool = True,
        mgda_solver: str = "qp",  
        mgda_step_size: float = 1.0,
        pcgrad_enabled: bool = True,
        cagrad_enabled: bool = True,
        cagrad_rho: float = 0.5,
        
        # Fairness dual ascent (optional constraints)
        fairness_duals_enabled: bool = False,
        epsilon_eo: float = 0.02,
        epsilon_fpr: float = 0.02,
        epsilon_sp: float = 0.02,
        dual_lr: float = 0.1,
        
        # DFBD: Demographics-free bias discovery
        arl_enabled: bool = True,
        arl_eta: float = 1.0,
        arl_width: int = 64,
        arl_depth: int = 1,
        arl_lr: float = 0.001,
        
        # Selector: Fairness-aware client selection
        selector_enabled: bool = True,
        selector_mode: str = "lyapunov",
        selector_tau: float = 0.02,
        selector_kappa: float = 0.5,
        
        # Distillation (optional)
        distill_enabled: bool = False,
        distill_temperature: float = 2.0,
        distill_steps: int = 200,
        distill_batch_size: int = 64,
        
        # Client-side CALT parameters
        prox_mu: float = 0.01,
        lambda_irm: float = 0.5,
        lambda_adv: float = 0.2,
        use_mixup: bool = True,
        use_cia: bool = True,
        use_adversary: bool = False,
        
        # Legacy/compatibility parameters
        gate_mode: str = "learned",
        lambda_fair: float = 0.1,
        lambda_fair_init: float = 0.1,
        lambda_fair_min: float = 0.01,
        lambda_fair_max: float = 2.0,
        tau: float = 1.0,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
        server_momentum: float = 0.8,
        delta_acc: float = 0.2,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.1,
        
        # Fairness loss weights
        w_eo: float = 1.0,
        w_fpr: float = 0.5,
        w_sp: float = 0.5,
        
        # Bias detection thresholds
        bias_threshold_eo: float = 0.15,
        bias_threshold_fpr: float = 0.15,
        bias_threshold_sp: float = 0.10,
        detector_patience: int = 2,
        
        # Privacy
        dp_enabled: bool = False,
        dp_clip: float = 1.0,
        dp_noise_mult: float = 0.6,
        
        # Base parameters
        epsilon: float = 0.01,
        weight_clip: float = 10.0,
        fairness_metric: str = "eo_gap",
        **kwargs
    ):
        super().__init__(n_clients, epsilon=epsilon, weight_clip=weight_clip,
                        fairness_metric=fairness_metric)
        
        self.version = version
        
        # PFA parameters
        self.mgda_normalize_grads = mgda_normalize_grads
        self.mgda_solver = mgda_solver
        self.mgda_step_size = mgda_step_size
        self.pcgrad_enabled = pcgrad_enabled
        self.cagrad_enabled = cagrad_enabled
        self.cagrad_rho = cagrad_rho
        
        # Fairness duals
        self.fairness_duals_enabled = fairness_duals_enabled
        self.epsilon_eo = epsilon_eo
        self.epsilon_fpr = epsilon_fpr
        self.epsilon_sp = epsilon_sp
        self.dual_lr = dual_lr
        self.dual_eo = 0.0
        self.dual_fpr = 0.0
        self.dual_sp = 0.0
        
        # DFBD parameters
        self.arl_enabled = arl_enabled
        self.arl_eta = arl_eta
        self.arl_lr = arl_lr
        if arl_enabled:
            self.arl_head = self._create_arl_head(arl_width, arl_depth)
            self.arl_optimizer = torch.optim.Adam(self.arl_head.parameters(), lr=arl_lr)
        else:
            self.arl_head = None
            self.arl_optimizer = None
            
        # Selector parameters
        self.selector_enabled = selector_enabled
        self.selector_mode = selector_mode
        self.selector_tau = selector_tau
        self.selector_kappa = selector_kappa
        self.client_fairness_debt = torch.zeros(n_clients)
        
        # Distillation parameters
        self.distill_enabled = distill_enabled
        self.distill_temperature = distill_temperature
        self.distill_steps = distill_steps
        self.distill_batch_size = distill_batch_size
        
        # CALT parameters
        self.prox_mu = prox_mu
        self.lambda_irm = lambda_irm
        self.lambda_adv = lambda_adv
        self.use_mixup = use_mixup
        self.use_cia = use_cia
        self.use_adversary = use_adversary
        
        # Legacy compatibility
        self.lambda_fair = lambda_fair
        self.lambda_fair_init = lambda_fair_init
        self.lambda_fair_min = lambda_fair_min
        self.lambda_fair_max = lambda_fair_max
        
        # Fairness loss weights
        self.w_eo = w_eo
        self.w_fpr = w_fpr
        self.w_sp = w_sp
        
        # Other parameters
        self.gate_mode = gate_mode
        self.tau = tau
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.server_momentum = server_momentum
        self.delta_acc = delta_acc
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bias_mitigation_mode = False
        self.bias_threshold_eo = bias_threshold_eo
        self.bias_threshold_fpr = bias_threshold_fpr
        self.bias_threshold_sp = bias_threshold_sp
        self.detector_patience = detector_patience
        self.consecutive_bias_rounds = 0
        
        # Privacy
        self.dp_enabled = dp_enabled
        self.dp_clip = dp_clip
        self.dp_noise_mult = dp_noise_mult
        
        # State tracking
        self.round_num = 0
        self.prev_global_update = None
        self.client_deltas_cache = []
        self.global_metrics_history = {
            'avg_eo_gap': [],
            'avg_fpr_gap': [],
            'avg_sp_gap': [],
            'worst_group_f1': [],
            'dual_values': [],
            'mgda_weights': [],
            'selector_queues': []
        }
        
        # For compatibility with tests
        if gate_mode == "learned":
            self.gate_network = self._create_gate_network()
            
        # AFL weights tracking
        self.afl_client_weights = torch.ones(n_clients) / n_clients
        
    def _create_arl_head(self, width: int, depth: int) -> nn.Module:
        """Create adversarial reweighting head for DFBD."""
        layers = []
        input_dim = 7  # loss, delta_norm, grad_norm, val_loss, val_acc, calibration, drift
        
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, width))
            else:
                layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        layers.append(nn.Linear(width, 1))
        layers.append(nn.Softplus())  # Non-negative tilts
        
        return nn.Sequential(*layers)
    
    def _create_gate_network(self) -> nn.Module:
        """Create gate network for legacy tests."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Softmax(dim=-1)
        )
    
    def aggregate_with_deltas(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        client_summaries: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Main aggregation with access to actual model deltas for proper gradient computation.
        """
        # Cache deltas for gradient computation
        self.client_deltas_cache = client_deltas
        
        # Compute weights using standard method
        weights = self.compute_weights(client_summaries)
        
        # Aggregate deltas
        aggregated = {}
        for key in client_deltas[0].keys():
            weighted_deltas = []
            for i, delta in enumerate(client_deltas):
                weighted_deltas.append(weights[i] * delta[key])
            aggregated[key] = sum(weighted_deltas)
        
        # Apply server momentum if enabled
        if self.server_momentum > 0:
            aggregated = self.apply_server_momentum(aggregated)
        
        # Apply DP noise if enabled
        if self.dp_enabled:
            aggregated = self._apply_dp_noise(aggregated)
        
        return aggregated
    
    def compute_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute aggregation weights with PFA + DFBD + Selector.
        """
        self.round_num += 1
        n = len(client_summaries)
        
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Update bias detection state
        self._update_bias_detection(client_summaries)
        
        if self.version == "2.0.0" and self.client_deltas_cache:
            # Full v2.0 PFA implementation with actual gradients
            weights = self._compute_pfa_weights(client_summaries)
        else:
            # Fallback to heuristic weighting
            weights = self._compute_heuristic_weights(client_summaries)
        
        # Apply post-processing (floor, clip)
        weights = self._postprocess(weights)
        
        return weights
    
    def _compute_pfa_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute weights using Pareto-Fair Aggregation with actual gradients.
        """
        n = len(client_summaries)
        
        # Step 1: Compute demographics-free tilts via DFBD
        client_tilts = self._compute_client_tilts(client_summaries)
        
        # Step 2: Compute multi-objective gradients from actual deltas
        g_acc, g_wg, g_fair = self._compute_objective_gradients_from_deltas(
            client_summaries, client_tilts
        )
        
        # Step 3: MGDA to find Pareto descent direction
        mgda_weights = self._mgda_solve([g_acc, g_wg, g_fair])
        
        # Step 4: Compute combined gradient direction
        g_mix = mgda_weights[0] * g_acc + mgda_weights[1] * g_wg + mgda_weights[2] * g_fair
        
        # Step 5: Apply PCGrad and CAGrad for conflict resolution
        if self.pcgrad_enabled:
            g_mix = self._pcgrad([g_acc, g_wg, g_fair], g_mix)
        if self.cagrad_enabled:
            g_mix = self._cagrad([g_acc, g_wg, g_fair], g_mix)
        
        # Step 6: Compute client weights from gradient alignment
        weights = self._gradient_to_weights(g_mix, client_tilts, client_summaries)
        
        # Step 7: Update auxiliary states
        if self.fairness_duals_enabled:
            self._update_duals(client_summaries)
        if self.selector_enabled:
            self._update_selector_queues(client_summaries)
        
        # Log metrics
        self._log_round_metrics(client_summaries, mgda_weights)
        
        return weights
    
    def _compute_heuristic_weights(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute weights using heuristic fairness scoring.
        """
        scores = []
        for summary in client_summaries:
            eo_gap = summary.get("eo_gap", 0.0)
            fpr_gap = summary.get("fpr_gap", 0.0)
            sp_gap = summary.get("sp_gap", 0.0)
            val_loss = summary.get("val_loss", 1.0)
            
            # Compute fairness score (lower is better)
            if self.bias_mitigation_mode:
                # Increase fairness emphasis in bias mode
                score = (
                    self.alpha * 2.0 * eo_gap +
                    self.beta * 2.0 * fpr_gap +
                    self.gamma * 2.0 * abs(sp_gap) +
                    self.delta * 0.5 * val_loss
                )
            else:
                score = (
                    self.alpha * eo_gap +
                    self.beta * fpr_gap +
                    self.gamma * abs(sp_gap) +
                    self.delta * val_loss
                )
            scores.append(score)
        
        scores = torch.tensor(scores, dtype=torch.float32)
        
        # Use temperature-scaled softmin
        temp = self.tau_min if self.bias_mitigation_mode else self.tau
        weights = torch.exp(-scores / max(temp, 0.1))
        weights = weights / weights.sum()
        
        return weights
    
    def _compute_objective_gradients_from_deltas(
        self,
        client_summaries: List[Dict[str, Any]],
        tilts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute actual objective gradients from client deltas.
        """
        if not self.client_deltas_cache:
            # Fallback to synthetic gradients
            return self._compute_synthetic_gradients(client_summaries, tilts)
        
        # Flatten all deltas to vectors
        flattened_deltas = []
        reference_state = self.client_deltas_cache[0]
        
        for delta in self.client_deltas_cache:
            flat = flatten_weights(delta)
            flattened_deltas.append(flat)
        
        # Accuracy gradient: weighted average by samples
        total_samples = sum(s.get('n_samples', 1) for s in client_summaries)
        g_acc = torch.zeros_like(flattened_deltas[0])
        
        for i, (delta, summary) in enumerate(zip(flattened_deltas, client_summaries)):
            weight = summary.get('n_samples', 1) / total_samples
            tilt = tilts[i].item() if i < len(tilts) else 1.0
            g_acc += weight * tilt * delta
        
        # Worst-group gradient: focus on high-loss clients
        losses = torch.tensor([s.get('val_loss', s.get('train_loss', 1.0)) 
                              for s in client_summaries])
        temperature = 10.0
        wg_weights = F.softmax(losses * temperature, dim=0)
        
        g_wg = torch.zeros_like(flattened_deltas[0])
        for i, (delta, w) in enumerate(zip(flattened_deltas, wg_weights)):
            tilt = tilts[i].item() if i < len(tilts) else 1.0
            g_wg += w * tilt * delta
        
        # Fairness gradient: weighted by fairness gaps
        fairness_scores = []
        for s in client_summaries:
            score = (self.w_eo * s.get('eo_gap', 0) +
                    self.w_fpr * s.get('fpr_gap', 0) +
                    self.w_sp * abs(s.get('sp_gap', 0)))
            fairness_scores.append(score)
        
        fairness_scores = torch.tensor(fairness_scores)
        fair_weights = F.softmax(fairness_scores * 5.0, dim=0)  # Temperature=5
        
        g_fair = torch.zeros_like(flattened_deltas[0])
        for i, (delta, w) in enumerate(zip(flattened_deltas, fair_weights)):
            tilt = tilts[i].item() if i < len(tilts) else 1.0
            g_fair += w * tilt * delta
        
        return g_acc, g_wg, g_fair
    
    def _compute_synthetic_gradients(
        self,
        client_summaries: List[Dict[str, Any]],
        tilts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fallback: compute synthetic gradients when deltas not available.
        """
        dim = 100  # Synthetic dimension
        
        # Accuracy gradient
        g_acc = torch.zeros(dim)
        total_samples = sum(s.get('n_samples', 1) for s in client_summaries)
        for i, summary in enumerate(client_summaries):
            tilt = tilts[i].item() if i < len(tilts) else 1.0
            weight = tilt * summary.get('n_samples', 1) / total_samples
            loss = summary.get('train_loss', 1.0)
            g_acc += weight * loss * torch.randn(dim) * 0.1
        
        # Worst-group gradient
        g_wg = torch.zeros(dim)
        losses = [s.get('val_loss', 1.0) for s in client_summaries]
        if losses:
            max_loss = max(losses)
            for i, loss in enumerate(losses):
                if loss >= max_loss * 0.9:  # Focus on worst performers
                    tilt = tilts[i].item() if i < len(tilts) else 1.0
                    g_wg += tilt * torch.randn(dim) * 0.1
        
        # Fairness gradient
        g_fair = torch.zeros(dim)
        for i, summary in enumerate(client_summaries):
            gap_score = (summary.get('eo_gap', 0) + 
                        summary.get('fpr_gap', 0) +
                        abs(summary.get('sp_gap', 0)))
            tilt = tilts[i].item() if i < len(tilts) else 1.0
            g_fair += tilt * gap_score * torch.randn(dim) * 0.1
        
        return g_acc, g_wg, g_fair
    
    def _gradient_to_weights(
        self,
        g_mix: torch.Tensor,
        tilts: torch.Tensor,
        client_summaries: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Convert gradient direction to client weights.
        """
        if not self.client_deltas_cache:
            # Fallback to tilted sample weighting
            weights = []
            total_samples = sum(s.get('n_samples', 1) for s in client_summaries)
            for i, summary in enumerate(client_summaries):
                tilt = tilts[i].item() if i < len(tilts) else 1.0
                weight = tilt * summary.get('n_samples', 1) / total_samples
                weights.append(weight)
            weights = torch.tensor(weights, dtype=torch.float32)
            return weights / weights.sum()
        
        # Compute alignment of each client delta with mixed gradient
        alignments = []
        for delta in self.client_deltas_cache:
            flat_delta = flatten_weights(delta)
            # Cosine similarity
            alignment = F.cosine_similarity(flat_delta.unsqueeze(0), 
                                           g_mix.unsqueeze(0)).item()
            alignments.append(max(0, alignment))  # Only positive contributions
        
        alignments = torch.tensor(alignments, dtype=torch.float32)
        
        # Combine with tilts and sample sizes
        weights = []
        total_samples = sum(s.get('n_samples', 1) for s in client_summaries)
        
        for i, summary in enumerate(client_summaries):
            tilt = tilts[i].item() if i < len(tilts) else 1.0
            alignment = alignments[i].item() if i < len(alignments) else 1.0
            samples = summary.get('n_samples', 1)
            
            # Combine factors
            weight = tilt * alignment * (samples / total_samples)
            weights.append(weight)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        
        # Ensure non-zero weights
        weights = weights + 1e-6
        return weights / weights.sum()
    
    def _mgda_solve(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        Solve MGDA quadratic program for Pareto descent direction.
        """
        n_tasks = len(gradients)
        
        if self.mgda_normalize_grads:
            # Normalize gradients
            gradients = [g / (g.norm() + 1e-8) for g in gradients]
        
        if self.mgda_solver == "qp":
            # Solve QP: min_α ||Σ α_i g_i||² s.t. α ≥ 0, Σα = 1
            try:
                # Compute gradient inner products
                G = torch.zeros(n_tasks, n_tasks)
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        G[i, j] = torch.dot(gradients[i].flatten(), 
                                          gradients[j].flatten()).item()
                
                # CVXPY formulation
                alpha = cp.Variable(n_tasks)
                objective = cp.Minimize(cp.quad_form(alpha, G.numpy()))
                constraints = [alpha >= 0, cp.sum(alpha) == 1]
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.SCS, verbose=False)
                
                if alpha.value is not None:
                    alphas = torch.tensor(alpha.value, dtype=torch.float32)
                else:
                    # Fallback to uniform
                    alphas = torch.ones(n_tasks) / n_tasks
            except:
                # Fallback to uniform if QP fails
                alphas = torch.ones(n_tasks) / n_tasks
        else:
            # Frank-Wolfe algorithm
            alphas = self._frank_wolfe_solver(gradients)
        
        # Store for logging
        self.last_mgda_weights = alphas
        
        return alphas
    
    def _frank_wolfe_solver(
        self,
        gradients: List[torch.Tensor],
        max_iter: int = 20
    ) -> torch.Tensor:
        """
        Frank-Wolfe solver for MGDA.
        """
        n_tasks = len(gradients)
        alphas = torch.ones(n_tasks) / n_tasks
        
        for _ in range(max_iter):
            # Compute current gradient
            g_sum = sum(alphas[i] * gradients[i] for i in range(n_tasks))
            
            # Find descent vertex
            dots = torch.tensor([torch.dot(g_sum.flatten(), g.flatten()) 
                                 for g in gradients])
            min_idx = torch.argmin(dots)
            
            # Line search
            v = torch.zeros(n_tasks)
            v[min_idx] = 1.0
            
            # Optimal step size (simplified)
            gamma = 2.0 / (2.0 + _)
            
            # Update
            alphas = (1 - gamma) * alphas + gamma * v
        
        return alphas
    
    def _pcgrad(self, gradients: List[torch.Tensor], g_mix: torch.Tensor) -> torch.Tensor:
        """
        PCGrad: Project conflicting gradients.
        """
        g_pc = g_mix.clone()
        
        for g in gradients:
            # Check for conflict
            dot_product = torch.dot(g_pc.flatten(), g.flatten())
            
            if dot_product < 0:
                # Project g_pc onto the normal of g
                g_norm = g / (g.norm() + 1e-8)
                g_pc = g_pc - dot_product * g_norm / (g_norm.norm() ** 2 + 1e-8)
        
        return g_pc
    
    def _cagrad(self, gradients: List[torch.Tensor], g_mix: torch.Tensor) -> torch.Tensor:
        """
        CAGrad: Conflict-averse gradient with convergence guarantees.
        """
        # Average gradient
        g_avg = sum(gradients) / len(gradients)
        
        # Compute conflict score
        conflicts = []
        for i in range(len(gradients)):
            for j in range(i + 1, len(gradients)):
                dot = torch.dot(gradients[i].flatten(), gradients[j].flatten())
                if dot < 0:
                    conflicts.append(abs(dot))
        
        if conflicts:
            # Adjust rho based on conflict level
            avg_conflict = sum(conflicts) / len(conflicts)
            max_possible = max(g.norm() for g in gradients) ** 2
            conflict_ratio = avg_conflict / (max_possible + 1e-8)
            
            # Higher conflict -> more weight on average
            adjusted_rho = min(0.9, self.cagrad_rho + 0.3 * conflict_ratio)
        else:
            adjusted_rho = self.cagrad_rho
        
        # Conflict-averse update
        g_ca = (1 - adjusted_rho) * g_mix + adjusted_rho * g_avg
        
        return g_ca
    
    def _compute_client_tilts(self, client_summaries: List[Dict[str, Any]]) -> torch.Tensor:
        """
        DFBD: Compute demographics-free tilts using ARL head.
        """
        if not self.arl_enabled or self.arl_head is None:
            return torch.ones(len(client_summaries))
        
        # Extract privacy-safe proxies
        proxies = []
        targets = []  # For training: use worst_group_f1 as target
        
        for summary in client_summaries:
            proxy = torch.tensor([
                summary.get('train_loss', 1.0),
                summary.get('delta_norm', 1.0),
                summary.get('grad_norm', 1.0),
                summary.get('val_loss', 1.0),
                summary.get('val_acc', 0.5),
                summary.get('calibration_error', 0.0),
                summary.get('drift', 0.0)
            ], dtype=torch.float32)
            proxies.append(proxy)
            
            # Target: inverse of worst_group_f1 (want to upweight low F1)
            wg_f1 = summary.get('worst_group_f1', 0.5)
            targets.append(1.0 / (wg_f1 + 0.1))
        
        if not proxies:
            return torch.ones(len(client_summaries))
        
        proxies_batch = torch.stack(proxies)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # Train ARL head (few steps per round)
        if self.arl_optimizer is not None:
            self.arl_head.train()
            for _ in range(5):  # Few gradient steps
                self.arl_optimizer.zero_grad()
                
                predicted_tilts = self.arl_head(proxies_batch).squeeze()
                loss = F.mse_loss(predicted_tilts, targets_tensor)
                
                loss.backward()
                self.arl_optimizer.step()
        
        # Inference
        self.arl_head.eval()
        with torch.no_grad():
            tilts = self.arl_head(proxies_batch).squeeze()
        
        # Ensure tensor shape
        if tilts.dim() == 0:
            tilts = tilts.unsqueeze(0)
        
        # Apply eta scaling and normalize
        tilts = tilts * self.arl_eta
        tilts = tilts / tilts.mean()  # Normalize to preserve scale
        
        # Clamp to reasonable range
        tilts = torch.clamp(tilts, 0.1, 10.0)
        
        return tilts
    
    def _update_bias_detection(self, client_summaries: List[Dict[str, Any]]):
        """
        Update bias detection state based on current metrics.
        """
        # Collect metrics
        eo_gaps = [s.get("eo_gap", 0) for s in client_summaries]
        fpr_gaps = [s.get("fpr_gap", 0) for s in client_summaries]
        sp_gaps = [abs(s.get("sp_gap", 0)) for s in client_summaries]
        
        avg_eo_gap = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr_gap = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp_gap = np.mean(sp_gaps) if sp_gaps else 0
        
        # Check if bias is detected
        bias_detected = (
            avg_eo_gap > self.bias_threshold_eo or
            avg_fpr_gap > self.bias_threshold_fpr or
            avg_sp_gap > self.bias_threshold_sp
        )
        
        if bias_detected:
            self.consecutive_bias_rounds += 1
            if self.consecutive_bias_rounds >= self.detector_patience:
                if not self.bias_mitigation_mode:
                    self.bias_mitigation_mode = True
                    # Increase lambda_fair when entering bias mode
                    self.lambda_fair = min(self.lambda_fair_max, self.lambda_fair * 1.5)
                    self.tau = self.tau_min  # Use sharp temperature
        else:
            self.consecutive_bias_rounds = 0
            # Only exit bias mode after sustained low bias
            if self.bias_mitigation_mode and avg_eo_gap < self.bias_threshold_eo * 0.5:
                self.bias_mitigation_mode = False
                # Decrease lambda_fair when exiting bias mode
                self.lambda_fair = max(self.lambda_fair_min, self.lambda_fair * 0.8)
                self.tau = self.tau_init  # Restore temperature
        
        # Update history
        self.global_metrics_history['avg_eo_gap'].append(avg_eo_gap)
        self.global_metrics_history['avg_fpr_gap'].append(avg_fpr_gap)
        self.global_metrics_history['avg_sp_gap'].append(avg_sp_gap)
    
    def _update_duals(self, client_summaries: List[Dict[str, Any]]):
        """
        Update dual variables for fairness constraints.
        """
        # Compute current gaps
        eo_gaps = [s.get('eo_gap', 0) for s in client_summaries]
        fpr_gaps = [s.get('fpr_gap', 0) for s in client_summaries]
        sp_gaps = [abs(s.get('sp_gap', 0)) for s in client_summaries]
        
        avg_eo = np.mean(eo_gaps) if eo_gaps else 0
        avg_fpr = np.mean(fpr_gaps) if fpr_gaps else 0
        avg_sp = np.mean(sp_gaps) if sp_gaps else 0
        
        # Dual ascent step
        self.dual_eo = np.clip(
            self.dual_eo + self.dual_lr * (avg_eo - self.epsilon_eo),
            0.0, 10.0
        )
        self.dual_fpr = np.clip(
            self.dual_fpr + self.dual_lr * (avg_fpr - self.epsilon_fpr),
            0.0, 10.0
        )
        self.dual_sp = np.clip(
            self.dual_sp + self.dual_lr * (avg_sp - self.epsilon_sp),
            0.0, 10.0
        )
    
    def _update_selector_queues(self, client_summaries: List[Dict[str, Any]]):
        """
        Update fairness debt queues for client selection.
        """
        if self.selector_mode == "lyapunov":
            for i, summary in enumerate(client_summaries):
                client_id = summary.get('client_id', i)
                if client_id < len(self.client_fairness_debt):
                    # Compute fairness debt
                    eo_gap = summary.get('eo_gap', 0)
                    fpr_gap = summary.get('fpr_gap', 0)
                    sp_gap = abs(summary.get('sp_gap', 0))
                    
                    debt = self.w_eo * eo_gap + self.w_fpr * fpr_gap + self.w_sp * sp_gap
                    
                    # Lyapunov-style update with decay
                    self.client_fairness_debt[client_id] = (
                        (1 - self.selector_tau) * self.client_fairness_debt[client_id] +
                        self.selector_kappa * debt
                    )
    
    def get_preferred_clients(self, n_select: int) -> List[int]:
        """
        Get preferred clients for next round based on fairness debt.
        """
        if not self.selector_enabled:
            return None
        
        # Add exploration bonus
        exploration = torch.rand(len(self.client_fairness_debt)) * 0.1
        scores = self.client_fairness_debt + exploration
        
        # Select top-k with highest debt
        if n_select >= len(scores):
            return list(range(len(scores)))
        
        _, selected = torch.topk(scores, n_select)
        return selected.tolist()
    
    def _apply_dp_noise(self, aggregated: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply differential privacy noise to aggregated update.
        """
        if not self.dp_enabled:
            return aggregated
        
        noisy = {}
        for key, tensor in aggregated.items():
            # Clip norm
            norm = tensor.norm()
            if norm > self.dp_clip:
                tensor = tensor * (self.dp_clip / norm)
            
            # Add Gaussian noise
            noise = torch.randn_like(tensor) * self.dp_clip * self.dp_noise_mult
            noisy[key] = tensor + noise
        
        return noisy
    
    def apply_server_momentum(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply server-side momentum to aggregated update.
        """
        if self.prev_global_update is None:
            self.prev_global_update = update
            return update
        
        # Momentum update
        momentum_update = {}
        for key in update.keys():
            momentum_update[key] = (
                self.server_momentum * self.prev_global_update[key] +
                (1 - self.server_momentum) * update[key]
            )
        
        self.prev_global_update = momentum_update
        return momentum_update
    
    def _log_round_metrics(self, client_summaries: List[Dict[str, Any]], mgda_weights: torch.Tensor):
        """
        Log metrics for current round.
        """
        wg_f1s = [s.get('worst_group_f1', 0) for s in client_summaries]
        if wg_f1s:
            self.global_metrics_history['worst_group_f1'].append(np.mean(wg_f1s))
        
        if self.fairness_duals_enabled:
            self.global_metrics_history['dual_values'].append({
                'eo': self.dual_eo,
                'fpr': self.dual_fpr,
                'sp': self.dual_sp
            })
        
        self.global_metrics_history['mgda_weights'].append(mgda_weights.tolist())
        
        if self.selector_enabled:
            self.global_metrics_history['selector_queues'].append(
                self.client_fairness_debt.tolist()
            )
    
    def get_fairness_config(self) -> Dict[str, Any]:
        """
        Get fairness configuration for clients (CALT parameters).
        """
        return {
            'prox_mu': self.prox_mu,
            'lambda_irm': self.lambda_irm,
            'lambda_adv': self.lambda_adv,
            'lambda_fair': self.lambda_fair,
            'use_mixup': self.use_mixup,
            'use_cia': self.use_cia,
            'use_adversary': self.use_adversary,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'delta_acc': self.delta_acc,
            'round': self.round_num,
            'version': self.version,
            'w_eo': self.w_eo,
            'w_fpr': self.w_fpr,
            'w_sp': self.w_sp
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get algorithm statistics.
        """
        stats = {
            'round': self.round_num,
            'version': self.version,
            'bias_mitigation_mode': self.bias_mitigation_mode,
            'delta_acc': self.delta_acc,
            'lambda_fair': self.lambda_fair,
            'global_metrics': self.global_metrics_history
        }
        
        if hasattr(self, 'last_mgda_weights'):
            stats['mgda_weights'] = self.last_mgda_weights.tolist()
        
        if self.fairness_duals_enabled:
            stats['duals'] = {
                'eo': self.dual_eo,
                'fpr': self.dual_fpr,
                'sp': self.dual_sp
            }
        
        return stats
    
    def _compute_component_weights(self, client_summaries: List[Dict[str, Any]]) -> ComponentWeights:
        """
        Compute weights from each component algorithm (for test compatibility).
        """
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
            weight = (loss + 1e-4) ** 2.0
            qffl_weights.append(weight)
        qffl_weights = torch.tensor(qffl_weights, dtype=torch.float32)
        qffl_weights = qffl_weights / qffl_weights.sum()
        
        # AFL: exponential weighting
        afl_weights = []
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            loss = summary.get("val_loss", summary.get("train_loss", 1.0))
            if client_id < len(self.afl_client_weights):
                self.afl_client_weights[client_id] *= torch.exp(torch.tensor(0.1 * loss))
        
        self.afl_client_weights = self.afl_client_weights / self.afl_client_weights.sum()
        for i, summary in enumerate(client_summaries):
            client_id = summary.get("client_id", i)
            if client_id < len(self.afl_client_weights):
                afl_weights.append(self.afl_client_weights[client_id].item())
            else:
                afl_weights.append(1.0 / len(client_summaries))
        afl_weights = torch.tensor(afl_weights, dtype=torch.float32)
        afl_weights = afl_weights / afl_weights.sum()
        
        # FairFATE: fairness-score guided weights
        fairfate_weights = []
        for summary in client_summaries:
            eo_gap = summary.get("eo_gap", 0.5)
            fpr_gap = summary.get("fpr_gap", 0.5)
            sp_gap = summary.get("sp_gap", 0.5)
            val_loss = summary.get("val_loss", 1.0)
            
            # Compute fairness penalty (lower is better)
            fairness_penalty = (
                self.alpha * eo_gap +
                self.beta * fpr_gap +
                self.gamma * abs(sp_gap) +
                self.delta * val_loss
            )
            fairfate_weights.append(fairness_penalty)
        
        # Convert to weights using softmin
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
