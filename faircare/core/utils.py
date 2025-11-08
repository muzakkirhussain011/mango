"""Utility functions for federated learning."""
from typing import Optional, Dict, Any, List, Tuple

import random
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import json
from pathlib import Path
import logging
from datetime import datetime


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def seed_context(seed: int):
    """Context manager for temporary seed setting."""
    old_random_state = random.getstate()
    old_np_state = np.random.get_state()
    old_torch_state = torch.get_rng_state()
    
    try:
        set_seed(seed)
        yield
    finally:
        random.setstate(old_random_state)
        np.random.set_state(old_np_state)
        torch.set_rng_state(old_torch_state)


def average_weights(weights_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Average model weights."""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = torch.stack([w[key] for w in weights_list]).mean(dim=0)
    return avg_weights


def weighted_average_weights(
    weights_list: List[Dict[str, torch.Tensor]], 
    coefficients: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute weighted average of model weights."""
    avg_weights = {}
    coefficients = coefficients.view(-1, 1)
    
    for key in weights_list[0].keys():
        stacked = torch.stack([w[key].float() for w in weights_list])
        # Reshape coefficients to match tensor dimensions
        shape = [len(coefficients)] + [1] * (stacked.dim() - 1)
        coef = coefficients.view(shape)
        avg_weights[key] = (stacked * coef).sum(dim=0)
    
    return avg_weights


def flatten_weights(weights: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten model weights to single vector."""
    return torch.cat([w.flatten() for w in weights.values()])


def unflatten_weights(
    flat_weights: torch.Tensor, 
    reference: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Unflatten vector to model weights dictionary."""
    weights = {}
    idx = 0
    for key, ref_tensor in reference.items():
        num_params = ref_tensor.numel()
        weights[key] = flat_weights[idx:idx + num_params].view(ref_tensor.shape)
        idx += num_params
    return weights


def compute_model_delta(
    weights_new: Dict[str, torch.Tensor],
    weights_old: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Compute difference between two model states."""
    return {k: weights_new[k] - weights_old[k] for k in weights_new.keys()}


def apply_model_delta(
    weights: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    lr: float = 1.0
) -> Dict[str, torch.Tensor]:
    """Apply delta update to model weights."""
    return {k: weights[k] + lr * delta[k] for k in weights.keys()}


def sample_clients(
    n_total: int,
    n_sample: int,
    availability: Optional[List[float]] = None,
    priority_scores: Optional[List[float]] = None,
    priority_fraction: float = 0.3,
    seed: Optional[int] = None
) -> List[int]:
    """Sample clients with optional priority-mixture (FedFair³ approach).

    Args:
        n_total: Total number of clients
        n_sample: Number of clients to sample
        availability: Client availability weights (optional)
        priority_scores: Priority scores for each client (optional)
            Higher score = higher priority (based on loss, fairness gaps, recency)
        priority_fraction: Fraction of samples from top-priority set (default 0.3)
        seed: Random seed for reproducibility

    Returns:
        List of selected client indices
    """
    if seed is not None:
        np.random.seed(seed)

    # Simple uniform or availability-weighted sampling
    if priority_scores is None:
        if availability is None:
            return np.random.choice(n_total, n_sample, replace=False).tolist()
        else:
            probs = np.array(availability) / np.sum(availability)
            return np.random.choice(n_total, n_sample, replace=False, p=probs).tolist()

    # Priority-mixture selection (FedFair³ approach)
    priority_scores = np.array(priority_scores)

    # Apply availability mask if provided
    if availability is not None:
        available_mask = np.array(availability) > 0
        priority_scores = priority_scores * available_mask

    # Determine split between priority and random
    n_priority = max(1, int(n_sample * priority_fraction))
    n_random = n_sample - n_priority

    # Select top-priority clients
    priority_indices = np.argsort(priority_scores)[-n_priority*2:]  # Top 2x for diversity
    priority_selected = np.random.choice(
        priority_indices,
        size=n_priority,
        replace=False
    ).tolist()

    # Select remaining from non-priority pool
    remaining_indices = [i for i in range(n_total) if i not in priority_selected]
    if len(remaining_indices) >= n_random:
        random_selected = np.random.choice(
            remaining_indices,
            size=n_random,
            replace=False
        ).tolist()
    else:
        random_selected = remaining_indices

    return priority_selected + random_selected


class Logger:
    """Simple logger for experiments."""
    
    def __init__(self, logdir: str, name: str = "experiment"):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.name = name
        
        # Setup file logging
        self.log_file = self.logdir / "console.txt"
        self.metrics_file = self.logdir / "metrics.jsonl"
        
        # Setup console logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # File handler
        self.file_handler = logging.FileHandler(self.log_file)
        self.file_handler.setLevel(logging.INFO)

        # Console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to JSONL file."""
        metrics_with_meta = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics_with_meta) + "\n")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Save configuration."""
        config_file = self.logdir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str)

    def close(self) -> None:
        """Close all file handlers to release file locks."""
        if hasattr(self, 'file_handler'):
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
        if hasattr(self, 'console_handler'):
            self.console_handler.close()
            self.logger.removeHandler(self.console_handler)


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    optimizer_type: str = "sgd"
) -> torch.optim.Optimizer:
    """Create optimizer for model."""
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
    elif optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
