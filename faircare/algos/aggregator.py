"""Base aggregator and registry."""

from typing import Protocol, Dict, List, Any, Optional
import torch
from abc import ABC, abstractmethod


class Aggregator(Protocol):
    """Aggregator protocol."""
    
    def client_weights_signal(self, n_clients: int) -> List[float]:
        """Signal for client sampling weights."""
        ...
    
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """Compute aggregation weights from client summaries."""
        ...
    
    def aggregate(
        self,
        client_updates: List[torch.Tensor],
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate client updates with weights."""
        ...


class BaseAggregator(ABC):
    """Base aggregator implementation."""
    
    def __init__(self, n_clients: int, **kwargs):
        self.n_clients = n_clients
        self.round = 0
    
    def client_weights_signal(self, n_clients: int) -> List[float]:
        """Default uniform sampling."""
        return [1.0 / n_clients] * n_clients
    
    @abstractmethod
    def compute_weights(self, client_summaries: List[Dict]) -> torch.Tensor:
        """Compute aggregation weights."""
        pass
    
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Standard weighted aggregation."""
        weights = weights / weights.sum()
        
        aggregated = {}
        for key in client_updates[0].keys():
            stacked = torch.stack([update[key] for update in client_updates])
            # Reshape weights for broadcasting
            shape = [len(weights)] + [1] * (stacked.dim() - 1)
            w = weights.view(shape)
            aggregated[key] = (stacked * w).sum(dim=0)
        
        self.round += 1
        return aggregated


# Algorithm registry
REGISTRY = {}


def register_aggregator(name: str):
    """Decorator to register aggregator."""
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator


def make_aggregator(
    name: str,
    n_clients: int,
    fairness_config: Optional[Dict] = None,
    algo_config: Optional[Dict] = None,
    **kwargs
) -> Aggregator:
    """Create aggregator by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown aggregator: {name}. Available: {list(REGISTRY.keys())}")
    
    cls = REGISTRY[name]
    
    # Merge configs
    all_kwargs = {"n_clients": n_clients}
    if fairness_config:
        all_kwargs.update(fairness_config)
    if algo_config:
        all_kwargs.update(algo_config)
    all_kwargs.update(kwargs)
    
    return cls(**all_kwargs)
