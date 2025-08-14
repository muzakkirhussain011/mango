
"""Core federated learning components."""

from faircare.core.client import Client
from faircare.core.server import Server
from faircare.core.secure_agg import SecureAggregator
from faircare.core.trainer import run_experiment
from faircare.core.evaluation import Evaluator
from faircare.core.utils import (
    set_seed,
    seed_context,
    Logger,
    sample_clients,
    average_weights,
    weighted_average_weights,
    compute_model_delta,
    apply_model_delta
)

__all__ = [
    "Client",
    "Server",
    "SecureAggregator",
    "run_experiment",
    "Evaluator",
    "set_seed",
    "seed_context",
    "Logger",
    "sample_clients",
    "average_weights",
    "weighted_average_weights",
    "compute_model_delta",
    "apply_model_delta"
]
