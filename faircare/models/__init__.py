
"""Neural network models for federated learning."""

from faircare.models.classifier import (
    MLP,
    LogisticRegression,
    create_model
)
from faircare.models.adversary import AdversarialDebiaser

__all__ = [
    "MLP",
    "LogisticRegression",
    "create_model",
    "AdversarialDebiaser"
]
