"""FairCare: Fair Federated Learning for Healthcare."""

__version__ = "0.1.0"

from faircare.core import Client, Server
from faircare.algos import REGISTRY as ALGO_REGISTRY
from faircare.fairness import fairness_report, group_confusion_counts

__all__ = [
    "Client",
    "Server", 
    "ALGO_REGISTRY",
    "fairness_report",
    "group_confusion_counts",
]
