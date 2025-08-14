
"""FairCare: Fair Federated Learning for Healthcare."""

__version__ = "0.1.0"

# Import after submodule init files are loaded
from faircare.core.client import Client
from faircare.core.server import Server
from faircare.algos.aggregator import REGISTRY as ALGO_REGISTRY
from faircare.fairness.metrics import fairness_report, group_confusion_counts

__all__ = [
    "Client",
    "Server", 
    "ALGO_REGISTRY",
    "fairness_report",
    "group_confusion_counts",
]
