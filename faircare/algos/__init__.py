"""Federated learning algorithms."""

from faircare.algos.aggregator import Aggregator, REGISTRY, make_aggregator
from faircare.algos.fedavg import FedAvgAggregator
from faircare.algos.fedprox import FedProxAggregator
from faircare.algos.qffl import QFFLAggregator
from faircare.algos.afl import AFLAggregator
from faircare.algos.fairfate import FairFateAggregator
from faircare.algos.fairfed import FairFedAggregator

__all__ = [
    "Aggregator",
    "REGISTRY",
    "make_aggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
    "QFFLAggregator",
    "AFLAggregator",
    "FairFateAggregator",
    "FairFedAggregator"
]
