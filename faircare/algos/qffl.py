from __future__ import annotations
from .aggregator import BaseAggregator

def make_aggregator(sens_present: bool) -> BaseAggregator:
    # q-FFL scales local loss; server uses FedAvg weights
    return BaseAggregator(sens_present)
