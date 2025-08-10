from __future__ import annotations
from .aggregator import BaseAggregator

def make_aggregator(sens_present: bool) -> BaseAggregator:
    # Momentum is applied in Server via FairMomentumAggregator; here weights uniform
    return BaseAggregator(sens_present)
