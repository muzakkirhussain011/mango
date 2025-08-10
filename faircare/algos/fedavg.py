from __future__ import annotations
from .aggregator import BaseAggregator

def make_aggregator(sens_present: bool) -> BaseAggregator:
    return BaseAggregator(sens_present)
