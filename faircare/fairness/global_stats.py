from __future__ import annotations
from typing import Dict, Any, List

class GlobalStats:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def update_and_summarize(self, report: Dict[str, Any]) -> Dict[str, Any]:
        self.history.append(report)
        # compute moving averages if desired; return last report enriched
        return report
