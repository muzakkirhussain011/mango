from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd

def summarize_metrics(history: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(history)
