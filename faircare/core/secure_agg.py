# faircare/core/secure_agg.py
from typing import List, Dict

def secure_sum_dicts(dicts: List[Dict[str, int]]) -> Dict[str, int]:
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


