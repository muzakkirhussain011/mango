# faircare/fairness/global_stats.py
import numpy as np

def aggregate_group_stats(list_of_stats):
    agg = {}
    for stats in list_of_stats:
        for g, d in stats.items():
            if g not in agg:
                agg[g] = {"tp":0,"fp":0,"fn":0,"tn":0,"n":0}
            for k in agg[g]:
                agg[g][k] += int(d.get(k,0))
    return agg

def eo_gap_from_stats(agg_stats):
    tprs = []
    for g, d in agg_stats.items():
        denom = d["tp"] + d["fn"]
        tprs.append(d["tp"]/max(1,denom))
    return float(max(tprs)-min(tprs)) if tprs else 0.0

def dp_gap_from_stats(agg_stats):
    srs = []
    for g, d in agg_stats.items():
        denom = d["tp"]+d["fp"]+d["tn"]+d["fn"]
        srs.append((d["tp"]+d["fp"])/max(1,denom))
    return float(max(srs)-min(srs)) if srs else 0.0
