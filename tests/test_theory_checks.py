from faircare.fairness.metrics import fairness_report

def test_composite_gap():
    rep = {"g0_tp":5,"g0_fp":5,"g0_fn":0,"g0_tn":0,"g0_n":10,
           "g1_tp":1,"g1_fp":1,"g1_fn":8,"g1_tn":0,"g1_n":10}
    out = fairness_report(rep)
    assert out["max_group_gap"] >= max(out["EO_gap"], out["FPR_gap"], out["SP_gap"])
