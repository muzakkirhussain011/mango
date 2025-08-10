from faircare.algos.faircare_fl import make_aggregator

def test_weights_sum_to_one():
    agg = make_aggregator(sens_present=True)
    w = agg.compute_weights([{"factor":1.0},{"factor":0.5},{"factor":2.0}])
    assert abs(sum(w) - 1.0) < 1e-6
    assert all(ww > 0 for ww in w)
