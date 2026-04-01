import numpy as np

from darkhunter_rv.binary_rv import fit_circular_binary, keplerian_circular


def test_fit_circular_binary_recovers_params():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 30)
    true = dict(gamma=10.0, k=5.0, t0=1.2, p=3.4)
    rv = keplerian_circular(t, true["gamma"], true["k"], true["t0"], true["p"]) + rng.normal(0, 0.2, size=t.size)
    err = np.full_like(rv, 0.2)

    res = fit_circular_binary(t, rv, err, period_days=true["p"])
    assert res.success
    assert abs(res.gamma - true["gamma"]) < 1.0
    assert abs(res.k - true["k"]) < 1.0
