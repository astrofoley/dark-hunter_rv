import numpy as np

from darkhunter_rv import rv_keplerian_plots as plots


def test_residual_ylim_caps_at_five_kms() -> None:
    y = np.array([20.0, -18.0])
    yerr = np.array([1.0, 1.0])
    lo, hi = plots._residual_ylim(y, yerr, [])
    assert hi <= 5.0
    assert lo >= -5.0


def test_xlim_from_data_spans_epochs() -> None:
    t = np.array([58000.0, 58100.0])
    report = {"now_mjd": 59000.0}
    lo, hi = plots._xlim_from_data(t, report)
    assert lo < 58000.0
    assert hi > 58100.0


def test_variant_param_lines_includes_m2sini() -> None:
    fit_variants = {
        "free": (
            np.zeros(6),
            {"P_days": 100.0, "e": 0.1, "K_kms": 50.0, "mass_function_msun": 0.01},
        ),
    }
    lines = plots._variant_param_lines(fit_variants, m1_msun=1.0)
    assert len(lines) == 1
    assert "RV only" in lines[0]
    assert "M₂ sin i" in lines[0]
