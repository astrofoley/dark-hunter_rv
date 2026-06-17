import numpy as np

from darkhunter_rv import rv_keplerian_plots as plots


def test_residual_ylim_caps_at_five_kms() -> None:
    y = np.array([20.0, -18.0])
    yerr = np.array([1.0, 1.0])
    lo, hi = plots._residual_ylim(y, yerr, [])
    assert hi <= 5.0
    assert lo >= -5.0


def test_xlim_uses_date_bounds_when_mjds_are_stale() -> None:
    from astropy.time import Time

    from darkhunter_rv.apf_observability import PLOT_HORIZON_DAYS

    now_mjd = float(Time("2026-06-17T14:00:00", scale="utc").mjd)
    t = np.array([60650.0, 61100.0])
    report = {
        "now_mjd": now_mjd,
        "observability_window": {
            "windows": [
                {
                    "start_date": "2026-06-17",
                    "end_date": "2027-06-17",
                    "start_mjd": float(Time("2026-06-17T20:00:00", scale="utc").mjd),
                    "end_mjd": float(Time("2026-06-18T06:00:00", scale="utc").mjd),
                }
            ]
        },
    }
    lo, hi = plots._xlim_from_data(t, report)
    assert hi > now_mjd + 30.0
    assert hi <= now_mjd + float(PLOT_HORIZON_DAYS) + 1.0


def test_xlim_from_data_caps_future_at_plot_horizon() -> None:
    from astropy.time import Time

    from darkhunter_rv.apf_observability import PLOT_HORIZON_DAYS

    now_mjd = float(Time("2026-06-12").mjd)
    t = np.array([60650.0, 61200.0])
    report = {
        "now_mjd": now_mjd,
        "observability_window": {
            "windows": [
                {
                    "start_date": "2026-06-17",
                    "end_date": "2027-06-17",
                    "start_mjd": float(Time("2026-06-17").mjd),
                    "end_mjd": float(Time("2027-06-17").mjd),
                }
            ]
        },
    }
    lo, hi = plots._xlim_from_data(t, report)
    assert hi <= now_mjd + float(PLOT_HORIZON_DAYS) + 1.0


def test_xlim_from_data_spans_epochs() -> None:
    t = np.array([58000.0, 58100.0])
    report = {"now_mjd": 59000.0}
    lo, hi = plots._xlim_from_data(t, report)
    assert lo < 58000.0
    assert hi > 58100.0


def test_xlim_from_data_single_epoch() -> None:
    now_mjd = 61200.0
    t = np.array([60900.0])
    report = {"now_mjd": now_mjd}
    lo, hi = plots._xlim_from_data(t, report)
    assert lo < 60900.0
    assert hi > 60900.0
    assert hi <= now_mjd + 91.0


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
