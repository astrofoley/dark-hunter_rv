"""Tests for fit_apf_rv_keplerian summary parsing (no network)."""

from pathlib import Path

import numpy as np
import pytest

import fit_apf_rv_keplerian as fitmod


def _write_star_summary(path: Path, *, period: float = 12.5, ecc: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "### STAR SUMMARY: 958479989998172288 ###\n\n"
        "[GAIA METADATA]\n"
        "Source_ID: 958479989998172288\n"
        "RA: 180.5\n"
        "Dec: -45.25\n"
        "NSS_Solution_Type: Orbital\n"
        f"Period: {period:.6f}\n"
        f"Eccentricity: {ecc:.6f}\n"
        "\n[EXTERNAL RV DATA]\n"
        "# No external data found.\n"
        "\n[PIPELINE RESULTS]\n"
        "# File | MJD | RV (km/s) | Err (km/s) | RMS | Fallback?\n"
        "Gaia_DR3_958479989998172288_epoch_1.txt 60000.1 -10.5 0.02 0.4 False\n"
        "Gaia_DR3_958479989998172288_epoch_2.txt 60012.3 -11.2 0.02 0.5 False\n"
        "Gaia_DR3_958479989998172288_epoch_3.txt 60024.0 -9.8 0.02 0.4 False\n"
        "Gaia_DR3_958479989998172288_epoch_4.txt 60036.2 -10.1 0.02 0.45 False\n"
        "Gaia_DR3_958479989998172288_epoch_5.txt 60048.5 -11.0 0.02 0.5 False\n"
        "Gaia_DR3_958479989998172288_epoch_6.txt 60060.0 -10.3 0.02 0.42 False\n"
        "Gaia_DR3_958479989998172288_epoch_7.txt 60072.4 -10.8 0.02 0.44 False\n"
    )


def test_parse_summary_pipeline_block(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_958479989998172288" / "Gaia_DR3_958479989998172288_summary.txt"
    _write_star_summary(summ)
    pts = fitmod.parse_summary(summ)
    assert len(pts) == 7
    assert pts[0].mjd == pytest.approx(60000.1)
    assert pts[0].rv == pytest.approx(-10.5)


def test_parse_object_id_nested_path(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_958479989998172288" / "Gaia_DR3_958479989998172288_summary.txt"
    _write_star_summary(summ)
    assert fitmod.parse_object_id_from_summary(summ) == "958479989998172288"


def test_load_nss_priors_from_summary_no_network(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_958479989998172288" / "Gaia_DR3_958479989998172288_summary.txt"
    _write_star_summary(summ, period=45.0, ecc=0.15)
    priors = fitmod.load_nss_priors_from_summary(summ)
    assert priors is not None
    assert priors["period_days"] == pytest.approx(45.0)
    assert priors["eccentricity"] == pytest.approx(0.15)


def test_discover_summary_files_nested_only(tmp_path: Path) -> None:
    _write_star_summary(tmp_path / "Gaia_DR3_111" / "Gaia_DR3_111_summary.txt")
    found = fitmod.discover_summary_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "Gaia_DR3_111_summary.txt"


def test_discover_prefers_flat_summary_over_nested_stub(tmp_path: Path) -> None:
    sid = "1702370142434513152"
    nested = tmp_path / f"Gaia_DR3_{sid}" / f"Gaia_DR3_{sid}_summary.txt"
    flat = tmp_path / f"Gaia_DR3_{sid}_summary.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text(
        "[GAIA METADATA]\nSource_ID: 1702370142434513152\nRA: 1.0\nDec: 2.0\n"
        "\n[PIPELINE RESULTS]\n# hdr\n"
        + "\n".join(
            f"Gaia_DR3_{sid}_epoch_{i}.txt {60000+i} -1.0 0.1 0.2 False" for i in range(1, 4)
        )
        + "\n"
    )
    flat.write_text(
        "[GAIA METADATA]\nSource_ID: 1702370142434513152\nRA: 1.0\nDec: 2.0\n"
        "\n[PIPELINE RESULTS]\n# hdr\n"
        + "\n".join(
            f"Gaia_DR3_{sid}_epoch_{i}.txt {60000+i} -1.0 0.1 0.2 False" for i in range(1, 21)
        )
        + "\n"
    )
    found = fitmod.discover_summary_files(tmp_path)
    assert len(found) == 1
    assert found[0].resolve() == flat.resolve()
    assert fitmod.count_pipeline_rows(found[0]) == 20


def test_discover_prefers_gaia_dr3_name_over_legacy_numeric(tmp_path: Path) -> None:
    sid = "77413727493690112"
    legacy = tmp_path / f"{sid}_summary.txt"
    gaia = tmp_path / f"Gaia_DR3_{sid}_summary.txt"
    body = (
        "[GAIA METADATA]\nSource_ID: 77413727493690112\nRA: 1.0\nDec: 2.0\n"
        "\n[PIPELINE RESULTS]\n# hdr\n"
        + "\n".join(
            f"Gaia_DR3_{sid}_epoch_{i}.txt {60000+i} -1.0 0.1 0.2 False" for i in range(1, 11)
        )
        + "\n"
    )
    legacy.write_text(body)
    gaia.write_text(body)
    found = fitmod.discover_summary_files(tmp_path)
    assert len(found) == 1
    assert found[0].resolve() == gaia.resolve()


def test_parse_summary_counts_nan_rv_epochs(tmp_path: Path) -> None:
    flat = tmp_path / "Gaia_DR3_1702370142434513152_summary.txt"
    flat.write_text(
        "[PIPELINE RESULTS]\n# File | MJD | RV | Err | RMS\n"
        "Gaia_DR3_1702370142434513152_epoch_1.txt 60613.38 -9.74 0.97 0.90 False\n"
        "Gaia_DR3_1702370142434513152_epoch_2.txt 60643.39 nan nan 0.94 False\n"
        "Gaia_DR3_1702370142434513152_epoch_3.txt 60654.21 nan nan 1.22 False\n"
        "Gaia_DR3_1702370142434513152_epoch_4.txt 60663.20 33.05 2.45 1.11 False\n"
        "Gaia_DR3_1702370142434513152_epoch_5.txt 60685.55 14.99 2.45 1.53 False\n"
        "Gaia_DR3_1702370142434513152_epoch_6.txt 60693.20 7.39 2.45 1.23 False\n"
        "Gaia_DR3_1702370142434513152_epoch_7.txt 60702.20 -1.65 2.48 1.28 False\n"
        "Gaia_DR3_1702370142434513152_epoch_8.txt 60716.36 -9.54 0.72 1.40 False\n"
    )
    pts = fitmod.parse_summary(flat)
    assert len(pts) == 6
    assert all(np.isfinite(p.rv) for p in pts)


def test_report_stem_uses_numeric_id(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_958479989998172288" / "Gaia_DR3_958479989998172288_summary.txt"
    _write_star_summary(summ)
    assert fitmod.report_stem(summ, "958479989998172288") == "958479989998172288"


def test_run_one_skips_all_nan_rv(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_999" / "Gaia_DR3_999_summary.txt"
    summ.parent.mkdir(parents=True)
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 999\nRA: 1.0\nDec: 2.0\n"
        "\n[PIPELINE RESULTS]\n"
        "# File | MJD | RV\n"
        + "\n".join(f"ep_{i}.txt {60000+i} nan 0.1 0.2" for i in range(8))
        + "\n"
    )
    out = tmp_path / "reports"
    assert fitmod.run_one(summ, out, min_points=7, max_points=None, m1_msun=None,
                          period_min=None, period_max=None, period_prior=None,
                          period_prior_sigma=0.15, fix_period=None, fix_e=None,
                          use_gaia_nss=False, gaia_cache_path=None,
                          observability_cache_path=None, query_gaia_online=False) is None


def test_fit_keplerian_clips_initial_guess() -> None:
    t = np.linspace(60000.0, 60070.0, 8)
    y = np.sin(2 * np.pi * t / 12.0) * 5.0
    yerr = np.full_like(t, 0.2)
    params, _ = fitmod.fit_keplerian(t, y, yerr, period_prior=1e6, period_prior_sigma=0.15)
    assert np.all(np.isfinite(params))


def test_parse_summary_excludes_legacy_sentinel_rv(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_958479989998172288_summary.txt"
    summ.write_text(
        "[PIPELINE RESULTS]\n"
        "# File | MJD | RV | Err | RMS\n"
        "f1.txt 60000.0 -10.0 0.1 0.2\n"
        "f2.txt 60010.0 -9999.0 0.1 0.2\n"
        "f3.txt 60020.0 -10.5 0.1 0.2\n"
        "f4.txt 60030.0 -11.0 0.1 0.2\n"
        "f5.txt 60040.0 -10.2 0.1 0.2\n"
    )
    pts = fitmod.parse_summary(summ)
    rvs = [p.rv for p in pts]
    assert -9999.0 not in rvs
    assert len(pts) == 4


def test_mass_function_msun_super_eccentric_returns_real() -> None:
    """e > 1 must not produce complex (1-e^2)^1.5 when coerced with float()."""
    fm = fitmod.mass_function_msun(100.0, 50.0, 1.2)
    assert isinstance(fm, float)
    assert np.isfinite(fm)


def test_mass_function_msun_negative_k_is_nan() -> None:
    assert not np.isfinite(fitmod.mass_function_msun(100.0, -50.0, 0.2))


def test_fit_report_json_has_no_circular_reference() -> None:
    t = np.linspace(60000, 60100, 10)
    y = 10.0 * np.sin(2 * np.pi * t / 20.0)
    yerr = np.full_like(y, 0.25)
    variants = fitmod.fit_all_variants(
        t, y, yerr, {"period_days": 20.0, "eccentricity": 0.1},
        period_min=5.0, period_max=200.0, period_prior_sigma=0.2,
    )
    import copy
    import json

    report = copy.deepcopy(variants["free"][1])
    report["fit_variants"] = {k: copy.deepcopy(v[1]) for k, v in variants.items()}
    json.dumps(report)  # must not raise


def test_fit_all_variants_includes_four_when_nss_present() -> None:
    t = np.linspace(60000, 60100, 12)
    y = 10.0 * np.sin(2 * np.pi * t / 20.0) + np.random.default_rng(0).normal(0, 0.2, t.size)
    yerr = np.full_like(y, 0.25)
    nss = {"period_days": 20.0, "eccentricity": 0.1}
    variants = fitmod.fit_all_variants(t, y, yerr, nss, period_min=5.0, period_max=200.0, period_prior_sigma=0.2)
    assert "free" in variants
    assert len(variants) == 4


def test_parse_summary_skips_literature_mjd_zero(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 99\nRA: 1.0\nDec: 2.0\n"
        "\n[EXTERNAL RV DATA]\n"
        "BAD_SURVEY 0.0 -20.0 1.2 z_meas\n"
        "GOOD_SURVEY 59000.0 -21.0 1.2 z_meas\n"
        "\n[PIPELINE RESULTS]\n"
        "Gaia_DR3_99_epoch_1.txt 60000.0 -10.0 0.2 0.3 False\n"
    )
    pts = fitmod.parse_summary(summ)
    lit = [p for p in pts if p.is_literature]
    assert len(lit) == 1
    assert lit[0].mjd == pytest.approx(59000.0)


def test_parse_summary_includes_external_rvs(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_123_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\nSource_ID: 123\nRA: 1.0\nDec: 2.0\n"
        "\n[EXTERNAL RV DATA]\n"
        "LAMOST_LRS 59000.0 -20.0 1.2 z_meas\n"
        "\n[PIPELINE RESULTS]\n"
        "Gaia_DR3_123_epoch_1.txt 60000.0 -10.0 0.2 0.3 False\n"
        "Gaia_DR3_123_epoch_2_kpf.txt 60001.0 -9.0 0.2 0.3 False\n"
    )
    pts = fitmod.parse_summary(summ)
    assert len(pts) == 3
    lit = [p for p in pts if p.is_literature]
    ours = [p for p in pts if not p.is_literature]
    assert len(lit) == 1
    assert lit[0].telescope == "LAMOST_LRS"
    assert sorted(p.telescope for p in ours) == ["APF", "KPF"]


def test_load_nss_priors_includes_inclination(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_1_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 1\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "NSS_Solution_Type: Orbital\n"
        "Period: 34.0\n"
        "Eccentricity: 0.25\n"
        "Inclination: 74.5\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    priors = fitmod.load_nss_priors_from_summary(summ)
    assert priors is not None
    assert priors["inclination_deg"] == pytest.approx(74.5)


def test_parse_m1_from_gaia_metadata_block(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "M1: 1.25\n"
        "Inclination: 60.0\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    assert fitmod.parse_m1_from_summary(summ) == pytest.approx(1.25)
    priors = fitmod.load_mass_priors_from_summary(summ)
    assert priors["m1_msun"] == pytest.approx(1.25)
    assert priors["inclination_deg"] == pytest.approx(60.0)


def test_resolve_m1_defaults_when_free_fit_present() -> None:
    rep = {
        "gaia_source_id": "99",
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    assert fitmod.resolve_m1_msun_for_rv_mass(rep) == pytest.approx(1.0)


def test_resolve_inclination_ignores_stale_json_90_fallback(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "Inclination: 60.0\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    rep = {
        "gaia_source_id": "99",
        "inclination_deg_used": 90.0,
        "m2sini_msun": 0.42,
        "m2_given_inclination_msun": 0.42,
    }
    assert fitmod.resolve_inclination_deg_for_rv_mass(rep, summary_path=summ) == pytest.approx(60.0)


def test_resolve_inclination_from_summary_metadata(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "Inclination: 60.0\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    rep = {"gaia_source_id": "99"}
    assert fitmod.resolve_inclination_deg_for_rv_mass(rep, summary_path=summ) == pytest.approx(60.0)


def test_website_table_masses_m2_at_i_from_inclination(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "M1: 1.0\n"
        "Inclination: 60.0\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    rep = {
        "gaia_source_id": "99",
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols = fitmod.website_table_masses_from_report(rep, summary_path=summ)
    assert cols["m2sin_i_msun"] is not None
    assert cols["m2_at_i_msun"] is not None
    assert cols["m2_at_i_msun"] > cols["m2sin_i_msun"]


def test_m2_msun_at_inclination_from_m2sin() -> None:
    m2 = fitmod.m2_msun_at_inclination(0.5, 60.0)
    assert m2 is not None
    assert m2 == pytest.approx(0.5 / np.sin(np.deg2rad(60.0)))


def test_website_table_masses_uses_json_inclination_for_m2_at_i() -> None:
    rep = {
        "gaia_source_id": "99",
        "inclination_deg_used": 60.0,
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols = fitmod.website_table_masses_from_report(rep, summary_path=None)
    assert cols["m2sin_i_msun"] is not None
    assert cols["m2_at_i_msun"] is not None
    assert cols["m2_at_i_msun"] > cols["m2sin_i_msun"]


def test_website_table_masses_m2_at_i_empty_without_inclination() -> None:
    rep = {
        "gaia_source_id": "99",
        "m2sini_msun": 0.4,
        "m2_given_inclination_msun": 0.4,
        "inclination_deg_used": 90.0,
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols = fitmod.website_table_masses_from_report(rep, summary_path=None)
    assert cols["m2sin_i_msun"] is not None
    assert cols["m2_at_i_msun"] is None


def test_website_table_masses_overrides_stale_json_90_with_summary(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "M1: 1.0\n"
        "Inclination: 60.0\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    rep = {
        "gaia_source_id": "99",
        "inclination_deg_used": 90.0,
        "m2sini_msun": 0.42,
        "m2_given_inclination_msun": 0.42,
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols = fitmod.website_table_masses_from_report(rep, summary_path=summ)
    assert cols["m2sin_i_msun"] is not None
    assert cols["m2_at_i_msun"] is not None
    assert cols["m2_at_i_msun"] > cols["m2sin_i_msun"]


def test_website_table_masses_with_teff_fallback(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "Teff: 5772.0\n"
        "logg: 4.44\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    rep = {
        "gaia_source_id": "99",
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols = fitmod.website_table_masses_from_report(rep, summary_path=summ)
    assert cols["m2sin_i_msun"] is not None


def test_website_table_masses_recompute_from_free_fit(tmp_path: Path) -> None:
    summ = tmp_path / "Gaia_DR3_99_summary.txt"
    summ.write_text(
        "[GAIA METADATA]\n"
        "Source_ID: 99\n"
        "RA: 1.0\n"
        "Dec: 2.0\n"
        "M1: 1.0\n"
        "Inclination: 90.0\n"
        "\n[PIPELINE RESULTS]\n"
        "ep_1.txt 60000 -1 0.1 0.2 False\n"
    )
    rep = {
        "used_m2_msun": 0.55,
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols = fitmod.website_table_masses_from_report(rep, summary_path=summ)
    assert cols["m2sin_i_msun"] is not None
    assert cols["m2_at_i_msun"] is not None
    assert cols["m2sin_i_msun"] == pytest.approx(cols["m2_at_i_msun"], rel=1e-5)


def test_lookup_fit_report_by_gaia_id() -> None:
    reports = {"Gaia_DR3_123": {"P_days": 1.0}}
    assert fitmod.lookup_fit_report_by_gaia_id(reports, "123") == reports["Gaia_DR3_123"]


def test_website_table_masses_from_report_separates_sources() -> None:
    rep = {
        "used_m2_msun": 0.55,
        "m2sini_msun": 0.42,
        "m2_given_inclination_msun": 0.48,
    }
    cols = fitmod.website_table_masses_from_report(rep)
    assert cols["m2_msun"] == pytest.approx(0.55)
    assert cols["m2sin_i_msun"] == pytest.approx(0.42)
    assert cols["m2_at_i_msun"] == pytest.approx(0.48)


def test_table_m1_overrides_default_and_nss() -> None:
    rep = {
        "gaia_source_id": "99",
        "used_m1_msun": 0.8,
        "gaia_nss": {"m1_msun": 0.9},
        "fit_variants": {
            "free": {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05},
        },
    }
    cols_default = fitmod.website_table_masses_from_report(rep)
    cols_table = fitmod.website_table_masses_from_report(rep, table_m1_msun=1.5)
    assert cols_default["m2sin_i_msun"] is not None
    assert cols_table["m2sin_i_msun"] is not None
    assert cols_table["m2sin_i_msun"] != cols_default["m2sin_i_msun"]


def test_fix_period_seeded_from_free_fit() -> None:
    rng = np.random.default_rng(42)
    p_true = 12.0
    t = np.sort(rng.uniform(60000, 60120, 18))
    y = 30.0 * np.sin(2 * np.pi * t / p_true) + rng.normal(0, 2.0, size=t.size)
    yerr = np.full_like(y, 2.0)
    gaia_nss = {"period_days": p_true, "eccentricity": 0.05}
    variants = fitmod.fit_all_variants(t, y, yerr, gaia_nss, period_min=1.0, period_max=100.0, period_prior_sigma=0.15)
    assert "free" in variants
    assert "fix_period" in variants
    assert "fix_period_ecc" in variants
    free_rep = variants["free"][1]
    fp_rep = variants["fix_period"][1]
    assert fp_rep["chi2_red"] <= 2.5 * free_rep["chi2_red"] + 0.05
    assert abs(fp_rep["e"] - free_rep["e"]) < 0.25


def test_rv_only_mass_estimates_use_free_fit() -> None:
    rep_free = {"P_days": 10.0, "K_kms": 40.0, "e": 0.1, "mass_function_msun": 0.05}
    m2s, m2i = fitmod.rv_only_mass_estimates(rep_free, m1_msun=1.0, inclination_deg=90.0)
    assert m2s is not None
    assert m2i is not None
    assert m2i == pytest.approx(m2s, rel=1e-6)


def test_solve_m2_with_inclination_matches_edge_on() -> None:
    # At i=90 deg, m2(with i) should match m2 sin(i).
    f_mass = 0.12
    m1 = 1.3
    m2sini = fitmod.solve_m2sini_msun(f_mass, m1)
    m2_i90 = fitmod.solve_m2_with_inclination_msun(f_mass, m1, 90.0)
    assert m2_i90 is not None
    assert m2_i90 == pytest.approx(m2sini, rel=1e-6)
