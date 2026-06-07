#!/usr/bin/env python3
"""
Phase A baseline: literature/APF overlap inventory, calibration gates, and diagnostic plots.

Absolute calibration: APF vs literature pairs within --pair-window-days (default 7), target |ΔRV| < 1 km/s.
Relative calibration: APF vs APF pairs on the same star, target improving toward 0.1 km/s after debias.

Re-run after pipeline changes and compare against ``calibration/phase_a_baseline/reference_manifest.json``.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.rv_phase_a_baseline \\
    --master calibration/literature_rv_master.csv \\
    --summary-dir output \\
    --diagnostics-glob 'output/Gaia_DR3_*_diagnostics.csv' \\
    --out-dir validation_output/rv_phase_a_baseline \\
    --bias-correction-applied
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.rv_overlap_lib import (  # noqa: E402
    PhaseAGoals,
    PhaseARunManifest,
    build_overlap_stars,
    enrich_pairs_with_deltas,
    file_sha256,
    find_pair_candidates,
    inventory_summary_counts,
    load_apf_epochs,
    load_literature_epochs,
    pair_counts_by_window,
    per_star_gate_table,
    summarize_absolute_gate,
    summarize_relative_gate,
)

logger = logging.getLogger(__name__)
_GOALS_PATH = _REPO_ROOT / "calibration" / "phase_a_baseline" / "goals.yaml"
_REFERENCE_MANIFEST = _REPO_ROOT / "calibration" / "phase_a_baseline" / "reference_manifest.json"


def _plot_inventory_counts(counts: dict[str, int], out_path: Path) -> None:
    labels = ["Literature stars", "APF stars", "Overlap stars"]
    keys = ["n_literature_stars", "n_apf_stars", "n_overlap_stars"]
    vals = [counts.get(k, 0) for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_ylabel("Count")
    ax.set_title("Phase A: star inventory")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_pair_type_counts(pairs: pd.DataFrame, out_path: Path) -> None:
    if pairs.empty:
        return
    vc = pairs["pair_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [str(x) for x in vc.index]
    ax.bar(labels, vc.values, color=["#c44e52", "#55a868", "#4c72b0"])
    ax.set_ylabel("Pair count")
    ax.set_title(f"Pair candidates (window applied)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_histogram(
    values: np.ndarray,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    thresholds: list[tuple[float, str, str]],
) -> None:
    if len(values) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=min(30, max(10, len(values))), color="#4c72b0", alpha=0.85, edgecolor="white")
    for thr, label, color in thresholds:
        ax.axvline(thr, color=color, ls="--", lw=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_abs_vs_days(pairs: pd.DataFrame, out_path: Path, threshold_kms: float) -> None:
    sub = pairs[pairs["pair_type"] == "apf_literature"]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    passed = sub["abs_delta_rv_kms"].astype(float) < threshold_kms
    ax.scatter(
        sub.loc[~passed, "delta_days"],
        sub.loc[~passed, "abs_delta_rv_kms"],
        c="#c44e52",
        label=f"fail (≥{threshold_kms} km/s)",
        alpha=0.8,
        s=36,
    )
    ax.scatter(
        sub.loc[passed, "delta_days"],
        sub.loc[passed, "abs_delta_rv_kms"],
        c="#55a868",
        label=f"pass (<{threshold_kms} km/s)",
        alpha=0.8,
        s=36,
    )
    ax.axhline(threshold_kms, color="gray", ls=":", lw=1)
    ax.set_xlabel("Δt (days)")
    ax.set_ylabel("|ΔRV| (km/s)")
    ax.set_title("Absolute gate: APF vs literature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_apf_vs_literature_scatter(pairs: pd.DataFrame, out_path: Path) -> None:
    sub = pairs[pairs["pair_type"] == "apf_literature"]
    if sub.empty:
        return
    x = sub["right_rv_kms"].astype(float)
    y = sub["left_rv_kms"].astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(x, y, s=40, alpha=0.75, c="#4c72b0")
    lims = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
    pad = 5.0
    lo, hi = lims[0] - pad, lims[1] + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Literature RV (km/s)")
    ax.set_ylabel("APF RV (km/s)")
    ax.set_title("Absolute calibration: APF vs literature")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_star_pass(per_star: pd.DataFrame, out_path: Path) -> None:
    sub = per_star[per_star["n_apf_literature_pairs"] > 0].copy()
    if sub.empty:
        return
    sub = sub.sort_values("abs_pass_rate")
    labels = [f"{r['name'] or r['gaia_dr3_id'][:8]}" for _, r in sub.iterrows()]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(sub))))
    ax.barh(labels, sub["abs_pass_rate"].astype(float), color="#4c72b0")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Absolute gate pass rate")
    ax.set_title("Per-star APF–literature pass rate")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_min_apf_lit_delta(overlap: pd.DataFrame, window_days: float, out_path: Path) -> None:
    if overlap.empty or "min_apf_literature_delta_days" not in overlap.columns:
        return
    sub = overlap.sort_values("min_apf_literature_delta_days")
    labels = [f"{r['name'] or str(r['gaia_dr3_id'])[:10]}" for _, r in sub.iterrows()]
    vals = sub["min_apf_literature_delta_days"].astype(float).values
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(sub))))
    colors = ["#55a868" if v <= window_days else "#c44e52" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.axvline(window_days, color="gray", ls="--", label=f"gate window {window_days} d")
    ax.set_xlabel("Min |ΔMJD| APF vs literature (days)")
    ax.set_title("Closest APF–literature epoch separation per overlap star")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_epoch_timeline(literature: pd.DataFrame, apf: pd.DataFrame, overlap: pd.DataFrame, out_path: Path) -> None:
    if overlap.empty:
        return
    ids = set(overlap["gaia_dr3_id"].astype(str))
    fig, ax = plt.subplots(figsize=(8, 5))
    all_rv: list[float] = []
    for gid in sorted(ids):
        lg = literature[literature["gaia_dr3_id"] == gid]
        ag = apf[apf["gaia_dr3_id"] == gid]
        if len(lg):
            ax.scatter(lg["mjd"], lg["rv_kms"], marker="s", s=30, alpha=0.7, c="#c44e52")
            all_rv.extend(lg["rv_kms"].astype(float).tolist())
        if len(ag):
            ax.scatter(ag["mjd"], ag["rv_kms"], marker="o", s=30, alpha=0.7, c="#4c72b0")
            all_rv.extend(ag["rv_kms"].astype(float).tolist())
    if all_rv:
        ylo, yhi = float(np.nanmin(all_rv)), float(np.nanmax(all_rv))
        pad = max(5.0, 0.05 * (yhi - ylo + 1e-6))
        ax.set_ylim(ylo - pad, yhi + pad)
    for gid in sorted(ids):
        name = str(overlap.loc[overlap["gaia_dr3_id"] == gid, "name"].iloc[0])
        lg = literature[literature["gaia_dr3_id"] == gid]
        ag = apf[apf["gaia_dr3_id"] == gid]
        x = float(lg["mjd"].median()) if len(lg) else float(ag["mjd"].median())
        ax.text(x, ax.get_ylim()[1], name or gid[:10], fontsize=7, ha="center", va="bottom")
    ax.scatter([], [], marker="s", c="#c44e52", label="Literature")
    ax.scatter([], [], marker="o", c="#4c72b0", label="APF")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.set_title("Overlap stars: epoch timeline")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_report_md(
    path: Path,
    *,
    counts: dict[str, int],
    abs_summary: dict,
    rel_summary: dict,
    goals: PhaseAGoals,
    bias_applied: bool,
) -> None:
    lines = [
        "# Phase A baseline report",
        "",
        "## Inventory",
        f"- Literature stars: {counts.get('n_literature_stars', 0)}",
        f"- APF stars: {counts.get('n_apf_stars', 0)}",
        f"- Overlap stars: {counts.get('n_overlap_stars', 0)}",
        f"- APF–literature pairs (≤{goals.pair_window_days} d): {counts.get('n_apf_literature_pairs', 0)}",
        f"- APF–APF pairs: {counts.get('n_apf_apf_pairs', 0)}",
        "",
        "## Absolute calibration (APF vs literature)",
        f"- Bias correction flagged as applied: **{bias_applied}**",
        f"- Threshold: |ΔRV| < {goals.absolute_gate_kms} km/s",
    ]
    if abs_summary.get("n_pairs", 0):
        lines.extend(
            [
                f"- Pass rate: {100 * abs_summary['pass_rate']:.1f}% ({abs_summary['n_pass']}/{abs_summary['n_pairs']})",
                f"- Median |ΔRV|: {abs_summary['median_abs_delta_rv_kms']:.3f} km/s",
                f"- p90 |ΔRV|: {abs_summary['p90_abs_delta_rv_kms']:.3f} km/s",
                f"- RMS ΔRV: {abs_summary['rms_delta_rv_kms']:.3f} km/s",
            ]
        )
    else:
        lines.append("- No APF–literature pairs in window.")
        lines.append(
            "- Likely cause: literature epochs predate APF coverage on overlap stars; "
            "see `plots/min_apf_literature_delta_days.png` and `pair_counts_by_window.csv`."
        )
    lines.extend(["", "## Relative calibration (APF–APF)", f"- Goal (post-debias target): {goals.relative_goal_kms} km/s"])
    if rel_summary.get("n_pairs", 0):
        lines.extend(
            [
                f"- Median |ΔRV|: {rel_summary['median_abs_delta_rv_kms']:.3f} km/s",
                f"- p90 |ΔRV|: {rel_summary['p90_abs_delta_rv_kms']:.3f} km/s",
                f"- RMS ΔRV: {rel_summary['rms_delta_rv_kms']:.3f} km/s",
                f"- Fraction below {goals.relative_goal_kms} km/s: {100 * rel_summary['frac_below_goal']:.1f}%",
            ]
        )
    else:
        lines.append("- No APF–APF pairs in window.")
    lines.extend(
        [
            "",
            "## Regression",
            "Compare `baseline_manifest.json` to `calibration/phase_a_baseline/reference_manifest.json`.",
            "Re-run with `--no-bias-correction-applied` after a `--no-bias` pipeline campaign for the canonical absolute gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase_a(
    *,
    master_path: Path,
    summary_dir: Path,
    diagnostics_glob: str | None,
    out_dir: Path,
    goals: PhaseAGoals,
    bias_correction_applied: bool,
    prefer_diagnostics_rv: bool,
    run_id: str | None = None,
) -> PhaseARunManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    literature = load_literature_epochs(master_path)
    apf = load_apf_epochs(
        summary_dir,
        diagnostics_glob=diagnostics_glob,
        bias_correction_applied=bias_correction_applied,
        prefer_diagnostics_rv=prefer_diagnostics_rv,
    )
    overlap = build_overlap_stars(literature, apf)
    pairs = find_pair_candidates(literature, apf, overlap, window_days=goals.pair_window_days)
    pairs = enrich_pairs_with_deltas(pairs)

    literature.to_csv(out_dir / "literature_epochs.csv", index=False)
    apf.to_csv(out_dir / "apf_epochs.csv", index=False)
    overlap.to_csv(out_dir / "overlap_stars.csv", index=False)
    window_tab = pair_counts_by_window(
        literature, apf, overlap, windows_days=[7, 30, 90, 180, 365]
    )
    window_tab.to_csv(out_dir / "pair_counts_by_window.csv", index=False)
    pairs.to_csv(out_dir / "pair_candidates.csv", index=False)

    abs_summary = summarize_absolute_gate(pairs, threshold_kms=goals.absolute_gate_kms)
    rel_summary = summarize_relative_gate(pairs, goal_kms=goals.relative_goal_kms)
    per_star = per_star_gate_table(pairs, absolute_threshold_kms=goals.absolute_gate_kms)
    per_star.to_csv(out_dir / "per_star_gates.csv", index=False)

    pd.DataFrame([abs_summary]).to_csv(out_dir / "absolute_gate_summary.csv", index=False)
    pd.DataFrame([rel_summary]).to_csv(out_dir / "relative_gate_summary.csv", index=False)

    counts = inventory_summary_counts(literature, apf, overlap, pairs)

    _plot_inventory_counts(counts, plots_dir / "inventory_star_counts.png")
    _plot_pair_type_counts(pairs, plots_dir / "pair_type_counts.png")
    _plot_histogram(
        pairs.loc[pairs["pair_type"] == "apf_literature", "abs_delta_rv_kms"].astype(float).values,
        plots_dir / "absolute_delta_rv_histogram.png",
        title="Absolute gate |ΔRV| (APF − literature)",
        xlabel="|ΔRV| (km/s)",
        thresholds=[(goals.absolute_gate_kms, f"gate {goals.absolute_gate_kms} km/s", "crimson")],
    )
    _plot_histogram(
        pairs.loc[pairs["pair_type"] == "apf_apf", "abs_delta_rv_kms"].astype(float).values,
        plots_dir / "relative_delta_rv_histogram.png",
        title="Relative gate |ΔRV| (APF − APF)",
        xlabel="|ΔRV| (km/s)",
        thresholds=[(goals.relative_goal_kms, f"goal {goals.relative_goal_kms} km/s", "darkgreen")],
    )
    _plot_histogram(
        pairs.loc[pairs["pair_type"] == "literature_literature", "abs_delta_rv_kms"].astype(float).values,
        plots_dir / "literature_literature_delta_histogram.png",
        title="Diagnostic: |ΔRV| literature–literature",
        xlabel="|ΔRV| (km/s)",
        thresholds=[],
    )
    _plot_abs_vs_days(pairs, plots_dir / "absolute_delta_rv_vs_delta_days.png", goals.absolute_gate_kms)
    _plot_apf_vs_literature_scatter(pairs, plots_dir / "apf_vs_literature_scatter.png")
    _plot_per_star_pass(per_star, plots_dir / "per_star_absolute_pass_rate.png")
    _plot_epoch_timeline(literature, apf, overlap, plots_dir / "overlap_epoch_timeline.png")
    _plot_min_apf_lit_delta(overlap, goals.pair_window_days, plots_dir / "min_apf_literature_delta_days.png")

    _write_report_md(
        out_dir / "REPORT.md",
        counts=counts,
        abs_summary=abs_summary,
        rel_summary=rel_summary,
        goals=goals,
        bias_applied=bias_correction_applied,
    )

    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = PhaseARunManifest(
        run_id=rid,
        created_utc=datetime.now(timezone.utc).isoformat(),
        master_path=str(master_path.resolve()),
        summary_dir=str(summary_dir.resolve()),
        diagnostics_glob=diagnostics_glob or "",
        bias_correction_applied=bias_correction_applied,
        goals={
            "pair_window_days": goals.pair_window_days,
            "absolute_gate_kms": goals.absolute_gate_kms,
            "relative_goal_kms": goals.relative_goal_kms,
        },
        inventory=counts,
        absolute_gate=abs_summary,
        relative_gate=rel_summary,
        output_files={},
    )
    manifest.inventory["pair_counts_by_window"] = window_tab.to_dict(orient="records")
    key_outputs = [
        "overlap_stars.csv",
        "pair_candidates.csv",
        "absolute_gate_summary.csv",
        "relative_gate_summary.csv",
        "per_star_gates.csv",
        "baseline_manifest.json",
        "REPORT.md",
    ]
    manifest.write_json(out_dir / "baseline_manifest.json")
    for name in key_outputs:
        p = out_dir / name
        if p.is_file():
            manifest.output_files[name] = file_sha256(p)
    manifest.write_json(out_dir / "baseline_manifest.json")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, default=_REPO_ROOT / "calibration" / "literature_rv_master.csv")
    ap.add_argument("--summary-dir", type=Path, default=_REPO_ROOT / "output")
    ap.add_argument("--diagnostics-glob", default="output/Gaia_DR3_*_diagnostics.csv")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "validation_output" / "rv_phase_a_baseline")
    ap.add_argument("--goals", type=Path, default=_GOALS_PATH)
    ap.add_argument("--pair-window-days", type=float, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--bias-correction-applied",
        action="store_true",
        default=None,
        help="APF RVs include order bias correction (default: true if neither bias flag set)",
    )
    ap.add_argument(
        "--no-bias-correction-applied",
        action="store_true",
        help="APF RVs are from a --no-bias pipeline run",
    )
    ap.add_argument("--summary-only-rv", action="store_true", help="Use summary RVs instead of diagnostics exposure_rv")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    goals = PhaseAGoals.from_yaml_path(args.goals)
    if args.pair_window_days is not None:
        goals.pair_window_days = float(args.pair_window_days)

    if args.no_bias_correction_applied:
        bias_applied = False
    elif args.bias_correction_applied:
        bias_applied = True
    else:
        bias_applied = True
        logger.warning(
            "Assuming bias_correction_applied=True. Re-run pipeline with --no-bias and pass "
            "--no-bias-correction-applied for the canonical absolute gate baseline."
        )

    manifest = run_phase_a(
        master_path=args.master,
        summary_dir=args.summary_dir,
        diagnostics_glob=args.diagnostics_glob or None,
        out_dir=args.out_dir,
        goals=goals,
        bias_correction_applied=bias_applied,
        prefer_diagnostics_rv=not args.summary_only_rv,
        run_id=args.run_id,
    )
    logger.info("Phase A baseline written to %s", args.out_dir)
    logger.info(
        "overlap=%d apf_lit_pairs=%d pass_rate=%.1f%% rel_pairs=%d",
        manifest.inventory.get("n_overlap_stars", 0),
        manifest.absolute_gate.get("n_pairs", 0),
        100.0 * manifest.absolute_gate.get("pass_rate", 0.0),
        manifest.relative_gate.get("n_pairs", 0),
    )


if __name__ == "__main__":
    main()
