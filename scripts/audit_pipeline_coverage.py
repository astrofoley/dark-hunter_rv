#!/usr/bin/env python3
"""Report gaps between spectra, summaries, Keplerian fits, and website RV assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from darkhunter_rv.rv_keplerian_plots import our_telescope_points
from darkhunter_rv.summary_paths import (
    count_pipeline_rows,
    discover_summary_files,
    discover_summary_path,
    parse_object_id_from_summary,
)
from darkhunter_rv.website_table_csv import gaia_id_from_row
from fit_apf_rv_keplerian import parse_summary

MIN_RV_PLOT_POINTS = 2


def _discover_spectra(spec_root: Path) -> dict[str, int]:
    """Gaia id -> spectrum file count under SPEC_ROOT."""
    counts: dict[str, int] = {}
    if not spec_root.is_dir():
        return counts
    patterns = (
        "Gaia_DR3_*_epoch_*.txt",
        "Gaia_DR3_*_*_ap1.flm",
        "Gaia_DR3_*_*_ap1.txt",
    )
    for pat in patterns:
        for p in spec_root.rglob(pat):
            if not p.is_file():
                continue
            stem = p.name
            if stem.startswith("Gaia_DR3_"):
                parts = stem.split("_")
                if len(parts) >= 3 and parts[1] == "DR3":
                    gid = parts[2]
                    if gid.isdigit():
                        counts[gid] = counts.get(gid, 0) + 1
    return counts


def _table_gaia_ids(data_csv: Path) -> list[str]:
    if not data_csv.is_file():
        return []
    with data_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows or "GAIA NAME" not in rows[0]:
        return []
    i = rows[0].index("GAIA NAME")
    out: list[str] = []
    for r in rows[1:]:
        if not r:
            continue
        gid = gaia_id_from_row(r[i] if i < len(r) else "")
        if gid:
            out.append(gid)
    return out


def _audit_one(
    gid: str,
    *,
    spec_counts: dict[str, int],
    out_dir: Path,
    reports_dir: Path,
    web_root: Path,
    min_points: int,
) -> dict[str, str]:
    n_spec = spec_counts.get(gid, 0)
    summ = discover_summary_path(out_dir, gid)
    summ_flat = out_dir / f"Gaia_DR3_{gid}_summary.txt"
    has_summary = summ is not None and summ.is_file()
    n_rows = count_pipeline_rows(summ) if has_summary and summ else 0
    n_finite = 0
    n_plot_points = 0
    if has_summary and summ:
        try:
            parsed = parse_summary(summ)
            n_finite = len(parsed)
            n_plot_points = len(our_telescope_points(parsed))
        except Exception:
            n_finite = 0
            n_plot_points = 0

    fit_json = reports_dir / f"{gid}_keplerian_fit.json"
    fit_png = reports_dir / f"{gid}_keplerian_fit.png"
    rv_plot = out_dir / f"Gaia_DR3_{gid}" / f"Gaia_DR3_{gid}_rv_plot.png"
    web_fit = web_root / "stars" / f"Gaia_DR3_{gid}" / "Gaia" / "RV_Fit" / f"{gid}_keplerian_fit.png"
    web_rv_plot = (
        web_root / "stars" / f"Gaia_DR3_{gid}" / "Gaia" / "Plots" / f"Gaia_DR3_{gid}_rv_plot.png"
    )
    web_summ = web_root / "stars" / f"Gaia_DR3_{gid}" / "Gaia" / f"{gid}_summary.txt"

    issues: list[str] = []
    if n_spec == 0:
        issues.append("no_spectra")
    if not has_summary:
        issues.append("no_summary")
    elif summ and summ.resolve() != summ_flat.resolve() and not summ_flat.is_file():
        issues.append("summary_nested_only")
    if has_summary and n_finite < min_points:
        issues.append(f"few_rv_points({n_finite}<{min_points})")
    if has_summary and n_finite >= min_points and not fit_json.is_file():
        issues.append("no_fit_json")
    if has_summary and not fit_png.is_file():
        issues.append("no_fit_png")
    if has_summary and n_plot_points >= MIN_RV_PLOT_POINTS and not rv_plot.is_file():
        issues.append("no_rv_data_plot")
    if web_summ.is_file() and not web_rv_plot.is_file() and n_plot_points >= MIN_RV_PLOT_POINTS:
        issues.append("website_summary_no_rv_plot")
    if web_summ.is_file() and not web_fit.is_file():
        issues.append("website_summary_no_rv_fit")

    return {
        "gaia_id": gid,
        "n_spectra": str(n_spec),
        "has_summary": "1" if has_summary else "0",
        "pipeline_rows": str(n_rows),
        "finite_rv_epochs": str(n_finite),
        "plot_rv_epochs": str(n_plot_points),
        "has_rv_data_plot": "1" if rv_plot.is_file() else "0",
        "has_fit_json": "1" if fit_json.is_file() else "0",
        "has_fit_png": "1" if fit_png.is_file() else "0",
        "has_website_rv_plot": "1" if web_rv_plot.is_file() else "0",
        "has_website_rv_fit": "1" if web_fit.is_file() else "0",
        "issues": ";".join(issues),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit spectra → summary → fit → website coverage.")
    ap.add_argument("--data-csv", default="/var/www/html/darkhunter/rv/tables/data.csv")
    ap.add_argument("--spec-root", default="/data2/gaia_stars/apf_reductions")
    ap.add_argument("--output-dir", default=None, help="Pipeline output (default: REPO/output)")
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--web-root", default="/var/www/html/darkhunter/rv")
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--out-csv", default=None, help="Write audit table (default: rv_fit_reports/qc/pipeline_audit.csv)")
    ap.add_argument("--only-issues", action="store_true", help="Write only rows with issues")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    reports_dir = Path(args.reports_dir) if args.reports_dir else repo / "rv_fit_reports"
    spec_root = Path(args.spec_root)
    web_root = Path(args.web_root)
    data_csv = Path(args.data_csv)

    spec_counts = _discover_spectra(spec_root)
    table_ids = _table_gaia_ids(data_csv)
    summary_ids: set[str] = set()
    for p in discover_summary_files(out_dir):
        sid = parse_object_id_from_summary(p)
        if sid:
            summary_ids.add(sid)

    all_ids = sorted(set(table_ids) | set(spec_counts) | summary_ids)

    rows = [
        _audit_one(
            gid,
            spec_counts=spec_counts,
            out_dir=out_dir,
            reports_dir=reports_dir,
            web_root=web_root,
            min_points=args.min_points,
        )
        for gid in all_ids
    ]

    out_csv = Path(args.out_csv) if args.out_csv else reports_dir / "qc" / "pipeline_audit.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            if args.only_issues and not row["issues"]:
                continue
            w.writerow(row)

    n_issue = sum(1 for r in rows if r["issues"])
    n_no_summ = sum(1 for r in rows if "no_summary" in r["issues"])
    n_no_rv_plot = sum(
        1
        for r in rows
        if "no_rv_data_plot" in r["issues"] or "website_summary_no_rv_plot" in r["issues"]
    )
    print(
        f"audit: {len(rows)} stars, {n_issue} with issues, "
        f"{n_no_summ} missing summary, {n_no_rv_plot} missing RV data plot assets"
    )
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
