#!/usr/bin/env python3
"""
Full chunk layout campaign: pipeline runs, measurement cache, grid search, stages B/C.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  python -m validation.chunk_campaign \\
    --spectrum-list validation_output/chunk_campaign/spectrum_list.txt \\
    --out-dir validation_output/chunk_campaign \\
    --run-pipeline
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.chunk_grid_search import composite_score  # noqa: E402
from validation.chunk_layout import (  # noqa: E402
    ChunkLayout,
    build_campaign_broad_grid,
    build_campaign_edge_grid,
    build_custom_edge_layout,
    load_campaign_layout_by_name,
    save_chunk_layout,
)
from validation.chunk_measurement_cache import (  # noqa: E402
    diagnostics_glob_for_layout,
    drop_layout_from_cache,
    find_cache_layout_collisions,
    ingest_layout_diagnostics_dir,
    layout_files_complete,
    load_cache,
    rebuild_cache_from_diagnostics,
    save_cache,
)
from validation.evaluate_chunk_layout import evaluate_layouts_from_glob  # noqa: E402
from validation.plot_chunk_residuals import _load_chunk_rows  # noqa: E402

logger = logging.getLogger(__name__)

CACHE_NAME = "measurement_cache.csv"


def write_spectrum_list_from_diagnostics(diagnostics_glob: str, out_path: Path) -> list[Path]:
    tab = _load_chunk_rows(diagnostics_glob)
    files = sorted({Path(str(f)) for f in tab["file"].astype(str).unique() if Path(str(f)).is_file()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")
    return files


def read_spectrum_list(path: Path) -> list[Path]:
    files: list[Path] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s).expanduser()
        if p.is_file():
            files.append(p.resolve())
    return files


def run_pipeline_for_layout(
    layout: ChunkLayout,
    spectrum_files: list[Path],
    *,
    campaign_dir: Path,
    instrument: str,
    log_level: str,
    batch_size: int = 20,
    extra_flags: list[str] | None = None,
) -> Path:
    layout_yaml = campaign_dir / "layouts" / f"{layout.name}.yaml"
    save_chunk_layout(layout, layout_yaml)
    out_dir = campaign_dir / "diagnostics" / layout.name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT),
        "DARKHUNTER_OUTPUT_DIR": str(out_dir),
    }
    flags = list(extra_flags or [])
    flags.extend(["--chunk-layout", str(layout_yaml), "--instrument", instrument, "--log-level", log_level])
    for i in range(0, len(spectrum_files), batch_size):
        batch = spectrum_files[i : i + batch_size]
        cmd = [sys.executable, "-m", "darkhunter_rv.pipeline", *[str(p) for p in batch], *flags]
        logger.info("pipeline %s batch %d..%d (%d files)", layout.name, i, i + len(batch), len(batch))
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if r.returncode != 0:
            raise RuntimeError(f"pipeline failed for {layout.name} code={r.returncode}")
    return out_dir


def coordinate_descent_edges(
    n_chunks: int,
    initial_edges: list[float],
    *,
    scan_points: int = 7,
    passes: int = 2,
) -> list[list[float]]:
    """Stage B: refine interior pixel edges by coordinate descent candidates."""
    edges = list(initial_edges)
    candidates: list[list[float]] = [edges.copy()]
    for _ in range(passes):
        for k in range(1, len(edges) - 1):
            lo = edges[k - 1] + 0.05
            hi = edges[k + 1] - 0.05
            if hi <= lo:
                continue
            best_e = edges[k]
            best_score = float("inf")
            for trial in np.linspace(lo, hi, scan_points):
                cand = edges.copy()
                cand[k] = float(trial)
                # score placeholder — actual scoring done after pipeline
                candidates.append(cand)
                edges[k] = float(trial)
    return candidates


def build_stage_b_layouts(n_chunks: int, base_edges: list[float]) -> list[ChunkLayout]:
    layouts: list[ChunkLayout] = []
    for i, edges in enumerate(coordinate_descent_edges(n_chunks, base_edges)):
        name = f"stageB_n{n_chunks}_e{i:02d}"
        layouts.append(build_custom_edge_layout(name, edges))
    return layouts


def _plot_campaign_summary(summary: pd.DataFrame, out_path: Path) -> None:
    valid = summary[summary.get("offline_eval_valid", True).astype(bool)].copy()
    if valid.empty:
        valid = summary.copy()
    valid = valid.sort_values("composite_score")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(valid))
    ax.bar(x, valid["composite_score"].astype(float), color="C0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(valid["layout"].astype(str), rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Composite score (lower better)")
    ax.set_title("Chunk campaign layout comparison")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    fig2, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in zip(
        axes,
        ["median_sigma_rv_kms", "p90_sigma_rv_kms", "relative_median_abs_delta_kms"],
        ["Median σ_RV", "p90 σ_RV", "Relative median |ΔRV|"],
    ):
        ax.scatter(
            valid["layout"].astype(str),
            valid[col].astype(float),
            s=50,
            alpha=0.8,
        )
        ax.set_title(title)
        ax.set_ylabel("km/s")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
    fig2.tight_layout()
    out_path2 = out_path.parent / "campaign_sigma_rv_vs_relative.png"
    fig2.savefig(out_path2, dpi=130)
    plt.close(fig2)


def write_next_steps(path: Path, *, best: pd.Series | None, stage_b_best: pd.Series | None) -> None:
    lines = [
        "# Chunk campaign — next steps",
        "",
        "## Completed",
        "- Broad grid: split N=2,3,4 and merge W=2,3,4 (full pipeline)",
        "- Edge presets: equal, blue-heavy, red-heavy, telluric-aware",
        "- Stage B coordinate-descent edge candidates",
        "- Measurement cache: `measurement_cache.csv` (dedupe by layout_name + measurement_id + file)",
        "",
        "## Best layouts so far",
    ]
    if best is not None:
        lines.append(
            f"- **Broad/edge winner:** `{best['layout']}` "
            f"(score={best['composite_score']:.4f}, median σ_RV={best.get('median_sigma_rv_kms', float('nan')):.3f} km/s, "
            f"rel={best['relative_median_abs_delta_kms']:.3f} km/s)"
        )
    if stage_b_best is not None:
        lines.append(
            f"- **Stage B refinement:** `{stage_b_best['layout']}` "
            f"(score={stage_b_best['composite_score']:.4f})"
        )
    lines.extend(
        [
            "",
            "## Recommended next work",
            "",
            "1. **Per-order variable edges** — allow different N+1 edges per echelle order in YAML; "
            "high bias-gradient orders get finer splits (use bias curve from chunk_bias_regression).",
            "2. **Telluric down-weighting** — persist `telluric_fraction` in diagnostics; zero-weight "
            "telluric-heavy chunks in stack (not just split).",
            "3. **Bad-line isolation** — flag chunks with high `mask_line_count` variance or CCF asymmetry; "
            "assign trust weights before IVW stack.",
            "4. **ML chunk quality model** — BDT on chunk features to predict |residual|; use to rank edge "
            "candidates before pipeline reruns.",
            "5. **Deploy winner** — install chosen layout YAML as default APF chunk config; rebuild bias table.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(
    *,
    spectrum_files: list[Path],
    campaign_dir: Path,
    run_pipeline: bool,
    instrument: str,
    log_level: str,
    edge_n_chunks: int,
    run_stage_b: bool,
    skip_pipeline_if_cached: bool,
    only_layouts: list[str] | None = None,
    force_layouts: set[str] | None = None,
) -> pd.DataFrame:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    cache_path = campaign_dir / CACHE_NAME
    cache = load_cache(cache_path)

    broad = build_campaign_broad_grid()
    edge = build_campaign_edge_grid(n_chunks=edge_n_chunks)
    layouts = broad + edge
    # Deduplicate by name
    seen: set[str] = set()
    unique_layouts: list[ChunkLayout] = []
    for lay in layouts:
        if lay.name not in seen:
            unique_layouts.append(lay)
            seen.add(lay.name)

    if only_layouts:
        want = set(only_layouts)
        from_grid = [lay for lay in unique_layouts if lay.name in want]
        found = {lay.name for lay in from_grid}
        extras: list[ChunkLayout] = []
        for name in sorted(want - found):
            lay = load_campaign_layout_by_name(name, campaign_dir)
            if lay is not None:
                extras.append(lay)
            else:
                logger.warning("Unknown layout %r (not in grid and no YAML found)", name)
        unique_layouts = from_grid + extras
        if not unique_layouts:
            logger.error("No layouts matched --only-layouts %s", sorted(want))
            return pd.DataFrame()

    force_layouts = force_layouts or set()

    for layout in unique_layouts:
        save_chunk_layout(layout, campaign_dir / "layouts" / f"{layout.name}.yaml")

    if run_pipeline:
        for layout in unique_layouts:
            if layout.name in force_layouts:
                cache = drop_layout_from_cache(cache, layout.name)
                save_cache(cache, cache_path)
            todo = spectrum_files
            if skip_pipeline_if_cached and layout.name not in force_layouts:
                done = layout_files_complete(cache, layout_name=layout.name, spectrum_files=spectrum_files)
                todo = [f for f in spectrum_files if f not in done]
            if not todo:
                logger.info("skip pipeline %s (cached)", layout.name)
                continue
            diag_dir = run_pipeline_for_layout(
                layout, todo, campaign_dir=campaign_dir, instrument=instrument, log_level=log_level
            )
            cache = ingest_layout_diagnostics_dir(diag_dir, layout=layout, cache=cache)
            save_cache(cache, cache_path)

    # Evaluate from per-layout diagnostics
    eval_rows = []
    for layout in unique_layouts:
        glob_pat = diagnostics_glob_for_layout(campaign_dir, layout.name)
        summary = evaluate_layouts_from_glob(
            [layout],
            glob_pat,
            clip_sigma=7.0,
            clip_max_delta_kms=20.0,
            min_chunks=3,
        )
        if not summary.empty:
            summary["offline_eval_valid"] = True
            eval_rows.append(summary)
    grid = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    if not grid.empty:
        grid["composite_score"] = grid.apply(composite_score, axis=1)
        grid = grid.sort_values("composite_score")

    stage_b_summary = pd.DataFrame()
    if run_stage_b and not grid.empty:
        # Pick best equal-split or edge preset with n_subchunks ~ edge_n_chunks
        split_candidates = grid[grid["layout"].astype(str).str.startswith(("subchunks_", "n"))]
        if len(split_candidates):
            base_name = str(split_candidates.iloc[0]["layout"])
        else:
            base_name = f"n{edge_n_chunks}_equal"
        from validation.chunk_layout import preset_pixel_edges

        base_edges = preset_pixel_edges(edge_n_chunks, "equal")
        stage_b_layouts = build_stage_b_layouts(edge_n_chunks, base_edges)
        if run_pipeline:
            for layout in stage_b_layouts[:8]:  # cap pipeline cost for stage B
                todo = spectrum_files
                if skip_pipeline_if_cached:
                    done = layout_files_complete(cache, layout_name=layout.name, spectrum_files=spectrum_files)
                    todo = [f for f in spectrum_files if f not in done]
                if not todo:
                    continue
                diag_dir = run_pipeline_for_layout(
                    layout, todo, campaign_dir=campaign_dir, instrument=instrument, log_level=log_level
                )
                cache = ingest_layout_diagnostics_dir(diag_dir, layout=layout, cache=cache)
                save_cache(cache, cache_path)
        sb_rows = []
        for layout in stage_b_layouts[:8]:
            glob_pat = diagnostics_glob_for_layout(campaign_dir, layout.name)
            s = evaluate_layouts_from_glob([layout], glob_pat, clip_sigma=7.0, clip_max_delta_kms=20.0, min_chunks=3)
            if not s.empty:
                s["offline_eval_valid"] = True
                sb_rows.append(s)
        if sb_rows:
            stage_b_summary = pd.concat(sb_rows, ignore_index=True)
            stage_b_summary["composite_score"] = stage_b_summary.apply(composite_score, axis=1)
            stage_b_summary = stage_b_summary.sort_values("composite_score")
            stage_b_summary.to_csv(campaign_dir / "stage_b_summary.csv", index=False)

    if not grid.empty:
        grid.to_csv(campaign_dir / "campaign_grid_summary.csv", index=False)
        _plot_campaign_summary(grid, campaign_dir / "plots" / "campaign_grid_scores.png")

    best = grid.iloc[0] if len(grid) else None
    sb_best = stage_b_summary.iloc[0] if len(stage_b_summary) else None
    write_next_steps(campaign_dir / "NEXT_STEPS.md", best=best, stage_b_best=sb_best)

    report_lines = [
        "# Chunk campaign report",
        "",
        f"- Spectra: {len(spectrum_files)}",
        f"- Layouts evaluated: {len(grid)}",
        f"- Cache rows: {len(cache)}",
        "",
    ]
    if best is not None:
        report_lines.append(f"**Best layout:** `{best['layout']}` (score={best['composite_score']:.4f})")
    report_lines.append("")
    report_lines.append("See `campaign_grid_summary.csv`, `NEXT_STEPS.md`, and `measurement_cache.csv`.")
    (campaign_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "n_spectra": len(spectrum_files),
        "n_layouts": len(grid),
        "n_cache_rows": int(len(cache)),
        "best_layout": None if best is None else str(best["layout"]),
    }
    (campaign_dir / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spectrum-list", type=Path, default=None)
    ap.add_argument(
        "--diagnostics-glob",
        default="output/Gaia_DR3_*_diagnostics.csv",
        help="Used to build spectrum list if --spectrum-list missing",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "validation_output" / "chunk_campaign")
    ap.add_argument("--run-pipeline", action="store_true")
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--edge-n-chunks", type=int, default=3, help="N for edge preset + stage B")
    ap.add_argument("--run-stage-b", action="store_true", default=True)
    ap.add_argument("--no-stage-b", action="store_false", dest="run_stage_b")
    ap.add_argument("--skip-pipeline-if-cached", action="store_true", default=True)
    ap.add_argument(
        "--no-skip-pipeline-if-cached",
        action="store_false",
        dest="skip_pipeline_if_cached",
        help="Re-run pipeline for every spectrum in each layout (ignore cache completeness).",
    )
    ap.add_argument(
        "--only-layouts",
        default="",
        help="Comma-separated layout names to run/evaluate (default: full campaign grid).",
    )
    ap.add_argument(
        "--force-layouts",
        default="",
        help="Comma-separated layouts to re-pipeline even if cached (drops their cache rows first).",
    )
    ap.add_argument(
        "--repair-cache",
        action="store_true",
        help="Rebuild measurement_cache.csv from diagnostics/*/ on disk, then exit unless --run-pipeline.",
    )
    ap.add_argument(
        "--repair-cache-layouts",
        default="",
        help="With --repair-cache: only rebuild these layouts (comma-separated); default all under layouts/.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.out_dir / CACHE_NAME
    if args.repair_cache:
        repair_layouts = [x.strip() for x in args.repair_cache_layouts.split(",") if x.strip()] or None
        cache = rebuild_cache_from_diagnostics(args.out_dir, layout_names=repair_layouts)
        save_cache(cache, cache_path)
        collisions = find_cache_layout_collisions(cache)
        logger.info(
            "Rebuilt cache -> %s (%d rows, %d layout(s))",
            cache_path,
            len(cache),
            cache["layout_name"].nunique() if len(cache) else 0,
        )
        if len(collisions):
            logger.warning(
                "Cross-layout cache collisions (same measurement_id+file, multiple layouts): %d",
                len(collisions),
            )
        if not args.run_pipeline:
            return

    only_layouts = [x.strip() for x in args.only_layouts.split(",") if x.strip()] or None
    force_layouts = {x.strip() for x in args.force_layouts.split(",") if x.strip()} or None

    list_path = args.spectrum_list or (args.out_dir / "spectrum_list.txt")
    if list_path.is_file():
        spectrum_files = read_spectrum_list(list_path)
    else:
        spectrum_files = write_spectrum_list_from_diagnostics(args.diagnostics_glob, list_path)
    if not spectrum_files:
        logger.error("No spectrum files")
        sys.exit(1)

    run_campaign(
        spectrum_files=spectrum_files,
        campaign_dir=args.out_dir,
        run_pipeline=args.run_pipeline,
        instrument=args.instrument,
        log_level=args.log_level,
        edge_n_chunks=args.edge_n_chunks,
        run_stage_b=args.run_stage_b,
        skip_pipeline_if_cached=args.skip_pipeline_if_cached,
        only_layouts=only_layouts,
        force_layouts=force_layouts,
    )
    logger.info("Campaign done -> %s", args.out_dir)


if __name__ == "__main__":
    main()
