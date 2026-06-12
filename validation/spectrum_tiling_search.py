#!/usr/bin/env python3
"""
CLI: exhaustive mixed-layout tiling search with pipeline σ_RV.

Example::

  cd /Users/rfoley/darkhunter/rvs/dark-hunter_rv
  PYTHONPATH=. python3 -m validation.spectrum_tiling_search \\
    --campaign-dir validation_output/chunk_campaign
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.spectrum_tiling import (  # noqa: E402
    DEFAULT_METRIC,
    TileRegistry,
    count_complete_tilings,
    file_meas_index,
    find_best_tiling_for_file,
    run_campaign_tiling_search,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Find best full-spectrum mixed chunk tiling (pipeline σ_RV) from campaign cache.",
    )
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=REPO_ROOT / "validation_output" / "chunk_campaign",
        help="Campaign directory with measurement_cache.csv and layouts/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <campaign-dir>/spectrum_tiling_search)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=("rv_err_calibrated_kms", "sigma_rv_core90_kms"),
        help="Objective for best tiling (pipeline stack outputs).",
    )
    parser.add_argument(
        "--max-tilings",
        type=int,
        default=None,
        help="Cap stack evaluations per exposure (default: no cap).",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count mixed tilings per file (no stack evaluation).",
    )
    parser.add_argument(
        "--mix-layouts",
        type=str,
        default="",
        help="Comma-separated layouts for mixed tiles (default: all cached layouts).",
    )
    parser.add_argument(
        "--whole-layouts",
        type=str,
        default="",
        help="Comma-separated whole-spectrum layouts (default: all cached layouts).",
    )
    parser.add_argument(
        "--only-file",
        type=str,
        default="",
        help="Run a single exposure basename/path for debugging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    campaign_dir = Path(args.campaign_dir)
    out_dir = Path(args.out_dir) if args.out_dir else campaign_dir / "spectrum_tiling_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    mix_layouts = [s.strip() for s in args.mix_layouts.split(",") if s.strip()] or None
    whole_layouts = [s.strip() for s in args.whole_layouts.split(",") if s.strip()] or None

    if args.only_file:
        from validation.chunk_adaptive_stack import (
            _index_by_file_layout,
            build_multi_layout_bias_tables,
            load_campaign_measurements,
            load_layouts,
        )
        from validation.chunk_bias_lib import load_stellar_metadata

        layouts = load_layouts(campaign_dir)
        meas_df = load_campaign_measurements(campaign_dir, layouts)
        needle = str(args.only_file)
        hits = meas_df[meas_df["file"].astype(str).str.endswith(needle)]
        if hits.empty:
            hits = meas_df[meas_df["file"].astype(str) == needle]
        if hits.empty:
            raise SystemExit(f"No exposures match --only-file={args.only_file!r}")
        file_label = str(hits.iloc[0]["file"])
        global_idx = _index_by_file_layout(meas_df)
        fidx = file_meas_index(global_idx, file_label)
        registry = TileRegistry.for_file(
            file_label,
            layouts,
            fidx,
            mix_layout_names=mix_layouts,
            whole_layout_names=whole_layouts,
        )
        if args.count_only:
            n_mixed = count_complete_tilings(registry)
            payload = {"file": file_label, "n_mixed_tilings": n_mixed}
            out_path = out_dir / "single_file_tiling_count.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload, indent=2))
            return 0

        per_object, fallback, intrinsic_model = build_multi_layout_bias_tables(campaign_dir, layouts)
        meta_tbl = load_stellar_metadata(REPO_ROOT / "output")
        gid = str(hits.iloc[0]["gaia_dr3_id"])
        star_meta = {"logg": np.nan, "mh": np.nan}
        if not meta_tbl.empty:
            sm = meta_tbl[meta_tbl["gaia_dr3_id"] == gid]
            if len(sm):
                star_meta["logg"] = float(sm.iloc[0].get("logg", np.nan))
                star_meta["mh"] = float(sm.iloc[0].get("mh", np.nan))

        result = find_best_tiling_for_file(
            registry,
            per_object=per_object,
            fallback=fallback,
            intrinsic_model=intrinsic_model,
            star_meta=star_meta,
            metric=args.metric,
            max_tilings=args.max_tilings,
        )
        if result is None:
            raise SystemExit("No valid tiling found")
        logger.info(
            "%s: %d mixed tilings, %d evaluated",
            file_label,
            result.n_mixed_tilings,
            result.n_tilings_evaluated,
        )
        payload = {
            "file": file_label,
            "tiling_name": result.tiling_name,
            "metric": args.metric,
            "metric_value": result.metric_value,
            "stack": {k: (float(v) if hasattr(v, "__float__") else v) for k, v in result.stack.items()},
            "tiles": [t.name for t in result.tiles],
            "n_mixed_tilings": result.n_mixed_tilings,
            "n_tilings_evaluated": result.n_tilings_evaluated,
            "search_truncated": result.search_truncated,
        }
        out_path = out_dir / "single_file_best_tiling.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        print(f"Wrote {out_path}")
        return 0

    epochs, summary, counts = run_campaign_tiling_search(
        campaign_dir,
        max_tilings_per_file=args.max_tilings,
        metric=args.metric,
        mix_layout_names=mix_layouts,
        whole_layout_names=whole_layouts,
        count_tilings=args.count_only,
    )

    if args.count_only:
        counts_path = out_dir / "tiling_search_counts.csv"
        counts.to_csv(counts_path, index=False)
        print(f"Wrote {counts_path}")
        return 0

    epochs_path = out_dir / "best_mixed_tiling_epochs.csv"
    summary_path = out_dir / "best_mixed_tiling_summary.csv"
    counts_path = out_dir / "tiling_search_counts.csv"
    epochs.to_csv(epochs_path, index=False)
    summary.to_csv(summary_path, index=False)
    counts.to_csv(counts_path, index=False)

    if not summary.empty and "median_sigma_rv_kms" in summary.columns:
        med = float(summary.iloc[0]["median_sigma_rv_kms"])
        logger.info(
            "Best mixed tiling: %d exposures, median %s=%.4f km/s",
            len(epochs),
            args.metric,
            med,
        )
    print(f"Wrote {epochs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
