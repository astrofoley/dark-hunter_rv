#!/usr/bin/env python3
"""Backfill output/Gaia_DR3_*_summary.txt for stars with spectra but missing summaries.

Cron runs ``pipeline --update``, which skips spectra whose diagnostics CSV is newer than
the input. When every epoch for a star is skipped and no summary ever existed in
``output/``, populate/website staging never sees that star.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from darkhunter_rv.summary_paths import (
    count_pipeline_rows,
    discover_primary_epoch_files,
    discover_spec_gaia_ids,
)


def _needs_summary(
    gaia_id: str,
    *,
    spec_root: Path,
    out_dir: Path,
) -> tuple[bool, Path, list[Path]]:
    epoch_files = discover_primary_epoch_files(spec_root, gaia_id)
    if not epoch_files:
        return False, out_dir / f"Gaia_DR3_{gaia_id}_summary.txt", epoch_files
    summary_path = out_dir / f"Gaia_DR3_{gaia_id}_summary.txt"
    n_epochs = len(epoch_files)
    n_rows = count_pipeline_rows(summary_path) if summary_path.is_file() else 0
    return n_rows < n_epochs, summary_path, epoch_files


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run pipeline on stars whose summaries are missing or incomplete."
    )
    ap.add_argument("--spec-root", default="/data2/gaia_stars/apf_reductions")
    ap.add_argument("--output-dir", default=None, help="Pipeline output (default: REPO/output)")
    ap.add_argument("--gaia-id", default=None, help="Single Gaia DR3 source id")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--with-plots",
        action="store_true",
        help="Pass --plots --plots-focus to pipeline (default: measure only, faster).",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else repo / "output"
    spec_root = Path(args.spec_root)
    py = os.environ.get("PY", sys.executable)

    if args.gaia_id:
        gaia_ids = [str(args.gaia_id).strip()]
    else:
        gaia_ids = sorted(discover_spec_gaia_ids(spec_root))

    pending: list[tuple[str, list[Path]]] = []
    for gid in gaia_ids:
        need, _summary, epoch_files = _needs_summary(gid, spec_root=spec_root, out_dir=out_dir)
        if need:
            pending.append((gid, epoch_files))

    if not pending:
        print("ensure_pipeline_summaries: all spectra have complete summaries")
        return 0

    print(f"ensure_pipeline_summaries: {len(pending)} star(s) need summary backfill")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["DARKHUNTER_OUTPUT_DIR"] = str(out_dir)

    rc = 0
    for gid, epoch_files in pending:
        cmd = [
            py,
            "-m",
            "darkhunter_rv.pipeline",
            "--instrument",
            "APF",
            "--update",
            "--no-run-all-methods",
        ]
        if args.with_plots:
            cmd.extend(["--plots", "--plots-focus"])
        cmd.extend(str(p) for p in epoch_files)
        print(f"  Gaia_DR3_{gid}: {len(epoch_files)} epoch file(s)")
        if args.dry_run:
            continue
        proc = subprocess.run(cmd, cwd=repo, env=env)
        if proc.returncode != 0:
            print(f"[WARN] pipeline exit {proc.returncode} for Gaia_DR3_{gid}", file=sys.stderr)
            rc = proc.returncode

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
