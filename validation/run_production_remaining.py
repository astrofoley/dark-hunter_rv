#!/usr/bin/env python3
"""
Production batch: measure RVs for spectra **not** already fully handled in the offset-calibration phase.

Reads ``calibration/manifest.json`` (from :mod:`validation.run_calibration_setup`). Any spectrum path
listed under ``offset_phase.spectrum_paths`` that already has ``output/<stem>_diagnostics.csv`` is
**skipped** by default, so Gaia/metadata are read from existing ``Gaia_DR3_*_summary.txt`` when the
pipeline runs on **new** files (limited archive queries).

Typical cron / batch::

  python -m validation.run_production_remaining \\
    --manifest calibration/manifest.json \\
    --spectrum-list all_spectra.txt \\
    --instrument APF \\
    --update

Use ``--include-offset-calibration`` to reprocess offset-training spectra too (e.g. after code change).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config as dh_config  # noqa: E402
from darkhunter_rv.calibration_manifest import load_manifest  # noqa: E402


def _read_path_list(path: Path) -> list[Path]:
    out: list[Path] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Listed file not found: {p}")
        out.append(p.resolve())
    return out


def _offset_training_resolved(manifest: dict) -> set[str]:
    raw = manifest.get("offset_phase", {}).get("spectrum_paths", [])
    return {str(Path(p).resolve()) for p in raw}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Process spectra, skipping offset-calibration training set")
    ap.add_argument("--manifest", type=Path, default=_REPO_ROOT / "calibration" / "manifest.json")
    ap.add_argument("--spectrum-list", type=Path, required=True, help="One spectrum path per line")
    ap.add_argument("--instrument", default="APF")
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--include-offset-calibration", action="store_true", help="Do not skip offset-training spectra")
    ap.add_argument("--update", action="store_true", help="Pass through to pipeline (skip up-to-date diagnostics)")
    ap.add_argument("--force", action="store_true", help="Pass through to pipeline with --update")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    repo_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest.resolve())
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(dh_config.OUTPUT_DIR).resolve()

    offset_set = _offset_training_resolved(manifest)
    all_paths = _read_path_list(args.spectrum_list.resolve())

    to_run: list[Path] = []
    skipped_offset = 0
    skipped_done = 0
    for p in all_paths:
        pr = str(p.resolve())
        diag = output_dir / f"{p.stem}_diagnostics.csv"
        if pr in offset_set and not args.include_offset_calibration and diag.is_file():
            logging.info("skip offset-training (already have diagnostics): %s", p.name)
            skipped_offset += 1
            continue
        if args.update and not args.force and diag.is_file():
            try:
                if diag.stat().st_mtime >= p.stat().st_mtime:
                    skipped_done += 1
                    continue
            except OSError:
                pass
        to_run.append(p)

    logging.info(
        "production: %d to run, %d skipped (offset training w/ diag), %d skipped (update mtime)",
        len(to_run),
        skipped_offset,
        skipped_done,
    )
    if not to_run:
        return 0

    batch_size = 40
    env = {**os.environ, "PYTHONPATH": str(repo_root)}

    for i in range(0, len(to_run), batch_size):
        chunk = to_run[i : i + batch_size]
        cmd = [
            sys.executable,
            "-m",
            "darkhunter_rv.pipeline",
            *[str(p) for p in chunk],
            "--instrument",
            args.instrument,
            "--log-level",
            args.log_level,
        ]
        if args.update:
            cmd.append("--update")
        if args.force:
            cmd.append("--force")
        logging.info("pipeline batch starting at %d (%d files)", i, len(chunk))
        r = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if r.returncode != 0:
            return r.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
