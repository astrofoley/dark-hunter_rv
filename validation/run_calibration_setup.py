#!/usr/bin/env python3
"""
One-shot **calibration setup**: mask-only bias training, build ``bias_statistics.txt``, optional
cleanup, then multi-method spectra for **method offsets**, compute ``method_rv_offsets.txt``, install
artifacts at the repo root, and write ``calibration/manifest.json``.

Stages
------

1. **Bias (mask only)** — For each path in ``--bias-list``, run::

     python -m darkhunter_rv.pipeline <spec> --mask-only --no-bias ...

   Chunk RVs in ``output/*_orders.txt`` are stellar-mask-only. Then
   :mod:`validation.build_bias_set` aggregates residuals → ``bias_statistics.txt`` (installed next
   to :data:`darkhunter_rv.config.REPO_ROOT`). If ``--clean-after-bias``, removes those spectra's
   ``*_orders.txt`` and ``*_diagnostics.csv`` from ``output/`` (not plots unless ``--clean-plots``).

2. **Method offsets** — For each path in ``--offset-list``, run the **full** default pipeline
   (multi-method, **with** bias). Write ``calibration/offset_diagnostics_list.txt`` and run
   :mod:`validation.compute_method_rv_offsets` → ``method_rv_offsets.txt`` at repo root.

3. **Optional** ``--rerun-offset-with-corrections`` — Re-run the offset-list spectra so adopted RVs
   pick up template/strong offsets (manifest records this).

**Note:** Root ``rv_bias.py`` targets an older ``*_orders.txt`` column layout (epoch/suborder).
This workflow uses :mod:`validation.build_bias_set`, which matches current chunk keys
(``0_a``, …).

Example (full calibration)::

  python -m validation.run_calibration_setup \\
    --bias-list calibration/bias_train.txt \\
    --offset-list calibration/offset_train.txt \\
    --instrument APF \\
    --clean-after-bias

Mask debias only (``subchunks_8`` production layout)::

  python -m validation.run_calibration_setup \\
    --bias-list calibration/bias_train.txt \\
    --bias-only \\
    --chunk-layout calibration/chunk_layouts/subchunks_8.yaml \\
    --clean-after-bias

Or: ``bash scripts/rebuild_mask_bias.sh`` (``SKIP_PIPELINE=1`` to re-aggregate existing ``*_orders.txt``).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config as dh_config  # noqa: E402
from darkhunter_rv.calibration_manifest import new_manifest, save_manifest  # noqa: E402
from validation.build_bias_set import build_bias  # noqa: E402


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


def _run_pipeline_batch(
    paths: list[Path],
    repo_root: Path,
    instrument: str,
    extra_flags: list[str],
    log_level: str,
) -> None:
    batch_size = 40
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        cmd = [
            sys.executable,
            "-m",
            "darkhunter_rv.pipeline",
            *[str(p) for p in chunk],
            "--instrument",
            instrument,
            "--log-level",
            log_level,
            *extra_flags,
        ]
        logging.info("pipeline batch %d..%d (%d files)", i, min(i + batch_size, len(paths)), len(chunk))
        r = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if r.returncode != 0:
            raise RuntimeError(f"pipeline failed with code {r.returncode}")


def _clean_bias_intermediates(
    paths: list[Path],
    output_dir: Path,
    plot_dir: Path | None,
    clean_plots: bool,
) -> None:
    for p in paths:
        stem = p.stem
        for suffix in ("_orders.txt", "_diagnostics.csv"):
            fp = output_dir / f"{stem}{suffix}"
            if fp.is_file():
                fp.unlink()
                logging.info("removed %s", fp)
        if clean_plots and plot_dir is not None and plot_dir.is_dir():
            for png in plot_dir.glob(f"{stem}*.png"):
                png.unlink()
                logging.info("removed %s", png)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Full calibration: mask bias + method offsets + manifest")
    ap.add_argument("--bias-list", type=Path, required=True, help="Text file: one spectrum path per line")
    ap.add_argument(
        "--offset-list",
        type=Path,
        default=None,
        help="Spectra in method-overlap region (required unless --bias-only)",
    )
    ap.add_argument(
        "--bias-only",
        action="store_true",
        help="Mask bias training + install bias_statistics.txt only (skip method offsets)",
    )
    ap.add_argument("--instrument", default="APF")
    ap.add_argument(
        "--chunk-layout",
        type=Path,
        default=None,
        help="YAML chunk layout for all pipeline batches (default: config.DEFAULT_CHUNK_LAYOUT)",
    )
    ap.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    ap.add_argument("--calibration-dir", type=Path, default=_REPO_ROOT / "calibration")
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: config.OUTPUT_DIR")
    ap.add_argument("--plot-dir", type=Path, default=None, help="Default: config.PLOT_DIR")
    ap.add_argument("--clean-after-bias", action="store_true", help="Delete *_orders.txt and *_diagnostics.csv for bias spectra")
    ap.add_argument("--clean-plots", action="store_true", help="With --clean-after-bias, also delete plot_dir/<stem>*.png")
    ap.add_argument("--skip-bias-pipeline", action="store_true", help="Assume mask-only pipeline already ran; only build bias table")
    ap.add_argument("--skip-offset-pipeline", action="store_true", help="Assume offset diagnostics exist; only compute offsets file")
    ap.add_argument(
        "--rerun-offset-with-corrections",
        action="store_true",
        help="After installing method_rv_offsets.txt, re-run pipeline on offset list (same inputs, new adopted RVs)",
    )
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    repo_root = args.repo_root.resolve()
    output_dir = Path(args.output_dir) if args.output_dir else dh_config.OUTPUT_DIR
    output_dir = output_dir.resolve()
    plot_dir = Path(args.plot_dir).resolve() if args.plot_dir else Path(dh_config.PLOT_DIR).resolve()

    bias_paths = _read_path_list(args.bias_list.resolve())
    bias_stems = {p.stem for p in bias_paths}

    if args.bias_only:
        offset_paths: list[Path] = []
    else:
        if args.offset_list is None:
            logging.error("--offset-list is required unless --bias-only")
            return 2
        offset_paths = _read_path_list(args.offset_list.resolve())
        if not offset_paths:
            logging.error("offset-list is empty (need at least one spectrum for method offsets)")
            return 2

    cal_dir = args.calibration_dir.resolve()
    cal_dir.mkdir(parents=True, exist_ok=True)
    bias_build = cal_dir / "bias_build"
    diag_list_path = cal_dir / "offset_diagnostics_list.txt"
    manifest_path = cal_dir / "manifest.json"

    manifest = new_manifest(instrument=args.instrument, repo_root=repo_root)

    bias_flags = ["--mask-only", "--no-bias"]
    full_flags: list[str] = []
    if args.chunk_layout is not None:
        layout_flag = ["--chunk-layout", str(args.chunk_layout.resolve())]
        bias_flags.extend(layout_flag)
        full_flags.extend(layout_flag)
    elif dh_config.DEFAULT_CHUNK_LAYOUT is not None:
        layout_flag = ["--chunk-layout", str(dh_config.DEFAULT_CHUNK_LAYOUT.resolve())]
        bias_flags.extend(layout_flag)
        full_flags.extend(layout_flag)

    if not bias_paths and not args.skip_bias_pipeline:
        logging.warning("Bias list is empty — bias file may be trivial")

    if not args.skip_bias_pipeline:
        _run_pipeline_batch(bias_paths, repo_root, args.instrument, bias_flags, args.log_level)

    summary = build_bias(
        output_dir,
        args.bootstrap,
        bias_build,
        spectrum_stems=bias_stems,
    )
    logging.info("bias build summary: %s", summary)
    if summary.get("n_files", 0) == 0:
        logging.error("No bias training *_orders.txt found for %d listed spectra", len(bias_paths))
        return 2
    for name in ("bias_summary.json", "bias_by_chunk.csv"):
        src = bias_build / name
        if src.is_file():
            shutil.copy2(src, cal_dir / name)
            logging.info("copied %s -> %s", src.name, cal_dir / name)

    bias_installed = repo_root / "bias_statistics.txt"
    built_bias = bias_build / "bias_statistics.txt"
    if built_bias.is_file():
        shutil.copy2(built_bias, bias_installed)
        logging.info("Installed %s", bias_installed)
    else:
        logging.error("Missing %s — cannot install bias file", built_bias)
        return 2

    manifest["bias_phase"]["spectrum_paths"] = [str(p) for p in bias_paths]
    try:
        manifest["bias_phase"]["bias_build_subdir"] = str(bias_build.relative_to(repo_root))
    except ValueError:
        manifest["bias_phase"]["bias_build_subdir"] = str(bias_build)
    manifest["bias_phase"]["bias_statistics_installed"] = str(bias_installed)
    if args.clean_after_bias:
        _clean_bias_intermediates(bias_paths, output_dir, plot_dir if args.clean_plots else None, args.clean_plots)
        manifest["bias_phase"]["cleaned_intermediates"] = True

    if args.bias_only:
        save_manifest(manifest_path, manifest)
        logging.info("Wrote %s (bias-only)", manifest_path)
        return 0

    if not args.skip_offset_pipeline:
        _run_pipeline_batch(offset_paths, repo_root, args.instrument, full_flags, args.log_level)

    diag_list_path.write_text(
        "\n".join(str((output_dir / f"{p.stem}_diagnostics.csv").resolve()) for p in offset_paths if p)
        + "\n",
        encoding="utf-8",
    )
    missing = [p for p in offset_paths if not (output_dir / f"{p.stem}_diagnostics.csv").is_file()]
    if missing:
        logging.error("Missing diagnostics for %d spectra (first: %s)", len(missing), missing[0])
        return 2

    mo_out = cal_dir / "method_rv_offsets.txt"
    cmd_mo = [
        sys.executable,
        "-m",
        "validation.compute_method_rv_offsets",
        "--diagnostics-list",
        str(diag_list_path),
        "--instrument",
        args.instrument,
        "--output",
        str(mo_out),
    ]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    r = subprocess.run(cmd_mo, cwd=str(repo_root), env=env)
    if r.returncode != 0:
        return r.returncode

    mo_installed = repo_root / "method_rv_offsets.txt"
    if mo_out.is_file():
        shutil.copy2(mo_out, mo_installed)
        logging.info("Installed %s", mo_installed)

    manifest["offset_phase"]["spectrum_paths"] = [str(p) for p in offset_paths]
    manifest["offset_phase"]["diagnostics_list_file"] = str(diag_list_path)
    manifest["offset_phase"]["method_rv_offsets_installed"] = str(mo_installed)

    if args.rerun_offset_with_corrections:
        _run_pipeline_batch(offset_paths, repo_root, args.instrument, full_flags, args.log_level)
        manifest["offset_phase"]["offsets_reprocessed"] = True

    save_manifest(manifest_path, manifest)
    logging.info("Wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
