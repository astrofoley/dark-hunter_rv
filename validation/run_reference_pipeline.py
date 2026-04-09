#!/usr/bin/env python3
"""
Full Dark Hunter pipeline with ``--run-all-methods`` and ``--plots``.

This is the same code path as ``python -m darkhunter_rv`` (``process_spectrum``): diagnostics CSV,
summary lines, mask + template FFT + strong_lines rows, and PNGs. Use it as the canonical
“real” reprocess when validating algorithm changes.

Gaia-style filenames (``Gaia_DR3_<source_id>_...``) pull Teff / log g / [M/H] from the cached star
summary when present, identical to the main CLI.

Examples::

  python validation/run_reference_pipeline.py data/Gaia_DR3_468391369318487040_epoch_1.txt

  python validation/run_reference_pipeline.py data/a.txt data/b.txt \\
    --plot-dir validation_output/my_batch_plots -- --qc-config order_chunk_qc.yaml

Any arguments after ``--`` are forwarded; without ``--``, unknown flags can be passed if listed
after the spectrum paths (use ``--`` if the first extra flag could be mistaken for this script’s).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from darkhunter_rv import config
from darkhunter_rv.pipeline import main as pipeline_main


def main() -> None:
    raw = sys.argv[1:]
    if "--" in raw:
        i = raw.index("--")
        main_part, passthrough = raw[:i], raw[i + 1 :]
    else:
        main_part, passthrough = raw, []

    parser = argparse.ArgumentParser(
        description="Run pipeline with --run-all-methods and --plots (reference / validation workflow)."
    )
    parser.add_argument(
        "spectra",
        nargs="+",
        type=Path,
        help="One or more spectrum files.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: validation_output/pipeline_plots_<first_stem>)",
    )
    parser.add_argument(
        "--instrument",
        default="APF",
        help="Instrument name (default: APF)",
    )
    args = parser.parse_args(main_part)

    first_stem = args.spectra[0].stem
    plot_dir = args.plot_dir
    if plot_dir is None:
        plot_dir = config.REPO_ROOT / "validation_output" / f"pipeline_plots_{first_stem}"
    plot_dir = plot_dir.resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    argv: list[str] = []
    for sp in args.spectra:
        argv.append(str(sp.resolve()))
    argv += [
        "--instrument",
        str(args.instrument),
        "--run-all-methods",
        "--plots",
        "--plot-dir",
        str(plot_dir),
    ]
    argv += passthrough
    pipeline_main(argv)


if __name__ == "__main__":
    main()
