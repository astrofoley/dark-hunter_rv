#!/usr/bin/env python3
"""
Calibration entrypoint (quick reference). **Full automation** lives in
:mod:`validation.run_calibration_setup` (bias + offsets + manifest).

Example::

  python -m validation.run_calibration_setup \\
    --bias-list calibration/bias_train.txt \\
    --offset-list calibration/offset_train.txt \\
    --instrument APF \\
    --clean-after-bias

Production (everything else, skipping offset-training spectra that already have diagnostics)::

  python -m validation.run_production_remaining \\
    --manifest calibration/manifest.json \\
    --spectrum-list all_spectra.txt \\
    --instrument APF \\
    --update

See ``docs/operations.md`` for environment variables and paths.
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print calibration pointers; use run_calibration_setup for the full workflow.",
    )
    parser.parse_args(argv)
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
