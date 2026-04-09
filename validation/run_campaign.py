#!/usr/bin/env python3
"""Run full validation campaign and emit a consolidated validation_report.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation import build_bias_set, evaluate_method_consistency, benchmark_broad_lines, calibrate_error_model


def main() -> None:
    ap = argparse.ArgumentParser(description='Run RV validation campaign')
    ap.add_argument('--orders-dir', type=Path, default=Path('output'))
    ap.add_argument('--diag-glob', default='output/*_diagnostics.csv')
    ap.add_argument('--out-dir', type=Path, default=Path('validation_output/campaign'))
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    bias = build_bias_set.build_bias(args.orders_dir, n_boot=200, out_dir=out / 'bias')
    import pandas as pd
    df = evaluate_method_consistency.load_diag(args.diag_glob)
    cons = evaluate_method_consistency.run(df, out / 'consistency')
    broad = benchmark_broad_lines.run(out_dir=out / 'broad_line')
    err = calibrate_error_model.calibrate(df, out / 'error_model')

    report = {
        'bias': bias,
        'consistency': cons,
        'broad_line': broad,
        'error_model': err,
    }
    (out / 'validation_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
