#!/usr/bin/env python3
"""Calibrate systematic error floors from diagnostics residuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_diag(glob_expr: str) -> pd.DataFrame:
    fps = list(Path('.').glob(glob_expr))
    if not fps:
        raise SystemExit('No diagnostics files')
    return pd.concat([pd.read_csv(fp) for fp in fps], ignore_index=True)


def calibrate(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    coverage = []
    for method, g in df.groupby('method'):
        g = g.dropna(subset=['rv_kms','rv_err_kms'])
        if len(g) < 2:
            continue
        # Use per-file median as proxy truth in absence of absolute truth
        med = g.groupby('file')['rv_kms'].transform('median')
        resid = g['rv_kms'] - med
        stat2 = np.clip(g['rv_err_kms'].values, 1e-6, np.inf) ** 2
        sys2 = float(max(0.0, np.nanmean(resid.values**2 - stat2)))
        sys = float(np.sqrt(sys2))
        tot = np.sqrt(stat2 + sys2)
        z = np.abs(resid.values) / np.clip(tot, 1e-6, np.inf)
        cov1 = float(np.mean(z <= 1.0))
        cov2 = float(np.mean(z <= 2.0))
        rows.append({'method': method, 'sys_floor_kms': sys, 'n': int(len(g))})
        coverage.append({'method': method, 'coverage_1sigma': cov1, 'coverage_2sigma': cov2})

    floors = pd.DataFrame(rows)
    cov = pd.DataFrame(coverage)
    floors.to_csv(out_dir / 'systematic_floors.csv', index=False)
    cov.to_csv(out_dir / 'coverage_report.csv', index=False)
    if len(cov):
        x = np.arange(len(cov))
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(x, cov['coverage_1sigma'], 'o-', label='1σ')
        ax.plot(x, cov['coverage_2sigma'], 'o-', label='2σ')
        ax.set_xticks(x)
        ax.set_xticklabels(cov['method'], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / 'coverage.png', dpi=130)
        plt.close(fig)

    summary = {
        'n_methods': int(len(floors)),
        'methods': floors['method'].tolist() if len(floors) else [],
    }
    (out_dir / 'error_model_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description='Calibrate RV systematic errors')
    ap.add_argument('--diag-glob', default='output/*_diagnostics.csv')
    ap.add_argument('--out-dir', type=Path, default=Path('validation_output/error_model'))
    args = ap.parse_args()

    df = load_diag(args.diag_glob)
    s = calibrate(df, args.out_dir)
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
