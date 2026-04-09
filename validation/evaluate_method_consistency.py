#!/usr/bin/env python3
"""Evaluate mask/template/line consistency from diagnostics CSV files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_val_dir = Path(__file__).resolve().parent
if str(_val_dir) not in sys.path:
    sys.path.insert(0, str(_val_dir))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from method_pair_stats import compute_method_pair_table


def load_diag(path_glob: str) -> pd.DataFrame:
    files = [Path(p) for p in Path('.').glob(path_glob)] if any(c in path_glob for c in '*?[') else [Path(path_glob)]
    frames = []
    for fp in files:
        if fp.exists() and fp.suffix == '.csv':
            df = pd.read_csv(fp)
            if 'method' in df.columns and 'rv_kms' in df.columns:
                frames.append(df)
    if not frames:
        raise SystemExit("No diagnostics CSV found")
    return pd.concat(frames, ignore_index=True)



def run(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pairs = compute_method_pair_table(df)
    methods = sorted(df['method'].dropna().unique().tolist())
    out_pairs.to_csv(out_dir / 'method_pair_offsets.csv', index=False)
    if len(out_pairs):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(np.arange(len(out_pairs)), out_pairs['median_offset_kms'])
        ax.set_xticks(np.arange(len(out_pairs)))
        ax.set_xticklabels([f"{a}-{b}" for a,b in zip(out_pairs['method_a'], out_pairs['method_b'])], rotation=45, ha='right')
        ax.set_ylabel('Median offset (km/s)')
        fig.tight_layout()
        fig.savefig(out_dir / 'method_pair_offsets.png', dpi=130)
        plt.close(fig)

    # rough trend diagnostics
    trend_rows = []
    for m in methods:
        g = df[df['method'] == m]
        for xcol in ['teff', 'telluric_fraction', 'mask_line_count']:
            if xcol in g.columns:
                tmp = g[[xcol, 'rv_kms']].dropna()
                if len(tmp) < 5:
                    continue
                x = tmp[xcol].values
                y = tmp['rv_kms'].values
                corr = np.corrcoef(x, y)[0,1] if len(x) > 2 else np.nan
                trend_rows.append({'method': m, 'x': xcol, 'corr': float(corr)})
    pd.DataFrame(trend_rows).to_csv(out_dir / 'method_trends.csv', index=False)

    summary = {
        'n_rows': int(len(df)),
        'methods': sorted(df['method'].dropna().unique().tolist()),
        'n_pairs': int(len(out_pairs)),
    }
    (out_dir / 'method_consistency_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description='Method consistency campaign')
    ap.add_argument('--diag-glob', default='output/*_diagnostics.csv')
    ap.add_argument('--out-dir', type=Path, default=Path('validation_output/consistency'))
    args = ap.parse_args()

    df = load_diag(args.diag_glob)
    s = run(df, args.out_dir)
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
