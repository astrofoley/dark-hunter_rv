#!/usr/bin/env python3
"""Benchmark broad-line RV estimators on synthetic data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

C = 299792.458


def gaussian(x, a, mu, sig, c):
    return c - a * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def lorentz(x, a, mu, gamma, c):
    return c - a * (gamma**2 / ((x - mu) ** 2 + gamma**2))


def voigt_profile(x, a, mu, sigma, gamma, c):
    z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
    return c - a * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))


def fit_method(name, w, f, rest):
    try:
        if name == 'gaussian':
            p0 = [1.0 - np.min(f), w[np.argmin(f)], 2.0, 1.0]
            p, _ = curve_fit(gaussian, w, f, p0=p0, maxfev=5000)
            mu = p[1]
        elif name == 'lorentzian':
            p0 = [1.0 - np.min(f), w[np.argmin(f)], 2.0, 1.0]
            p, _ = curve_fit(lorentz, w, f, p0=p0, maxfev=5000)
            mu = p[1]
        elif name == 'voigt':
            p0 = [1.0 - np.min(f), w[np.argmin(f)], 1.5, 1.0, 1.0]
            p, _ = curve_fit(voigt_profile, w, f, p0=p0, maxfev=8000)
            mu = p[1]
        elif name == 'core':
            m = (w > rest - 3) & (w < rest + 3)
            wc, fc = w[m], f[m]
            p0 = [1.0 - np.min(fc), wc[np.argmin(fc)], 1.0, 1.0]
            p, _ = curve_fit(gaussian, wc, fc, p0=p0, maxfev=5000)
            mu = p[1]
        elif name == 'smoothed_min':
            k = 9
            y = np.convolve(f, np.ones(k)/k, mode='same')
            mu = w[np.argmin(y)]
        else:
            return np.nan
        return C * (mu - rest) / rest
    except Exception:
        return np.nan


def run(n=200, snr=80.0, vsig=2.0, out_dir=Path('validation_output/broad_line')):
    rng = np.random.default_rng(0)
    rest = 4861.3
    methods = ['gaussian', 'lorentzian', 'voigt', 'core', 'smoothed_min']
    rows = []
    for _ in range(n):
        true_rv = rng.normal(0, 20)
        mu = rest * (1 + true_rv / C)
        w = np.linspace(rest - 20, rest + 20, 400)
        f = 1.0 - 0.5 * np.exp(-0.5 * ((w - mu) / vsig) ** 2)
        noise = rng.normal(0, 1.0/snr, size=len(w))
        fn = f + noise
        for m in methods:
            rv = fit_method(m, w, fn, rest)
            rows.append({'method': m, 'true_rv': true_rv, 'est_rv': rv, 'err': rv - true_rv})
    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'broad_line_trials.csv', index=False)
    rep = df.groupby('method').agg(bias_kms=('err','mean'), rms_kms=('err','std'), n=('err','count')).reset_index()
    rep.to_csv(out_dir / 'broad_line_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    data = [df[df['method']==m]['err'].dropna().values for m in rep['method']]
    ax.boxplot(data, labels=rep['method'], showfliers=False)
    ax.axhline(0, color='k', ls='--', lw=0.8)
    ax.set_ylabel('RV error (km/s)')
    ax.set_title('Broad-line estimator error distribution')
    fig.tight_layout()
    fig.savefig(out_dir / 'broad_line_errors.png', dpi=130)
    plt.close(fig)
    best = rep.sort_values('rms_kms').iloc[0].to_dict()
    rec = f"# Broad-line method recommendation\n\nBest by synthetic RMS: **{best['method']}** (bias={best['bias_kms']:.3f} km/s, rms={best['rms_kms']:.3f} km/s).\n"
    Path('docs').mkdir(exist_ok=True)
    Path('docs/broad_line_method.md').write_text(rec)
    summary = {'best_method': best['method'], 'n_trials': int(n)}
    (out_dir / 'broad_line_summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description='Benchmark broad-line RV estimators')
    ap.add_argument('--n-trials', type=int, default=200)
    ap.add_argument('--snr', type=float, default=80.0)
    ap.add_argument('--sigma', type=float, default=2.0)
    ap.add_argument('--out-dir', type=Path, default=Path('validation_output/broad_line'))
    args = ap.parse_args()
    s = run(n=args.n_trials, snr=args.snr, vsig=args.sigma, out_dir=args.out_dir)
    print(json.dumps(s, indent=2))


if __name__ == '__main__':
    main()
