"""CLI for fitting a simple spectroscopic-binary RV model (SB1, circular)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .binary_rv import fit_circular_binary

logger = logging.getLogger(__name__)


def load_summary(path: Path):
    """Load whitespace table: file mjd rv rv_err rv_rms (as written by pipeline)."""
    mjd = []
    rv = []
    err = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            mjd.append(float(parts[1]))
            rv.append(float(parts[2]))
            err.append(float(parts[3]))
    return np.array(mjd), np.array(rv), np.array(err)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit SB1 circular orbit to RV summary")
    parser.add_argument("summary_file")
    parser.add_argument("--period", type=float, required=True, help="Period (days)")
    parser.add_argument("--out", default=None, help="Output prefix")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s", force=True)

    summ = Path(args.summary_file)
    t, rv, err = load_summary(summ)
    res = fit_circular_binary(t, rv, err, period_days=args.period)

    out = Path(args.out) if args.out else summ.with_suffix("")
    txt = out.with_suffix(".sb1.txt")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(f"gamma_kms {res.gamma:.6f}\n")
        fh.write(f"k_kms {res.k:.6f}\n")
        fh.write(f"period_days {res.period:.6f}\n")
        fh.write(f"t0_days {res.t0:.6f}\n")
        fh.write(f"success {res.success}\n")
        fh.write(f"message {res.message}\n")

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(t, rv, yerr=err, fmt="o", ms=4, capsize=2, label="data")
    ax.plot(t, res.rv_pred, "-", lw=1.2, label="SB1 circular")
    ax.set_xlabel("MJD")
    ax.set_ylabel("RV (km/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".sb1.png"), dpi=140)
    plt.close(fig)

    logger.info("Wrote %s and %s", txt, out.with_suffix(".sb1.png"))


if __name__ == "__main__":
    main()
