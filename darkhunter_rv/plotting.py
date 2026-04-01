"""Diagnostic figures for RV pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def plot_normalized_order(
    wave: np.ndarray,
    flux_norm: np.ndarray,
    continuum: np.ndarray | None,
    outpath: Path,
    title: str = "",
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wave, flux_norm, "k-", lw=0.6, label="norm flux")
    if continuum is not None and len(continuum) == len(wave):
        ax.plot(wave, continuum / np.nanmedian(continuum), "r--", lw=0.8, alpha=0.7, label="continuum (scaled)")
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Norm flux")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_ccf(vel: np.ndarray, ccf: np.ndarray, outpath: Path, title: str = "", peak_vel: float | None = None) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(vel, ccf, "b-", lw=0.8)
    if peak_vel is not None:
        ax.axvline(peak_vel, color="r", ls="--", label=f"peak ~ {peak_vel:.2f} km/s")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("CCF")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    logger.debug("saved %s", outpath)


def plot_chunk_rvs(
    chunk_keys: list,
    rvs: np.ndarray,
    errs: np.ndarray,
    outpath: Path,
    title: str = "",
) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(chunk_keys))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(x, rvs, yerr=errs, fmt="o", capsize=2, ms=3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in chunk_keys], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("RV (km/s)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_tournament_scores(names: list, scores: list, outpath: Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.2)))
    y = np.arange(len(names))
    ax.barh(y, scores)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Tournament score (sum CCF peaks)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def plot_balmer_panels(spec_data: dict, mjd: float, rv: float, outpath: Path) -> None:
    lines = {"Ha": 6562.8, "Hb": 4861.3, "Hg": 4340.5, "Hd": 4101.7}
    all_w, all_f = [], []
    for _o, d in spec_data.items():
        all_w.append(np.array(d["wavelength"]))
        all_f.append(np.array(d["flux"]))
    if not all_w:
        return
    flat_w = np.concatenate(all_w)
    flat_f = np.concatenate(all_f)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (name, rest) in zip(axes, lines.items()):
        m = (flat_w > rest - 50) & (flat_w < rest + 50)
        if np.sum(m) < 10:
            ax.set_visible(False)
            continue
        w, f = flat_w[m], flat_f[m]
        f = f / np.nanpercentile(f, 95)
        vel = config.C_KMS * (w - rest) / rest
        ax.plot(vel, f, lw=0.8)
        ax.axvline(rv, color="k", ls="--")
        ax.set_title(name)
        ax.set_xlabel("km/s")
    fig.suptitle(f"MJD {mjd:.4f}  RV ~ {rv:.1f} km/s")
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
