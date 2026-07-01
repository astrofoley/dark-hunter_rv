"""Copy Gaia DR3 plot PNGs from pipeline output into the website star tree."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

LEGACY_HBETA_GLOBS = (
    "*_h_beta_rv.png",
    "*_h_beta_order*.png",
    "*_h_beta_three*.png",
)


def gaia_plot_prefix(gaia_id: str) -> str:
    return f"Gaia_DR3_{gaia_id}"


def website_plots_dir(web_root: Path, gaia_id: str) -> Path:
    return web_root / "stars" / gaia_plot_prefix(gaia_id) / "Gaia" / "Plots"


def resolve_web_root(explicit: Optional[str] = None, *, sync_enabled: bool = True) -> Optional[Path]:
    """Return WEB_ROOT (or explicit path) when sync is enabled and the directory exists."""
    if not sync_enabled:
        return None
    raw = (explicit or os.environ.get("WEB_ROOT") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def prune_legacy_gaia_plots(plot_dir: Path) -> None:
    if not plot_dir.is_dir():
        return
    for pattern in LEGACY_HBETA_GLOBS:
        for path in plot_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def stage_gaia_plots(
    gaia_id: str,
    src_plot_dir: Path,
    web_root: Path,
    *,
    reports_dir: Optional[Path] = None,
) -> int:
    """Mirror scripts/lib/website_plot_sync.sh: stage contract PNGs for one star."""
    dest_dir = website_plots_dir(web_root, gaia_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = gaia_plot_prefix(gaia_id)
    staged = 0
    for name in (
        f"{prefix}_28_hbeta.png",
        f"{prefix}_rv_plot.png",
        f"{prefix}_keplerian_residuals.png",
    ):
        src = src_plot_dir / name
        if not src.is_file() and name.endswith("_keplerian_residuals.png") and reports_dir is not None:
            src = reports_dir / f"{gaia_id}_keplerian_residuals.png"
        if not src.is_file():
            continue
        shutil.copy2(src, dest_dir / name)
        staged += 1
    prune_legacy_gaia_plots(dest_dir)
    return staged


def maybe_stage_gaia_plots(
    gaia_id: str,
    src_plot_dir: Path,
    *,
    web_root: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
) -> int:
    if web_root is None:
        return 0
    n = stage_gaia_plots(gaia_id, src_plot_dir, web_root, reports_dir=reports_dir)
    if n:
        print(
            f"[sync] Gaia_DR3_{gaia_id}: staged {n} plot(s) -> {website_plots_dir(web_root, gaia_id)}",
            flush=True,
        )
    return n
