"""Phase gates: precision metrics must not regress between CCF estimator study phases."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRIC_LOWER_IS_BETTER = frozenset(
    {
        "median_sigma_rv_kms",
        "p90_sigma_rv_kms",
        "median_chunk_scatter_kms",
        "bias_curve_rms_kms",
        "stellar_bias_cv_rmse_kms",
    }
)

METRIC_HIGHER_IS_BETTER = frozenset({"low_snr_finite_rate"})

DEFAULT_TOLERANCE = {
    "median_sigma_rv_kms": 0.005,
    "p90_sigma_rv_kms": 0.008,
    "median_chunk_scatter_kms": 0.005,
    "bias_curve_rms_kms": 0.003,
    "stellar_bias_cv_rmse_kms": 0.003,
    "low_snr_finite_rate": 0.02,
}


def load_baseline(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in data:
        raise ValueError(f"baseline missing metrics: {path}")
    return data


def save_baseline(
    path: Path,
    *,
    phase: str,
    metrics: dict[str, float],
    estimator: str = "gauss_offset",
    tolerance: dict[str, float] | None = None,
    notes: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "estimator": estimator,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: float(v) for k, v in metrics.items() if np.isfinite(float(v))},
        "tolerance": dict(tolerance or DEFAULT_TOLERANCE),
        "notes": notes,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metric_ok(
    key: str,
    new_val: float,
    prior_val: float,
    tol: float,
) -> tuple[bool, str]:
    if not np.isfinite(new_val) or not np.isfinite(prior_val):
        return True, f"{key}: skip (non-finite)"
    if key in METRIC_LOWER_IS_BETTER:
        ok = new_val <= prior_val + tol
        return ok, f"{key}: {new_val:.4f} vs prior {prior_val:.4f} (tol +{tol})"
    if key in METRIC_HIGHER_IS_BETTER:
        ok = new_val >= prior_val - tol
        return ok, f"{key}: {new_val:.4f} vs prior {prior_val:.4f} (tol -{tol})"
    return True, f"{key}: ungated"


def check_phase_gate(
    phase: str,
    metrics: dict[str, float],
    prior: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Compare ``metrics`` against ``prior`` baseline.

    Returns dict with ``passed``, ``failures``, ``checks``.
    """
    prior_metrics = prior.get("metrics", prior)
    tolerance = prior.get("tolerance", DEFAULT_TOLERANCE)
    checks: list[str] = []
    failures: list[str] = []

    keys = set(prior_metrics.keys()) | set(metrics.keys())
    gate_keys = METRIC_LOWER_IS_BETTER | METRIC_HIGHER_IS_BETTER

    for key in sorted(keys):
        if key not in gate_keys:
            continue
        new_val = float(metrics.get(key, np.nan))
        prior_val = float(prior_metrics.get(key, np.nan))
        tol = float(tolerance.get(key, DEFAULT_TOLERANCE.get(key, 0.0)))
        ok, msg = _metric_ok(key, new_val, prior_val, tol)
        checks.append(msg)
        if not ok:
            failures.append(msg)

    passed = len(failures) == 0
    if strict and not passed:
        return {
            "phase": phase,
            "passed": False,
            "failures": failures,
            "checks": checks,
        }
    return {
        "phase": phase,
        "passed": passed,
        "failures": failures,
        "checks": checks,
    }


def update_baseline_if_passed(
    baseline_path: Path,
    *,
    phase: str,
    metrics: dict[str, float],
    gate_result: dict[str, Any],
    estimator: str,
) -> bool:
    if not gate_result.get("passed"):
        return False
    save_baseline(
        baseline_path,
        phase=phase,
        metrics=metrics,
        estimator=estimator,
        notes=f"Updated after phase {phase} gate pass",
    )
    return True
