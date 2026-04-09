"""JSON manifest for calibration setup vs production processing (path bookkeeping)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def new_manifest(*, instrument: str, repo_root: Path) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instrument": instrument,
        "repo_root": str(repo_root.resolve()),
        "bias_phase": {
            "spectrum_paths": [],
            "bias_build_subdir": "bias_build",
            "bias_statistics_installed": "",
            "cleaned_intermediates": False,
        },
        "offset_phase": {
            "spectrum_paths": [],
            "diagnostics_list_file": "",
            "method_rv_offsets_installed": "",
            "offsets_reprocessed": False,
        },
    }


def load_manifest(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(path: Path | str, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
