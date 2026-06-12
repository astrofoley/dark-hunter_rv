#!/usr/bin/env python3
"""Query DESI MWS RVs — thin wrapper around update_summary_external_rvs.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    prior = list(sys.argv)
    try:
        sys.argv = ["update_summary_external_rvs.py", "--sources", "desi", *prior[1:]]
        ext_path = _REPO / "scripts" / "update_summary_external_rvs.py"
        spec = importlib.util.spec_from_file_location("update_summary_external_rvs", ext_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {ext_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return int(mod.main())
    finally:
        sys.argv = prior


if __name__ == "__main__":
    print("update_summary_desi_rvs", flush=True)
    raise SystemExit(main())
