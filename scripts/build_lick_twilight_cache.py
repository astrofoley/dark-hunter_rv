#!/usr/bin/env python3
"""Download UCO/Lick nautical (-12 deg) twilight tables into a local JSON cache."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from darkhunter_rv.lick_twilight_cache import (
    build_cache_years,
    build_doy_anchor_table,
    default_cache_path,
    load_cache,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build rv_fit_reports/lick_twilight_cache.json from "
            "https://mthamilton.ucolick.org/cgi-bin/lick_calendar_form.pl/ "
            "(nautical 12 deg tables)."
        )
    )
    ap.add_argument("--cache", default=None, help="Output JSON path")
    ap.add_argument(
        "--years",
        default=None,
        help="Comma-separated years (default: last year, this year, next year)",
    )
    args = ap.parse_args()

    cache_path = Path(args.cache) if args.cache else default_cache_path()
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    else:
        y = datetime.utcnow().year
        years = [y - 1, y, y + 1]

    payload = build_cache_years(years, cache_path=cache_path)
    doy_path = cache_path.parent / "lick_doy_anchors.json"
    doy_payload = build_doy_anchor_table(cache_path=cache_path, doy_path=doy_path)
    print(f"Wrote {len(payload['nights'])} nautical nights to {cache_path} (years={years}).")
    print(f"Wrote {len(doy_payload.get('rows') or [])} DOY anchors to {doy_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
