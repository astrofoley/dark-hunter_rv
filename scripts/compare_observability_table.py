#!/usr/bin/env python3
"""Print our Lick visibility numbers vs the Gaia-4491 planning-table reference."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from darkhunter_rv.observability_table_compare import compare_reference_table


def main() -> int:
    for line in compare_reference_table():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
