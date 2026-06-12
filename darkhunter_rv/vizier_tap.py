"""VizieR TAP sync queries (ADQL → CSV)."""

from __future__ import annotations

import csv
import io
import logging

import requests

logger = logging.getLogger(__name__)

VIZIER_TAP_SYNC = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


def execute_vizier_adql(query: str, *, timeout: float = 180.0) -> list[dict[str, str]]:
    """Run ADQL on VizieR TAP (sync, CSV)."""
    resp = requests.post(
        VIZIER_TAP_SYNC,
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        },
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text.strip()
        if detail.startswith("<"):
            detail = detail[:800]
        raise RuntimeError(f"VizieR TAP HTTP {resp.status_code}: {detail[:500]}")
    text = resp.text.strip()
    if text.startswith("<?xml") or text.startswith("<"):
        raise RuntimeError(f"VizieR TAP query failed: {text[:500]}")
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def sql_in_ints(values: list[int]) -> str:
    return ", ".join(str(int(v)) for v in values)


def parse_gaia_source_id(val: object) -> int | None:
    """Parse Gaia source_id from TAP CSV (string int; avoid float round-off)."""
    if val in (None, "", "null", "nan"):
        return None
    try:
        return int(str(val).strip())
    except (TypeError, ValueError):
        return None


def sql_max_err_clause(column: str, max_rv_err: float | None) -> str:
    """Optional ADQL ``AND column <= max`` fragment; empty when ``max_rv_err`` is None."""
    if max_rv_err is None:
        return ""
    return f"  AND {column} <= {float(max_rv_err)}\n"
