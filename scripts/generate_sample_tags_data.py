#!/usr/bin/env python3
"""Regenerate website/rv/sample_tags_data.js from tables/sample_tags.json."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "website/rv/tables/sample_tags.json"
    dst = repo / "website/rv/sample_tags_data.js"
    data = json.loads(src.read_text(encoding="utf-8"))
    dst.write_text(
        "window.__SAMPLE_TAG_DATA__ = " + json.dumps(data, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
