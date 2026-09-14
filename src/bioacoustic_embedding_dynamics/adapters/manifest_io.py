"""Shared JSONL manifest writer."""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest_jsonl(rows: list[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)
