"""Aggregate branch JSONL files without selecting on Meta-Test."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


def summarize(root, output="summary.json") -> dict:
    root = Path(root)
    result = {}
    for log in sorted(root.glob("*/metrics.jsonl")):
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            continue
        result[log.parent.name] = {
            "steps": len(rows),
            "mean_total_loss": mean(r["total_loss"] for r in rows),
            "mean_matching_loss": mean(r["matching_loss"] for r in rows),
            "mean_kl_weight": mean(r["kl_weight"] for r in rows),
            "selected_steps": sum(bool(r.get("selected", False)) for r in rows),
            "last": rows[-1],
        }
    destination = root / output
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
