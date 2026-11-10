"""Offline paired analysis; it never invokes the model or reads feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import paired_condition_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="JSON list or JSONL query-level branch output")
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = ap.parse_args()
    path = Path(args.rows)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    conditions = sorted({str(r["condition"]) for r in rows if str(r["condition"]) != "E0"})
    report = {
        "source": str(path),
        "feature_cache_used": False,
        "conditions": {
            c: paired_condition_summary(rows, condition=c, bootstrap_repeats=args.bootstrap_repeats)
            for c in conditions
        },
        "case_counts": {},
    }
    for c in conditions:
        summary = report["conditions"][c]
        report["case_counts"][c] = {
            "correct_to_wrong": summary["correct_to_wrong"]["estimate"],
            "wrong_to_correct": summary["wrong_to_correct"]["estimate"],
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

