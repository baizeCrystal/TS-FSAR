"""CLI smoke runner for Experiment A.

The production entry point is intentionally an API: construct the real
``TS_FSAR`` and optimizer, then pass a closure returning ``total``, ``kl`` and
``matching`` to :func:`counterfactual_update`.  ``--demo`` exercises the same
real AdamW path on CPU and is safe for CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .counterfactual import counterfactual_update


def _demo(output):
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x, y = torch.randn(8, 4), torch.randint(0, 3, (8,))

    def losses(branch):
        logits = model(x)
        matching = torch.nn.functional.cross_entropy(logits, y)
        kl = (logits - torch.zeros_like(logits)).square().mean()
        weight = {"A0": 1.0, "A1": 0.0, "A2": 0.5}[branch]
        return {"total": matching + weight * kl, "matching": matching, "kl": kl}

    result = counterfactual_update(model, optimizer, losses)
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run the CPU synthetic path")
    parser.add_argument("--output", default="counterfactual_result.json")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("provide --demo for the safe CPU smoke path; production runs use the Python API")
    _demo(args.output)


if __name__ == "__main__":
    main()
