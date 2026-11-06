"""CLI smoke runner for Experiment B using a tiny CPU model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .controller import KLControlConfig, KLWeightController
from .short_train import run_short_branch


def _demo(output, mode):
    torch.manual_seed(11)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    episodes = [{"x": torch.randn(8, 4), "y": torch.randint(0, 3, (8,)),
                 "relative_projection": -0.1 if i % 2 == 0 else 0.2} for i in range(8)]
    controller = KLWeightController(KLControlConfig(mode=mode, attenuation=0.5,
                                                     interference_threshold=0.0, seed=3))

    def loss_fn(m, episode, weight):
        logits = m(episode["x"])
        matching = torch.nn.functional.cross_entropy(logits, episode["y"])
        kl = logits.square().mean()
        return {"total": matching + weight * kl, "matching": matching, "kl": kl}

    return run_short_branch(model, optimizer, episodes, loss_fn, controller=controller,
                            output_dir=output, branch_name=mode, checkpoint_every=3,
                            reference_projections=[e["relative_projection"] for e in episodes])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--mode", choices=("original", "selective", "random", "uniform"), default="selective")
    parser.add_argument("--output", default="output/blocking_validation_demo")
    args = parser.parse_args(argv)
    if not args.demo:
        parser.error("provide --demo for the safe CPU smoke path; production runs use the Python API")
    print(_demo(args.output, args.mode))


if __name__ == "__main__":
    main()
