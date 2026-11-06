"""Experiment B: short, paired continuation from one checkpoint."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .controller import KLWeightController
from .protocol import episode_sequence_hash


def run_short_branch(model, optimizer, episodes: Iterable[object], loss_fn: Callable,
                     *, controller: KLWeightController, output_dir, branch_name: str,
                     scheduler=None, scaler=None, start_episode: int = 0,
                     max_steps: Optional[int] = None, checkpoint_every: int = 50,
                     evaluator: Optional[Callable] = None, metadata: Optional[dict] = None,
                     stop_on_nonfinite: bool = True,
                     reference_projections: Optional[Iterable[float]] = None) -> dict:
    """Train one B branch with resumable checkpoints and JSONL diagnostics.

    ``loss_fn(model, episode, kl_weight)`` returns a mapping containing
    ``total``, ``kl``, ``matching`` and optionally ``metrics``.  The episode
    sequence is consumed identically by each caller; its hash is written to the
    manifest so accidental branch drift is detectable.
    """
    episodes = list(episodes)
    out = Path(output_dir) / branch_name
    out.mkdir(parents=True, exist_ok=True)
    projections = []
    if reference_projections is not None:
        projections = [float(x) for x in reference_projections]
    elif controller.config.mode in {"selective", "random", "uniform"}:
        # Selective decisions are based on values measured before the run.
        projections = [float(getattr(e, "relative_projection", 0.0)) if not isinstance(e, Mapping)
                       else float(e.get("relative_projection", 0.0)) for e in episodes]
    decisions = controller.plan(projections if projections else [0.0] * len(episodes))
    if start_episode:
        decisions = decisions[start_episode:]
    log_path = out / "metrics.jsonl"
    steps = 0
    for offset, episode in enumerate(episodes[start_episode:], start=start_episode):
        if max_steps is not None and steps >= max_steps:
            break
        decision = decisions[offset - start_episode]
        optimizer.zero_grad(set_to_none=True)
        result = dict(loss_fn(model, episode, decision.weight))
        total = result["total"]
        if stop_on_nonfinite and not torch.isfinite(total.detach()):
            raise FloatingPointError(f"non-finite loss at episode {offset}: {total}")
        if scaler is None:
            total.backward()
            optimizer.step()
        else:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        row = {"episode": offset, "branch": branch_name, "kl_weight": decision.weight,
               "selected": decision.selected, "reason": decision.reason,
               "kl_loss": float(result["kl"].detach()),
               "matching_loss": float(result["matching"].detach()),
               "total_loss": float(total.detach()), **dict(result.get("metrics", {}))}
        if evaluator is not None:
            row.update({f"eval_{k}": v for k, v in evaluator(model, episode).items()})
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        steps += 1
        if checkpoint_every and (steps % checkpoint_every == 0):
            save_checkpoint(out / "latest.pt", model, optimizer, scheduler=scheduler,
                            scaler=scaler, next_episode=offset + 1,
                            metadata={**(metadata or {}), "branch": branch_name,
                                      "episode_hash": episode_sequence_hash(episodes)})
    save_checkpoint(out / "latest.pt", model, optimizer, scheduler=scheduler, scaler=scaler,
                    next_episode=start_episode + steps,
                    metadata={**(metadata or {}), "branch": branch_name,
                              "episode_hash": episode_sequence_hash(episodes)})
    summary = {"branch": branch_name, "steps": steps, "episode_hash": episode_sequence_hash(episodes),
               "checkpoint": str(out / "latest.pt"), "status": "completed"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
