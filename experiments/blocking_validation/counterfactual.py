"""Experiment A: paired, real-optimizer counterfactual episode updates."""

from __future__ import annotations

import copy
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch

from .metrics import gradient_metrics
from .state import restore, snapshot


def _grads(loss, params, retain_graph=True):
    return torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                allow_unused=True, create_graph=False)


def counterfactual_update(model, optimizer, loss_fn: Callable[[str], Mapping[str, torch.Tensor]],
                          *, branches: Sequence[str] = ("A0", "A1", "A2"),
                          evaluator: Optional[Callable[[str, object], Mapping[str, float]]] = None,
                          scheduler=None, scaler=None, grad_clip_norm: Optional[float] = None,
                          scheduler_step: Optional[Callable[[], None]] = None) -> Dict[str, object]:
    """Run each branch from one exact state and return diagnostics.

    ``loss_fn(branch)`` must return ``total``, ``kl`` and ``matching`` tensors.
    The branch itself controls only KL's coefficient; all other terms stay in
    the closure.  The optimizer, scheduler, AMP scaler, and RNG are restored
    before every branch, so AdamW momentum and adaptive scaling are retained.
    """
    base = snapshot(model, optimizer, scheduler, scaler)
    params = [p for p in model.parameters() if p.requires_grad]
    results: Dict[str, object] = {}
    for branch in branches:
        restore(base, model, optimizer, scheduler, scaler)
        optimizer.zero_grad(set_to_none=True)
        losses = dict(loss_fn(branch))
        for key in ("total", "kl", "matching"):
            if key not in losses:
                raise KeyError(f"loss_fn must provide {key!r}")
        kl_grads = _grads(losses["kl"], params)
        matching_grads = _grads(losses["matching"], params)
        metrics = dict(gradient_metrics(kl_grads, matching_grads))
        if scaler is None:
            losses["total"].backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            optimizer.step()
        else:
            scaled = scaler.scale(losses["total"])
            scaled.backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        if scheduler_step is not None:
            scheduler_step()
        eval_metrics = {} if evaluator is None else dict(evaluator(branch, model))
        metrics.update({"kl_loss": float(losses["kl"].detach()),
                        "matching_loss": float(losses["matching"].detach()),
                        "total_loss": float(losses["total"].detach()),
                        "post_update": eval_metrics})
        results[branch] = metrics
    restore(base, model, optimizer, scheduler, scaler)
    return results
