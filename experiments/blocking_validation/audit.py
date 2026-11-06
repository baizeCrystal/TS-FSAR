"""Runtime loss-graph audit; complements the checked-in production audit note."""

from __future__ import annotations

from typing import Mapping

import torch


def audit_loss_graph(model, losses: Mapping[str, torch.Tensor]) -> dict:
    """Report which trainable parameters each loss reaches.

    This uses ``autograd.grad`` only and never calls ``optimizer.step``.  The
    caller should pass the actual production tensors (Episode CE, Student CE,
    KL, Teacher CE) from one episode.
    """
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    result = {}
    for loss_name, loss in losses.items():
        grads = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True,
                                    allow_unused=True)
        result[loss_name] = {name: {"connected": g is not None,
                                    "norm": 0.0 if g is None else float(g.detach().float().norm())}
                             for (name, _), g in zip(params, grads)}
    return result
