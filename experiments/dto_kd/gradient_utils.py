"""Side-effect-free gradient diagnostics for DTO-KD."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable, Mapping, Sequence

import torch

from .controller import flatten_gradients, gradient_metrics


def activation_context(mode: str, sampled: bool = False):
    """Offload saved backward tensors without changing the forward/RNG stream."""
    if mode not in ("off", "diagnostics", "all"):
        raise ValueError(f"unknown activation offload mode: {mode}")
    if mode == "all" or (mode == "diagnostics" and sampled):
        # Pageable host memory avoids pinning many GB per parallel experiment.
        return torch.autograd.graph.save_on_cpu(pin_memory=False)
    return nullcontext()


def selected_parameters(model: torch.nn.Module, prefixes: Sequence[str]):
    return [
        p for name, p in model.named_parameters()
        if p.requires_grad and any(name.startswith(prefix) for prefix in prefixes)
    ]


def loss_gradients(
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> Dict[str, torch.Tensor]:
    """Return detached CPU vectors without changing parameter ``.grad`` fields.

    Release each objective's GPU gradients before computing the next objective.
    These vectors are for diagnostics only; the normal model backward is separate.
    """
    params = list(parameters)
    result = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        result[name] = torch.cat([
            g.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            if g is not None else torch.zeros(p.numel(), device="cpu", dtype=torch.float32)
            for g, p in zip(grads, params)
        ]) if params else torch.zeros(1)
        del grads
    return result


def pair_metrics(losses: Mapping[str, torch.Tensor], parameters: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
    grads = loss_gradients(losses, parameters)
    return gradient_metrics(grads["distill"], grads["task"])
