"""Side-effect-free gradient diagnostics for DTO-KD."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import torch

from .controller import flatten_gradients, gradient_metrics


def selected_parameters(model: torch.nn.Module, prefixes: Sequence[str]):
    return [
        p for name, p in model.named_parameters()
        if p.requires_grad and any(name.startswith(prefix) for prefix in prefixes)
    ]


def loss_gradients(
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> Dict[str, torch.Tensor]:
    """Compute independent gradients without changing ``.grad`` fields."""
    params = list(parameters)
    result = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        result[name] = torch.cat([
            (g if g is not None else torch.zeros_like(p)).detach().float().reshape(-1)
            for g, p in zip(grads, params)
        ]) if params else torch.zeros(1)
    return result


def pair_metrics(losses: Mapping[str, torch.Tensor], parameters: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
    grads = loss_gradients(losses, parameters)
    return gradient_metrics(grads["distill"], grads["task"])
