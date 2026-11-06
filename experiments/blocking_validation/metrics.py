"""Gradient diagnostics used by the causal validation, without parameter updates."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import torch


def _flat(grads: Iterable[Optional[torch.Tensor]]) -> torch.Tensor:
    values = [g.detach().reshape(-1).float() for g in grads if g is not None]
    return torch.cat(values) if values else torch.zeros(1)


def gradient_metrics(kl_grads: Iterable[Optional[torch.Tensor]],
                     matching_grads: Iterable[Optional[torch.Tensor]],
                     *, eps: float = 1e-12) -> Mapping[str, float]:
    """Return cosine, norms, and KL projection onto the matching direction."""
    gk, gm = _flat(kl_grads), _flat(matching_grads)
    size = max(gk.numel(), gm.numel())
    if gk.numel() != size:
        gk = torch.nn.functional.pad(gk, (0, size - gk.numel()))
    if gm.numel() != size:
        gm = torch.nn.functional.pad(gm, (0, size - gm.numel()))
    dot = torch.dot(gk, gm)
    nk, nm = torch.linalg.vector_norm(gk), torch.linalg.vector_norm(gm)
    return {
        "kl_norm": float(nk),
        "matching_norm": float(nm),
        "dot": float(dot),
        "cosine": float(dot / (nk * nm + eps)),
        "relative_projection": float(dot / (nm.square() + eps)),
        "kl_opposes_matching": bool(dot < 0),
    }
