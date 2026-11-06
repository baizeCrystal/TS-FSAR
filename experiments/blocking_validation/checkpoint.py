"""Atomic checkpoints for B short runs and interruption recovery."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from .state import snapshot


def save_checkpoint(path, model, optimizer, *, scheduler=None, scaler=None,
                    next_episode: int = 0, metadata: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": None if scheduler is None else scheduler.state_dict(),
               "scaler": None if scaler is None else scaler.state_dict(),
               "next_episode": int(next_episode), "metadata": metadata or {},
               "rng": snapshot(model, optimizer, scheduler, scaler)}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, *, scheduler=None, scaler=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload
