"""Exact model/optimizer/RNG snapshots for paired counterfactual updates."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class TrainingSnapshot:
    model: Any
    optimizer: Any
    scheduler: Any
    scaler: Any
    torch_rng: torch.Tensor
    cuda_rng: Optional[list]
    numpy_rng: Any
    python_rng: Any


def snapshot(model, optimizer, scheduler=None, scaler=None) -> TrainingSnapshot:
    return TrainingSnapshot(
        copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
        None if scheduler is None else copy.deepcopy(scheduler.state_dict()),
        None if scaler is None else copy.deepcopy(scaler.state_dict()),
        torch.get_rng_state().clone(),
        [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        copy.deepcopy(np.random.get_state()), copy.deepcopy(random.getstate()))


def restore(snapshot: TrainingSnapshot, model, optimizer, scheduler=None, scaler=None) -> None:
    model.load_state_dict(snapshot.model)
    optimizer.load_state_dict(snapshot.optimizer)
    if scheduler is not None and snapshot.scheduler is not None:
        scheduler.load_state_dict(snapshot.scheduler)
    if scaler is not None and snapshot.scaler is not None:
        scaler.load_state_dict(snapshot.scaler)
    torch.set_rng_state(snapshot.torch_rng)
    if snapshot.cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot.cuda_rng)
    np.random.set_state(snapshot.numpy_rng)
    random.setstate(snapshot.python_rng)
