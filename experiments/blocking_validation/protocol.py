"""Split and episode identity guards for reproducible causal runs."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping, Sequence


def episode_sequence_hash(episodes: Sequence[object]) -> str:
    """Hash stable episode metadata or tensors without serialising model data."""
    canonical = []
    for episode in episodes:
        if isinstance(episode, Mapping):
            item = {str(k): _normalise(v) for k, v in sorted(episode.items(), key=lambda x: str(x[0]))}
        else:
            item = _normalise(episode)
        canonical.append(item)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _normalise(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.numpy().tobytes().hex()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    return value


def assert_split_disjoint(meta_train: Iterable[object], meta_val: Iterable[object],
                          meta_test: Iterable[object]) -> None:
    splits = [set(meta_train), set(meta_val), set(meta_test)]
    names = ["Meta-Train", "Meta-Val", "Meta-Test"]
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = splits[i] & splits[j]
            if overlap:
                raise ValueError(f"{names[i]} and {names[j]} overlap: {sorted(overlap)!r}")
