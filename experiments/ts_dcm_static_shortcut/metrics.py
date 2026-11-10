"""Paired episode-level statistics for frozen counterfactual outputs."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> list:
    return [float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))]


def episode_bootstrap(
    episode_ids: Iterable[int],
    values: Iterable[float],
    *,
    seed: int = 20261110,
    repeats: int = 5000,
) -> Dict[str, object]:
    """Bootstrap a query-level quantity by resampling whole episodes."""
    e = np.asarray(list(episode_ids))
    v = np.asarray(list(values), dtype=np.float64)
    if e.shape != v.shape:
        raise ValueError("episode_ids and values must have identical length")
    unique = np.unique(e)
    if not len(unique):
        raise ValueError("empty input")
    per_episode = np.asarray([v[e == eid].mean() for eid in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(int(repeats), dtype=np.float64)
    for i in range(int(repeats)):
        samples[i] = per_episode[rng.integers(0, len(per_episode), size=len(per_episode))].mean()
    return {
        "estimate": float(per_episode.mean()),
        "ci95": _percentile_ci(samples),
        "episodes": int(len(unique)),
        "queries": int(len(v)),
        "repeats": int(repeats),
        "seed": int(seed),
    }


def paired_condition_summary(
    rows: Iterable[Mapping[str, object]],
    *,
    condition: str,
    baseline: str = "E0",
    bootstrap_repeats: int = 5000,
    seed: int = 20261110,
) -> Dict[str, object]:
    """Summarise paired predictions for one condition.

    Input rows are query-level records with ``episode_id``, ``label``,
    ``prediction``, ``condition`` and optional ``margin``.  The baseline rows
    must be present for the same query keys.
    """
    rows = list(rows)
    key = lambda r: (int(r["episode_id"]), str(r["query_id"]))
    by = {(str(r["condition"]), key(r)): r for r in rows}
    pairs = []
    for (c, k), r in by.items():
        if c != condition:
            continue
        b = by.get((baseline, k))
        if b is None:
            raise ValueError(f"missing {baseline} row for {condition} key {k}")
        label = int(r["label"])
        rcorrect, bcorrect = int(r["prediction"]) == label, int(b["prediction"]) == label
        pairs.append({
            "episode_id": int(r["episode_id"]),
            "query_id": str(r["query_id"]),
            "baseline_correct": bcorrect,
            "condition_correct": rcorrect,
            "prediction_changed": int(r["prediction"]) != int(b["prediction"]),
            "correct_to_wrong": bcorrect and not rcorrect,
            "wrong_to_correct": (not bcorrect) and rcorrect,
            "margin_delta": float(r.get("margin", np.nan)) - float(b.get("margin", np.nan)),
        })
    if not pairs:
        raise ValueError(f"no paired rows for {condition}")
    def stat(name):
        vals = np.asarray([float(p[name]) for p in pairs], dtype=np.float64)
        # Keep the stream stable across Python processes (built-in hash is
        # intentionally randomized by Python).
        offset = sum((i + 1) * ord(ch) for i, ch in enumerate(name)) % 1000
        return episode_bootstrap(
            [p["episode_id"] for p in pairs], vals, seed=seed + offset, repeats=bootstrap_repeats
        )
    return {
        "condition": condition,
        "baseline": baseline,
        "queries": len(pairs),
        "accuracy": stat("condition_correct"),
        "prediction_change": stat("prediction_changed"),
        "correct_to_wrong": stat("correct_to_wrong"),
        "wrong_to_correct": stat("wrong_to_correct"),
        "margin_delta": stat("margin_delta"),
    }


def score_margin(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """True-class score minus strongest wrong-class score."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or labels.shape != (scores.shape[0],):
        raise ValueError("scores must be [N,way] and labels [N]")
    out = np.empty(len(labels), dtype=np.float64)
    for i, y in enumerate(labels):
        wrong = np.delete(scores[i], y)
        out[i] = scores[i, y] - wrong.max()
    return out

