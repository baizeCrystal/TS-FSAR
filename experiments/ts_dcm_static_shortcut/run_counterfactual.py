"""Guarded campus runner for pixel counterfactual materialization.

The runner is intentionally conservative.  It decodes the exact frame indices
from the frozen episode manifest, applies interventions before TS-FSAR's
resize/crop/normalization, and writes auditable inputs.  A model evaluator is
loaded only when supplied as ``module:function``; its contract is documented in
``README.md``.  Existing feature caches are never read.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from .protocol import read_jsonl, validate_mask_array, validate_protocol
from .transforms import apply_condition, sha256_array


def _decode(video_path: str, indices: Iterable[int]) -> np.ndarray:
    try:
        from decord import VideoReader
    except ImportError as exc:  # pragma: no cover - campus dependency
        raise RuntimeError("decord is required on the campus runner") from exc
    vr = VideoReader(video_path, num_threads=1)
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.ndim != 1 or len(idx) == 0 or int(idx.min()) < 0 or int(idx.max()) >= len(vr):
        raise ValueError(f"invalid frame indices for {video_path}: {idx.tolist()}")
    return np.asarray(vr.get_batch(idx).asnumpy(), dtype=np.uint8)


def _load_mask(entry: Dict[str, Any], frame_indices: list[int], shape: tuple[int, ...]) -> np.ndarray:
    path = Path(entry["mask_path"])
    data = np.load(path, allow_pickle=False)
    if "mask" not in data:
        raise ValueError(f"mask file {path} must contain an array named 'mask'")
    mask = np.asarray(data["mask"])
    if mask.ndim != 3:
        raise ValueError(f"mask file {path} must be [T,H,W]")
    if "frame_indices" not in data:
        raise ValueError(f"mask file {path} must contain frame_indices for auditability")
    if list(map(int, data["frame_indices"].tolist())) != list(map(int, frame_indices)):
        raise ValueError(f"mask frame_indices mismatch for {entry.get('video_id')}")
    validate_mask_array(mask, shape[0], shape[1], shape[2])
    return mask.astype(bool)


def _load_video_root(manifest: Dict[str, Any]) -> Path:
    root = manifest.get("video_root") or os.environ.get("TS_FSAR_VIDEO_ROOT")
    if not root:
        raise ValueError("set video_root in protocol manifest or TS_FSAR_VIDEO_ROOT")
    return Path(root)


def _load_mask_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("videos", payload)
    if not isinstance(entries, dict):
        raise ValueError("mask manifest must be a JSON object keyed by video_id")
    return {str(k): dict(v) for k, v in entries.items()}


def _call_evaluator(spec: str, episode: Dict[str, Any], condition_paths: Dict[str, str], out_dir: Path) -> Dict[str, Any]:
    module_name, func_name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module_name), func_name)
    result = fn(episode=episode, condition_paths=condition_paths, output_dir=str(out_dir))
    if not isinstance(result, dict):
        raise TypeError("evaluator must return a dict of branch outputs")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--evaluator", help="optional module:function evaluator hook")
    ap.add_argument("--allow-pilot", action="store_true", help="allow pending masks for materialization-only pilot")
    ap.add_argument("--materialize-only", action="store_true")
    args = ap.parse_args()
    protocol_path = Path(args.protocol)
    manifest = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_protocol(manifest, allow_pilot=args.allow_pilot)
    if not args.materialize_only and not args.evaluator:
        raise ValueError("formal run needs --evaluator; use --materialize-only for an input-only pilot")
    episodes = read_jsonl(manifest["episode_manifest"])
    limit = min(len(episodes), int(manifest.get("episodes_limit", len(episodes))))
    masks = _load_mask_manifest(manifest["mask_manifest"])
    video_root = _load_video_root(manifest)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(manifest["max_runtime_minutes"]) * 60.0
    rows = []
    status = "complete"
    try:
        for episode in episodes[:limit]:
            if time.monotonic() >= deadline:
                status = "partial_timeout"
                break
            episode_id = int(episode["episode"])
            query_paths: Dict[str, Dict[str, str]] = {}
            for query in episode["query"]:
                if time.monotonic() >= deadline:
                    status = "partial_timeout"
                    break
                video_id = str(query["video_id"])
                entry = masks.get(video_id)
                if entry is None:
                    raise ValueError(f"no reviewed mask entry for query video {video_id}")
                frame_indices = list(map(int, query["frame_indices"]))
                frames = _decode(str(video_root / f"{video_id}.mp4"), frame_indices)
                mask = _load_mask(entry, frame_indices, frames.shape)
                per_condition = {}
                for condition in manifest["conditions"]:
                    transformed, record = apply_condition(frames, mask, condition, seed=int(manifest["seed"]) + episode_id)
                    path = out_dir / f"episode_{episode_id:05d}_{video_id}_{condition}.npz"
                    np.savez_compressed(path, frames=transformed, frame_indices=np.asarray(frame_indices, dtype=np.int64), mask=mask)
                    per_condition[condition] = str(path)
                    rows.append({"episode_id": episode_id, "query_id": video_id, **record, "path": str(path)})
                query_paths[video_id] = per_condition
            if status == "partial_timeout":
                break
            if args.evaluator and not args.materialize_only:
                result = _call_evaluator(args.evaluator, episode, query_paths, out_dir)
                (out_dir / f"episode_{episode_id:05d}_outputs.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                if "rows" in result:
                    with (out_dir / "query_rows.jsonl").open("a", encoding="utf-8") as stream:
                        for row in result["rows"]:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    except TimeoutError:
        status = "partial_timeout"
    (out_dir / "materialization_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    run_status = {
        "status": status,
        "episodes_requested": limit,
        "episodes_materialized": len({r["episode_id"] for r in rows}),
        "query_conditions_materialized": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "strict_reencode_required": True,
        "feature_cache_reuse": False,
        "protocol": str(protocol_path),
    }
    (out_dir / "status.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    print(json.dumps(run_status, indent=2))


if __name__ == "__main__":
    main()

