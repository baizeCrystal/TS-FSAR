"""Protocol and manifest checks shared by the CPU and campus runners."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_TOP_LEVEL = {
    "dataset", "checkpoint", "episode_manifest", "mask_manifest", "mask_type",
    "mask_review_status", "query_only", "conditions", "seed", "max_runtime_minutes",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"empty episode manifest: {path}")
    return rows


def validate_protocol(manifest: Dict[str, Any], *, allow_pilot: bool = False) -> Dict[str, Any]:
    missing = sorted(REQUIRED_TOP_LEVEL - set(manifest))
    if missing:
        raise ValueError("protocol manifest missing keys: " + ", ".join(missing))
    if manifest["mask_type"] not in ("per_frame_binary", "per_frame_soft"):
        raise ValueError("formal run requires an explicit per-frame pixel mask")
    if manifest["mask_review_status"] != "approved" and not allow_pilot:
        raise ValueError("mask_review_status must be approved for a formal run")
    if manifest.get("mask_type") == "per_frame_soft":
        raise ValueError("soft masks are not accepted for the strict pixel intervention")
    if manifest.get("query_only") is not True:
        raise ValueError("the primary protocol must set query_only=true")
    conditions = manifest["conditions"]
    required = {"E0", "E1_background_blur", "E2_background_texture", "E3_foreground_blur"}
    if not required.issubset(set(conditions)):
        raise ValueError("conditions must include E0, E1, E2_background_texture and E3_foreground_blur")
    if float(manifest["max_runtime_minutes"]) > 60:
        raise ValueError("max_runtime_minutes must be <= 60")
    return {
        "formal": manifest["mask_review_status"] == "approved",
        "mask_type": manifest["mask_type"],
        "query_only": True,
        "conditions": list(conditions),
        "max_runtime_minutes": float(manifest["max_runtime_minutes"]),
    }


def validate_mask_array(mask, frame_count: int, height: int, width: int) -> None:
    shape = tuple(getattr(mask, "shape", ()))
    if shape != (frame_count, height, width):
        raise ValueError(f"mask shape {shape} does not match {(frame_count, height, width)}")
    if getattr(mask, "dtype", None) is None or str(mask.dtype) not in ("bool", "uint8"):
        raise ValueError("mask must be bool or uint8")
    if not bool(mask.any(axis=(1, 2)).all()):
        raise ValueError("each frame needs protected foreground pixels")
    if not bool((~mask.astype(bool)).any(axis=(1, 2)).all()):
        raise ValueError("each frame needs background pixels")

