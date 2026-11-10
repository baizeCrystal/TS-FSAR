"""Small deterministic pixel interventions used by the diagnostic.

The functions deliberately operate on native RGB frames before the repository's
resize/crop/normalization pipeline.  They do not use a segmentation model and
therefore require an explicit per-frame foreground mask supplied by the caller.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np


CONDITIONS = ("E0", "E1_background_blur", "E2_background_texture", "E3_foreground_blur")


def _as_float(frames: np.ndarray) -> Tuple[np.ndarray, bool, np.dtype]:
    arr = np.asarray(frames)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError("frames must have shape [T,H,W,3]")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("frames must be numeric")
    was_uint8 = arr.dtype == np.uint8
    original_dtype = arr.dtype
    out = arr.astype(np.float32, copy=True)
    if was_uint8:
        out /= 255.0
    elif out.size and (float(out.min()) < -1e-5 or float(out.max()) > 1.00001):
        raise ValueError("float frames must be in [0,1]")
    return out, was_uint8, original_dtype


def _restore(arr: np.ndarray, was_uint8: bool, dtype: np.dtype) -> np.ndarray:
    if was_uint8:
        return np.rint(np.clip(arr, 0.0, 1.0) * 255.0).astype(dtype)
    return np.clip(arr, 0.0, 1.0).astype(dtype, copy=False)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(np.ceil(3.0 * float(sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def gaussian_blur(frames: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Apply a channel-preserving separable Gaussian blur without scipy."""
    arr = np.asarray(frames, dtype=np.float32)
    k = _gaussian_kernel(sigma)
    r = len(k) // 2
    padded = np.pad(arr, ((0, 0), (r, r), (r, r), (0, 0)), mode="reflect")
    # tensordot keeps the implementation dependency-free and deterministic.
    tmp = np.empty_like(arr)
    for y in range(arr.shape[1]):
        tmp[:, y] = np.tensordot(padded[:, y : y + 2 * r + 1, r : r + arr.shape[2]], k, axes=([1], [0]))
    padded2 = np.pad(tmp, ((0, 0), (0, 0), (r, r), (0, 0)), mode="reflect")
    out = np.empty_like(arr)
    for x in range(arr.shape[2]):
        out[:, :, x] = np.tensordot(padded2[:, :, x : x + 2 * r + 1], k, axes=([2], [0]))
    return np.clip(out, 0.0, 1.0)


def _validate_mask(frames: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(foreground_mask)
    if mask.shape != np.asarray(frames).shape[:3]:
        raise ValueError(
            "foreground_mask must have shape [T,H,W] matching frames; "
            f"got {mask.shape} for {np.asarray(frames).shape[:3]}"
        )
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)
    if not np.all(mask.any(axis=(1, 2))):
        raise ValueError("every frame needs at least one protected foreground pixel")
    if not np.all((~mask).any(axis=(1, 2))):
        raise ValueError("every frame needs at least one background pixel")
    return mask


def apply_condition(
    frames: np.ndarray,
    foreground_mask: np.ndarray,
    condition: str,
    *,
    seed: int = 0,
    sigma: float = 3.0,
    texture_strength: float = 0.12,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply E0/E1/E2/E3 and return transformed frames plus an audit record.

    E1 and E2 alter only ``~foreground_mask``.  E3 alters only the foreground
    and is the matched perturbation control.  The operation is frame aligned:
    no frames are dropped, duplicated, reordered, or resampled.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    original = np.asarray(frames)
    arr, was_uint8, dtype = _as_float(original)
    mask = _validate_mask(arr, foreground_mask)
    if condition == "E0":
        changed = np.zeros(mask.shape, dtype=bool)
        out = arr
    else:
        rng = np.random.default_rng(int(seed))
        if condition in ("E1_background_blur", "E3_foreground_blur"):
            altered = gaussian_blur(arr, sigma=sigma)
        else:
            # A smooth deterministic low-frequency field plus a small global
            # colour shift.  It is intentionally local and non-generative.
            h, w = arr.shape[1:3]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
            field = np.stack(
                [np.sin(2.0 * np.pi * (xx / max(w, 1) + yy / max(h, 1)) + p) for p in phase],
                axis=-1,
            )
            field = field[None]
            shift = rng.uniform(-1.0, 1.0, size=(arr.shape[0], 1, 1, 3)).astype(np.float32)
            altered = np.clip(arr + float(texture_strength) * (0.65 * field + 0.35 * shift), 0.0, 1.0)
        target = ~mask if condition.startswith("E1_") or condition.startswith("E2_") else mask
        out = arr.copy()
        out[target] = altered[target]
        changed = np.any(np.abs(out - arr) > (1.0 / 255.0 if was_uint8 else 1e-7), axis=-1)
        if np.any(changed & mask) and condition.startswith(("E1_", "E2_")):
            raise AssertionError("background intervention changed protected foreground")
        if np.any(changed & ~mask) and condition == "E3_foreground_blur":
            raise AssertionError("foreground control changed background")
    restored = _restore(out, was_uint8, dtype)
    record = {
        "condition": condition,
        "seed": int(seed),
        "sigma": float(sigma),
        "texture_strength": float(texture_strength),
        "shape": list(restored.shape),
        "frame_count_preserved": int(restored.shape[0]) == int(original.shape[0]),
        "foreground_pixels": int(mask.sum()),
        "changed_pixels": int(changed.sum()),
        "foreground_changed_pixels": int((changed & mask).sum()),
        "background_changed_pixels": int((changed & ~mask).sum()),
        "input_sha256": sha256_array(original),
        "output_sha256": sha256_array(restored),
    }
    return restored, record


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def boxes_to_mask(boxes: np.ndarray, height: int, width: int, *, dilate: int = 0) -> np.ndarray:
    """Convert normalized ``[x0,y0,x1,y1]`` boxes to a mask.

    This helper is for preparing a review artifact only.  A box-derived mask
    must be marked ``review_only`` in a manifest and is rejected by the formal
    runner unless explicitly overridden for a pilot.
    """
    b = np.asarray(boxes, dtype=np.float32)
    if b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("boxes must have shape [T,4]")
    out = np.zeros((len(b), height, width), dtype=bool)
    for t, (x0, y0, x1, y1) in enumerate(b):
        xa, xb = sorted((int(np.floor(x0 * width)), int(np.ceil(x1 * width))))
        ya, yb = sorted((int(np.floor(y0 * height)), int(np.ceil(y1 * height))))
        xa, xb = max(0, xa - dilate), min(width, xb + dilate)
        ya, yb = max(0, ya - dilate), min(height, yb + dilate)
        out[t, ya:yb, xa:xb] = True
    return out

