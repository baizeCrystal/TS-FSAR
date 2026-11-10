"""Render a compact E0/E1/E2/E3 and mask contact sheet for visual approval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.astype(np.float32).copy()
    out[mask] = 0.55 * out[mask] + 0.45 * np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return np.clip(out, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="one materialized file per condition")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for visual review") from exc
    items = []
    for path in args.npz:
        with np.load(path, allow_pickle=False) as data:
            original = data["frames"]
            frames = original.astype(np.float32)
            if original.dtype == np.uint8:
                frames /= 255.0
            mask = data["mask"].astype(bool)
        items.append((Path(path).stem, frames, mask))
    if not items:
        raise ValueError("no input files")
    t = min(x[1].shape[0] for x in items)
    thumb_w, thumb_h = 224, 224
    label_h = 24
    canvas = Image.new("RGB", (thumb_w * t, (thumb_h + label_h) * len(items)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (name, frames, mask) in enumerate(items):
        draw.text((2, row * (thumb_h + label_h) + 2), name, fill="black")
        for col in range(t):
            image = Image.fromarray(np.uint8(np.clip(_overlay(frames[col], mask[col]), 0, 1) * 255))
            image = image.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            canvas.paste(image, (col * thumb_w, row * (thumb_h + label_h) + label_h))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()

