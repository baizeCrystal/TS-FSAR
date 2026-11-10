"""Create a reviewable protocol skeleton; never silently promotes coarse boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mask-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="ssv2-small")
    ap.add_argument("--shot", type=int, choices=(1, 5), required=True)
    ap.add_argument("--episodes-limit", type=int, default=0)
    args = ap.parse_args()
    n = args.episodes_limit or (500 if args.shot == 1 else 300)
    manifest = {
        "dataset": args.dataset,
        "shot": args.shot,
        "way": 5,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_manifest": str(Path(args.episodes).resolve()),
        "mask_manifest": str(Path(args.mask_manifest).resolve()),
        "mask_type": "per_frame_binary",
        "mask_review_status": "pending",
        "mask_source": "manual_or_reviewed_tracking_required",
        "query_only": True,
        "conditions": ["E0", "E1_background_blur", "E2_background_texture", "E3_foreground_blur"],
        "seed": 20261110,
        "episodes_limit": n,
        "max_runtime_minutes": 55,
        "strict_reencode_required": True,
        "feature_cache_reuse": False,
        "notes": "Set mask_review_status=approved only after frame-level visual audit.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

