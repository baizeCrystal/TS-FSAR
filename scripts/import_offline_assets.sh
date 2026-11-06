#!/usr/bin/env bash
set -euo pipefail
asset_dir="${1:-${TSFSAR_ASSET_DIR:-}}"; [[ -n "$asset_dir" ]] || { echo "usage: $0 ASSET_DIR" >&2; exit 2; }
root="$(cd "$(dirname "$0")/.." && pwd)"; archive="$asset_dir/CLIP-vit-b-16.tar.gz"; [[ -f "$archive" ]] || archive="$asset_dir/CLIP-vit-b-16-v2.tar.gz"; [[ -f "$archive" ]] || { echo 'CLIP archive missing' >&2; exit 1; }
mkdir -p "$root/model"; tar -tzf "$archive" >/dev/null; tar -xzf "$archive" -C "$root/model"; mkdir -p "$root/checkpoints/dto_kd"
for f in E0_global_KL_model_best.pth B_global_DIST_model_best.pth C_episode_KL_model_best.pth D_episode_DIST_model_best.pth; do [[ -f "$asset_dir/$f" ]] || { echo "missing $f" >&2; exit 1; }; cp "$asset_dir/$f" "$root/checkpoints/dto_kd/$f"; done
echo 'Offline assets imported; run scripts/campus_preflight.sh next.'
