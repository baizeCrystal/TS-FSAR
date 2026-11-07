#!/usr/bin/env bash
set -euo pipefail

# Independent evaluation only. Does not stop/inspect/reconfigure training queues.
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METHOD_DIR/../.."

SHOT=""; GPU=""; DATA=""; CHECKPOINT=""; OUTPUT=""; MANIFEST_ROOT=""
DISABLE=()
while (($#)); do
  case "$1" in
    --shot) SHOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --data-root) DATA="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --manifest-root) MANIFEST_ROOT="$2"; shift 2 ;;
    --disable-psds) DISABLE=(--disable-psds); shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$SHOT" =~ ^(1|5)$ || -z "$GPU" || -z "$DATA" || -z "$CHECKPOINT" || -z "$OUTPUT" ]]; then
  echo 'Usage: bash research/ps_ds_subspace_experiment/run_gpu.sh --shot 1|5 --gpu N --data-root DIR --checkpoint FILE --output DIR [--disable-psds]' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-python}"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No visible GPU; CPU model inference is disabled"'
MANIFEST_ROOT="${MANIFEST_ROOT:-$METHOD_DIR/results/shot$SHOT}"
"$PYTHON" -u -m research.ps_ds_subspace_experiment.evaluate \
  --stage prepare --shot "$SHOT" --data-root "$DATA" --output "$MANIFEST_ROOT"
for stage in smoke explore formal; do
  "$PYTHON" -u -m research.ps_ds_subspace_experiment.evaluate \
    --stage "$stage" --shot "$SHOT" --data-root "$DATA" \
    --checkpoint "$CHECKPOINT" --manifest "$MANIFEST_ROOT/episodes.jsonl.gz" \
    --output "$OUTPUT" "${DISABLE[@]}"
done
