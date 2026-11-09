#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METHOD_DIR/../.."
GPU=''
ARGS=()
while (($#)); do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --shot|--data-root|--output)
      ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo '--gpu requires an allocated GPU index'; exit 2; }
PYTHON="${PYTHON:-python}"
# Cooperate with the existing campus queues; the FD survives exec and remains
# held throughout preflight, training and evaluation subprocesses.
LOCK_DIR="${TSFSAR_GPU_LOCK_DIR:-/tmp/tsfsar-gpu-locks-${UID}}"
mkdir -p "$LOCK_DIR"
exec 8>"$LOCK_DIR/gpu-${GPU}.lock"
flock -n 8 || { echo "GPU $GPU reserved by another campus queue"; exit 1; }
"$PYTHON" -c 'import sys; from research.cls_ctm_joint_experiment.launch import check_gpu; check_gpu(int(sys.argv[1]))' "$GPU"
export CUDA_VISIBLE_DEVICES="$GPU"
export TSFSAR_CPU_THREADS="${TSFSAR_CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -u -m research.cls_ctm_joint_experiment.pipeline "${ARGS[@]}"
