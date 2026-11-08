#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METHOD_DIR/../.."
GPU=''
ARGS=()
while (($#)); do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --shot|--data-root|--checkpoint|--manifest|--output)
      ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo '--gpu requires an allocated GPU index'; exit 2; }
export CUDA_VISIBLE_DEVICES="$GPU"
export TSFSAR_CPU_THREADS="${TSFSAR_CPU_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -u -m research.sam_qam_experiment.pipeline "${ARGS[@]}"
