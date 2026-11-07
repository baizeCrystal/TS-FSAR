#!/usr/bin/env bash
# One allocated campus GPU, complete logs; never starts formal training.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
shot=''
gpu=''
seed=0
out=''
reuse_report=''
adaptation_checks=false
dto_reference_checks=false
selection_protocol=meta-val-fixed
while [[ $# -gt 0 ]]; do
  case "$1" in
    --shot) shot="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --output) out="$2"; shift 2 ;;
    --reuse-initialization-report) reuse_report="$2"; shift 2 ;;
    --adaptation-checks) adaptation_checks=true; shift ;;
    --dto-reference-checks) dto_reference_checks=true; shift ;;
    --selection-protocol) selection_protocol="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$shot" == 1 || "$shot" == 5 ]] || { echo '--shot must be 1 or 5' >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ && -n "$out" ]] || { echo '--gpu and --output required' >&2; exit 2; }
[[ ! -e "$out" && ! -e "${out}.console.log" ]] || { echo 'choose a fresh acceptance output' >&2; exit 1; }
mkdir -p "$(dirname "$out")"
threads="${TSFSAR_CPU_THREADS:-2}"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || exit 2
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads" BLIS_NUM_THREADS="$threads" VECLIB_MAXIMUM_THREADS="$threads"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
runtime_base="${TSFSAR_RUNTIME_TMPDIR:-/tmp}"
mkdir -p "$runtime_base"
export TMPDIR
TMPDIR="$(mktemp -d "$runtime_base/tsfsar-accept-${UID}.XXXXXX")"
gpu_locks="${TSFSAR_GPU_LOCK_DIR:-/tmp/tsfsar-gpu-locks-${UID}}"
mkdir -p "$gpu_locks"
exec 8>"$gpu_locks/gpu-${gpu}.lock"
flock -n 8 || { echo "GPU $gpu is reserved by a campus launcher" >&2; exit 1; }
exec 9>"${out}.lock"
flock -n 9 || { echo 'another acceptance uses this output' >&2; exit 1; }
exec > >(tee -a "${out}.console.log") 2>&1
echo "[$(date -Is)] CUDA acceptance only: shot=$shot seed=$seed GPU=$gpu output=$out"
args=(--shot "$shot" --seed "$seed" --gpu "$gpu" --output-dir "$out")
args+=(--selection-protocol "$selection_protocol")
if [[ -n "$reuse_report" ]]; then args+=(--reuse-initialization-report "$reuse_report"); fi
if [[ "$adaptation_checks" == true ]]; then args+=(--adaptation-checks); fi
if [[ "$dto_reference_checks" == true ]]; then args+=(--dto-reference-checks); fi
exec python -u scripts/campus_acceptance.py "${args[@]}"
