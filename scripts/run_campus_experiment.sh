#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
experiment=E0
shot=1
gpu=0
seed=0
mode=train
out=''
checkpoint=''
config=''
smoke=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) experiment="$2"; shift 2 ;;
    --shot) shot="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --output) out="$2"; shift 2 ;;
    --checkpoint) checkpoint="$2"; shift 2 ;;
    --config) config="$2"; shift 2 ;;
    --smoke) smoke=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$experiment" in E0|E1|E2|E3) ;; *) echo 'experiment must be E0/E1/E2/E3' >&2; exit 2 ;; esac
case "$shot" in 1|5) ;; *) echo 'shot must be 1 or 5' >&2; exit 2 ;; esac
case "$mode" in train|eval) ;; *) echo 'mode must be train or eval' >&2; exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo '--gpu must name one allocated GPU index' >&2; exit 2; }
if [[ "$experiment" == E2 && -z "$config" ]]; then
  echo '--config is required for E2 (choose 0.05, 0.1, or 0.2)' >&2
  exit 2
fi
cfg="${config:-configs/dto_kd/ssv2_small_${shot}shot_${experiment}.yaml}"
[[ -f "$cfg" ]] || { echo "missing config $cfg" >&2; exit 1; }
if [[ "$mode" == eval && ! -f "$checkpoint" ]]; then
  echo '--checkpoint must name an existing file for eval' >&2
  exit 2
fi

# Keep the original episode batch and DataLoader worker count.
# Bound BLAS/OpenMP threads during CPU PCA initialization on shared hosts.
threads="${TSFSAR_CPU_THREADS:-2}"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo 'TSFSAR_CPU_THREADS must be positive' >&2; exit 2; }
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads" BLIS_NUM_THREADS="$threads"
export VECLIB_MAXIMUM_THREADS="$threads" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Runtime sockets must not inherit the NFS pip-install TMPDIR. Environments
# and pip downloads can stay on /data; these small runtime files use local /tmp.
runtime_base="${TSFSAR_RUNTIME_TMPDIR:-/tmp}"
mkdir -p "$runtime_base"
export TMPDIR
TMPDIR="$(mktemp -d "$runtime_base/tsfsar-${UID}.XXXXXX")"

out="${out:-./runs/campus/${experiment}_${shot}shot_seed${seed}}"
if [[ "$smoke" == true && "/$out/" != */smoke/* ]]; then
  echo '--smoke requires --output inside a smoke/ directory' >&2
  exit 2
fi
mkdir -p "$out"
command -v flock >/dev/null || { echo 'flock is required' >&2; exit 1; }
exec 9>"$out/.${mode}.lock"
flock -n 9 || { echo "another $mode process is using $out" >&2; exit 1; }
gpu_locks="${TSFSAR_GPU_LOCK_DIR:-/tmp/tsfsar-gpu-locks-${UID}}"
mkdir -p "$gpu_locks"
exec 8>"$gpu_locks/gpu-${gpu}.lock"
flock -n 8 || { echo "GPU $gpu already reserved by another campus launcher" >&2; exit 1; }
console="$out/console.log"
tmp="$out/config.yaml"
if [[ "$mode" == eval ]]; then
  console="$out/eval_console.log"
  tmp="$out/config_eval.yaml"
fi
exec > >(tee -a "$console") 2>&1
echo "[$(date -Is)] START experiment=$experiment shot=$shot gpu=$gpu seed=$seed mode=$mode pid=$$"
echo "python=$(command -v python) CPU_THREADS=$threads runtime_TMPDIR=$TMPDIR"
echo "config=$cfg output=$out console=$console"
echo 'LSN training initialization performs CPU PCA before GPU episode processing.'

args=(--config "$cfg" --output "$tmp" --seed "$seed" --output-dir "$out" --mode "$mode")
if [[ "$smoke" == true ]]; then
  echo 'SMOKE CHECK ONLY: 2 train episodes, 2 validation episodes, gradients every step.'
  args+=(--smoke)
fi
if [[ "$mode" == eval ]]; then args+=(--checkpoint "$checkpoint"); fi
python -u scripts/prepare_run_config.py "${args[@]}"
export CUDA_VISIBLE_DEVICES="$gpu"
if [[ "$experiment" == E0 ]]; then
  exec python -u main.py --cfg "$tmp"
else
  exec python -u -m experiments.dto_kd.entry --cfg "$tmp"
fi
