#!/usr/bin/env bash
# Run from an activated tsfsar-dto-kd environment; never stop another queue.
set -euo pipefail
METHOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METHOD_DIR/../.."
GPUS=4,5
ASSETS=/data/fsar_workspace/psds_assets
DATA=/data/fsar_workspace/datasets/SSv2-Small
OUTPUT=''
PREPARE=false
while (($#)); do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --assets-dir) ASSETS="$2"; shift 2 ;;
    --data-root) DATA="$2"; shift 2 ;;
    --output-root) OUTPUT="$2"; shift 2 ;;
    --prepare-in-background) PREPARE=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TSFSAR_CPU_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-/data/fsar_workspace/clsctm_results/seed0_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$OUTPUT" ]] || { echo "Output already exists: $OUTPUT"; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"
if [[ "$PREPARE" == false ]]; then
  # Include CPU checks and asset preparation in nohup too: closing Terminal
  # during these stages must not kill the pipeline.
  LOG="${OUTPUT}.launcher.log"
  [[ ! -e "$LOG" ]] || { echo "Launcher log already exists: $LOG"; exit 1; }
  nohup bash "$METHOD_DIR/launch_campus.sh" --prepare-in-background \
    --gpus "$GPUS" --assets-dir "$ASSETS" --data-root "$DATA" --output-root "$OUTPUT" \
    > "$LOG" 2>&1 < /dev/null &
  echo "后台管理进程 PID：$!"
  echo "任务目录：$OUTPUT"
  echo "管理日志：$LOG"
  echo "查看启动：tail -n 40 -F '$LOG'"
  exit 0
fi
"$PYTHON" -c 'import sys; from research.cls_ctm_experiment.launch import check_gpu; g=[int(x) for x in sys.argv[1].split(",")]; assert len(g)==2 and len(set(g))==2 and min(g)>=0, "需要两张不同的已分配 GPU"; [check_gpu(x) for x in g]' "$GPUS"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m unittest discover -s research/cls_ctm_experiment/tests -v
"$PYTHON" -u -m research.ps_ds_subspace_experiment.github_assets --assets-dir "$ASSETS"
exec "$PYTHON" -u -m research.cls_ctm_experiment.launch \
  --gpus "$GPUS" --assets-dir "$ASSETS" --data-root "$DATA" --output-root "$OUTPUT" \
  < /dev/null
