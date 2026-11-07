#!/usr/bin/env bash
# Run on campus only. New worktree + CPU checks + CUDA acceptance, NO formal jobs.
set -euo pipefail
revision=''
gpus=0,2
stop_known=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --revision) revision="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --stop-known-queue) stop_known=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo '--revision requires the fetched full commit hash' >&2; exit 2; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd
export TSFSAR_SSV2_ROOT=/data/fsar_workspace/datasets/SSv2-Small
export TSFSAR_RUNTIME_TMPDIR=/tmp TSFSAR_CPU_THREADS=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OLD="$(cat /data/fsar_workspace/latest_tsfsar_official_tree.txt)"
PREVIOUS_RUN="$(cat "$OLD/runs/campus/latest_official_test_root.txt")"
PREVIOUS="$PREVIOUS_RUN/acceptance"
NEW="${OLD}-dto-reference-$(date +%Y%m%d_%H%M%S)"
git -C "$OLD" worktree add --detach "$NEW" "$revision"
mkdir -p "$NEW/model" "$NEW/runs/campus"
ln -s "$OLD/model/vit-b-16" "$NEW/model/vit-b-16"
cd "$NEW"
python scripts/verify_weights.py
CUDA_VISIBLE_DEVICES='' python -m unittest tests.test_dto_kd_reference_cpu tests.test_evaluation_progress_cpu -v
RUN="$NEW/runs/campus/dto_reference_seed0_$(date +%Y%m%d_%H%M%S)"
# File-only plan checks initialization hashes, configurations and capacity before
# any stop. This never queries or computes with CUDA.
python scripts/run_campus_dto_reference.py --previous-acceptance-root "$PREVIOUS" \
  --gpus "$gpus" --seed 0 --output-root "$RUN"
if [[ "$stop_known" == true ]]; then
  JOBS="$(python - "$PREVIOUS_RUN/formal/status.json" <<'PY'
import json, sys
from pathlib import Path
print(','.join(j['name'] for j in json.loads(Path(sys.argv[1]).read_text())['jobs']))
PY
)"
  python "$OLD/scripts/stop_campus_jobs.py" --run-root "$PREVIOUS_RUN/formal" --jobs "$JOBS" --apply
fi
python - "$gpus" <<'PY'
import sys, time
from scripts.restart_campus_official_test import gpu_idle
gpus = [int(s) for s in sys.argv[1].split(',')]
for attempt in range(20):
    busy = [gpu for gpu in gpus if not gpu_idle(gpu)]
    if not busy:
        break
    print('WAIT_GPUS', busy, flush=True)
    time.sleep(3)
else:
    raise SystemExit('Selected GPUs remain occupied; no unrelated process was stopped. No acceptance started.')
PY
echo "$NEW" > /data/fsar_workspace/latest_tsfsar_dto_reference_tree.txt
nohup python -u scripts/run_campus_dto_reference.py \
  --previous-acceptance-root "$PREVIOUS" --gpus "$gpus" --seed 0 \
  --output-root "$RUN" --apply > "${RUN}.driver.log" 2>&1 < /dev/null &
echo "验收管理进程 PID：$!"
echo "新代码目录：$NEW"
echo "总日志：${RUN}.driver.log"
echo '当前仅运行 CUDA 验收；不会自动启动正式训练。'
tail -n 40 -F "${RUN}.driver.log"
