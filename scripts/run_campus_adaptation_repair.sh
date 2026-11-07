#!/usr/bin/env bash
# Campus only: new CUDA certification, then a NEW queue from common initial state.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
previous=''
out=''
gpus=2,3
seed=0
experiments=E1_safe
while [[ $# -gt 0 ]]; do
  case "$1" in
    --previous-acceptance-root) previous="$2"; shift 2 ;;
    --output-root) out="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --experiments) experiments="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$out" && -n "$previous" ]] || { echo '--previous-acceptance-root and --output-root required' >&2; exit 2; }
[[ "$gpus" =~ ^[0-9]+,[0-9]+$ ]] || { echo '--gpus must name two distinct idle allocated GPUs' >&2; exit 2; }
IFS=, read -r gpu1 gpu5 <<< "$gpus"
[[ "$gpu1" != "$gpu5" ]] || exit 2
for shot in 1 5; do
  [[ -f "$previous/shot$shot/acceptance.json" ]] || { echo "missing previous $shot-shot report" >&2; exit 1; }
done
IFS=, read -ra selections <<< "$experiments"
for selection in "${selections[@]}"; do
  case "$selection" in E0_detach|E1_detach|E1_safe) ;; *) echo 'select new adaptations only' >&2; exit 2 ;; esac
done
[[ ! -e "$out" ]] || { echo 'choose a new --output-root; nothing overwritten' >&2; exit 1; }
mkdir -p "$out/acceptance"
out="$(cd "$out" && pwd)"
mkdir -p runs/campus
echo "$out" > runs/campus/latest_adaptation_root.txt
echo "$$" > "$out/launcher.pid"
echo "ADAPTATION_ROOT=$out experiments=$experiments GPUs=$gpus"
acceptance_pids=()
cleanup() {
  for pid in "${acceptance_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
bash scripts/run_campus_acceptance.sh --shot 1 --gpu "$gpu1" --seed "$seed" \
  --adaptation-checks --output "$out/acceptance/shot1" \
  --reuse-initialization-report "$previous/shot1/acceptance.json" &
acceptance_pids+=("$!")
bash scripts/run_campus_acceptance.sh --shot 5 --gpu "$gpu5" --seed "$seed" \
  --adaptation-checks --output "$out/acceptance/shot5" \
  --reuse-initialization-report "$previous/shot5/acceptance.json" &
acceptance_pids+=("$!")
failed=0
for pid in "${acceptance_pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
acceptance_pids=()
[[ "$failed" == 0 ]] || { echo 'CUDA acceptance failed; formal queue NOT started' >&2; exit 1; }
exec python -u scripts/launch_campus_suite.py --protocol v2 \
  --experiments "$experiments" --gpus "$gpus" --seed "$seed" \
  --acceptance-1shot "$out/acceptance/shot1/acceptance.json" \
  --acceptance-5shot "$out/acceptance/shot5/acceptance.json" \
  --activation-offload diagnostics --launch-delay 20 --output-root "$out/formal"
