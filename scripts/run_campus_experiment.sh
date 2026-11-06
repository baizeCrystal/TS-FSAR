#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)";cd "$root";experiment=E0;shot=1;gpu=0;seed=0;mode=train;out='';checkpoint='';config=''
while [[ $# -gt 0 ]]; do case "$1" in --experiment)experiment="$2";shift 2;;--shot)shot="$2";shift 2;;--gpu)gpu="$2";shift 2;;--seed)seed="$2";shift 2;;--mode)mode="$2";shift 2;;--output)out="$2";shift 2;;--checkpoint)checkpoint="$2";shift 2;;--config)config="$2";shift 2;;*)echo "unknown argument: $1" >&2;exit 2;;esac;done
case "$experiment" in E0|E1|E2|E3);;*)echo 'experiment must be E0/E1/E2/E3' >&2;exit 2;;esac
if [[ "$experiment" == E2 && -z "$config" ]];then echo '--config is required for E2 (choose 0.05, 0.1, or 0.2 candidate)' >&2;exit 2;fi
cfg="${config:-configs/dto_kd/ssv2_small_${shot}shot_${experiment}.yaml}";[[ -f "$cfg" ]]||{ echo "missing config $cfg" >&2;exit 1;};out="${out:-./runs/campus/${experiment}_${shot}shot_seed${seed}}";tmp="$out/config.yaml";mkdir -p "$out"
args=(--config "$cfg" --output "$tmp" --seed "$seed" --output-dir "$out" --mode "$mode");if [[ "$mode" == eval ]];then [[ -n "$checkpoint" ]]||{ echo --checkpoint required for eval >&2;exit 2;};args+=(--checkpoint "$checkpoint");fi
python scripts/prepare_run_config.py "${args[@]}";export CUDA_VISIBLE_DEVICES="$gpu";if [[ "$experiment" == E0 ]];then exec python main.py --cfg "$tmp";else exec python -m experiments.dto_kd.entry --cfg "$tmp";fi
