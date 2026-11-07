# DTO-KD FSAR adaptation repair — 2026-11-07

This is a NEW experiment definition. The historical E1, E0 and E2 configurations
remain available. Neither a historical trained checkpoint nor an interrupted E1
state initializes a new run. Existing original-server data and jobs are untouched.

## What is fixed, and what is still a research question

The original GLAC KL sends gradients into both student and teacher Adapter. The
old E1 also multiplies KL gradients by `pi_distill / KL`. Small disagreement can
therefore coexist with weak supervised classification. This is a plausible
mechanism for the supplied logs, not an established causal attribution or proof
of uniform predictions.

The new configurations close the KL target gradient path. Teacher CE STILL
trains the Adapter; its targets can evolve under label supervision across steps.
This is not the same as a wholly frozen pretrained teacher in the DTO-KD paper.

| Variant | KL target gradient | Objective |
| --- | --- | --- |
| E0 | original bidirectional | original released TS-FSAR objective |
| E1 | original bidirectional | historical log-KL/log-DCM plus external CE |
| E0_detach | stopped | same scalar objective as E0 |
| E1_detach | stopped | same scalar objective as historical E1; retains inverse-KL risk |
| E1_safe | stopped | shifted logs of complete supervised loss and base-weighted KL |

For E1_safe define, with the released coefficients for the selected shot:

```
S = VIS_ALPHA * DCM_CE + CLS_ALPHA * student_CE
    + KD_ALPHA * teacher_CE + SIDE_CLS_ALPHA * side_CE
D = KD_ALPHA * KL(student, stop_gradient(teacher))
L = pi_task * log1p(S) + pi_distill * log1p(D)
```

All three CE terms remain present and keep their released ratios to DCM inside
S. Their common outer multiplier changes: E1_safe is not E0-equivalent. The
existing direct probability update is applied to the SAME two log1p values used
by this objective. Stable log1p is used directly, not FP32 `log(1 + tiny_loss)`.
The additive 1 is a declared design choice, not fitted on Test or selected from
the nine reported training samples. It has not been validated as optimal.

The actual KL gradient coefficient is

```
pi_distill * KD_ALPHA / (1 + KD_ALPHA * KL) <= KD_ALPHA
```

Thus KL approaching zero does not create an inverse-KL singularity. This is a
coefficient bound, not a bound on the underlying gradient or optimizer update.
Very small negative KL roundoff (>= -1e-6) is clamped to zero in E1_safe; larger
negative or nonfinite KL is an error. This tolerance is not a model selection
hyperparameter. The supervised multiplier is `pi_task / (1 + S)` and is logged
explicitly. Absolute optimization scales differ, so no direction-only benefit
or automatic recovery of CE/accuracy is claimed.

E0_detach isolates the KL target path relative to E0. E1_detach isolates that
path relative to the old E1, but may retain the old small-KL problem. E1_safe
changes both loss transformation and supervised grouping relative to E1_detach;
that comparison does not isolate either factor. New experiments must start
fresh, use declared budgets, and retain all negative results. If E1 is stopped
early, report its completed budget; do not present it as an equal-budget final
comparison. Never change fusion weights from these observations or tune on Test.

## Stop old E1, keep the source of running controls intact

Do not pull new code into the directory used by still-running E0/E2. The
following commands fetch the branch and check it out in a NEW worktree. They
reuse only the verified local CLIP files via a local symlink.

Run in the existing campus repository, using the activated tsfsar-dto-kd env:

```bash
set -e
OLD_TREE="$PWD"
OLD_RUN="$(realpath "$(cat runs/campus/latest_rigorous_root.txt)")"
OLD_ACCEPT="$(realpath "$(cat runs/campus/latest_acceptance_root.txt)")"

git fetch origin research/dto-kd-fsar-gradient-coordination
NEW_TREE="${OLD_TREE}-dto-safe-$(date +%Y%m%d_%H%M%S)"
git worktree add --detach "$NEW_TREE" FETCH_HEAD
mkdir -p "$NEW_TREE/model"
ln -s "$OLD_TREE/model/vit-b-16" "$NEW_TREE/model/vit-b-16"

python "$NEW_TREE/scripts/stop_campus_jobs.py" --run-root "$OLD_RUN" --apply
python - "$OLD_RUN" <<'PY'
import json, sys
from pathlib import Path
for job in json.loads((Path(sys.argv[1]) / 'status.json').read_text())['jobs']:
    print(job['name'], job['status'], 'gpu=', job.get('gpu'), 'pid=', job.get('pid'))
PY

cd "$NEW_TREE"
mkdir -p runs/campus
printf '%s\n' "$OLD_ACCEPT" > runs/campus/previous_acceptance_root.txt
python scripts/verify_weights.py
```

The stopping tool validates live PID ownership, process session and exact
`--cfg` path before sending SIGTERM to E1_1shot/E1_5shot only. It records an
interruption request and retains all existing files. It never sends SIGKILL,
signals the old launcher, or rewrites its status.json. A process still exiting
may take time to release its GPU; check `nvidia-smi` before the next step.

**Old launcher consequence:** it treats an interrupted E1 as a failure, keeps
other running E0/E2 jobs alive, and cancels every not-yet-started job (possibly
E2_5shot_kl0.02). The stopping tool prints that list. Do not mistake an unstarted
cancelled E2 for a completed control. Requeue it separately later if still needed,
without overwriting old output; no original-server queue is changed by this work.

Omit `--apply` for a read-only preview. If an identity check fails, the tool stops
without signalling any target. Do not replace it with a broad pkill command.

## Campus verification and optional fresh training

New code changes the source fingerprint. Old acceptance reports can supply
their hash-verified INITIAL model/RNG only, never their pass status. The new
full-model acceptance reruns E0 equivalence, then 32-episode smoke checks for
E0, old E1, E0_detach, E1_detach and E1_safe, all from the common initialization.
It verifies teacher KL gradient zero, teacher CE gradient nonzero, and the safe
KL coefficient bound. These checks establish engineering properties, not final
accuracy, causal attribution or full-budget stability.

On idle, allocated, ECC-clean GPUs 2 and 3 (the original E1 allocations), run:

```bash
TSFSAR_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=2 \
  python -m unittest tests.test_dto_kd_cuda -v
```

Only after that passes, the following command performs both-shot acceptance in
parallel and starts a NEW E1_safe queue ONLY IF BOTH pass. Nothing is resumed
from an E1 training checkpoint. The reused initialization and Episode protocol
remain common with the existing controls. A10 batch/frame/query settings,
optimizer, schedule, validation budget and fusion settings remain unchanged.

```bash
export TSFSAR_SSV2_ROOT=/data/fsar_workspace/datasets/SSv2-Small
export TSFSAR_RUNTIME_TMPDIR=/tmp
export TSFSAR_CPU_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OLD_ACCEPT="$(cat runs/campus/previous_acceptance_root.txt)"
REPAIR_ROOT="runs/campus/adaptation_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_campus_adaptation_repair.sh \
  --previous-acceptance-root "$OLD_ACCEPT" --gpus 2,3 --seed 0 \
  --experiments E1_safe --output-root "$REPAIR_ROOT" \
  > "${REPAIR_ROOT}.driver.log" 2>&1 < /dev/null &
echo "driver pid=$! log=${REPAIR_ROOT}.driver.log"
```

To also obtain a fresh E0_detach control, select
`--experiments E0_detach,E1_safe` in that command (four formal tasks, two GPUs).
E1_detach is available explicitly but is not automatically added as a full run.
Do not duplicate the still-running E0/E2 or use GPU7 with its reported ECC fault.

Monitor without following large model printouts:

```bash
REPAIR_ROOT="$(cat runs/campus/latest_adaptation_root.txt)"
tail -F "$REPAIR_ROOT/acceptance/shot1.console.log" \
        "$REPAIR_ROOT/acceptance/shot5.console.log"
# Once acceptance passes and the fresh formal queue starts:
tail -F "$REPAIR_ROOT/formal/E1_safe_1shot/console.log" \
        "$REPAIR_ROOT/formal/E1_safe_5shot/console.log"
```

Every training step writes raw CE/KL, actual coefficients, controller weights,
AMP status and global response entropy/max probability/accuracy to
`train_trace.jsonl`. Those accuracy values are fractions, not percentages.
Train console cadence is still 100 Episodes. Existing parameter-group probes
now also include KL-vs-auxiliary and KL-vs-complete-supervised gradient metrics;
for the teacher group the KL vector is zero by construction, so its cosine is
not evidence of direction agreement. Compare raw CE and branch accuracy rather
than total losses across changed objective definitions. A tiny KL alone is never
an acceptance criterion for research success.

## Local validation scope

CPU tests use synthetic tiny LSN/features with the real loss paths. They test
target-gradient isolation without deleting teacher CE, unchanged scalar outputs
for detach-only controls, bounded small-KL derivatives, correct effective-gradient
diagnostics and exact complete-state resume. Original E0/E2 loss equivalence
regressions remain covered. No original-server training, real-model inference,
GPU computation or queue modification was performed. Actual A10 acceptance and
scientific recovery of CE/final accuracy remain unverified until run on campus.
