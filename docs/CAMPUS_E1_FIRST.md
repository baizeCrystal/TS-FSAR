# Prioritize E1_safe in the existing campus adaptation queue

This transition defers the two E0_detach controls and starts ONLY E1_safe_1shot
and E1_safe_5shot, on GPUs 2 and 3. It does not rerun the original E0/E2, repeat
CUDA certification, or initialize E1 from an E0 training checkpoint. Old control
outputs remain intact and must be labelled interrupted if their budget was not
completed. Their later continuation is a separate decision; no automatic restart
of those controls is scheduled.

Fetch the repository ref and extract ONLY the stdlib scheduling helper into
/tmp. Do not pull/checkout new model code in the active worktree. Because the
training source/configuration is unchanged, the existing certificates remain
applicable. The original formal entry still validates their source fingerprint
before starting each new E1 job.

```bash
(
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/fsar_workspace/envs/tsfsar-dto-kd

TREE=/data/fsar_workspace/datasets/TS-FSAR-blocking/TS-FSAR-dto-safe-20261107_145849
OLD_ROOT="$TREE/runs/campus/adaptation_20261107_150707"
git -C "$TREE" fetch origin research/dto-kd-fsar-gradient-coordination
PRIORITY_TOOL="$(mktemp /tmp/tsfsar-e1-first.XXXXXX.py)"
git -C "$TREE" show FETCH_HEAD:scripts/prioritize_campus_e1.py > "$PRIORITY_TOOL"

NEW_ROOT="$TREE/runs/campus/e1_first_$(date +%Y%m%d_%H%M%S)"
nohup python -u "$PRIORITY_TOOL" \
  --project-root "$TREE" --run-root "$OLD_ROOT" \
  --output-root "$NEW_ROOT" --apply \
  > "${NEW_ROOT}.driver.log" 2>&1 < /dev/null &

echo "Transition PID=$! log=${NEW_ROOT}.driver.log"
tail -n 60 -F "${NEW_ROOT}.driver.log"
)
```

The helper verifies the old launcher PID, owner, working directory and exact
formal output path. It accepts only the expected four-job adaptation queue and
requires both E1 jobs to still be queued. It sends SIGTERM to that launcher;
the launcher's own handler stops its active E0_detach process groups. It waits
up to 60 seconds for both the old launcher and GPU resources/locks to release.
Other queues are not signalled. Any mismatch, a previously started E1, or a
resource timeout aborts the transition without starting another E1 queue.

The switch request is saved in the old run root. The new run root records the
transition, reuses the old acceptance directory via a symlink, and becomes the
latest_adaptation_root pointer. No old status.json or checkpoint is overwritten.
If a timeout occurs, inspect the recorded state before retrying: this helper
does not force-kill remaining processes or blindly restart an interrupted queue.

Closing the terminal or Ctrl+C during tail does not stop the nohup transition
or its eventual E1 queue. The driver log should show two E1_safe jobs and their
START entries. Actual TRAIN steps are the evidence that updates have begun.
