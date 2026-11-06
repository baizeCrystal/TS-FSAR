# FSAR blocking validation

This package implements the controlled validation protocol for the hypothesis
that the original GLAC teacher KL can suppress later task-discriminative
learning. It does not change `TS_FSAR`, D, PCGrad, or existing experiment
outputs. The package injects the production model and episode loader through a
small Python API, which also makes CPU tests independent of CLIP and video
datasets.

The checked-in audit in [`training_graph_audit.md`](training_graph_audit.md)
records the production graph. The teacher Adapter is trainable: the original
KL has paths to both the α-D student head/LSN and the Adapter. Episode CE goes
through TS-DCM/LSN; Teacher CE reaches the Adapter. Therefore KL regulation
changes only its coefficient and keeps all other GLAC terms intact.

Experiment A calls `counterfactual_update` with one model, optimizer, scheduler,
AMP scaler, and episode closure. Every branch is restored from one exact
model/optimizer/RNG snapshot and uses the real optimizer step. The closure
returns `total`, `kl`, and `matching`; diagnostics include cosine, norms, and
relative projection before the update, plus evaluator outputs after it.

Experiment B calls `run_short_branch` from a formal checkpoint. B0 uses
`KLControlConfig(mode="original")`; B1 uses `selective`; B2 uses `random` with
the same reference intervention count; B3 uses `uniform`, spreading the same
KL budget over all steps. The reference projections and episode hash are
written to each branch checkpoint. Meta-Val is an evaluator only and is never
used to choose a per-episode weight.

Safe CPU smoke commands (they do not touch a GPU) are:

```bash
python -m experiments.blocking_validation.run_counterfactual --demo
python -m experiments.blocking_validation.run_short_branch --demo --mode selective
.conda-env/bin/python -m experiments.blocking_validation.run_cpu_tests
# If pytest is installed, the equivalent discovery command is:
# pytest -q experiments/blocking_validation/tests
```

For a formal run, build the existing `TS_FSAR` and optimizer from the selected
1-shot config, restore the same checkpoint and scheduler state for B0--B3,
construct fixed Meta-Train/Val episode lists, and call the API. Keep one output
directory per branch and preserve the emitted `episode_hash`. A typical
production closure is:

```python
def loss_fn(model, episode, kl_weight):
    parts = model(..., glac_kl_weight=kl_weight)  # adapter added by runner
    return {"total": parts["episode_ce"] + 2*parts["student_ce"]
            + .1*parts["lsn_ce"] + .1*(kl_weight*parts["kl"] + parts["teacher_ce"]),
            "matching": parts["episode_ce"], "kl": parts["kl"]}
```

The current production `TS_FSAR.forward` does not yet expose a KL-weight
argument; the integration point must be added in the experiment adapter (or a
loss closure can reproduce its four measured components). Do not silently
freeze the Adapter when adding that adapter. No formal GPU training is started
by this repository change.
