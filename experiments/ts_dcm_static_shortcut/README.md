# TS-FSAR static-shortcut counterfactual diagnostic

This directory is a frozen-checkpoint, read-only diagnostic. It tests whether
changing pixels outside a reviewed action/interaction mask changes TS-FSAR's
decision while keeping the same episode, labels, frame indices, and support
videos. It does not train, alter weights, add a model branch, or consume the
existing feature caches.

The primary protocol is query-only and contains four paired conditions:

* `E0`: original sampled frames;
* `E1_background_blur`: Gaussian blur on background pixels only;
* `E2_background_texture`: deterministic low-frequency colour/texture change on
  background pixels only;
* `E3_foreground_blur`: the matched foreground control.

The intervention happens on native decoded RGB frames before the repository's
`ToTensorVideo -> KineticsResizedCropFewshot -> NormalizeVideo` test pipeline.
The output is then required to pass through the ordinary TS-FSAR forward. A
cached LSN/alpha-D/R/W tensor is never a valid input for E1--E3.

## Current evidence gate

The existing S02 evidence is explicitly a post-LSN feature-selection diagnostic;
its `region_conditions.json` says `not_pixel_intervention: true` and
`outside_ROI_is_approximate_background: true`. It therefore cannot be used as a
formal mask for this experiment. `prepare_manifest.py` creates a manifest with
`mask_review_status: pending`, and the runner refuses a formal run until a
reviewed per-frame binary mask is supplied. This is intentional: a center box or
an approximate ROI would not establish a background causal intervention.

## Mask format

Create a JSON object keyed by video ID. Each value must contain a `mask_path`
pointing to an `.npz` with `mask` shaped `[8,H,W]` (native decoded dimensions)
and `frame_indices` equal to the episode manifest. Every frame must contain both
foreground and background pixels. Record the review protocol and set
`mask_review_status` to `approved` only after checking an overlay for every
selected query video. The mask must include the hand, manipulated object, and
their visible trajectory; a coarse rectangle may be used for a pilot, but must
remain marked pending and cannot support the formal result.

## Campus server commands

From the repository root, after adding approved masks and setting the video root:

```bash
export TS_FSAR_VIDEO_ROOT=/path/to/ssv2/videos
export CUDA_VISIBLE_DEVICES=0
python -m experiments.ts_dcm_static_shortcut.prepare_manifest \
  --episodes /path/to/fixed_val_1shot/episodes.jsonl \
  --checkpoint /path/to/E0/model_best.pth \
  --mask-manifest /path/to/approved_masks.json \
  --shot 1 --output /path/to/static_shortcut_1shot.json
```

Edit the generated manifest after the visual audit (`mask_review_status` must
be `approved`, and add `video_root`). A materialization pilot can run on CPU or
GPU without the model:

```bash
python -m experiments.ts_dcm_static_shortcut.run_counterfactual \
  --protocol /path/to/static_shortcut_1shot.json \
  --output output/ts_dcm_static_shortcut_diagnosis/pilot \
  --materialize-only
```

The repository includes a frozen evaluator hook. Set the same config that was
used to produce the fixed episodes (the supplied S02 1-shot config is an
example), then run:

```bash
export TS_FSAR_STATIC_CONFIG=research/three_question_counterfactual_audit_20261105/sources/S02_local_evidence/config.yaml
export TS_FSAR_STATIC_CHECKPOINT=/path/to/E0/model_best.pth
python -m experiments.ts_dcm_static_shortcut.run_counterfactual \
  --protocol /path/to/static_shortcut_1shot.json \
  --output output/ts_dcm_static_shortcut_diagnosis/formal \
  --evaluator experiments.ts_dcm_static_shortcut.campus_evaluator:evaluate
```

The hook strictly loads the checkpoint, runs the normal LSN/TS-DCM forward on
every condition, and exports query-level rows for `ts_dcm`, `lsn_text`,
`adapter_text`, and `fusion`, including scores/probabilities, predictions,
labels, true-vs-rival margins, alpha-D matrices, and TSM matrices. It does not
read the existing feature caches. The runner enforces a 55-minute deadline and
writes `partial_timeout` status if it is reached; it never labels a partial run
complete. For 5-shot, use `Metatest_5shot.yaml`, its matching checkpoint, and a
5-shot fixed episode manifest.

Offline paired statistics use:

```bash
python -m experiments.ts_dcm_static_shortcut.analyze \
  --rows output/ts_dcm_static_shortcut_diagnosis/formal/query_rows.jsonl \
  --output output/ts_dcm_static_shortcut_diagnosis/formal/summary.json
```

Run the CPU checks with `pytest -q experiments/ts_dcm_static_shortcut/tests`.

