# Read-only audit

The repository was inspected without resetting the dirty worktree. The relevant
paths and actual functions are:

The previously recorded E0 1-shot checkpoint is
`output/ablation_frame_tsm/ssv2_small_1shot_E0_gpu0_20261021_095537/E0/model_best.pth`
(SHA256 `78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf`).
The fixed episode cache used in that audit is
`output/failure_scenario_diagnostics/E0_val1shot_context_then_appearance_20261103_104126`.
The repository configs for the two requested protocols are
`configs/ssv2_small/Metatest_1shot.yaml` and `configs/ssv2_small/Metatest_5shot.yaml`;
their checkpoint paths are the `TEST.PRETRAIN_MODEL_PATH` values and must be
replaced with the exact E0 best checkpoint selected for the campus run.

* `datasets/base_dataset.py::_decode_video` samples the fixed indices returned
  by `_get_video_frames_list`; no pixel intervention happens there.
* `datasets/video_few_shot.py::get_seq` and `::_config_transform` apply
  `ToTensorVideo`, `KineticsResizedCropFewshot` (224 crop, test center crop),
  and `NormalizeVideo`. The intervention runner therefore operates on decoded
  native RGB frames before this pipeline.
* `trainers/model.py::LSN.forward` produces the adapted LSN sequence and CLIP
  CLS output. The LSN output has one CLS token followed by patch tokens.
* `trainers/ts_fsar.py::Alpha_DC` uses `DC_ALPHA` (0.4 in the SSv2-small
  configs), double centering, `+1e-5`, and upper-triangle extraction. In
  `TS_DCM.forward`, signed CLS--patch cosine weights form alpha-D features;
  class support features are averaged over shots before query-to-prototype
  frame correlations, and `generator(query_cls_mean + global_support_cls_mean)`
  produces the task-specific 8x8 TSM matrix.
* `TS_FSAR.forward` returns the TS-DCM branch and, at evaluation, combines it
  with Adapter--Text and LSN--Text probabilities using the configured fusion
  powers. The exact 1-shot/5-shot weights are in
  `configs/ssv2_small/Metatest_1shot.yaml` and `Metatest_5shot.yaml`.

The historical caches under `output/failure_scenario_diagnostics` and
`research/three_question_counterfactual_audit_20261105/sources/S02_local_evidence`
contain features or post-LSN region selections. They can validate episode
identity and the old E0 readout, but using their features for E1--E3 would
bypass the altered pixels and invalidate the causal comparison.

The existing S02 annotations are coarse rectangles and explicitly state
`not_pixel_intervention: true`; no reliable reviewed native-resolution mask was
found in the repository. Consequently no strict H1/H2/H3 result is claimed by
this implementation. The formal runner fails closed until such masks exist.

