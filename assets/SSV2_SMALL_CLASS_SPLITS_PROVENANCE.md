# SSv2-Small class-index mapping provenance

Copied from the existing `output/glac_loss_ablation/metadata/class_mapping.json`
in the source TS-FSAR workspace. Source SHA256: `c60c731300c6056041eb7ddf5e045ec608de47304366993dc1c88a081090d14d`.

The order gives the original train0..63, val0..11, test0..23 label mapping.
Val prompts use the existing class-name-only recipe from
`research/blocking_causal_validation_20261105/inputs/validation_prompt_names.json`.
This supplies previously missing validation text metadata. It does not modify the
training or final Test prompt files. All prospective experiment groups use this
same frozen Val text recipe. This protocol corrects the previous campus entry
which selected checkpoints using Test; historical Test-selected checkpoints are
reference artifacts, not matched prospective controls.
