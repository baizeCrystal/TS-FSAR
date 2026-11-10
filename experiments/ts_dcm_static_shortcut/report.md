# Static-shortcut diagnostic report template

## Execution identity

Record the checkpoint path and SHA256, protocol manifest hash, episode manifest
hash, mask manifest hash, video-root identity, CUDA device, wall time, and
whether the status is `complete` or `partial_timeout`. Record the exact model
config and verify strict checkpoint loading.

## Evidence gate

Report how many query videos have approved frame-level masks, the visual review
pass rate, and whether the hand, manipulated object, and visible trajectory are
protected. If any of these are missing, stop at the pilot and state that the
experiment is not a strict background counterfactual.

## Paired results

For 1-shot (up to 500 episodes) and 5-shot (up to 300 episodes), report E0,
E1, E2 and E3 for TS-DCM, LSN--Text, Adapter--Text and full fusion. Include
accuracy, prediction-change rate, correct-to-wrong, wrong-to-correct, and
true-class minus strongest-wrong-class margin with episode-cluster bootstrap
95% intervals. Keep query-only as the primary analysis; support-only and
query+support are supplementary.

## Interpretation rule

Call H1 supported only when a reviewed background-only intervention changes
predictions systematically, the foreground control cannot explain it, and the
paired interval is consistent with a directional effect. Attribute H2 to a
branch only when that branch changes in the same paired cases; inference alone
cannot establish a causal GLAC training effect. Call H3 supported only when
alpha-D/TSM changes track the true-versus-rival TS-DCM margin in reviewed cases.
Otherwise use the conservative labels: appearance-distribution sensitivity,
text-branch sensitivity, or no stable evidence. Background accuracy loss alone
does not establish a static shortcut.

## Current status

Before a reviewed mask manifest and a complete campus forward are available,
the correct conclusion is: **formal H1/H2/H3 remain untested; existing S02
feature-region results are not pixel-level causal evidence.**

