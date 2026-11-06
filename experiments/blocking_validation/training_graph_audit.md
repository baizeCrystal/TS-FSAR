# GLAC training graph audit

The existing implementation computes `loss_vision` from the TS-DCM query to
support score, `loss_gc` from the α-D representation and LSN `Linear` head,
`loss_text_lsn` from LSN CLS features, `loss_text_clip` from the trainable
`model.Adapter`, and `loss_gk = DistillKL(logits_lsn_adc, logits_v2t_clip)`.

`utils.functions.DistillKL` applies `T² KL(teacher || student)` without a
detach. With the production optimizer, names containing `Linear`, `up_linear`,
`up_ln`, `side`, `generator`, and `Adapter` are trainable. Consequently:

| loss | trainable path |
| --- | --- |
| Episode CE | TS-DCM generator and LSN features |
| Student CE | α-D features, LSN side path, and `Linear` |
| LSN text CE | LSN `up_ln`/`up_linear` and side path |
| Teacher CE | `Adapter` |
| KL | Student CE path **and** `Adapter` because teacher logits are attached |

The controller must multiply only `loss_gk`. B0 is the exact original sum;
the teacher CE remains active in every B group. If an optional frozen-teacher
factor is studied later, it must be a separate factor and cannot be folded into
B0--B3.

The historical diagnostics in `output/glac_diagnostics/` and
`research/blocking_causal_validation_20261105/` report Level-1 gradient
conflict only. They do not establish Level 3 learning suppression. This
package records the additional paired-update and short-run evidence needed to
separate those claims.
