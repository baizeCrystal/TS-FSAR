"""Raw and effective weighted gradients on declared parameter groups."""
import torch

from .controller import gradient_metrics
from .gradient_utils import loss_gradients
from .parameter_groups import is_lsn_parameter


def coefficients(model, cfg, task, distill):
    if model.dto_mode in ('safe_dynamic', 'paper_exact', 'paper_amortized'):
        return model.dto_last_metrics['task_coefficient'], model.dto_last_metrics['distill_coefficient']
    if model.dto_mode == 'dynamic':
        # These are the weights used by this forward, before the next update.
        d, t = model.dto_last_metrics['pi_distill'], model.dto_last_metrics['pi_task']
        eps = model.dto_controller.eps
        return (t / float(task.detach()) if float(task.detach()) >= eps else 0.,
                d / float(distill.detach()) if float(distill.detach()) >= eps else 0.)
    alpha = cfg.TRAIN.KD_ALPHA if model.dto_mode == 'off' else model.dto_static_kl_alpha
    return float(cfg.TRAIN.VIS_ALPHA), float(alpha)


def parameter_group(name):
    if is_lsn_parameter(name):
        return 'lsn'
    if name.startswith(('model.up_linear', 'model.up_ln')):
        return 'lsn_text_head'
    if name.startswith('model.Adapter'):
        return 'teacher_adapter'
    if name.startswith('model.Linear'):
        return 'global_head'
    if name.startswith('ts_dcm.generator'):
        return 'tsm_generator'
    return 'other'


def diagnose(model, cfg, target_labels):
    task, distill, student, teacher, side = model._components(target_labels)
    auxiliary = (cfg.TRAIN.CLS_ALPHA * student + cfg.TRAIN.SIDE_CLS_ALPHA * side
                 + cfg.TRAIN.KD_ALPHA * teacher)
    names_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    vectors = loss_gradients({'distill': distill, 'task': task, 'auxiliary': auxiliary},
                             [p for _, p in names_params])
    task_weight, distill_weight = coefficients(model, cfg, task, distill)
    auxiliary_weight = model.dto_last_metrics.get('auxiliary_coefficient', 1.) if model.dto_mode == 'safe_dynamic' else 1.
    slices, offset = {}, 0
    for name, param in names_params:
        slices.setdefault(parameter_group(name), []).append((offset, offset + param.numel()))
        offset += param.numel()
    slices['all_trainable'] = [(0, offset)]
    rows = []
    for group, ranges in slices.items():
        selected = {key: torch.cat([v[start:stop] for start, stop in ranges]) for key, v in vectors.items()}
        if not all(bool(torch.isfinite(v).all()) for v in selected.values()):
            rows.append({'parameter_group': group, 'finite': False,
                         'task_coefficient': task_weight, 'distill_coefficient': distill_weight})
            continue
        raw = gradient_metrics(selected['distill'], selected['task'])
        effective_distill = distill_weight * selected['distill'].double()
        effective_task = task_weight * selected['task'].double()
        weighted = gradient_metrics(effective_distill, effective_task)
        effective_auxiliary = auxiliary_weight * selected['auxiliary'].double()
        auxiliary_metrics = gradient_metrics(effective_auxiliary, effective_task)
        all_auxiliary = gradient_metrics(effective_auxiliary + effective_distill, effective_task)
        distill_vs_auxiliary = gradient_metrics(effective_distill, effective_auxiliary)
        distill_vs_supervised = gradient_metrics(effective_distill, effective_task + effective_auxiliary)
        total = effective_distill + effective_task + effective_auxiliary
        pair = effective_distill + effective_task
        def alignment(vector):
            return {'dot_raw_kl': float(torch.dot(vector, selected['distill'].double())),
                    'dot_raw_dcm': float(torch.dot(vector, selected['task'].double()))}
        rows.append({'parameter_group': group, 'finite': True, 'task_coefficient': task_weight,
                     'distill_coefficient': distill_weight,
                     'auxiliary_coefficient': auxiliary_weight,
                     'detach_teacher': model.dto_detach_teacher,
                     **{'raw_' + k: v for k, v in raw.items()},
                     **{'weighted_' + k: v for k, v in weighted.items()},
                     **{'auxiliary_' + k: v for k, v in auxiliary_metrics.items()},
                     **{'all_auxiliary_' + k: v for k, v in all_auxiliary.items()},
                     **{'distill_vs_auxiliary_' + k: v for k, v in distill_vs_auxiliary.items()},
                     **{'distill_vs_supervised_' + k: v for k, v in distill_vs_supervised.items()},
                     'pair_gradient_alignment': alignment(pair),
                     'total_gradient_alignment': alignment(total),
                     'alignment_sign': 'negative dot means -gradient is locally ascent for that raw objective; before AdamW',
                     'both_objective_gradients_nonzero': raw['task_norm'] > 0 and raw['distill_norm'] > 0})
    return rows
