"""Raw and effective weighted gradients on declared parameter groups."""
import torch

from .controller import gradient_metrics
from .gradient_utils import loss_gradients


def coefficients(model, cfg, task, distill):
    if model.dto_mode == 'dynamic':
        # These are the weights used by this forward, before the next update.
        d, t = model.dto_last_metrics['pi_distill'], model.dto_last_metrics['pi_task']
        eps = model.dto_controller.eps
        return (t / float(task.detach()) if float(task.detach()) >= eps else 0.,
                d / float(distill.detach()) if float(distill.detach()) >= eps else 0.)
    alpha = cfg.TRAIN.KD_ALPHA if model.dto_mode == 'off' else model.dto_static_kl_alpha
    return float(cfg.TRAIN.VIS_ALPHA), float(alpha)


def parameter_group(name):
    if name.startswith('model.side'):
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
        auxiliary_metrics = gradient_metrics(selected['auxiliary'], effective_task)
        all_auxiliary = gradient_metrics(selected['auxiliary'].double() + effective_distill, effective_task)
        rows.append({'parameter_group': group, 'finite': True, 'task_coefficient': task_weight,
                     'distill_coefficient': distill_weight,
                     **{'raw_' + k: v for k, v in raw.items()},
                     **{'weighted_' + k: v for k, v in weighted.items()},
                     **{'auxiliary_' + k: v for k, v in auxiliary_metrics.items()},
                     **{'all_auxiliary_' + k: v for k, v in all_auxiliary.items()},
                     'both_objective_gradients_nonzero': raw['task_norm'] > 0 and raw['distill_norm'] > 0})
    return rows
