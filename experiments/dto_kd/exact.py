"""DTO-KD Eq.8--15 on actual shared-LSN log gradients.

This is the paper's two-gradient solver, not its Eq.18 approximation. The
teacher target is detached by the caller; auxiliary CE remains outside this
two-objective problem. Consequently its KKT checks do NOT certify the complete
AdamW update. No model forward, random draw, gradient clipping or .grad mutation
occurs here. Detached gradient vectors/reductions live on CPU to fit A10 memory.
"""
import math
import time

import torch

from .parameter_groups import is_lsn_parameter


def log_loss(loss, eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('DTO log-loss floor must be finite and positive')
    value = loss.float().reshape(())
    if not bool(torch.isfinite(value.detach())) or float(value.detach()) < 0:
        raise FloatingPointError('DTO reference requires finite nonnegative losses')
    return value.clamp_min(eps).log()


def solve_gram(g11, g12, g22, difference_squared):
    """Minimise ||p*g1+(1-p)*g2||² for 0<=p<=1, including boundaries.

    difference_squared is computed directly from vectors to avoid catastrophic
    cancellation in g11+g22-2*g12. Only exactly equal vectors use the .5 tie.
    Normalising the Gram by a common positive scale preserves the solution.
    """
    if not all(math.isfinite(v) for v in (g11, g12, g22, difference_squared)):
        raise FloatingPointError('nonfinite DTO log-gradient Gram matrix')
    if min(g11, g22, difference_squared) < 0:
        raise ValueError('negative squared gradient norm')
    scale = max(g11, g22, abs(g12), difference_squared)
    if difference_squared == 0:
        p = .5
    else:
        p = min(1., max(0., (g22 / scale - g12 / scale) / (difference_squared / scale)))
    q = 1. - p
    dot1, dot2 = p * g11 + q * g12, p * g12 + q * g22
    norm2 = max(0., p * dot1 + q * dot2)
    tolerance = 1e-10 * scale
    # Projection onto a convex segment: <g_i, g*> >= ||g*||².
    if dot1 < norm2 - tolerance or dot2 < norm2 - tolerance:
        raise RuntimeError('DTO simplex solution failed its first-order KKT check')
    return {'pi_distill': p, 'pi_task': q,
            'log_gradient_distill_squared': g11,
            'log_gradient_task_squared': g22,
            'log_gradient_cross': g12,
            'log_gradient_difference_squared': difference_squared,
            'pair_direction_squared': norm2,
            'pair_direction_dot_log_distill': dot1,
            'pair_direction_dot_log_task': dot2,
            'simplex_boundary': p in (0., 1.),
            'pair_stationary': norm2 == 0.,
            'pair_norm_squared_relative_to_gram': norm2 / scale if scale else 0.,
            'pair_kkt_verified': True}


def shared_lsn_weights(model, distill, task, eps):
    """Compute raw gradients twice, then divide by losses in FP64 on CPU.

    Parameters must be graph-connected to BOTH objectives; zero numerical
    gradients are kept. TSM, private heads, teacher Adapter and frozen backbone
    are not included in the shared-LSN Gram. A floor is numerical protection,
    not log1p: below eps the matching clamped log has zero derivative. A zero
    objective gradient can give a stationary boundary solution, logged openly.
    """
    started = time.perf_counter()
    log_loss(distill, eps)
    log_loss(task, eps)
    selected = [(n, p) for n, p in model.named_parameters()
                if is_lsn_parameter(n) and p.requires_grad]
    if not selected:
        raise RuntimeError('DTO exact solver found no trainable shared-LSN candidates')
    params = [p for _, p in selected]
    def gradients(loss):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        values = [None if g is None else g.detach().to(device='cpu', dtype=torch.float32)
                  for g in grads]
        del grads
        return values
    distill_grads = gradients(distill)
    task_grads = gradients(task)
    d, t = float(distill.detach()), float(task.detach())
    wd = 1. / d if d >= eps else 0.
    wt = 1. / t if t >= eps else 0.
    g11 = g12 = g22 = difference_squared = 0.
    names, elements = [], 0
    for (name, param), gd, gt in zip(selected, distill_grads, task_grads):
        if gd is None or gt is None:
            continue
        a, b = gd.double().reshape(-1) * wd, gt.double().reshape(-1) * wt
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise FloatingPointError('nonfinite raw DTO solver gradients: ' + name)
        delta = a - b
        g11 += float(torch.dot(a, a))
        g12 += float(torch.dot(a, b))
        g22 += float(torch.dot(b, b))
        difference_squared += float(torch.dot(delta, delta))
        names.append(name)
        elements += param.numel()
    if not names:
        raise RuntimeError('no LSN parameters are connected to both DTO objectives')
    result = solve_gram(g11, g12, g22, difference_squared)
    result.update(solver='paper-closed-form-simplex', solver_group='shared_lsn',
                  solver_parameter_names=names, solver_numel=elements,
                  solver_seconds=time.perf_counter() - started,
                  distill_log_floor_active=d < eps, task_log_floor_active=t < eps,
                  alignment_scope='KL/DCM pair on shared LSN before auxiliary CE, AMP and AdamW')
    return result
