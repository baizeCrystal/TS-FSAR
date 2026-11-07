"""SRL eqs. 1/2/4/5/6, adapted from images to weighted video patch sets.

There is deliberately no query argument in fit_templates. PS -> DS is our
migration design, not a claim that SRL used this sequential initialization.
Paper ambiguities/sign corrections are documented in METHOD.md.
"""
from dataclasses import dataclass, asdict
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Settings:
    rank: int = 5
    ps_steps: int = 50
    ds_steps: int = 50
    ps_lr: float = 0.1
    ds_lr: float = 0.01
    distance_floor: float = 1e-12
    orthogonality_tolerance: float = 1e-7


def orth_error(u):
    eye = torch.eye(u.shape[-1], dtype=u.dtype, device=u.device)
    return (u.transpose(-1, -2) @ u - eye).abs().amax()


def svd_subspace(h, rank):
    """Left singular vectors of H[C,M] via HH^T eigendecomposition (float64).

    Mathematically identical SVD subspace/singular values, avoiding large V^T.
    No centering, L2 preprocessing, whitening or randomized truncation.
    """
    h = h.detach().double()
    if not torch.isfinite(h).all() or not 1 <= rank <= min(h.shape[-2:]):
        raise ValueError('invalid feature matrix or rank')
    values, vectors = torch.linalg.eigh(h @ h.transpose(-1, -2))
    values = values.flip(-1).clamp_min(0)
    singular = values.sqrt()
    # HH^T squares the condition number. Use its resolvable numerical rank,
    # rather than pretending its tiny positive eigenvalues are reliable SVD modes.
    cutoff = values[..., :1] * (max(h.shape[-2:]) * torch.finfo(h.dtype).eps)
    effective = (values > cutoff).sum(-1)
    if (effective < rank).any():
        raise ValueError('support effective rank below locked rank; no silent fallback')
    u = vectors.flip(-1)[..., :rank]
    s = singular[..., :rank]
    return u, s / s.sum(-1, keepdim=True), effective


def wsd(a, wa, b, wb, floor=1e-12):
    """All pairwise SRL WSD; a[N,C,r], b[K,C,r], weights sum to one.

    D^2 = 1 - sum_ij sqrt(lambda'_ai lambda'_bj) (u_ai^T u_bj)^2.
    Singular values (not squared singular values) define lambda'.
    """
    overlap = torch.einsum('nci,kcj->nkij', a, b)
    weights = torch.sqrt(wa[:, None, :, None] * wb[None, :, None, :])
    squared = 1 - (overlap.square() * weights).sum((-1, -2))
    if squared.detach().amin() < -1e-7:
        raise FloatingPointError('WSD left valid domain; orthogonality failure')
    return squared.clamp_min(floor).sqrt()


def cayley_descent(u, z, lr):
    """Exact paper tangent generator + descending Cayley, Woodbury rank-2r solve.

    W = ZU^T-UZ^T - .5 U(U^TZ-Z^TU)U^T.
    U+ = (I+alpha W/2)^-1 (I-alpha W/2)U.
    No QR retraction, skipped steps, explicit channel-space inverse or damping.
    """
    ut = u.transpose(-1, -2)
    zt = z.transpose(-1, -2)
    skew = ut @ z - zt @ u
    a = torch.cat((z, u), dim=-1)
    bt = torch.cat((ut, -zt - .5 * skew @ ut), dim=-2)
    eye = torch.eye(a.shape[-1], dtype=u.dtype, device=u.device)
    return u - lr * a @ torch.linalg.solve(eye + .5 * lr * (bt @ a), bt @ u)


def ps_objective(u, weights, samples, sample_weights, labels, floor):
    distances = wsd(samples, sample_weights, u, weights, floor)
    return distances.gather(1, labels[:, None]).sum()


def ds_objective(u, weights, samples, sample_weights, labels, floor):
    # Negative log likelihood, SUM over support samples, temperature exactly 1.
    return F.cross_entropy(-wsd(samples, sample_weights, u, weights, floor),
                           labels, reduction='sum')


def fit_templates(support, labels, way, settings=Settings()):
    """support[N,T,C,M]; support labels only. Returns detached U[way,C,r]."""
    if support.ndim != 4 or labels.shape != (support.shape[0],):
        raise ValueError('expected video-major support and one label per video')
    if not torch.equal(torch.unique(labels), torch.arange(way, device=labels.device)):
        raise ValueError('episode labels must be contiguous 0..way-1')
    shot = support.shape[0] // way
    if any(int((labels == c).sum()) != shot for c in range(way)):
        raise ValueError('unbalanced supports')
    h = support.detach().permute(0, 2, 1, 3).flatten(2)  # all frames/patches per video
    samples, sample_weights, sample_rank = svd_subspace(h, settings.rank)
    unions = torch.stack([h[labels == c].permute(1, 0, 2).flatten(1) for c in range(way)])
    u, weights, union_rank = svd_subspace(unions, settings.rank)
    trace = {}
    max_orth = float(orth_error(u))
    # Only these newly allocated episode template tensors require gradients.
    # This also works inside outer torch.no_grad(), but NOT inference_mode().
    with torch.enable_grad():
        for name, objective, steps, lr in (
                ('ps', ps_objective, settings.ps_steps, settings.ps_lr),
                ('ds', ds_objective, settings.ds_steps, settings.ds_lr)):
            trace[name + '_initial'] = float(objective(
                u, weights, samples, sample_weights, labels, settings.distance_floor))
            norms = []
            for _ in range(steps):
                u = u.detach().requires_grad_(True)
                loss = objective(u, weights, samples, sample_weights, labels, settings.distance_floor)
                z, = torch.autograd.grad(loss, (u,))
                with torch.no_grad():
                    u = cayley_descent(u, z, lr)
                    error = float(orth_error(u))
                    max_orth = max(max_orth, error)
                    if not torch.isfinite(u).all() or error > settings.orthogonality_tolerance:
                        raise FloatingPointError('non-finite / non-orthonormal Cayley update')
                    norms.append(float(z.norm()))
            trace[name + '_final'] = float(objective(
                u, weights, samples, sample_weights, labels, settings.distance_floor))
            trace[name + '_objective_increased'] = trace[name + '_final'] > trace[name + '_initial'] + 1e-9
            trace[name + '_steps'] = steps
            trace[name + '_gradient_norm_max'] = max(norms, default=0.)
    trace.update(settings=asdict(settings), template_shape=list(u.shape),
                 sample_effective_rank=sample_rank.tolist(), union_effective_rank=union_rank.tolist(),
                 max_orthogonality_error=max_orth, final_orthogonality_error=float(orth_error(u)),
                 template_singular_weights=weights.tolist(), query_labels_used=False)
    return u.detach(), trace
