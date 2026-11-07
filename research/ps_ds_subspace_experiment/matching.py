"""Class-conditional projection; original Alpha_DC, cos_sim and TSM generator."""
import time
import torch
import torch.nn.functional as F
from .subspace import Settings, fit_templates


def project(tokens, u):
    """[B,C,M] -> [B,C,M]; equivalent to (UU^T) tokens."""
    return u @ (u.transpose(-1, -2) @ tokens)


def validate_original(dcm):
    if (dcm.tsm_variant != 'original' or dcm.support_aggregation_mode != 'mean'
            or dcm.residual_enabled):
        raise ValueError('PSDS requires original frozen TS-DCM/TSM, no other variants')


def match_from_templates(dcm, features, support_labels, way, shot, query_per_class, bases):
    """No labels for queries. TSM always from unprojected CLS features."""
    from utils.functions import cos_sim
    validate_original(dcm)
    t = dcm.cfg.DATA.NUM_INPUT_FRAMES
    ns, nq = way * shot, way * query_per_class
    cls, patches = features[:, 0], features[:, 1:]
    weighted = (F.cosine_similarity(cls[:, None], patches, dim=-1)[..., None] * patches).transpose(1, 2)
    videos = weighted.reshape(ns + nq, t, weighted.shape[1], weighted.shape[2])
    support_cls = cls[:ns*t].reshape(ns, t, -1).mean(1).mean(0)
    query_cls = cls[ns*t:].reshape(nq, t, -1).mean(1)
    # Preserve original repeat/add operations and the generator input exactly.
    prototype = (support_cls.reshape(1, 1, -1).repeat(nq, way, 1)
                 + query_cls[:, None].repeat(1, way, 1))
    weights = dcm.generator(prototype)
    columns, stats = [], []
    for c in range(way):
        own = videos[:ns][support_labels == c].reshape(shot*t, weighted.shape[1], weighted.shape[2])
        query = videos[ns:].reshape(nq*t, weighted.shape[1], weighted.shape[2])
        # Float64 template optimization; original feature dtype for alpha-D and TSM.
        u = bases[c].to(dtype=features.dtype)
        projected = project(torch.cat((own, query)), u)
        if projected.shape != (shot*t+nq*t, weighted.shape[1], weighted.shape[2]):
            raise AssertionError('projection changed patch shape')
        adc = dcm.alpha_dc(projected)  # call ORIGINAL alpha-D, including its epsilon
        norms = adc.norm(dim=-1)
        if not torch.isfinite(adc).all() or (norms == 0).any():
            raise FloatingPointError('alpha-D non-finite or zero norm')
        proto = adc[:shot*t].reshape(shot, t, -1).mean(0)
        corr = cos_sim(adc[shot*t:], proto).reshape(nq, t*t)
        columns.append((corr * weights[:, c]).sum(-1))
        stats.append(dict(class_id=c, alpha_d_norm_min=float(norms.min()),
                          alpha_d_norm_max=float(norms.max())))
    score = torch.stack(columns, dim=1)
    if score.shape != (nq, way) or not torch.isfinite(score).all():
        raise FloatingPointError('invalid PSDS score')
    return score, stats


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def matching(dcm, features, support_labels, way, shot, query_per_class,
             enabled=True, settings=Settings()):
    validate_original(dcm)
    if not enabled:
        # Exact original path, no reimplementation and no template construction.
        return dcm(features, support_labels, way, shot, query_per_class)[0], {'enabled': False}
    synchronize(features.device)
    started = time.perf_counter()
    t = dcm.cfg.DATA.NUM_INPUT_FRAMES
    cls, patch = features[:, 0], features[:, 1:]
    weighted = (F.cosine_similarity(cls[:, None], patch, dim=-1)[..., None]*patch).transpose(1, 2)
    support = weighted[:way*shot*t].reshape(way*shot, t, weighted.shape[1], weighted.shape[2])
    u, trace = fit_templates(support, support_labels, way, settings)
    synchronize(features.device)
    fitted = time.perf_counter()
    score, numeric = match_from_templates(dcm, features, support_labels, way, shot, query_per_class, u)
    synchronize(features.device)
    trace.update(enabled=True, fit_seconds=fitted-started,
                 projection_matching_seconds=time.perf_counter()-fitted,
                 total_additional_seconds=time.perf_counter()-started,
                 projected_token_channels=features.shape[-1], projected_patch_count=features.shape[1]-1,
                 alpha_d=numeric)
    return score, trace
