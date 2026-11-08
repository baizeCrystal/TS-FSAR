"""One shared gate, original alpha-D, original cosine and unmasked TSM.

Unlike candidate-specific SAM/QAM, each video's alpha-D is computed only once.
Frame chunks bound memory and do not change per-frame alpha-D operations.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from research.ps_ds_subspace_experiment.matching import validate_original
from utils.functions import cos_sim, extract_class_indices


def original_tsm(dcm, features, way, shot, queries):
    t = dcm.cfg.DATA.NUM_INPUT_FRAMES
    cls = features[:, 0]
    support = cls[:way*shot*t].reshape(way*shot, t, -1).mean(1).mean(0)
    query = cls[way*shot*t:].reshape(way*queries, t, -1).mean(1)
    condition = (support.reshape(1, 1, -1).repeat(way*queries, way, 1)
                 + query[:, None].repeat(1, way, 1))
    return dcm.generator(condition)


def matching(dcm, features, labels, way, shot, queries, module=None,
             enabled=True, feature_frames_per_chunk=8, checkpoint_chunks=False,
             gate_override=None):
    validate_original(dcm)
    if not enabled:
        return dcm(features, labels, way, shot, queries)[0], {'enabled': False}
    t = dcm.cfg.DATA.NUM_INPUT_FRAMES
    ns, nq = way*shot, way*queries
    if features.ndim != 3 or features.shape[0] != (ns+nq)*t:
        raise ValueError('invalid episode feature shape')
    if labels.shape != (ns,) or feature_frames_per_chunk < 1:
        raise ValueError('invalid support labels or frame chunk')
    cls = features[:ns*t, 0].reshape(ns, t, -1)
    if gate_override is None:
        if module is None:
            raise ValueError('trained CLS-CTM module required')
        gate, detail = module(cls, labels, way)
    else:
        gate, detail = gate_override, {}
    if gate.shape != (features.shape[-1],) or not bool(torch.isfinite(gate).all()):
        raise ValueError('gate must be one finite [channel] vector per episode')

    def compute(part, shared_gate):
        enhanced = part * shared_gate
        frame_cls, patch = enhanced[:, 0], enhanced[:, 1:]
        weighted = F.cosine_similarity(frame_cls[:, None], patch, dim=-1)[..., None] * patch
        return dcm.alpha_dc(weighted.transpose(1, 2))

    outputs = []
    for part in features.split(feature_frames_per_chunk):
        if checkpoint_chunks and torch.is_grad_enabled():
            value = checkpoint(compute, part, gate, use_reentrant=False)
        else:
            value = compute(part, gate)
        outputs.append(value)
    adc = torch.cat(outputs)
    if not bool(torch.isfinite(adc).all()) or bool((adc.norm(dim=-1) == 0).any()):
        raise FloatingPointError('nonfinite or zero-norm alpha-D after channel weighting')
    support = adc[:ns*t].reshape(ns, t, -1)
    prototypes = torch.stack([support.index_select(0, extract_class_indices(labels, c)).mean(0)
                              for c in torch.unique(labels)])
    correlation = cos_sim(adc[ns*t:], prototypes.flatten(0, 1)).reshape(
        nq, t, way, t).permute(0, 2, 1, 3).reshape(nq, way, t*t)
    # Always build the temporal condition from unweighted features.
    weights = original_tsm(dcm, features, way, shot, queries)
    score = (correlation * weights).sum(-1)
    if not bool(torch.isfinite(score).all()):
        raise FloatingPointError('nonfinite matching score')
    detached = gate.detach()
    detail.update(enabled=True, gate_shape=list(gate.shape),
        gate_min=float(detached.min()), gate_max=float(detached.max()),
        gate_mean=float(detached.mean()), gate_std=float(detached.std(unbiased=False)),
        gate_mean_absolute_deviation_from_one=float((detached-1).abs().mean()),
        query_labels_used=False, query_features_used_for_gate=False,
        original_tsm_condition=True, episode_shared_gate=True,
        token_shape=list(features.shape), checkpoint_chunks=checkpoint_chunks)
    return score, detail
