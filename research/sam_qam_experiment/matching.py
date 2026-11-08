"""Candidate-specific channel gates, ORIGINAL alpha-D / cosine / signed TSM."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from research.ps_ds_subspace_experiment.matching import validate_original


def original_tsm(dcm, features, way, shot, queries):
    """Exact original CLS mean/repeat/add, with NO masked CLS in the generator."""
    t = dcm.cfg.DATA.NUM_INPUT_FRAMES
    cls = features[:, 0]
    support = cls[:way*shot*t].reshape(way*shot, t, -1).mean(1).mean(0)
    query = cls[way*shot*t:].reshape(way*queries, t, -1).mean(1)
    condition = (support.reshape(1, 1, -1).repeat(way*queries, way, 1)
                 + query[:, None].repeat(1, way, 1))
    return dcm.generator(condition)


def pair_scores(dcm, own, queries, gate, weights):
    """Batch independent query/class pairs, then shot-mean ORIGINAL alpha-D.

own[K,T,197,C], queries[B,T,197,C], gate[B,C], weights[B,64].
Each query uses its OWN gate on both sides, before CLS--Patch weighting.
"""
    from utils.functions import cos_sim
    batch, t, tokens, channels = queries.shape
    shot = len(own)
    pair = torch.cat((own[None].expand(batch, -1, -1, -1, -1), queries[:, None]), 1)
    pair = (pair * gate[:, None, None, None, :]).reshape(-1, tokens, channels)
    cls, patches = pair[:, 0], pair[:, 1:]
    weighted = F.cosine_similarity(cls[:, None], patches, dim=-1)[..., None] * patches
    adc = dcm.alpha_dc(weighted.transpose(1, 2)).reshape(batch, shot+1, t, -1)
    if not bool(torch.isfinite(adc).all()) or bool((adc.norm(dim=-1) == 0).any()):
        raise FloatingPointError('non-finite or zero-norm alpha-D after channel weighting')
    proto = adc[:, :shot].mean(1)
    # cos_sim accepts batch axes, retains its original norm-product + .01.
    correlation = cos_sim(adc[:, shot], proto).reshape(batch, t*t)
    return (correlation * weights).sum(-1)


def matching(dcm, features, labels, way, shot, queries, module=None,
             enabled=True, query_chunk=1, checkpoint_chunks=False, gate_override=None):
    """Only support labels are accepted; query labels cannot enter gate generation.

Chunking changes batching only. Non-reentrant activation checkpointing recomputes
alpha-D in backward to bound A10 activation memory; SAM/QAM/BN/noise run ONCE.
"""
    validate_original(dcm)
    if not enabled:
        return dcm(features, labels, way, shot, queries)[0], {'enabled': False}
    if query_chunk < 1:
        raise ValueError('positive query chunk required')
    t = dcm.cfg.DATA.NUM_INPUT_FRAMES
    ns, nq = way*shot, way*queries
    if features.ndim != 3 or features.shape[0] != (ns+nq)*t:
        raise ValueError('invalid episode feature shape')
    videos = features.reshape(ns+nq, t, features.shape[1], features.shape[2])
    if gate_override is None:
        if module is None:
            raise ValueError('trained SAM/QAM required when enabled')
        gates = module(videos[:ns, :, 1:].permute(0, 1, 3, 2), labels,
                       videos[ns:, :, 1:].permute(0, 1, 3, 2), way)
    else:
        gates = gate_override
    if gates.shape != (nq, way, features.shape[-1]):
        raise ValueError('gate shape must be [query, class, channel]')
    weights = original_tsm(dcm, features, way, shot, queries)
    columns = []
    for c in range(way):
        own = videos[:ns][labels == c]
        if len(own) != shot:
            raise ValueError('support shot mismatch')
        parts = []
        for start in range(0, nq, query_chunk):
            end = min(start+query_chunk, nq)
            # Bind dcm only; tensor inputs are explicit to avoid late-bound
            # class/chunk closure bugs in backward recomputation.
            def compute(s, q, g, w):
                return pair_scores(dcm, s, q, g, w)
            inputs = (own, videos[ns+start:ns+end], gates[start:end, c], weights[start:end, c])
            if checkpoint_chunks and torch.is_grad_enabled():
                value = checkpoint(compute, *inputs, use_reentrant=False)
            else:
                value = compute(*inputs)
            parts.append(value)
        columns.append(torch.cat(parts))
    score = torch.stack(columns, 1)
    detached = gates.detach()
    return score, dict(enabled=True, gate_shape=list(gates.shape),
        gate_min=float(detached.min()), gate_max=float(detached.max()),
        gate_mean=float(detached.mean()), gate_std=float(detached.std(unbiased=False)),
        class_gate_difference=float((detached-detached.mean(1, keepdim=True)).abs().mean()),
        query_gate_difference=float((detached-detached.mean(0, keepdim=True)).abs().mean()),
        query_labels_used=False, original_tsm_condition=True,
        token_shape=list(features.shape), checkpoint_chunks=checkpoint_chunks)
