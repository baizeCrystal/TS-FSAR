"""CPU-only sampling, identity, feature comparisons, and paired statistics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMES = {'e0': 1., 'p075': .75, 'p133': 1.33, 'p05': .5, 'p2': 2.,
           'r075': .75, 'r133': 1.33}
SCOPES = ('query', 'support', 'both')
BRANCHES = ('ts_dcm', 'adapter_text', 'lsn_text', 'fusion')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_episode(ep, shot):
    if len(ep['support']) != 5 * shot or len(ep['query']) != 5:
        raise ValueError('Requires 5-way, 1 query/class, and the specified shot')
    class_map = {}
    for role in ('support', 'query'):
        labels = []
        for v in ep[role]:
            label, cls = int(v['episode_label']), int(v['class_id'])
            if label in class_map and class_map[label] != cls:
                raise ValueError('Inconsistent global/episode label map')
            class_map[label] = cls
            labels.append(label)
            sample_indices(v, 'e0')
        expected = shot if role == 'support' else 1
        if sorted(labels) != sorted(list(range(5)) * expected):
            raise ValueError('Missing/duplicate episode labels')
    for role, field in [('support', 'support_labels'), ('query', 'target_labels')]:
        if field in ep and ep[field] != [v['episode_label'] for v in ep[role]]:
            raise ValueError('Stored labels do not agree with video records')
    if len(set(class_map.values())) != 5:
        raise ValueError('Episode classes must be distinct')
    support_ids = {str(v['video_id']) for v in ep['support']}
    if len(support_ids) != 5 * shot or support_ids & {str(v['video_id']) for v in ep['query']}:
        raise ValueError('Duplicate/leaked support video')
    return [class_map[c] for c in range(5)]


def sample_indices(record, scheme='e0', seed=20261111):
    """Warp locked source positions within the OFFICIAL sampler's eligible range.

    Rate=50 uses floor(N/8) bins; frames in the trailing remainder are excluded
    by the original sampler. Use [0,8*floor(N/8)-1], not a new [0,N-1] range.
    Historical padded clips of N<8 use [0,N-1]. Keep repeats (no hidden repair).
    Gamma=1 bypasses quantization and returns the original exact indices.
    """
    if scheme not in SCHEMES:
        raise ValueError('Unknown scheme: ' + scheme)
    n = int(record['source_num_frames'])
    original = np.asarray(record['frame_indices'], dtype=np.int64)
    hi = n // 8 * 8 - 1 if n >= 8 else n - 1
    if n < 1 or original.shape != (8,) or np.any(np.diff(original) < 0):
        raise ValueError('Requires a nonempty video and 8 chronological locked indices')
    if original.min() < 0 or original.max() > hi:
        raise ValueError('Locked indices exceed official rate=50 sampling range')
    u = original / max(hi, 1)
    gamma = SCHEMES[scheme]
    warped = u ** gamma
    control_info = {}
    if scheme.startswith('r'):
        # Random signs redistribute absolute E1 offsets across time positions.
        # Sorting/clipping/rounding changes RMS; report the realized difference.
        key = int(digest({'seed': seed, 'video': str(record['video_id']),
                          'indices': original.tolist(), 'scheme': scheme})[:16], 16)
        rng = np.random.default_rng(key)
        target_indices = np.rint(warped * hi).astype(np.int64)
        target_rms = float(np.sqrt(np.square(target_indices / max(hi, 1) - u).mean()))
        target_unique = len(np.unique(target_indices))
        candidates = []
        for _ in range(32):
            offsets = rng.permutation(np.abs(warped - u)) * rng.choice([-1., 1.], 8)
            candidate = np.sort(np.clip(u + offsets, 0, 1))
            rounded = np.rint(candidate * hi).astype(np.int64)
            rms = float(np.sqrt(np.square(rounded / max(hi, 1) - u).mean()))
            candidates.append((abs(len(np.unique(rounded)) - target_unique), abs(rms - target_rms), candidate))
        best = min(candidates, key=lambda c: (c[0], c[1]))
        warped = best[2]
        control_info = {'control_candidates': 32, 'control_target_rms': target_rms,
                        'control_target_unique_frames': target_unique, 'control_rms_residual': best[1],
                        'control_unique_frame_residual': best[0],
                        'control_rule': 'minimize unique-frame-count difference, then RMS shift difference; no labels/scores'}
    indices = original.copy() if gamma == 1 else np.rint(warped * hi).astype(np.int64)
    if len(indices) != 8 or np.any(np.diff(indices) < 0):
        raise AssertionError('Sampler changed count or time direction')
    actual = indices / max(hi, 1)
    return {
        'indices': indices.tolist(), 'original_indices': original.tolist(),
        'u': u.tolist(), 'requested_u': warped.tolist(), 'actual_u': actual.tolist(),
        'scheme': scheme, 'gamma': gamma, 'eligible_range': [0, hi],
        'unique_frames': int(len(np.unique(indices))),
        'duplicate_fraction': float(1 - len(np.unique(indices)) / 8),
        'mean_abs_shift': float(np.abs(actual - u).mean()),
        'rms_shift': float(np.sqrt(np.square(actual - u).mean())),
        'duplicate_policy': 'keep_all_eight', 'seed': seed, **control_info,
    }


def cache_key(record, sampling, extraction):
    return digest({'video': str(record['video_id']), 'split': record['split'],
                   'n': record['source_num_frames'], 'sampling': sampling,
                   'extraction': extraction})


def cosine_distance(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    return 1 - (a * b).sum(-1) / denom


def compare_features(original, changed, original_times, changed_times):
    """Sequence-level change, with an interpolation sensitivity check.

    Collapse duplicate source times by averaging, evaluate only the common
    temporal support (no extrapolation). Neither metric establishes semantics.
    """
    a, b = np.asarray(original), np.asarray(changed)
    ta, tb = np.asarray(original_times), np.asarray(changed_times)
    if a.shape != b.shape or len(ta) != len(a) or len(tb) != len(b):
        raise ValueError('Feature/time dimensions disagree')
    ua, ub = np.unique(ta), np.unique(tb)
    va = np.stack([a[ta == t].mean(0) for t in ua])
    vb = np.stack([b[tb == t].mean(0) for t in ub])
    common = np.unique(np.concatenate([ua, ub]))
    common = common[(common >= max(ua[0], ub[0])) & (common <= min(ua[-1], ub[-1]))]
    aligned = None
    if len(common):
        ia = np.stack([np.interp(common, ua, va[:, c]) for c in range(a.shape[1])], axis=1)
        ib = np.stack([np.interp(common, ub, vb[:, c]) for c in range(b.shape[1])], axis=1)
        aligned = float(cosine_distance(ia, ib).mean())
    shared = np.intersect1d(ua, ub)
    shared_dist = None
    if len(shared):
        shared_dist = float(cosine_distance(va[np.searchsorted(ua, shared)], vb[np.searchsorted(ub, shared)]).mean())
    return {'index_cosine_distance': float(cosine_distance(a, b).mean()),
            'time_interpolated_cosine_distance': aligned, 'common_time_points': len(common),
            'shared_frame_cosine_distance': shared_dist, 'shared_frames': len(shared),
            'interpretation': 'sampling_sequence_representation_change_not_semantic_change'}


def margin(scores, labels):
    x = np.asarray(scores, dtype=np.float64).copy()
    y = np.asarray(labels, dtype=np.int64)
    truth = x[np.arange(len(y)), y].copy()
    x[np.arange(len(y)), y] = -np.inf
    return truth - x.max(-1)


def paired_bootstrap(values, episode_ids, repeats=5000, seed=20261111):
    """Resample whole episodes, including all their queries as one cluster."""
    v, ids = np.asarray(values, dtype=float), np.asarray(episode_ids)
    means = np.array([v[ids == eid].mean() for eid in np.unique(ids)])
    if not len(means) or not np.isfinite(means).all() or repeats < 1:
        raise ValueError('Nonempty finite data and positive bootstrap repeats required')
    rng = np.random.default_rng(seed)
    draws = np.array([means[rng.integers(len(means), size=len(means))].mean() for _ in range(repeats)])
    return {'estimate': float(means.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
            'episodes': len(means), 'queries': len(v), 'repeats': repeats, 'seed': seed}


def validate_output(arrays):
    required = {'labels', 'R', 'W', *BRANCHES}
    if not required.issubset(arrays):
        raise ValueError('Missing output keys: ' + str(required - set(arrays)))
    if arrays['labels'].shape != (5,) or arrays['R'].shape != (5, 5, 8, 8) or arrays['W'].shape != (5, 5, 8, 8):
        raise ValueError('Output dimension mismatch')
    for branch in BRANCHES:
        if arrays[branch].shape != (5, 5):
            raise ValueError('Invalid branch score shape')
    if any(not np.isfinite(v).all() for v in arrays.values() if np.issubdtype(v.dtype, np.number)):
        raise ValueError('Non-finite model output')
