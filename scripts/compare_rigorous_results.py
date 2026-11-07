#!/usr/bin/env python3
"""Strict paired episode comparison of saved predictions; never selects weights."""
import argparse
import json
from pathlib import Path

import numpy as np

BRANCHES = ('matching', 'clip', 'side', 'fusion')


def read(path):
    rows = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row['episode']
            if key in rows:
                raise ValueError('duplicate episode: ' + str(key))
            if not row['labels'] or set(row['predictions']) != set(BRANCHES):
                raise ValueError('missing labels/branch predictions')
            if any(len(row['labels']) != len(row['predictions'][b]) for b in BRANCHES):
                raise ValueError('query counts do not match')
            rows[key] = row
    if not rows:
        raise ValueError('empty predictions')
    return rows


def compare(reference, candidate, repetitions=5000, seed=0):
    if repetitions < 5000:
        raise ValueError('at least 5000 episode bootstrap replicates required')
    if set(reference) != set(candidate):
        raise ValueError('episode IDs differ; refusing a forced paired analysis')
    ids = sorted(reference)
    for key in ids:
        a, b = reference[key], candidate[key]
        if a['episode_sha256'] != b['episode_sha256'] or a['labels'] != b['labels']:
            raise ValueError(f'episode {key}: IDs/frames/inputs/query labels differ')
    results = {}
    for branch in BRANCHES:
        correct_a = [np.asarray(reference[i]['predictions'][branch]) == reference[i]['labels'] for i in ids]
        correct_b = [np.asarray(candidate[i]['predictions'][branch]) == candidate[i]['labels'] for i in ids]
        a = np.array([x.mean() for x in correct_a])
        b = np.array([x.mean() for x in correct_b])
        difference = 100. * (b - a)
        rng = np.random.default_rng(seed)
        bootstrap = np.empty(repetitions)
        for start in range(0, repetitions, 128):
            stop = min(start + 128, repetitions)
            indices = rng.integers(0, len(ids), size=(stop - start, len(ids)))
            bootstrap[start:stop] = difference[indices].mean(1)
        ac, bc = np.concatenate(correct_a), np.concatenate(correct_b)
        results[branch] = {'reference_percent': 100. * a.mean(), 'candidate_percent': 100. * b.mean(),
                           'difference_pp': difference.mean(),
                           'ci95_pp': np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                           'query_counts': {'both_correct': int((ac & bc).sum()),
                                            'both_wrong': int((~ac & ~bc).sum()),
                                            'corrected': int((~ac & bc).sum()),
                                            'new_errors': int((ac & ~bc).sum())}}
    return {'episodes': len(ids), 'bootstrap_replicates': repetitions, 'bootstrap_seed': seed,
            'resampling_unit': 'episode', 'branches': results,
            'limitations': ['Unadjusted per-contrast CIs; a single training seed does not measure seed variability.',
                            'Historical Test-selected checkpoints provide descriptive comparisons only.',
                            'These saved Test results must not choose a method, fusion weight or hyperparameter.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--historical-descriptive', action='store_true',
                        help='explicitly label missing/unmatched training provenance as descriptive only')
    args = parser.parse_args()
    reference_meta = json.loads((args.reference.parent.parent / 'protocol_v2.json').read_text())
    candidate_meta = json.loads((args.candidate.parent.parent / 'protocol_v2.json').read_text())
    for key in ('version', 'source_sha256', 'shot', 'seed', 'evaluation_protocol_sha256'):
        if reference_meta[key] != candidate_meta[key]:
            raise ValueError('evaluation provenance mismatch: ' + key)
    a, b = reference_meta.get('training_protocol'), candidate_meta.get('training_protocol')
    matched_training = (a is not None and b is not None and all(
        a.get(key) == b.get(key) for key in ('version', 'source_sha256', 'initial_model_sha256', 'shot', 'seed')))
    if not matched_training and not args.historical_descriptive:
        raise ValueError('matched v2 training provenance absent; use --historical-descriptive for historical comparisons')
    result = compare(read(args.reference), read(args.candidate), args.bootstrap, args.seed)
    result['matched_training_provenance'] = matched_training
    result['comparison_kind'] = 'controlled_v2' if matched_training else 'historical_descriptive'
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
