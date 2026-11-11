"""Package historical episode identities and E0 references; CPU, no forward.

This mechanical export makes the first-stage protocols available after git pull
without assuming the campus machine already has ignored output directories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .core import file_hash, read_rows, validate_episode, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='multi_metric diagnostic root')
    p.add_argument('--historical', type=Path, required=True, help='tsm_class_discriminability root')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=200)
    a = p.parse_args()
    if a.limit < 1 or a.output.exists():
        p.error('positive limit and a new output directory required')
    a.output.mkdir(parents=True)
    for shot in (1, 5):
        source, history = a.source / f'novel_{shot}shot', a.historical / f'novel_{shot}shot'
        episodes = read_rows(source / 'episodes.jsonl')[:a.limit]
        if len(episodes) != a.limit:
            raise ValueError('Source has fewer episodes than requested')
        for ep in episodes:
            validate_episode(ep, shot)
        config = yaml.safe_load((source / 'effective_config.yaml').read_text())
        folder = a.output / f'{shot}shot'
        folder.mkdir()
        (folder / 'episodes.jsonl').write_text(''.join(json.dumps(ep, separators=(',', ':')) + '\n' for ep in episodes))
        write_json(folder / 'config.json', config)
        pixels = json.loads((source / 'pixel_hashes.json').read_text())[:a.limit]
        ref = {name: np.load(history / (name + '.npy'), mmap_mode='r')[:a.limit].tolist()
               for name in ('matching_score', 'clip_probability', 'side_probability', 'full_log_score', 'targets')}
        ref['pixel_hashes'] = pixels
        write_json(folder / 'reference.json', ref)
        write_json(folder / 'provenance.json', {
            'source': str(source.resolve()), 'historical': str(history.resolve()),
            'source_episodes_sha256': file_hash(source / 'episodes.jsonl'),
            'selection': f'first {a.limit} unchanged episodes; no outcome selection',
            'checkpoint_sha256': '78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf',
            'checkpoint_training_shot': 1, 'inference_shot': shot,
            'E0_sampling': 'official uniform-bin random (rate=50), replay locked indices',
            'historical_names_not_reused': 'old E1/E2 denote different metrics; only old E0 is imported',
            'files': {name: file_hash(folder / name) for name in ('episodes.jsonl', 'config.json', 'reference.json')},
        })
    print(a.output)


if __name__ == '__main__':
    main()
