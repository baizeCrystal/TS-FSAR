"""Protocol fingerprints and complete restart state; no credentials captured."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hash():
    files = sorted({p for folder in ('experiments/dto_kd', 'trainers', 'datasets', 'utils',
                                     'configs', 'prompts', 'assets')
                    for p in (ROOT / folder).rglob('*')
                    if p.suffix in ('.py', '.yaml', '.json', '.txt')})
    # Include the entry/config tools used to create or certify a run.
    files += [ROOT / name for name in ('main.py', 'scripts/prepare_run_config.py',
                                      'scripts/verify_weights.py', 'scripts/campus_acceptance.py',
                                      'scripts/check_ssv2_files.py')]
    return hashlib.sha256(json.dumps([(str(p.relative_to(ROOT)), file_hash(p))
                                     for p in files], sort_keys=True).encode()).hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state(cuda=False):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if cuda else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def atomic_torch_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def append_json(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


class EpisodeLedger:
    """Append new episodes; replay after resume must match exactly."""
    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        if self.path.exists():
            with self.path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    previous = self.records.setdefault(row['episode'], row['episode_sha256'])
                    if previous != row['episode_sha256']:
                        raise RuntimeError('conflicting episode ledger rows')

    def observe(self, row):
        key, value = row['episode'], row['episode_sha256']
        if key in self.records:
            if self.records[key] != value:
                raise RuntimeError(f'episode {key} changed on replay')
        else:
            append_json(self.path, row)
            self.records[key] = value

    def sequence_hash(self, count):
        if not all(i in self.records for i in range(count)):
            raise RuntimeError('incomplete episode sequence')
        return hashlib.sha256(''.join(self.records[i] for i in range(count)).encode()).hexdigest()
