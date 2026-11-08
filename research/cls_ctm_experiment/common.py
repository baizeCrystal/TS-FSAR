"""Locked assets, isolated outputs and shared reproducibility checks."""
import contextlib
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from research.ps_ds_subspace_experiment.protocol import (
    ROOT, canonical, digest, sha_file, write_new, seed_all, model_digest,
    load_config, source_audit)
from .module import Settings

HERE = Path(__file__).resolve().parent


def specification():
    return json.loads((HERE/'configs/ssv2_small.json').read_text())


def implementation_hash():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('configs/*.json'))
    # Pin the shared official sampler/configuration/statistical utilities too.
    shared = ROOT/'research/ps_ds_subspace_experiment'
    paths += [shared/n for n in ['protocol.py', 'episodes.py', 'evaluate.py',
                                'analyze.py', 'matching.py', 'subspace.py', 'configs/locked.json']]
    return digest({str(p.relative_to(ROOT)): sha_file(p) for p in paths})


@contextlib.contextmanager
def output_lock(root):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.clsctm.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another process owns '+str(root))
        yield


def identity_file(path, identity):
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise RuntimeError('output identity changed; use a fresh directory: '+str(path))
    else:
        write_new(path, identity)


def atomic_save(value, path):
    import torch
    temp = path.with_name(path.name+'.partial')
    if temp.exists():
        raise RuntimeError('unfinished checkpoint preserved: '+str(temp))
    with temp.open('xb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)  # only the experiment's explicitly mutable latest file


def load_frozen(shot, data_root, output, checkpoint, workers):
    import torch
    from trainers.model import build_model
    from trainers.ts_fsar import TS_FSAR
    from research.ps_ds_subspace_experiment.evaluate import verify_clip
    from research.ps_ds_subspace_experiment.matching import validate_original
    if not torch.cuda.is_available():
        raise RuntimeError('real video model execution is allowed only on the campus GPU')
    torch.set_num_threads(int(os.environ.get('TSFSAR_CPU_THREADS', '2')))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg, raw, lock = load_config(shot, data_root, output, workers)
    cp = Path(checkpoint).resolve()
    if sha_file(cp) != lock['checkpoints'][str(shot)]['sha256']:
        raise RuntimeError('wrong E0 checkpoint for this shot: '+str(cp))
    verify_clip(lock)
    vision, text, tokenizer = build_model(cfg)
    # TEST.ENABLE remains True to skip PCA; strict E0 state replaces every value.
    model = TS_FSAR(cfg, vision)
    state = torch.load(cp, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state, vision
    model.eval().requires_grad_(False).cuda()
    validate_original(model.ts_dcm)
    return model, text, tokenizer, cfg, raw, lock


def fresh_module(device, seed):
    import torch
    from .module import CLSCTM
    # The encoder's random module construction must not affect head initialization.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        module = CLSCTM(Settings(**specification()['module']))
    return module.to(device)


def load_module(path, shot, checkpoint_sha, allow_smoke=False):
    import torch
    path = Path(path).resolve()
    value = torch.load(path, map_location='cpu', weights_only=True)
    identity = value['identity']
    if (identity['shot'] != shot or identity['checkpoint_sha256'] != checkpoint_sha
            or identity['implementation_sha256'] != implementation_hash()
            or identity['settings'] != asdict(Settings(**specification()['module']))
            or (identity['smoke'] and not allow_smoke)):
        raise RuntimeError('module belongs to a different source/checkpoint/shot/protocol')
    if (not value.get('training_complete') or
            value['completed_steps'] != identity['training']['steps']):
        raise RuntimeError('module training is incomplete')
    module = fresh_module('cuda', identity['seed'])
    module.load_state_dict(value['module'], strict=True)
    module.eval()
    return module, value, sha_file(path)


def encode(model, images, frames, videos_per_chunk):
    import torch
    if len(images) % frames or videos_per_chunk < 1:
        raise ValueError('encoder chunks must contain complete videos')
    with torch.no_grad():
        features = [model.model(part.cuda())[0].detach()
                    for part in images.split(frames*videos_per_chunk)]
    return torch.cat(features)
