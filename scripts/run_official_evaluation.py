#!/usr/bin/env python3
"""Evaluate a checkpoint with a pinned, unmodified public TS-FSAR checkout.

Preparation is CPU/file-only. Only --execute starts the official GPU evaluator.
This does not retroactively change a checkpoint's training/selection protocol.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'https://github.com/yaozhang0417/TS-FSAR.git'
COMMIT = '1a9105323b3e8cbf51a3ea59768a3d4618bd884f'
LOCK = ROOT / 'assets/OFFICIAL_EVALUATION_SOURCE_SHA256.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_source(source):
    lock = json.loads(LOCK.read_text())
    if lock['commit'] != COMMIT or lock['repository'] != UPSTREAM:
        raise RuntimeError('official source lock identity mismatch')
    for name, expected in lock['files'].items():
        path = source / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError('official source hash mismatch: ' + name)
    return lock


def evaluation_config(source, shot, dataset, checkpoint, output):
    # Start from the actual official Metatest YAML, not the campus DTO config.
    original = yaml.safe_load((source / f'configs/ssv2_small/Metatest_{shot}shot.yaml').read_text())
    cfg = copy.deepcopy(original)
    if (cfg['TRAIN']['ENABLE'] or not cfg['TEST']['ENABLE']
            or not cfg['TEST']['LOAD_PRETRAIN']):
        raise RuntimeError('unexpected official evaluation mode')
    cfg['DATA']['DATA_ROOT_DIR'] = str(dataset)
    cfg['TEST']['PRETRAIN_MODEL_PATH'] = str(checkpoint)
    cfg['OUTPUT_DIR'] = cfg['OUTPUT_DIR_SAVE'] = str(output)
    # All labels, prompts, RNG policy, workers, episode counts, transforms,
    # temperatures, shot settings and fusion coefficients remain official.
    return cfg


def prepare(args):
    dataset = args.dataset_root.resolve(strict=True)
    clip = args.clip_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    if not dataset.is_dir() or not clip.is_dir() or not checkpoint.is_file():
        raise ValueError('dataset/CLIP must be directories and checkpoint must be a file')
    for name in ('config.json', 'pytorch_model.bin', 'vocab.json', 'merges.txt'):
        if not (clip / name).is_file():
            raise FileNotFoundError('incomplete CLIP directory: ' + name)
    clip_hashes = json.loads((ROOT / 'assets/CLIP_VIT_B16_SHA256SUMS.json').read_text())['files']
    for name, expected in clip_hashes.items():
        if not (clip / name).is_file() or sha256(clip / name) != expected:
            raise RuntimeError('CLIP hash mismatch: ' + name)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('use a fresh --output; existing results are never overwritten')
    if shutil.disk_usage(output.parent).free < checkpoint.stat().st_size + 128 * 1024**2:
        raise RuntimeError('insufficient space for checkpoint snapshot and official source')
    output.mkdir()
    source = output / 'official_source'
    source.mkdir()
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    repository = str(args.official_repository.resolve()) if args.official_repository else UPSTREAM
    subprocess.run(['git', '-C', str(source), 'fetch', '--depth', '1', repository, COMMIT], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', '-q', '--detach', COMMIT], check=True)
    source_lock = verify_source(source)
    # The official model loader uses ./model/vit-b-16, relative to its own cwd.
    (source / 'model').mkdir(exist_ok=True)
    (source / 'model/vit-b-16').symlink_to(clip, target_is_directory=True)
    expected = sha256(checkpoint)
    snapshot = output / 'evaluation_checkpoint.pth'
    shutil.copyfile(checkpoint, snapshot)
    if sha256(snapshot) != expected or sha256(checkpoint) != expected:
        raise RuntimeError('checkpoint changed during snapshot; evaluation refused')
    cfg = evaluation_config(source, args.shot, dataset, snapshot, output)
    # Verify all required video paths using the exact official Test list.
    rows = (source / 'configs/ssv2_small/test_few_shot.txt').read_text().splitlines()
    for row in rows:
        if not row.strip():
            continue
        video = dataset / (row.strip().split('/')[-1] + '.mp4')
        if not video.is_file() or not os.access(video, os.R_OK) or not video.stat().st_size:
            raise FileNotFoundError('missing/empty/unreadable official Test video: ' + str(video))
    config_path = output / 'official_eval.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [sys.executable, '-u', str(source / 'main.py'), '--cfg', str(config_path)]
    previous = checkpoint.parent / 'protocol_v2.json'
    selection = json.loads(previous.read_text()).get('selection_split', 'unknown') if previous.is_file() else 'unknown'
    report = {'repository': UPSTREAM, 'commit': COMMIT, 'shot': args.shot,
              'source_lock_sha256': sha256(LOCK), 'verified_source_files': len(source_lock['files']),
              'original_checkpoint': str(checkpoint), 'checkpoint_sha256': expected,
              'checkpoint_snapshot': str(snapshot), 'checkpoint_training_selection': selection,
              'clip_directory': str(clip), 'verified_clip_sha256': clip_hashes,
              'evaluation_split': 'official 24-class Test',
              'episodes': cfg['TRAIN']['NUM_TEST_TASKS'],
              'rng_policy': 'unmodified official loader/worker RNG; no fixed episode bank',
              'config_sha256': sha256(config_path), 'cwd': str(source), 'command': command,
              'status': 'prepared_only', 'training_selection_changed': False,
              'limitations': ['Official evaluation does not establish official training/selection.',
                              'Upstream prints rounded accuracy; no paired-bootstrap evidence.',
                              'CUDA execution and accuracy require campus-side validation.']}
    (output / 'official_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    return source, command, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', type=int, choices=(1, 5), required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path,
                        default=Path(os.environ.get('TSFSAR_SSV2_ROOT', '/data/fsar_workspace/datasets/SSv2-Small')))
    parser.add_argument('--clip-root', type=Path, default=ROOT / 'model/vit-b-16')
    parser.add_argument('--official-repository', type=Path,
                        help='optional local official git checkout for offline preparation')
    parser.add_argument('--execute', action='store_true', help='start official evaluation on the specified campus GPU')
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU index must be nonnegative')
    source, command, report = prepare(args)
    output = args.output.resolve()
    print('OFFICIAL_EVALUATION_PREPARED', output, flush=True)
    if not args.execute:
        print('CPU preparation only; no torch import, model loading or GPU computation.', flush=True)
        return
    import fcntl
    locks = Path(os.environ.get('TSFSAR_GPU_LOCK_DIR', '/tmp/tsfsar-gpu-locks-' + str(os.getuid())))
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f'gpu-{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                                         '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if active.strip():
            raise RuntimeError('GPU is occupied; use a free allocated GPU and a fresh output')
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        # Avoid inheriting a local checkout through PYTHONPATH.
        env.pop('PYTHONPATH', None)
        env['PYTHONUNBUFFERED'] = '1'
        path = output / 'official_evaluation.json'
        report['status'] = 'running'
        path.write_text(json.dumps(report, indent=2) + '\n')
        print('OFFICIAL_EVALUATION_LOG', output / 'console.log', flush=True)
        with (output / 'console.log').open('x') as log:
            result = subprocess.run(command, cwd=str(source), env=env, stdout=log, stderr=subprocess.STDOUT)
        report.update(status='completed' if result.returncode == 0 else 'failed', returncode=result.returncode)
        path.write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            raise SystemExit(result.returncode)
        print('OFFICIAL_EVALUATION_COMPLETED', output, flush=True)


if __name__ == '__main__':
    main()
