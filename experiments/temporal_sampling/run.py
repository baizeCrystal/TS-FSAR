"""Explicit CPU preflight/sampling smoke and opt-in GPU evaluation CLI."""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .core import (BRANCHES, SCHEMES, SCOPES, compare_features, digest, file_hash,
                   read_rows, sample_indices, validate_episode, validate_output, write_json)

HERE = Path(__file__).resolve().parent
EXPECTED_CHECKPOINT = '78aae7edcbea1f5a48f17aa28d9631ccf680f69c08535b40083cfaa4a46cf5cf'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight', 'sample', 'evaluate'))
    p.add_argument('--protocol', type=Path, required=True, help='bundled data/1shot or data/5shot')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--video-root', type=Path)
    p.add_argument('--model-dir', type=Path, default=Path('model/vit-b-16'))
    p.add_argument('--prompts', type=Path, default=Path('prompts/Ssv2small_test.json'))
    p.add_argument('--output', type=Path)
    p.add_argument('--cache', type=Path)
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--schemes', nargs='+', choices=SCHEMES, default=['e0', 'p075', 'p133'])
    p.add_argument('--scopes', nargs='+', choices=SCOPES, default=['query', 'support'])
    p.add_argument('--batch-videos', type=int, default=2)
    p.add_argument('--cache-disk-mb', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--max-minutes', type=float, default=55)
    p.add_argument('--decode', action='store_true', help='decode one video in sampling smoke test')
    return p


def load_protocol(path, limit):
    frozen = json.loads((HERE / 'frozen_source.json').read_text())
    if file_hash(HERE / frozen['snapshot']) != frozen['snapshot_sha256']:
        raise ValueError('Frozen original E0 implementation hash mismatch')
    provenance = json.loads((path / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        if file_hash(path / name) != expected:
            raise ValueError('Bundled protocol changed: ' + name)
    config = json.loads((path / 'config.json').read_text())
    episodes = read_rows(path / 'episodes.jsonl')
    if limit < 1 or limit > len(episodes):
        raise ValueError(f'limit must be within [1,{len(episodes)}]; export a larger protocol for phase 2')
    shot = int(config['TRAIN']['SHOT'])
    if shot not in (1, 5) or config['DATA']['NUM_INPUT_FRAMES'] != 8:
        raise ValueError('Requires 1/5-shot, 8-frame E0 protocol')
    if config['DATA']['SAMPLING_RATE'] != 50 or config['DATA']['SAMPLING_MODE'] != 'interval_based':
        raise ValueError('This protocol is defined for the audited official rate=50 sampler')
    for ep in episodes[:limit]:
        validate_episode(ep, shot)
    if len({ep['episode'] for ep in episodes[:limit]}) != limit:
        raise ValueError('Episode IDs must be unique')
    return config, episodes[:limit], json.loads((path / 'reference.json').read_text())


def check_paths(args, config, episodes):
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise ValueError('--checkpoint must point to E0 best checkpoint')
    if file_hash(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Checkpoint is not the locked E0 checkpoint (both shots use the same one)')
    if args.video_root is None or not args.video_root.is_dir():
        raise ValueError('--video-root directory required')
    for p in (args.model_dir / 'config.json', args.prompts):
        if not p.is_file():
            raise ValueError('Missing asset: ' + str(p))
    if not any((args.model_dir / name).is_file() for name in ('pytorch_model.bin', 'model.safetensors')):
        raise ValueError('Missing local CLIP model weights')
    prompt = json.loads(args.prompts.read_text())
    if any(n.replace('_', ' ') not in prompt for n in config['TEST']['CLASS_NAME']):
        raise ValueError('Prompt classes incomplete')
    videos = {str(v['video_id']) for ep in episodes for v in ep['support'] + ep['query']}
    missing = [v for v in videos if not (args.video_root / (v + '.mp4')).is_file()]
    if missing:
        raise ValueError(f'{len(missing)} missing videos (first: {missing[:5]})')
    return len(videos)


def verify_historical(index, output, pixels, reference):
    if pixels != reference['pixel_hashes'][index]:
        raise RuntimeError('E0 preprocessed pixels do not match historical hash')
    np.testing.assert_array_equal(output['labels'], reference['targets'][index])
    errors = {}
    for key, ref in [('ts_dcm', 'matching_score'), ('adapter_probability', 'clip_probability'),
                     ('lsn_probability', 'side_probability')]:
        old = np.asarray(reference[ref][index])
        # History used full-episode FP32; tiny library/GEMM drift is permitted,
        # but any prediction flip is a failed reproduction, not a new baseline.
        atol = 5e-4 if key == 'ts_dcm' else 1e-4
        np.testing.assert_allclose(output[key], old, atol=atol, rtol=5e-5)
        np.testing.assert_array_equal(output[key].argmax(-1), old.argmax(-1))
        errors[key] = float(np.max(np.abs(output[key] - old)))
    historical_fusion = np.exp(np.asarray(reference['full_log_score'][index]))
    np.testing.assert_allclose(output['fusion'], historical_fusion, atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(output['fusion'].argmax(-1), historical_fusion.argmax(-1))
    return errors


def sampling_for(ep, scheme, scope, seed):
    return [sample_indices(v, scheme if scope == 'both' or scope == role else 'e0', seed)
            for role in ('support', 'query') for v in ep[role]]


def evaluate(args, config, episodes, reference):
    if args.output is None or args.cache is None:
        raise ValueError('evaluate needs --output (new directory) and --cache')
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing experiment directory')
    # Keep new cache material isolated from existing diagnostics.
    if args.cache == args.output or args.output in args.cache.parents:
        raise ValueError('--cache must be separate from the new run directory')
    args.output.mkdir(parents=True)
    config['TEST']['LLM_test_json'] = str(args.prompts.resolve())
    config['DATA']['DATA_ROOT_DIR'] = str(args.video_root.resolve())
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError('Experiment time budget reached')
    def alarm_handler(signum, frame):
        raise TimeoutError('Experiment hard wall-clock limit reached')
    old_handler = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, args.max_minutes * 60)
    status = {'status': 'initializing', 'e0_completed': 0, 'conditions_completed': 0,
              'requested_episodes': len(episodes), 'training': False, 'gpu_verified': False}
    try:
        from .runtime import Encoder
        encoder = Encoder(config, args.checkpoint, args.model_dir, args.video_root,
                          args.cache, args.cache_disk_mb)
        status['device'] = encoder.device
        manifest = {'config': config, 'checkpoint': str(args.checkpoint.resolve()),
                    'checkpoint_sha256': file_hash(args.checkpoint), 'episode_digest': digest(episodes),
                    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'git_tracked_diff_sha256': digest(subprocess.check_output(['git', 'diff', '--', 'trainers', 'datasets', 'utils'], text=True)),
                    'extraction_identity': encoder.identity,
                    'schemes': args.schemes, 'scopes': args.scopes, 'seed': args.seed,
                    'max_minutes': args.max_minutes, 'batch_videos': args.batch_videos,
                    'same_checkpoint_for_both_shots': True,
                    'E0': 'official equal-bin random locked indices; not exact equidistant',
                    'hypotheses': 'pending_real_forward', 'phase': '200' if len(episodes) <= 200 else 'optional_expanded'}
        write_json(args.output / 'manifest.json', manifest)
        (args.output / 'episodes.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in episodes))
        # Complete ALL E0 checks before producing any intervention result.
        e0_accuracies = {branch: [] for branch in BRANCHES}
        for i, ep in enumerate(episodes):
            check_deadline()
            chunks, pixels = encoder.baseline(ep)
            output = encoder.score(ep, chunks)
            validate_output(output)
            errors = verify_historical(i, output, pixels, reference)
            for branch in BRANCHES:
                e0_accuracies[branch].append(float((output[branch].argmax(1) == output['labels']).mean()))
            output.update(clip_frames=np.stack([v['clip'] for v in chunks]),
                          lsn_frames=np.stack([v['lsn'] for v in chunks]))
            np.savez(args.output / f"e0_{ep['episode']:05d}.npz", **output)
            write_json(args.output / f"e0_{ep['episode']:05d}.json", {'pixel_hash': pixels, 'historical_max_error': errors})
            status.update(e0_completed=i+1, status='verifying_e0')
            write_json(args.output / 'status.json', {**status, 'seconds': time.monotonic() - started})
            if i % 10 == 0:
                print(f'E0 verified {i+1}/{len(episodes)}', flush=True)
        status['gpu_verified'] = encoder.device == 'cuda'
        write_json(args.output / 'e0_verification.json', {'status': 'passed', 'episodes': len(episodes),
                    'native_forward': encoder.native_check, 'all_pixels_and_predictions_match': True})
        write_json(args.output / 'e0_summary.json', {'episodes': len(episodes), 'queries': 5*len(episodes),
            'accuracy_percent': {branch: float(np.mean(values)*100) for branch, values in e0_accuracies.items()}})
        schemes = [s for s in dict.fromkeys(args.schemes) if s != 'e0']
        for scheme in schemes:
            for scope in dict.fromkeys(args.scopes):
                for i, ep in enumerate(episodes):
                    check_deadline()
                    videos = ep['support'] + ep['query']
                    base_sampling = sampling_for(ep, 'e0', 'both', args.seed)
                    base_chunks = encoder.changed(videos, base_sampling, args.batch_videos, check_deadline)
                    with np.load(args.output / f"e0_{ep['episode']:05d}.npz", allow_pickle=False) as z:
                        original = {k: z[k] for k in z.files}
                    rebuilt = encoder.score(ep, base_chunks)
                    # Detect library/batch-size drift or a bad cache before it
                    # can masquerade as an intervention effect.
                    for branch in BRANCHES:
                        np.testing.assert_allclose(rebuilt[branch], original[branch], atol=5e-4, rtol=5e-5)
                        np.testing.assert_array_equal(rebuilt[branch].argmax(1), original[branch].argmax(1))
                    samplings = sampling_for(ep, scheme, scope, args.seed)
                    chunks = encoder.changed(videos, samplings, args.batch_videos, check_deadline)
                    output = encoder.score(ep, chunks)
                    validate_output(output)
                    feature_changes = []
                    for j, (v, s, b, change) in enumerate(zip(videos, samplings, base_sampling, chunks)):
                        if s['scheme'] == 'e0':
                            continue
                        feature_changes.append({'role': 'support' if j < len(ep['support']) else 'query',
                            'video_id': v['video_id'], 'sampling': s,
                            'clip': compare_features(original['clip_frames'][j], change['clip'], b['actual_u'], s['actual_u']),
                            'lsn': compare_features(original['lsn_frames'][j], change['lsn'], b['actual_u'], s['actual_u'])})
                    output['clip_frames'] = np.stack([v['clip'] for v in chunks])
                    output['lsn_frames'] = np.stack([v['lsn'] for v in chunks])
                    name = f"{scheme}_{scope}_{ep['episode']:05d}"
                    np.savez(args.output / (name + '.npz'), **output)
                    write_json(args.output / (name + '.json'), {'episode': ep['episode'], 'scheme': scheme, 'scope': scope,
                        'feature_changes': feature_changes, 'sampling': samplings,
                        'episode_identity_unchanged': True})
                    status.update(status='evaluating', conditions_completed=status['conditions_completed']+1)
                    write_json(args.output / 'status.json', {**status, 'seconds': time.monotonic() - started})
                    if i % 10 == 0:
                        print(f'{scheme}/{scope} {i+1}/{len(episodes)}', flush=True)
        status['status'] = 'complete'
    except TimeoutError as exc:
        status.update(status='partial_timeout', error=str(exc))
    except Exception as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        status['seconds'] = time.monotonic() - started
        write_json(args.output / 'status.json', status)
    print(json.dumps(status, indent=2))
    if status['status'] != 'complete':
        raise SystemExit(124)


def main():
    p = parser()
    args = p.parse_args()
    if not 0 < args.max_minutes <= 60 or args.batch_videos < 1 or args.cache_disk_mb < 0:
        p.error('0 < max-minutes <= 60, batch-videos >= 1, cache-disk-mb >= 0 required')
    config, episodes, reference = load_protocol(args.protocol, args.limit)
    if args.mode == 'sample':
        v = episodes[0]['query'][0]
        info = {'video': v, 'sampling': {s: sample_indices(v, s, args.seed) for s in SCHEMES}}
        if args.decode:
            if args.video_root is None:
                p.error('--decode requires --video-root')
            from .runtime import decode
            info['decoded_shapes'] = {s: list(decode(args.video_root, v, x['indices']).shape)
                                      for s, x in info['sampling'].items()}
        print(json.dumps(info, indent=2))
        return
    n = check_paths(args, config, episodes)
    if args.mode == 'preflight':
        print(json.dumps({'status': 'paths_and_identity_passed', 'episodes': len(episodes),
                          'videos': n, 'shot': config['TRAIN']['SHOT'], 'model_forward': 'not_run'}, indent=2))
        return
    evaluate(args, config, episodes, reference)


if __name__ == '__main__':
    main()
