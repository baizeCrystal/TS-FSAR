"""One GPU, frozen E0 encoder, paired E0/E1 evaluation; no training entry.

Run as: python -m research.ps_ds_subspace_experiment.evaluate --help
"""
import argparse
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from .protocol import (HERE, ROOT, canonical, digest, sha_file, write_new, read_records,
                       source_audit, load_config, check_manifest, seed_all, model_digest)
from .subspace import Settings


def verify_manifest_bundle(path, shot, seed, workers=None):
    rows = read_records(path)
    check_manifest(rows, shot)
    meta = json.loads((path.parent/'manifest_meta.json').read_text())
    if (meta['manifest_sha256'] != sha_file(path) or meta['seed'] != seed or meta['shot'] != shot
            or meta['lock_sha256'] != sha_file(HERE/'configs/locked.json')
            or (workers is not None and meta['workers'] != workers)):
        raise RuntimeError('manifest identity/settings mismatch')
    for filename,key in [('video_metadata.json','video_metadata_sha256'),
                         ('sampler_pixel_audit.json','sampler_pixel_audit_sha256')]:
        if sha_file(path.parent/filename) != meta[key]:
            raise RuntimeError('manifest dependency corrupt: '+filename)
    audit=json.loads((path.parent/'sampler_pixel_audit.json').read_text())
    if not audit['passed'] or [r['episode_id'] for r in rows[:3]] != [r['episode_id'] for r in audit['records']]:
        raise RuntimeError('manifest does not match audited sampler prefix')
    return rows,meta,audit


def prepare(args, cfg, raw, lock):
    from .episodes import Capture, Plan, Replay, stream
    path = args.output/'episodes.jsonl.gz'
    if path.exists() or (args.output/'manifest_meta.json').exists():
        verify_manifest_bundle(path,args.shot,args.seed,args.workers)
        print('VERIFIED_EXISTING_MANIFEST', path, flush=True)
        return
    if shutil.disk_usage(args.output).free < 512*1024**2:
        raise RuntimeError('need at least 512 MiB free for manifest capture; use data volume')
    seed_all(args.seed)
    dataset = Capture(cfg, 'test')
    # Do not let missing videos cause a retry / replacement episode.
    missing = [v for v in dataset.split_few_shot.videos
               if not (Path(args.data_root)/(v+'.mp4')).is_file()]
    if missing:
        raise RuntimeError('missing official Test videos: '+repr(missing[:20]))
    metadata_path = args.output/'video_metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        from decord import VideoReader
        metadata = {}
        for i,v in enumerate(dataset.split_few_shot.videos):
            reader = VideoReader(str(Path(args.data_root)/(v+'.mp4')), num_threads=1)
            metadata[v] = dict(source_num_frames=len(reader),source_fps=float(reader.get_avg_fps()))
            del reader
            if (i+1)%400 == 0:
                print('VIDEO_METADATA',i+1,'/ 2400',flush=True)
        write_new(metadata_path,metadata)
    planned = Plan(cfg, metadata)
    # Audit the fast plan against full original decoding/transform/RNG for 3
    # episodes using the SAME worker seeds and official sampler implementation.
    from itertools import islice
    seed_all(args.seed)
    audited = list(islice(stream(dataset,args.workers,args.seed),3))
    seed_all(args.seed)
    probes = list(islice(stream(planned,args.workers,args.seed),3))
    pixel_audit = []
    for original,plan in zip(audited,probes):
        original.pop('episode_id')
        pixels = original.pop('pixels_sha256')
        for role in ['support','query']:
            for video in original[role]:
                video.pop('pixel_sha256')
        original['episode_id'] = digest(original)
        if original != plan:
            raise AssertionError('fast metadata plan does not match original sampler exactly')
        pixel_audit.append(dict(episode_id=plan['episode_id'],pixels_sha256=pixels))
    replay = list(islice(stream(Replay(cfg,probes),args.workers,args.seed),3))
    if [x['pixels_sha256'] for x in replay] != [x['pixels_sha256'] for x in pixel_audit]:
        raise AssertionError('original sampled pixels differ from replay pixels')
    # The audited pixels are external evidence, not part of immutable episode IDs.
    audit_path=args.output/'sampler_pixel_audit.json'
    if not audit_path.exists():
        write_new(audit_path,dict(passed=True,episodes=3,records=pixel_audit,
                 same_original_sampler=True,same_frames=True,same_preprocessed_pixels=True))
    seed_all(args.seed)
    partial = args.output/'episodes.partial.jsonl.gz'
    if partial.exists():
        raise RuntimeError('unfinished capture exists; choose fresh --output, preserve failure evidence')
    started = time.perf_counter()
    with partial.open('xb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', mtime=0) as f:
            for i, row in enumerate(stream(planned, args.workers, args.seed)):
                f.write((canonical(row)+'\n').encode())
                if (i+1) % 100 == 0:
                    print('CAPTURE', args.shot, i+1, '/ 10000', flush=True)
    rows = read_records(partial)
    check_manifest(rows, args.shot)
    if [r['episode_id'] for r in rows[:3]] != [r['episode_id'] for r in pixel_audit]:
        raise AssertionError('completed manifest differs from audited sampler prefix')
    os.rename(partial, path)
    write_new(args.output/'manifest_meta.json', dict(
        schema=1, shot=args.shot, seed=args.seed, workers=args.workers,
        episodes=len(rows), manifest_sha256=sha_file(path),
        lock_sha256=sha_file(HERE/'configs/locked.json'), source_objects=source_audit(),
        capture_seconds=time.perf_counter()-started, configuration=raw,
        sampling='actual official Video_few_shot test sampler, dedicated seed-0 DataLoader generator',
        capture_mode='metadata plan; original sampler/frame RNG, audited full decode/pixel replay on 3 episodes',
        video_metadata_sha256=sha_file(metadata_path),sampler_pixel_audit_sha256=sha_file(audit_path),
        historical_selected_episode_stream_replayed=False))
    print('FIXED_MANIFEST_READY', path, flush=True)


def checkpoint_path(args, lock):
    spec = lock['checkpoints'][str(args.shot)]
    path = Path(args.checkpoint or ROOT/spec['relative_path']).resolve()
    if sha_file(path) != spec['sha256']:
        raise RuntimeError('wrong checkpoint for shot; must use historical formal E0: '+str(path))
    return path, spec


def verify_clip(lock):
    for name, expected in lock['clip_files'].items():
        if sha_file(ROOT/'model/vit-b-16'/name) != expected:
            raise RuntimeError('CLIP file missing/corrupt: '+name)


def tensor_summary(x):
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
                std=float(x.std(unbiased=False)))


def run(args, cfg, raw, lock):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from trainers.model import build_model
    from trainers.ts_fsar import TS_FSAR
    from utils.functions import build_embeddings, extract_class_indices
    from .episodes import Replay, stream
    from .matching import matching, synchronize, validate_original
    if not torch.cuda.is_available():
        raise RuntimeError('GPU evaluation only; do not run real model inference on original CPU server')
    if args.manifest is None:
        raise ValueError('--manifest is required')
    manifest = args.manifest.resolve()
    rows,meta,audit = verify_manifest_bundle(manifest,args.shot,args.seed)
    audited_pixels = {r['episode_id']:r['pixels_sha256'] for r in audit['records']}
    cp, cp_spec = checkpoint_path(args, lock)
    cfg.TEST.PRETRAIN_MODEL_PATH = str(cp)
    raw['TEST']['PRETRAIN_MODEL_PATH'] = str(cp)
    verify_clip(lock)
    n = {'smoke': 3, 'explore': 200, 'formal': 10000}[args.stage]
    settings = Settings(**lock['subspace'])
    # Formal work is gated on engineering checks, never on achieved Test accuracy.
    if args.stage != 'smoke':
        prior = 'smoke' if args.stage == 'explore' else 'explore'
        gate = args.output/prior/'completion.json'
        if not gate.is_file():
            raise RuntimeError('run '+prior+' first in the same output directory')
        completion = json.loads(gate.read_text())
        if (not completion.get('passed') or completion['manifest_sha256'] != sha_file(manifest)
                or completion['checkpoint_sha256'] != cp_spec['sha256']
                or completion['method_disabled'] != args.disable_psds
                or completion['implementation_sha256'] != implementation_hash()):
            raise RuntimeError('preceding engineering gate does not match this run')
    folder = args.output/args.stage
    folder.mkdir(exist_ok=True)
    if not (folder/'effective_config.json').exists():
        write_new(folder/'effective_config.json',raw)
    identity = dict(schema=1, shot=args.shot, stage=args.stage, episodes=n, seed=args.seed,
                    manifest_sha256=sha_file(manifest), checkpoint_sha256=cp_spec['sha256'],
                    lock_sha256=sha_file(HERE/'configs/locked.json'),
                    implementation_sha256=implementation_hash(), method_disabled=args.disable_psds,
                    settings=asdict(settings), source_objects=source_audit())
    header = folder/'run_identity.json'
    if header.exists():
        if json.loads(header.read_text()) != identity:
            raise RuntimeError('output belongs to a different experiment; never overwrite')
    else:
        write_new(header, identity)
    result_path = folder/'paired_predictions.jsonl'
    previous = read_records(result_path) if result_path.exists() else []
    for i, result in enumerate(previous):
        if i >= n or result['episode_id'] != rows[i]['episode_id']:
            raise RuntimeError('resume prefix differs from manifest')
    if len(previous) == n and (folder/'completion.json').exists():
        if sha_file(result_path) != json.loads((folder/'completion.json').read_text())['predictions_sha256']:
            raise RuntimeError('completed predictions changed')
        print('ALREADY_COMPLETE', folder, flush=True)
        return
    if shutil.disk_usage(folder).free < 1024**3:
        raise RuntimeError('need at least 1 GiB free for predictions/audits')
    seed_all(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vision, text, tokenizer = build_model(cfg)
    model = TS_FSAR(cfg, vision).cuda()
    state = torch.load(cp, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state, vision
    model.eval().requires_grad_(False)
    validate_original(model.ts_dcm)
    with torch.no_grad():
        embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text, tokenizer)
    if embeddings.shape[0] != 24:
        raise RuntimeError('text encoder omitted a Novel Test class')
    del text, tokenizer
    before = model_digest(model)
    capture = {}
    hooks = [
        model.model.register_forward_hook(lambda m, a, out: capture.update(lsn=out[0])),
        model.ts_dcm.register_forward_hook(lambda m, a, out: capture.update(score=out[0])),
        model.model.Adapter.register_forward_hook(lambda m, a, out: capture.update(clip=out)),
        model.model.up_linear.register_forward_hook(lambda m, a, out: capture.update(side=out)),
    ]
    loader = stream(Replay(cfg, rows[len(previous):n]), args.workers, args.seed)
    counts = {name: sum(sum(r[name]['correct']) for r in previous)
              for name in ['e0_matching','e1_matching','e0_fusion','e1_fusion']}
    started = time.perf_counter()
    try:
        with result_path.open('a') as output, torch.no_grad():
            for local_index, item in enumerate(loader):
                index = len(previous)+local_index
                row = item['record']
                if row['episode_id'] in audited_pixels and item['pixels_sha256'] != audited_pixels[row['episode_id']]:
                    raise AssertionError('campus pixel preprocessing differs from audited original pixels')
                capture.clear()
                images = item['images'].cuda()
                support = torch.tensor([v['episode_label'] for v in row['support']], device='cuda')
                target = torch.tensor([v['episode_label'] for v in row['query']], device='cuda')
                real_support = torch.tensor([v['class_id'] for v in row['support']], device='cuda')
                # Exactly main.test: aggregate text via support label mapping/mean.
                episode_embeddings = torch.stack([
                    torch.mean(torch.index_select(embeddings[real_support], 0,
                        extract_class_indices(support, c)), dim=0) for c in torch.unique(support)])
                synchronize(images.device)
                t0 = time.perf_counter()
                reference = model(images, support, target, None, episode_embeddings)
                synchronize(images.device)
                e0_seconds = time.perf_counter()-t0
                score0 = capture['score']
                clip, side = capture['clip'].mean(1), capture['side'].mean(1)
                pclip = ((clip/clip.norm(dim=-1,keepdim=True))@episode_embeddings.T*cfg.TEMP_CLIP).softmax(1)
                pside = ((side/side.norm(dim=-1,keepdim=True))@episode_embeddings.T*cfg.TEMP_SIDE).softmax(1)
                def fusion(score):
                    return (score.softmax(1).pow(cfg.TEST.FUSE_VISION)
                            *pclip.pow(cfg.TEST.FUSE_CLIP)*pside.pow(cfg.TEST.FUSE_SIDE))
                fused0 = fusion(score0)
                torch.testing.assert_close(F.nll_loss(torch.log(fused0+1e-8), target),
                                           reference['loss'], atol=0, rtol=0)
                if float((fused0.argmax(1)==target).float().mean()) != reference['acc']:
                    raise AssertionError('official full-model parity failed')
                # For EVERY smoke episode, disabled path must be bitwise identical.
                if args.stage == 'smoke':
                    noop, _ = matching(model.ts_dcm, capture['lsn'], support, 5, args.shot, 1, enabled=False)
                    if not torch.equal(noop, score0):
                        raise AssertionError('disabled PSDS does not reproduce E0 exactly')
                score1, diagnostic = matching(model.ts_dcm, capture['lsn'], support, 5, args.shot, 1,
                                              enabled=not args.disable_psds, settings=settings)
                fused1 = fusion(score1)
                if not all(torch.isfinite(x).all() for x in [score0,score1,pclip,pside,fused0,fused1]):
                    raise FloatingPointError('non-finite score/probability')
                result = dict(episode_index=index, episode_id=row['episode_id'],
                              pixels_sha256=item['pixels_sha256'],
                              query_video_ids=[v['video_id'] for v in row['query']],
                              support_video_ids=[v['video_id'] for v in row['support']],
                              class_order=row['class_order'], target=target.tolist(),
                              e0_seconds=e0_seconds, diagnostics=diagnostic,
                              scale=dict(e0_logits=tensor_summary(score0),e1_logits=tensor_summary(score1),
                                         e0_entropy=float(-(score0.softmax(1)*score0.log_softmax(1)).sum(1).mean()),
                                         e1_entropy=float(-(score1.softmax(1)*score1.log_softmax(1)).sum(1).mean())),
                              clip_probability=pclip.tolist(), side_probability=pside.tolist(),
                              original_forward_parity=True)
                for name, values in [('e0_matching',score0),('e1_matching',score1),
                                     ('e0_fusion',fused0),('e1_fusion',fused1)]:
                    pred = values.argmax(1)
                    correct = (pred == target).tolist()
                    result[name] = dict(scores=values.tolist(), prediction=pred.tolist(), correct=correct)
                    counts[name] += sum(correct)
                output.write(canonical(result)+'\n')
                output.flush()
                if (index+1)%10 == 0 or index+1 == n:
                    print('PAIRED', args.stage, args.shot, index+1, '/', n,
                          {k:round(100*v/((index+1)*5),3) for k,v in counts.items()},
                          'PSDS_seconds', round(diagnostic.get('total_additional_seconds',0),3), flush=True)
                capture.clear()
                del images, reference, score0, score1, fused0, fused1
        if before != model_digest(model) or any(p.grad is not None for p in model.parameters()):
            raise AssertionError('frozen model parameters/buffers/gradients changed')
        if sha_file(cp) != cp_spec['sha256']:
            raise AssertionError('checkpoint changed during evaluation')
        complete = dict(identity, passed=True, model_unchanged=True, no_encoder_gradients=True,
                        checkpoint_unchanged=True, official_forward_parity=True,
                        accuracy_percent={k:100*v/(n*5) for k,v in counts.items()},
                        predictions_sha256=sha_file(result_path), resumed_episodes=len(previous),
                        seconds_this_invocation=time.perf_counter()-started,
                        torch_version=torch.__version__, python_version=sys.version,
                        gpu=torch.cuda.get_device_name(0), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_new(folder/'completion.json', complete)
        print('PAIRED_EVALUATION_COMPLETE', folder, flush=True)
    finally:
        for h in hooks:
            h.remove()


def implementation_hash():
    return digest({p.name:sha_file(p) for p in HERE.glob('*.py')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare','smoke','explore','formal'])
    parser.add_argument('--shot', required=True, type=int, choices=[1,5])
    parser.add_argument('--data-root', required=True, help='directory directly containing <video_id>.mp4')
    parser.add_argument('--checkpoint', help='must match the locked per-shot SHA256')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='one independent root per shot')
    parser.add_argument('--seed', type=int, default=0, choices=[0])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--disable-psds', action='store_true', help='one switch back to original E0')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(int(os.environ.get('TSFSAR_CPU_THREADS','2')))
    cfg, raw, lock = load_config(args.shot,args.data_root,args.output,args.workers)
    import fcntl
    with (args.output/'.run.lock').open('a') as lease:
        try:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another process owns this output directory; no duplicate run started')
        if args.stage == 'prepare':
            prepare(args,cfg,raw,lock)
        else:
            run(args,cfg,raw,lock)


if __name__ == '__main__':
    main()
