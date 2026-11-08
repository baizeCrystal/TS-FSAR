"""Paired official Novel-Test replay: original E0 versus trained CLS-CTM."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F
from .common import (ROOT, canonical, digest, sha_file, write_new, seed_all,
    model_digest, source_audit, specification, implementation_hash, output_lock,
    identity_file, load_frozen, load_module)
from .matching import matching
from research.ps_ds_subspace_experiment.episodes import Replay, stream
from research.ps_ds_subspace_experiment.evaluate import verify_manifest_bundle, tensor_summary
from research.ps_ds_subspace_experiment.matching import synchronize


def predictions(values, target):
    pred = values.argmax(1)
    return dict(scores=values.tolist(), prediction=pred.tolist(), correct=(pred == target).tolist())


def run(args):
    rows, _, audit = verify_manifest_bundle(args.manifest, args.shot, 0)
    audited = {r['episode_id']: r['pixels_sha256'] for r in audit['records']}
    n = dict(smoke=3, explore=200, formal=10000)[args.stage]
    model, text, tokenizer, cfg, raw, lock = load_frozen(
        args.shot, args.data_root, args.output, args.checkpoint, args.workers)
    checkpoint_sha = sha_file(args.checkpoint)
    module, module_value, module_sha = load_module(args.module_checkpoint, args.shot,
        checkpoint_sha, allow_smoke=args.allow_smoke_module)
    if args.allow_smoke_module and args.stage != 'smoke':
        raise ValueError('a preflight module may only run engineering smoke episodes')
    identity = dict(schema=1, method='CLS-CTM-discriminative', shot=args.shot,
        stage=args.stage, episodes=n, seed=0, manifest_sha256=sha_file(args.manifest),
        checkpoint_sha256=checkpoint_sha, trained_module_sha256=module_sha,
        implementation_sha256=implementation_hash(), method_disabled=args.disable_clsctm,
        settings=asdict(module.settings), source_objects=source_audit(),
        preflight_module=module_value['identity']['smoke'])
    if args.stage != 'smoke':
        previous_stage = 'smoke' if args.stage == 'explore' else 'explore'
        prior = json.loads((args.output/previous_stage/'completion.json').read_text())
        if not prior['passed'] or any(prior[k] != identity[k] for k in
                ('manifest_sha256','checkpoint_sha256','trained_module_sha256',
                 'implementation_sha256','method_disabled','preflight_module')):
            raise RuntimeError('preceding engineering gate belongs to another run')
    folder = args.output/args.stage
    folder.mkdir(exist_ok=True)
    identity_file(folder/'run_identity.json', identity)
    identity_file(folder/'effective_config.json', raw)
    path = folder/'paired_predictions.jsonl'
    previous = []
    if path.exists():
        with path.open() as f:
            previous = [json.loads(line) for line in f if line.strip()]
    if len(previous) > n:
        raise RuntimeError('too many saved episodes')
    for i, row in enumerate(previous):
        if (row['episode_index'] != i or row['episode_id'] != rows[i]['episode_id']
                or not row['original_forward_parity']):
            raise RuntimeError('saved episode prefix differs from official manifest')
    if (folder/'completion.json').exists():
        c = json.loads((folder/'completion.json').read_text())
        if len(previous) != n or sha_file(path) != c['predictions_sha256'] or not c['passed']:
            raise RuntimeError('completed prediction data changed')
        print('CLSCTM_ALREADY_COMPLETE', folder, flush=True)
        return
    if shutil.disk_usage(folder).free < 1024**3:
        raise RuntimeError('need 1 GiB free for prediction records')
    from utils.functions import build_embeddings, extract_class_indices
    with torch.no_grad():
        embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text, tokenizer)
    if embeddings.shape[0] != 24:
        raise RuntimeError('expected 24 official Novel-Test classes')
    del text, tokenizer
    original, module_before = model_digest(model), model_digest(module)
    capture = {}
    hooks = [
        model.model.register_forward_hook(lambda m, a, out: capture.update(lsn=out[0])),
        model.ts_dcm.register_forward_hook(lambda m, a, out: capture.update(score=out[0])),
        model.model.Adapter.register_forward_hook(lambda m, a, out: capture.update(clip=out)),
        model.model.up_linear.register_forward_hook(lambda m, a, out: capture.update(side=out))]
    counts = {name: sum(sum(r[name]['correct']) for r in previous)
              for name in ['e0_matching','e1_matching','e0_fusion','e1_fusion']}
    loader = stream(Replay(cfg, rows[len(previous):n]), args.workers, 0)
    started = time.perf_counter()
    try:
        with path.open('a') as output, torch.no_grad():
            for local_index, item in enumerate(loader):
                index = len(previous)+local_index
                row = item['record']
                if row['episode_id'] in audited and item['pixels_sha256'] != audited[row['episode_id']]:
                    raise RuntimeError('campus frame preprocessing differs from audited pixels')
                capture.clear()
                images = item['images'].cuda()
                support = torch.tensor([v['episode_label'] for v in row['support']], device='cuda')
                target = torch.tensor([v['episode_label'] for v in row['query']], device='cuda')
                real = torch.tensor([v['class_id'] for v in row['support']], device='cuda')
                episode_embeddings = torch.stack([
                    torch.mean(torch.index_select(embeddings[real], 0,
                        extract_class_indices(support, c)), dim=0) for c in torch.unique(support)])
                synchronize(images.device)
                tick = time.perf_counter()
                reference = model(images, support, target, None, episode_embeddings)
                synchronize(images.device)
                e0_seconds = time.perf_counter()-tick
                score0 = capture['score']
                clip, side = capture['clip'].mean(1), capture['side'].mean(1)
                pclip = ((clip/clip.norm(dim=-1, keepdim=True))@episode_embeddings.T*cfg.TEMP_CLIP).softmax(1)
                pside = ((side/side.norm(dim=-1, keepdim=True))@episode_embeddings.T*cfg.TEMP_SIDE).softmax(1)
                def fusion(score):
                    return (score.softmax(1).pow(cfg.TEST.FUSE_VISION)
                            *pclip.pow(cfg.TEST.FUSE_CLIP)*pside.pow(cfg.TEST.FUSE_SIDE))
                fused0 = fusion(score0)
                torch.testing.assert_close(F.nll_loss(torch.log(fused0+1e-8), target),
                                           reference['loss'], atol=0, rtol=0)
                if float((fused0.argmax(1) == target).float().mean()) != reference['acc']:
                    raise AssertionError('original complete-model prediction mismatch')
                if args.stage == 'smoke':
                    disabled, _ = matching(model.ts_dcm, capture['lsn'], support, 5, args.shot, 1,
                                            enabled=False)
                    if not torch.equal(disabled, score0):
                        raise AssertionError('disabled path is not bitwise E0')
                    ones = torch.ones((256,), device=images.device)
                    neutral, _ = matching(model.ts_dcm, capture['lsn'], support, 5, args.shot, 1,
                                         gate_override=ones)
                    torch.testing.assert_close(neutral, score0, rtol=1e-5, atol=1e-5)
                synchronize(images.device)
                tick = time.perf_counter()
                score1, diagnostic = matching(model.ts_dcm, capture['lsn'], support, 5, args.shot, 1,
                    module, enabled=not args.disable_clsctm,
                    feature_frames_per_chunk=specification()['training']['feature_frames_per_chunk'])
                synchronize(images.device)
                diagnostic['total_additional_seconds'] = time.perf_counter()-tick
                fused1 = fusion(score1)
                if not all(bool(torch.isfinite(x).all()) for x in [score0,score1,pclip,pside,fused0,fused1]):
                    raise FloatingPointError('non-finite matching or fusion score')
                result = dict(episode_index=index, episode_id=row['episode_id'],
                    pixels_sha256=item['pixels_sha256'], class_order=row['class_order'],
                    query_video_ids=[v['video_id'] for v in row['query']],
                    support_video_ids=[v['video_id'] for v in row['support']], target=target.tolist(),
                    original_forward_parity=True, e0_seconds=e0_seconds, diagnostics=diagnostic,
                    clip_probability=pclip.tolist(), side_probability=pside.tolist(),
                    scale=dict(e0_logits=tensor_summary(score0), e1_logits=tensor_summary(score1),
                        e0_entropy=float(-(score0.softmax(1)*score0.log_softmax(1)).sum(1).mean()),
                        e1_entropy=float(-(score1.softmax(1)*score1.log_softmax(1)).sum(1).mean())))
                for name, values in [('e0_matching',score0), ('e1_matching',score1),
                                     ('e0_fusion',fused0), ('e1_fusion',fused1)]:
                    result[name] = predictions(values, target)
                    counts[name] += sum(result[name]['correct'])
                output.write(canonical(result)+'\n'); output.flush()
                if (index+1) % 10 == 0 or index+1 == n:
                    print('CLSCTM_PAIRED', args.stage, args.shot, index+1, '/', n,
                        {k: round(100*v/((index+1)*5),3) for k,v in counts.items()},
                        'additional_seconds', round(diagnostic['total_additional_seconds'],3), flush=True)
                capture.clear()
                del images, reference, score0, score1, fused0, fused1
        if model_digest(model) != original or model_digest(module) != module_before:
            raise RuntimeError('frozen E0 or CLS-CTM state changed during evaluation')
        if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in module.parameters()):
            raise RuntimeError('evaluation created unexpected gradients')
        if sha_file(args.checkpoint) != checkpoint_sha or sha_file(args.module_checkpoint) != module_sha:
            raise RuntimeError('checkpoint changed during evaluation')
        write_new(folder/'completion.json', dict(identity, passed=True,
            predictions_sha256=sha_file(path), model_unchanged=True, module_unchanged=True,
            no_encoder_gradients=True, original_forward_parity=True,
            accuracy_percent={k:100*v/(n*5) for k,v in counts.items()},
            seconds_this_invocation=time.perf_counter()-started, resumed_episodes=len(previous),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(0)))
        print('CLSCTM_EVALUATION_COMPLETE', folder, flush=True)
    finally:
        for hook in hooks:
            hook.remove()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('smoke','explore','formal'), required=True)
    p.add_argument('--shot', type=int, choices=(1,5), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--module-checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--disable-clsctm', action='store_true')
    p.add_argument('--allow-smoke-module', action='store_true', help='engineering preflight only')
    args = p.parse_args()
    os.chdir(ROOT)
    for name in ('data_root','checkpoint','module_checkpoint','manifest','output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.workers < 0:
        p.error('workers must be nonnegative')
    seed_all(0)
    with output_lock(args.output):
        run(args)


if __name__ == '__main__':
    main()
