"""Frozen-E0 Base-Train adaptation. No Meta-Test labels or checkpoint selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from .common import (ROOT, canonical, digest, sha_file, write_new, seed_all,
    model_digest, source_audit, specification, implementation_hash, output_lock,
    identity_file, atomic_save, load_frozen, fresh_module, encode)
from .data import training_config, train_loader
from .matching import matching


def rng_state():
    import random
    s = np.random.get_state()
    return dict(python=random.getstate(), numpy=[s[0], s[1].tolist(), s[2], s[3], s[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(value):
    import random
    random.setstate(value['python'])
    s = value['numpy']
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    torch.set_rng_state(value['torch'])
    torch.cuda.set_rng_state_all(value['cuda'])


def cpu_state(module):
    return {name: t.detach().cpu().clone() for name, t in module.state_dict().items()}


def preserve_prefix(path, completed):
    """Keep post-checkpoint failure evidence before rebuilding the exact prefix."""
    if not path.exists():
        if completed:
            raise RuntimeError('missing resume ledger: '+str(path))
        return
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if len(rows) < completed:
                rows.append(row)
    if len(rows) != completed or [r['step'] for r in rows] != list(range(1, completed+1)):
        raise RuntimeError('resume ledger prefix is incomplete or unordered')
    backup = path.with_name(path.name+f'.before_resume_{time.time_ns()}')
    path.rename(backup)
    with path.open('x') as stream:
        for row in rows:
            stream.write(canonical(row)+'\n')


def run(args):
    out = args.output
    spec = specification()
    training = dict(spec['training'])
    if args.smoke:
        training['steps'] = 2
    if shutil.disk_usage(out).free < 2*1024**3:
        raise RuntimeError('need at least 2 GiB free for logs/checkpoints')
    seed_all(args.seed)
    model, text, tokenizer, cfg, raw, lock = load_frozen(
        args.shot, args.data_root, out, args.checkpoint, args.workers)
    del text, tokenizer
    torch.cuda.empty_cache()
    original = model_digest(model)
    module = fresh_module('cuda', args.seed)
    optimizer = torch.optim.AdamW(module.parameters(), lr=training['learning_rate'],
                                 weight_decay=training['weight_decay'])
    identity = dict(schema=1, method='CLS-CTM-discriminative', shot=args.shot, seed=args.seed,
        smoke=args.smoke, settings=asdict(module.settings), training=training,
        checkpoint_sha256=sha_file(args.checkpoint), implementation_sha256=implementation_hash(),
        source_objects=source_audit(), frozen_model_sha256=original,
        data_root=str(args.data_root), workers=args.workers, torch_version=str(torch.__version__),
        gpu=torch.cuda.get_device_name(0), precision='FP32; no AMP or encoder gradients')
    identity_file(out/'train_identity.json', identity)
    train_cfg, train_raw = training_config(args.shot, args.data_root, out, args.workers)
    identity_file(out/'training_data_config.json', train_raw)
    if (out/'training_complete.json').exists():
        complete = json.loads((out/'training_complete.json').read_text())
        if sha_file(out/'module_final.pth') != complete['module_sha256']:
            raise RuntimeError('completed module checksum changed')
        print('CLSCTM_TRAIN_ALREADY_COMPLETE', out, flush=True)
        return
    start, previous_elapsed = 0, 0.
    sequence = hashlib.sha256()
    seed_all(args.seed)
    if args.resume:
        saved = torch.load(out/'resume_latest.pth', map_location='cpu', weights_only=True)
        if saved['identity'] != identity:
            raise RuntimeError('resume identity does not match this run')
        module.load_state_dict(saved['module'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start, previous_elapsed = saved['completed_steps'], saved['elapsed_seconds']
        for name in ('train_trace.jsonl', 'train_episodes.jsonl'):
            preserve_prefix(out/name, start)
        with (out/'train_episodes.jsonl').open() as stream:
            for line in stream:
                sequence.update(json.loads(line)['episode_id'].encode())
        if sequence.hexdigest() != saved['train_sequence_sha256']:
            raise RuntimeError('resume episode ledger changed')
        restore_rng(saved['rng'])
    elif any((out/name).exists() for name in ('resume_latest.pth','train_trace.jsonl','train_episodes.jsonl')):
        raise RuntimeError('existing training output; use --resume or a fresh directory')
    initial_module = cpu_state(module)
    module.train()  # The E0 model stays eval(), with ALL parameters frozen.
    started = time.perf_counter()
    window = []
    loader = train_loader(train_cfg, args.seed, start, training['steps']-start, args.workers)
    with (out/'train_trace.jsonl').open('a') as trace, (out/'train_episodes.jsonl').open('a') as ledger:
        for step, batch in enumerate(loader, start=start+1):
            torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            lr = training['learning_rate']*.5*(1.+math.cos(math.pi*(step-1)/training['steps']))
            for group in optimizer.param_groups:
                group['lr'] = lr
            images = torch.cat((batch['support_set'], batch['target_set']))
            features = encode(model, images, 8, training['encoder_videos_per_chunk'])
            labels = batch['support_labels'].long().cuda()
            target = batch['target_labels'].long().cuda()
            optimizer.zero_grad(set_to_none=True)
            score, detail = matching(model.ts_dcm, features, labels, 5, args.shot, 5,
                module, feature_frames_per_chunk=training['feature_frames_per_chunk'],
                checkpoint_chunks=training['activation_checkpoint'])
            loss = F.cross_entropy(score, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite matching CE')
            loss.backward()
            gradient_norms = {}
            for name in ('projector',):
                grads = [p.grad for p in getattr(module, name).parameters() if p.grad is not None]
                if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError('missing/non-finite gradient in '+name)
                gradient_norms[name] = math.sqrt(sum(float(g.double().square().sum()) for g in grads))
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('gradient leaked into frozen E0')
            optimizer.step()
            record = dict(batch['episode_record'], step=step)
            sequence.update(record['episode_id'].encode())
            accuracy = float((score.argmax(1) == target).float().mean())
            window.append((float(loss.detach()), accuracy))
            row = dict(step=step, matching_ce=window[-1][0], matching_accuracy=accuracy,
                learning_rate=lr, gate=detail, gradient_norms=gradient_norms,
                effective_update=True, seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                train_episode_id=record['episode_id'])
            trace.write(canonical(row)+'\n'); trace.flush()
            ledger.write(canonical(record)+'\n'); ledger.flush()
            if step % 100 == 0 or step == training['steps'] or args.smoke:
                print('CLSCTM_TRAIN', canonical(dict(step=step, steps=training['steps'],
                    mean_matching_ce=sum(x[0] for x in window)/len(window),
                    mean_matching_accuracy=sum(x[1] for x in window)/len(window),
                    averaging_episodes=len(window), learning_rate=lr, gradient_norms=gradient_norms,
                    gate_min=detail['gate_min'], gate_max=detail['gate_max'],
                    peak_allocated_GiB=row['peak_allocated_bytes']/1024**3,
                    elapsed_seconds=previous_elapsed+time.perf_counter()-started)), flush=True)
                window.clear()
            # Do not retain a finished feature graph between episodes.
            del batch, features, images, score, loss, labels, target
            if step % 500 == 0 or step == training['steps']:
                atomic_save(dict(identity=identity, module=cpu_state(module),
                    optimizer=optimizer.state_dict(), completed_steps=step,
                    rng=rng_state(), train_sequence_sha256=sequence.hexdigest(),
                    elapsed_seconds=previous_elapsed+time.perf_counter()-started), out/'resume_latest.pth')
    if model_digest(model) != original or sha_file(args.checkpoint) != identity['checkpoint_sha256']:
        raise RuntimeError('frozen model or original checkpoint changed')
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('frozen E0 unexpectedly has gradients')
    changed = {name: not torch.equal(value.cpu(), initial_module[name])
               for name, value in module.state_dict().items() if name.endswith('weight')}
    if start < training['steps'] and not any(changed.values()):
        raise RuntimeError('module parameters did not update')
    final = dict(identity=identity, module=cpu_state(module), completed_steps=training['steps'],
                 training_complete=True, frozen_model_unchanged=True,
                 original_checkpoint_unchanged=True, new_linear_weights_changed=changed)
    if (out/'module_final.pth').exists():
        raise RuntimeError('final module exists without completion; preserved, do not overwrite')
    atomic_save(final, out/'module_final.pth')
    write_new(out/'training_complete.json', dict(passed=True,
        module_sha256=sha_file(out/'module_final.pth'), identity=identity,
        frozen_model_unchanged=True, original_checkpoint_unchanged=True,
        train_sequence_sha256=sequence.hexdigest(), completed_steps=training['steps'],
        elapsed_seconds=previous_elapsed+time.perf_counter()-started,
        trainable_parameters=sum(p.numel() for p in module.parameters())))
    print('CLSCTM_TRAIN_COMPLETE', out/'module_final.pth', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shot', type=int, choices=(1,5), required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--smoke', action='store_true', help='two Base Train updates, never a formal module')
    p.add_argument('--resume', action='store_true', help='resume this exact run at the last saved update')
    args = p.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.workers < 0:
        p.error('workers must be nonnegative')
    with output_lock(args.output):
        run(args)


if __name__ == '__main__':
    main()
