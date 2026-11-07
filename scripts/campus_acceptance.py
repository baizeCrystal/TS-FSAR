#!/usr/bin/env python3
"""Opt-in campus-only full-model CUDA acceptance, never formal training.

Runs an actual training episode through released E0 and unchanged adapter loss
paths, then three-step E0/E1 smoke checks with diagnostic step 2 and two Val
episodes. A passed, source-hashed report is required by protocol v2 training.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shot', type=int, choices=(1, 5), required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output-dir exists; choose a fresh directory')
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Check the selected physical allocation before CUDA initializes.
    ecc = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                                  '--query-gpu=ecc.errors.uncorrected.volatile.total',
                                  '--format=csv,noheader,nounits'], text=True).strip()
    if not ecc.isdigit() or int(ecc) != 0:
        parser.error('GPU ECC status is not a verified zero: ' + ecc)
    active = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                                     '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if active:
        parser.error('GPU already has compute processes; stop the old queue or choose an idle allocation')
    out = args.output_dir.resolve()
    out.mkdir(parents=True)
    try:
        run_acceptance(args, out)
    except BaseException as error:
        (out / 'acceptance_failed.json').write_text(json.dumps({
            'passed': False, 'error': str(error), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise


def run_acceptance(args, out):
    import numpy as np
    import torch
    import yaml
    from utils.config import Config
    from utils.functions import build_embeddings, cosine_lr
    from utils.optimizer import construct_optimizer
    from trainers.model import build_model
    from trainers.ts_fsar import TS_FSAR
    from experiments.dto_kd.model import DTOTSFSAR
    from experiments.dto_kd.episodes import loader, unpack
    from experiments.dto_kd.gradient_utils import activation_context
    from experiments.dto_kd.rigorous import train, configuration_signature
    from experiments.dto_kd.state import (atomic_torch_save, cpu_state, file_hash, model_hash,
                                         restore_rng, rng_state, source_hash)
    from scripts.audit_campus_protocol import merge
    from scripts.prepare_run_config import prepare_protocol
    from scripts.check_ssv2_files import check

    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_weights.py')], check=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    raw = merge(yaml.safe_load((ROOT / 'configs/base.yaml').read_text()), yaml.safe_load(
        (ROOT / f'configs/dto_kd/ssv2_small_{args.shot}shot_E0.yaml').read_text()))
    raw['TRAIN']['RANDOM_SEED'] = args.seed
    raw['OUTPUT_DIR'] = raw['OUTPUT_DIR_SAVE'] = str(out)
    raw['DTO_KD'].update(UPDATE_RULE='paper_pi_softmax', ACTIVATION_OFFLOAD='diagnostics', TRACE_INTERVAL=0)
    prepare_protocol(raw, out / 'config.yaml', 'train')
    dataset_check = check(raw['DATA']['DATA_ROOT_DIR'])
    (out / 'dataset_check.json').write_text(json.dumps(dataset_check, indent=2) + '\n')
    (out / 'config.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = Config(load=False, cfg_dict=copy.deepcopy(raw))
    cfg.cfg_dict = copy.deepcopy(raw)
    image, text, tokenizer = build_model(cfg)
    device = torch.device('cuda:0')
    original = TS_FSAR(cfg, image)
    initial = cpu_state(original)
    initial_hash = model_hash(original)
    initial_rng = rng_state(cuda=True)
    bank = out / 'initialization.pth'
    atomic_torch_save({'model': initial, 'rng': initial_rng}, bank)
    embeddings = build_embeddings(cfg.TRAIN.CLASS_NAME, cfg.TRAIN.LLM_train_json, text, tokenizer)
    val_embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text, tokenizer)
    # Per-episode RNG makes this identical to worker-loaded formal episode 0.
    cfg.DATA_LOADER.NUM_WORKERS = 0
    batch = next(iter(loader(cfg, 'train')))
    inputs = unpack(batch, device)
    record = inputs[-1]

    def step(model):
        model.to(device).train()
        optimizer = construct_optimizer(model, cfg)
        scaler = torch.cuda.amp.GradScaler()
        restore_rng(initial_rng)
        with activation_context('all'), torch.cuda.amp.autocast():
            result = model(*inputs[:4], embeddings)
        scaler.scale(result['loss']).backward()
        scaler.unscale_(optimizer)
        grads = {n: p.grad.detach().cpu().clone() if p.grad is not None else None
                 for n, p in model.named_parameters() if p.requires_grad}
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError('first-step AMP overflow: one-update acceptance not established')
        values = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        summary = {'loss': float(result['loss'].detach()), 'accuracy': result['acc']}
        if hasattr(model, 'clear_dto_cache'):
            model.clear_dto_cache()
        model.cpu()
        del optimizer, result
        return grads, values, summary

    reference_gradients, reference_parameters, reference = step(original)
    del original
    gc.collect()
    torch.cuda.empty_cache()
    comparisons = []
    for mode in ('off', 'static_kl'):
        cfg.DTO_KD.MODE = mode
        cfg.DTO_KD.STATIC_KL_ALPHA = cfg.TRAIN.KD_ALPHA
        cfg.TEST.ENABLE = True  # skip PCA, then replace EVERY tensor strictly
        model = DTOTSFSAR(cfg, image)
        cfg.TEST.ENABLE = False
        model.load_state_dict(initial, strict=True)
        if model_hash(model) != initial_hash:
            raise RuntimeError('adapter initialization differs from original')
        gradients, parameters, summary = step(model)
        torch.testing.assert_close(torch.tensor(summary['loss']), torch.tensor(reference['loss']), rtol=1e-5, atol=1e-6)
        if summary['accuracy'] != reference['accuracy']:
            raise RuntimeError('unchanged objective changed training predictions')
        maximum = {'gradient_max_abs': 0., 'parameter_max_abs': 0.}
        for name, expected in reference_gradients.items():
            actual = gradients[name]
            if expected is None:
                if actual is not None:
                    raise RuntimeError('gradient presence changed: ' + name)
            else:
                if actual is None:
                    raise RuntimeError('gradient missing: ' + name)
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-5)
                maximum['gradient_max_abs'] = max(maximum['gradient_max_abs'], float((actual - expected).abs().max()))
            torch.testing.assert_close(parameters[name], reference_parameters[name], rtol=1e-5, atol=1e-6)
            maximum['parameter_max_abs'] = max(maximum['parameter_max_abs'], float((parameters[name] - reference_parameters[name]).abs().max()))
        comparisons.append({'mode': mode, 'passed': True, **maximum, **summary})
        del model, gradients, parameters
        gc.collect()
        torch.cuda.empty_cache()
    del inputs, batch, reference_gradients, reference_parameters
    smokes = []
    for mode in ('off', 'dynamic'):
        smoke_out = out / ('smoke_' + mode)
        smoke_out.mkdir()
        smoke_raw = copy.deepcopy(raw)
        smoke_raw['OUTPUT_DIR'] = smoke_raw['OUTPUT_DIR_SAVE'] = str(smoke_out)
        smoke_raw['TRAIN'].update(NUM_TRAIN_TASKS=3, NUM_TEST_TASKS=2, VAL_FRE_ITER=3)
        smoke_raw['DTO_KD'].update(MODE=mode, GRADIENT_SAMPLE_INTERVAL=2)
        smoke_raw['DATA_LOADER']['NUM_WORKERS'] = 0
        smoke_raw['LOG_PERIOD'] = 1
        cfg = Config(load=False, cfg_dict=copy.deepcopy(smoke_raw))
        cfg.cfg_dict = copy.deepcopy(smoke_raw)
        cfg.TEST.ENABLE = True
        model = DTOTSFSAR(cfg, image)
        cfg.TEST.ENABLE = False
        model.load_state_dict(initial, strict=True)
        model.to(device)
        optimizer = construct_optimizer(model, cfg)
        scheduler = cosine_lr(optimizer, 0, 3)
        restore_rng(initial_rng)
        result = train(loader(cfg, 'train'), loader(cfg, 'test', 'val'), model, embeddings,
                       val_embeddings, optimizer, scheduler, torch.cuda.amp.GradScaler(), cfg, device,
                       {'configuration_signature': configuration_signature(cfg), 'source_sha256': source_hash()})
        if result['successful_updates'] != 3:
            raise RuntimeError('smoke did not exercise three successful optimizer updates')
        smokes.append({'mode': mode, **result})
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    report = {'passed': True, 'source_sha256': source_hash(), 'shot': args.shot, 'seed': args.seed,
              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'physical_gpu_argument': args.gpu,
              'initial_model_sha256': initial_hash, 'initialization_file': bank.name,
              'initialization_file_sha256': file_hash(bank), 'episode0_sha256': record['episode_sha256'],
              'comparisons': comparisons, 'smokes': smokes,
              'dataset_check': dataset_check,
              'tolerances': {'loss_rtol': 1e-5, 'loss_atol': 1e-6, 'gradient_rtol': 1e-3,
                             'gradient_atol': 1e-5, 'parameter_rtol': 1e-5, 'parameter_atol': 1e-6},
              'scope': 'one real episode equality and 3-step E0/E1 smoke; not final accuracy reproduction'}
    (out / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CAMPUS_ACCEPTANCE_PASSED', out / 'acceptance.json', flush=True)


if __name__ == '__main__':
    main()
