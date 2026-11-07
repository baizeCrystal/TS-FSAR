#!/usr/bin/env python3
"""Opt-in campus-only full-model CUDA acceptance, never formal training.

Runs an actual training episode through released E0 and unchanged adapter loss
paths at one common finite AMP scale, then bounded E0/E1 smoke checks with
default AMP, diagnostic step 2 and two declared-split evaluation episodes. A passed, source-hashed
report is required by protocol v2 training.
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
    parser.add_argument('--reuse-initialization-report', type=Path,
                        help='reuse only model/RNG initialization from a passed hashed report; rerun all checks')
    parser.add_argument('--adaptation-checks', action='store_true',
                        help='also certify E0_detach, E1_detach and E1_safe with common initialization')
    parser.add_argument('--selection-protocol', choices=('meta-val-fixed', 'official-test'), default='meta-val-fixed')
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
    print('ACCEPTANCE_STAGE importing dependencies', flush=True)
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
    from experiments.dto_kd.acceptance import (one_successful_update, load_certified_initialization,
                                               verify_smoke_schedule)
    from experiments.dto_kd.rigorous import train, configuration_signature
    from experiments.dto_kd.state import (atomic_torch_save, cpu_state, file_hash, model_hash,
                                         restore_rng, rng_state, source_hash, append_json)
    from scripts.audit_campus_protocol import merge
    from scripts.prepare_run_config import prepare_protocol
    from scripts.check_ssv2_files import check

    print('ACCEPTANCE_STAGE verifying CLIP weights', flush=True)
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
    raw['RIGOROUS'] = {'VERSION': 2, 'SELECTION_PROTOCOL': args.selection_protocol}
    prepare_protocol(raw, out / 'config.yaml', 'train')
    print('ACCEPTANCE_STAGE checking dataset files', flush=True)
    dataset_check = check(raw['DATA']['DATA_ROOT_DIR'])
    (out / 'dataset_check.json').write_text(json.dumps(dataset_check, indent=2) + '\n')
    (out / 'config.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = Config(load=False, cfg_dict=copy.deepcopy(raw))
    cfg.cfg_dict = copy.deepcopy(raw)
    print('ACCEPTANCE_STAGE loading CLIP', flush=True)
    image, text, tokenizer = build_model(cfg)
    device = torch.device('cuda:0')
    reused = None
    if args.reuse_initialization_report:
        cached, reused = load_certified_initialization(
            args.reuse_initialization_report, args.shot, args.seed, torch.__version__,
            torch.cuda.get_device_name(0))
        print('ACCEPTANCE_STAGE strictly loading previously certified PCA initialization', flush=True)
        original_flag = cfg.TEST.ENABLE
        cfg.TEST.ENABLE = True
        original = TS_FSAR(cfg, image)
        cfg.TEST.ENABLE = original_flag
        original.load_state_dict(cached['model'], strict=True)
        if model_hash(original) != reused['initial_model_sha256']:
            raise RuntimeError('reused initialization model hash mismatch')
        initial_rng = cached['rng']
        del cached
    else:
        print('ACCEPTANCE_STAGE building original PCA initialization', flush=True)
        original = TS_FSAR(cfg, image)
        initial_rng = rng_state(cuda=True)
    initial = cpu_state(original)
    initial_hash = model_hash(original)
    bank = out / 'initialization.pth'
    atomic_torch_save({'model': initial, 'rng': initial_rng}, bank)
    embeddings = build_embeddings(cfg.TRAIN.CLASS_NAME, cfg.TRAIN.LLM_train_json, text, tokenizer)
    val_embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text, tokenizer)
    # Per-episode RNG makes this identical to worker-loaded formal episode 0.
    cfg.DATA_LOADER.NUM_WORKERS = 0
    batch = next(iter(loader(cfg, 'train')))
    inputs = unpack(batch, device)
    record = inputs[-1]

    def step(model, name, init_scale=None, max_attempts=1):
        model.to(device).train()
        optimizer = construct_optimizer(model, cfg)
        scaler = torch.cuda.amp.GradScaler(**({'init_scale': init_scale} if init_scale is not None else {}))

        def forward():
            with activation_context('all'), torch.cuda.amp.autocast():
                return model(*inputs[:4], embeddings)

        def record_attempt(row):
            row = {'model': name, **row}
            append_json(out / 'amp_attempts.jsonl', row)
            print('ACCEPTANCE_AMP', json.dumps(row), flush=True)

        print('ACCEPTANCE_STAGE one effective update:', name, flush=True)
        grads, values, summary = one_successful_update(
            model, initial, optimizer, scaler, forward, lambda: restore_rng(initial_rng),
            record_attempt, max_attempts=max_attempts)
        model.cpu()
        del optimizer, scaler
        return grads, values, summary

    reference_gradients, reference_parameters, reference = step(original, 'original_E0', max_attempts=17)
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
        gradients, parameters, summary = step(model, mode, init_scale=reference['successful_scale'])
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
    variants = [('off', False, 'off'), ('dynamic', False, 'dynamic')]
    if args.adaptation_checks:
        variants += [('off', True, 'E0_detach'), ('dynamic', True, 'E1_detach'),
                     ('safe_dynamic', True, 'E1_safe')]
        if args.selection_protocol == 'official-test':
            variants += [('scale_matched', True, 'E3_detach')]
    for mode, detach_teacher, variant in variants:
        print('ACCEPTANCE_STAGE default-AMP smoke:', variant, flush=True)
        smoke_out = out / ('smoke_' + variant)
        smoke_out.mkdir()
        smoke_raw = copy.deepcopy(raw)
        smoke_raw['OUTPUT_DIR'] = smoke_raw['OUTPUT_DIR_SAVE'] = str(smoke_out)
        # Exercise the formal default scaler, including its initial skipped
        # updates. This is a fixed acceptance-only budget, not a new recipe.
        smoke_raw['TRAIN'].update(NUM_TRAIN_TASKS=32, NUM_TEST_TASKS=2, VAL_FRE_ITER=32)
        smoke_raw['DTO_KD'].update(MODE=mode, GRADIENT_SAMPLE_INTERVAL=0,
                                   GRADIENT_EARLY_STEPS=2, GRADIENT_EARLY_INTERVAL=2,
                                   DETACH_TEACHER=detach_teacher)
        smoke_raw['DATA_LOADER']['NUM_WORKERS'] = 0
        # Exercise the real progress-logging path even in a two-episode smoke.
        # Otherwise a missing dependency here fails only at formal episode 1000.
        smoke_raw['TEST']['PROGRESS_LOG_PERIOD'] = 1
        smoke_raw['LOG_PERIOD'] = 1
        cfg = Config(load=False, cfg_dict=copy.deepcopy(smoke_raw))
        cfg.cfg_dict = copy.deepcopy(smoke_raw)
        cfg.TEST.ENABLE = True
        model = DTOTSFSAR(cfg, image)
        cfg.TEST.ENABLE = False
        model.load_state_dict(initial, strict=True)
        model.to(device)
        optimizer = construct_optimizer(model, cfg)
        base_lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler = cosine_lr(optimizer, 0, cfg.TRAIN.NUM_TRAIN_TASKS)
        restore_rng(initial_rng)
        result = train(loader(cfg, 'train'), loader(cfg, 'test', 'test' if args.selection_protocol == 'official-test' else 'val'), model, embeddings,
                       val_embeddings, optimizer, scheduler, torch.cuda.amp.GradScaler(), cfg, device,
                       {'configuration_signature': configuration_signature(cfg), 'source_sha256': source_hash()})
        if result['successful_updates'] < 3:
            raise RuntimeError('32-episode default-AMP smoke had fewer than three effective updates')
        trace = [json.loads(line) for line in (smoke_out / 'train_trace.jsonl').read_text().splitlines()]
        schedule_check = verify_smoke_schedule(trace, base_lrs, cfg.TRAIN.NUM_TRAIN_TASKS)
        adaptation_check = None
        if detach_teacher:
            probes = [json.loads(line) for line in (smoke_out / 'gradient_groups.jsonl').read_text().splitlines()]
            teacher_probes = [p for p in probes if p['parameter_group'] == 'teacher_adapter']
            if not teacher_probes or not all(p.get('finite') and p['raw_distill_norm'] == 0.
                                             and p['auxiliary_distill_norm'] > 0. for p in teacher_probes):
                raise RuntimeError('KL target not stopped or teacher CE gradient missing')
            if mode == 'safe_dynamic' and not all(
                    0. <= row['distill_coefficient'] <= raw['TRAIN']['KD_ALPHA'] + 1e-12
                    for row in trace):
                raise RuntimeError('safe KL coefficient exceeded released alpha')
            adaptation_check = {'passed': True, 'teacher_kl_gradient_zero': True,
                                'teacher_ce_gradient_nonzero': True,
                                'bounded_kl_coefficient_checked': mode == 'safe_dynamic'}
        smokes.append({'mode': mode, 'variant': variant, 'detach_teacher': detach_teacher,
                       'adaptation_check': adaptation_check,
                       'minimum_required_successful_updates': 3,
                       'schedule_check': schedule_check,
                       'initial_amp_scale': 65536., 'skipped_updates': 32 - result['successful_updates'], **result})
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    report = {'passed': True, 'source_sha256': source_hash(), 'shot': args.shot, 'seed': args.seed,
              'selection_protocol': args.selection_protocol,
              'selection_split': 'Novel-Test' if args.selection_protocol == 'official-test' else 'Meta-Val',
              'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'physical_gpu_argument': args.gpu,
              'initial_model_sha256': initial_hash, 'initialization_file': bank.name,
              'initialization_file_sha256': file_hash(bank), 'episode0_sha256': record['episode_sha256'],
              'comparisons': comparisons, 'smokes': smokes,
              'initialization_reuse': reused,
              'amp_reference': reference,
              'amp_policy': 'Reference only: <=17 fixed-input backoff attempts from default scale. '
                            'Equivalent candidates use the same successful scale, one attempt. '
                            '32-episode smokes and formal training use the unchanged default scaler.',
              'dataset_check': dataset_check,
              'tolerances': {'loss_rtol': 1e-5, 'loss_atol': 1e-6, 'gradient_rtol': 1e-3,
                             'gradient_atol': 1e-5, 'parameter_rtol': 1e-5, 'parameter_atol': 1e-6},
              'scope': 'one effective real-episode update equality and 32-episode declared-variant smokes; '
                       'not final accuracy reproduction or full-run stability proof'}
    (out / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('CAMPUS_ACCEPTANCE_PASSED', out / 'acceptance.json', flush=True)


if __name__ == '__main__':
    main()
