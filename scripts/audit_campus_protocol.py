#!/usr/bin/env python3
"""CPU/file-only checks against the repository's released SSv2-Small recipe.

This checks configuration/provenance, not CUDA numerics or achieved accuracy.
It never imports torch, starts jobs, or modifies an existing run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = prefix + '.' + key if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def compare_config(path, generated=False):
    base = yaml.safe_load((ROOT / 'configs/base.yaml').read_text())
    cfg = merge(base, yaml.safe_load(path.read_text()))
    shot = cfg['TRAIN']['SHOT']
    if shot not in (1, 5):
        raise ValueError('unsupported shot: ' + str(shot))
    released = ROOT / f'configs/ssv2_small/Metatrain_{shot}shot.yaml'
    expected = merge(base, yaml.safe_load(released.read_text()))
    actual, reference = flatten(cfg), flatten(expected)
    # Compare the complete scientific sections, except runtime paths/logging,
    # optional EVAL_START_EPISODE (not present in the released config), and the
    # explicitly changed DTO objective. KL's base coefficient remains unchanged.
    keys = {key for key in reference if key.startswith((
        'TRAIN.', 'LSN.', 'SOLVER.', 'AUGMENTATION.', 'VIDEO.', 'DATA.'))}
    keys -= {'DATA.DATA_ROOT_DIR', 'DATA.ANNO_DIR', 'TRAIN.LOG_FILE'}
    keys |= {'NUM_GPUS', 'TEMP_CLIP', 'TEMP_SIDE', 'TEST.BATCH_SIZE',
             'TEST.USE_TEXT', 'TEST.FUSE_CLIP', 'TEST.FUSE_VISION', 'TEST.FUSE_SIDE'}
    # These were added for deployment, not part of the released sampling recipe.
    keys -= {'DATA.DECODE_THREADS', 'DATA.FAIL_ON_DECODE_ERROR'}
    if generated and not cfg['TRAIN']['ENABLE']:
        keys.discard('TRAIN.ENABLE')
    differences = {key: {'released': reference.get(key), 'actual': actual.get(key)}
                   for key in sorted(keys) if reference.get(key) != actual.get(key)}
    checks = {}
    dto = cfg.get('DTO_KD', {})
    checks['original_tsm'] = cfg.get('TSM_VARIANT', 'original') == 'original'
    checks['mean_support'] = cfg.get('TSM_SUPPORT_AGGREGATION', 'mean') == 'mean'
    checks['no_residual'] = not cfg.get('TSM_RESIDUAL', {}).get('ENABLE', False)
    glac = cfg.get('GLAC', {})
    checks['all_original_glac_terms'] = all(glac.get(k, True) for k in (
        'ENABLE', 'ENABLE_KL', 'ENABLE_STUDENT_CE', 'ENABLE_TEACHER_CE'))
    protocol = None
    if generated:
        ann = Path(cfg['DATA']['ANNO_DIR'])
        protocol_path = ann / 'protocol.json'
        if protocol_path.is_file():
            protocol = json.loads(protocol_path.read_text())
        is_train = bool(cfg['TRAIN']['ENABLE'])
        official = cfg.get('RIGOROUS', {}).get('SELECTION_PROTOCOL') == 'official-test'
        split = 'val' if is_train and not official else 'test'
        canonical = (ROOT / f'configs/ssv2_small/{split}_few_shot.txt').read_text()
        if split == 'val':
            canonical = '\n'.join('test' + row[len('val'):] for row in canonical.splitlines()
                                  if row.strip()) + '\n'
        checks['evaluation_list_matches_split'] = (
            (ann / 'test_few_shot.txt').is_file()
            and (ann / 'test_few_shot.txt').read_text() == canonical)
        checks['training_list_matches'] = (
            (ann / 'train_few_shot.txt').is_file()
            and (ann / 'train_few_shot.txt').read_bytes()
            == (ROOT / 'configs/ssv2_small/train_few_shot.txt').read_bytes())
        mapping = json.loads((ROOT / 'assets/SSV2_SMALL_CLASS_SPLITS.json').read_text())
        checks['evaluation_class_names_match'] = cfg['TEST']['CLASS_NAME'] == mapping[split]
        prompt_path = Path(cfg['TEST']['LLM_test_json'])
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        checks['evaluation_prompt_matches'] = (
            prompt_path.is_file() and prompt_path.read_bytes()
            == (ROOT / f'prompts/Ssv2small_{split}.json').read_bytes())
        checks['protocol_records_split'] = (
            protocol is not None and protocol.get('current_evaluation_split') == split
            and protocol.get('checkpoint_selection_split') == ('Novel-Test' if official else 'Meta-Val'))
        if official:
            checks['official_rng_policy_recorded'] = protocol.get('rng_policy') == 'official-worker-stream'
    return {'config': str(path.resolve()), 'shot': shot, 'dto_mode': dto.get('MODE', 'off'),
            'detach_teacher': bool(dto.get('DETACH_TEACHER', False)),
            'objective_scope': 'shifted full-supervised and base-weighted KL' if dto.get('MODE') == 'safe_dynamic'
                               else 'original log KL/DCM; unchanged auxiliary CE; shared-LSN exact solve'
                               if dto.get('MODE') == 'paper_exact'
                               else 'original log KL/DCM; unchanged auxiliary CE; direct Eq18 approximation'
                               if dto.get('MODE') == 'paper_amortized'
                               else 'task-anchored DCM with bounded KL; unchanged auxiliary CE'
                               if dto.get('MODE') == 'task_anchored'
                               else 'original scalar objective or declared DTO control',
            'protocol_version': cfg.get('RIGOROUS', {}).get('VERSION', 'legacy'),
            'selection_protocol': cfg.get('RIGOROUS', {}).get('SELECTION_PROTOCOL', 'meta-val-fixed'),
            'controller_update': 'paper-closed-form-simplex' if dto.get('MODE') == 'paper_exact'
                                 else 'task-anchored-kl-cap' if dto.get('MODE') == 'task_anchored'
                                 else dto.get('UPDATE_RULE', 'legacy_logits'),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'released_recipe_differences': differences, 'checks': checks,
            'configuration_checks_pass': not differences and all(checks.values()),
            'protocol': protocol}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, help='audit existing per-job config.yaml files')
    parser.add_argument('--output', type=Path, help='new JSON report (never overwrite)')
    args = parser.parse_args()
    paths = sorted(args.run_root.glob('*/config.yaml') if args.run_root else
                   (ROOT / 'configs/dto_kd').glob('ssv2_small_*.yaml'))
    if not paths:
        parser.error('no configurations found')
    results = [compare_config(path, generated=args.run_root is not None) for path in paths]
    report = {
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                text=True).strip(),
        'scope': 'CPU/files only; configuration checks do not certify experiment validity',
        'limitations': [
            'Historical meta-val-fixed differs from official selection; official-test uses released Test and worker RNG.',
            'legacy_logits is not the direct Eq.18 update; protocol v2 uses paper_pi_softmax.',
            'Configuration checks do not certify full-model CUDA numerics or final accuracy.',
            'Pair actual episode/input hashes and evaluation protocol before a paired analysis.',
            'Inspect weighted parameter-group probes and effective AMP updates in protocol v2.',
            'Full CUDA resume equivalence remains unverified; model_best is not a full resume file.',
        ],
        'configurations': results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')
    return 0 if all(row['configuration_checks_pass'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
