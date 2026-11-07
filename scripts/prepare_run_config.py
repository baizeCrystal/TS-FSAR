#!/usr/bin/env python3
"""Generate isolated run configs with an explicitly recorded selection protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def prepare_protocol(c, output, mode):
    data_root = Path(os.environ.get('TSFSAR_SSV2_ROOT', c['DATA']['DATA_ROOT_DIR'])).resolve()
    c['DATA']['DATA_ROOT_DIR'] = str(data_root)
    # Use canonical code-versioned split lists, independent of stale output files.
    source = ROOT / 'configs/ssv2_small'
    mapping = json.loads((ROOT / 'assets/SSV2_SMALL_CLASS_SPLITS.json').read_text())
    texts = {}
    videos = {}
    provenance = {}
    for split in ('train', 'val', 'test'):
        path = source / f'{split}_few_shot.txt'
        text = path.read_text()
        rows = [row for row in text.splitlines() if row.strip()]
        labels = set()
        ids = set()
        for row in rows:
            label, vid = row.split('/')
            if not label.startswith(split):
                raise ValueError(f'invalid split label in {path}: {row}')
            labels.add(int(label[len(split):]))
            ids.add(vid)
        if labels != set(range(len(mapping[split]))):
            raise ValueError(f'label mapping does not match {path}')
        texts[split], videos[split] = text, ids
        provenance[split] = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'rows': len(rows), 'classes': len(labels)}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if videos[a] & videos[b] or set(mapping[a]) & set(mapping[b]):
            raise ValueError(f'overlap between {a} and {b}')
    # Legacy main calls the eval loader "test" for both validation and final Test.
    # Alias ONLY the prefix in a run-private file, preserving video IDs and labels.
    official = c.get('RIGOROUS', {}).get('SELECTION_PROTOCOL') == 'official-test'
    if official:
        lock = json.loads((ROOT / 'assets/OFFICIAL_EVALUATION_SOURCE_SHA256.json').read_text())
        required = [f'configs/ssv2_small/{split}_few_shot.txt' for split in ('train', 'test')]
        required += ['prompts/Ssv2small_train.json', 'prompts/Ssv2small_test.json']
        required += [f'configs/ssv2_small/Meta{mode}_{shot}shot.yaml'
                     for mode in ('train', 'test') for shot in (1, 5)]
        for name in required:
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != lock['files'][name]:
                raise ValueError('canonical official source changed: ' + name)
        released = yaml.safe_load((ROOT / 'configs/ssv2_small/Metatest_1shot.yaml').read_text())
        if mapping['test'] != released['TEST']['CLASS_NAME'] or mapping['train'] != released['TRAIN']['CLASS_NAME']:
            raise ValueError('class-name mapping differs from the official configuration')
    evaluation = 'val' if mode == 'train' and not official else 'test'
    ann = output.parent / ('official_test_annotations' if official else
                           ('meta_val_annotations' if mode == 'train' else 'meta_test_annotations'))
    ann.mkdir(parents=True, exist_ok=True)
    (ann / 'train_few_shot.txt').write_text(texts['train'])
    selected = texts[evaluation]
    if evaluation == 'val':
        selected = '\n'.join('test' + line[len('val'):] for line in selected.splitlines() if line.strip()) + '\n'
    (ann / 'test_few_shot.txt').write_text(selected)
    # Dataset checks OUTPUT_DIR before ANNO_DIR; reject stale overrides.
    for name in ('train_few_shot.txt', 'test_few_shot.txt'):
        if (Path(c['OUTPUT_DIR']) / name).exists():
            raise ValueError(f'stale split override in output directory: {name}; choose a fresh output')
    c['DATA']['ANNO_DIR'] = str(ann)
    c['TEST']['CLASS_NAME'] = mapping[evaluation]
    c['TEST']['LLM_test_json'] = str(ROOT / f'prompts/Ssv2small_{evaluation}.json')
    provenance.update(checkpoint_selection_split='Novel-Test' if official else 'Meta-Val',
                      selection_protocol='official-test' if official else 'meta-val-fixed',
                      rng_policy='official-worker-stream' if official else 'fixed-per-episode',
                      current_evaluation_split=evaluation,
                      evaluation_prompt=c['TEST']['LLM_test_json'],
                      label_prefix_alias='valN -> testN for legacy loader only' if evaluation == 'val' else None)
    if official:
        provenance['official_source_commit'] = lock['commit']
    (ann / 'protocol.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'PROTOCOL: mode={mode}, evaluation={evaluation}, annotations={ann}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--output-dir')
    p.add_argument('--mode', choices=('train', 'eval'), default='train')
    p.add_argument('--checkpoint')
    p.add_argument('--activation-offload', choices=('off', 'diagnostics', 'all'), default='diagnostics')
    p.add_argument('--smoke', action='store_true', help='three train/two validation episodes; not formal results')
    p.add_argument('--protocol', choices=('legacy', 'v2'), default='legacy')
    p.add_argument('--selection-protocol', choices=('meta-val-fixed', 'official-test'), default='meta-val-fixed')
    p.add_argument('--acceptance-report', help='passed full-model campus CUDA certificate for this shot/seed')
    p.add_argument('--resume', help='protocol-v2 complete resume_latest.pth, not model_best.pth')
    a = p.parse_args()
    if a.selection_protocol == 'official-test' and a.protocol != 'v2':
        raise SystemExit('official-test requires the common v2 training loop')
    if a.selection_protocol == 'official-test' and a.resume:
        raise SystemExit('official worker RNG cannot resume exactly; start a fresh run')
    c = yaml.safe_load(Path(a.config).read_text())
    if a.protocol != 'v2' and (c.get('DTO_KD', {}).get('DETACH_TEACHER')
                               or c.get('DTO_KD', {}).get('MODE') == 'safe_dynamic'):
        raise SystemExit('new FSAR adaptations require protocol v2')
    if a.seed is not None:
        c['TRAIN']['RANDOM_SEED'] = a.seed
    if a.output_dir:
        c['OUTPUT_DIR'] = a.output_dir
        c['OUTPUT_DIR_SAVE'] = a.output_dir
    if a.smoke:
        if a.mode != 'train' or not a.output_dir or 'smoke' not in Path(a.output_dir).parts:
            raise SystemExit('--smoke requires train mode and an output directory inside smoke/')
        c['TRAIN'].update(NUM_TRAIN_TASKS=3, NUM_TEST_TASKS=2, VAL_FRE_ITER=3, EVAL_START_EPISODE=1)
        c['LOG_PERIOD'] = 1
        c.setdefault('DTO_KD', {})['GRADIENT_SAMPLE_INTERVAL'] = 2
    c.setdefault('DTO_KD', {})['ACTIVATION_OFFLOAD'] = a.activation_offload
    if a.protocol == 'v2':
        if a.mode == 'train' and not a.smoke and not a.acceptance_report:
            raise SystemExit('protocol v2 training requires --acceptance-report from campus_acceptance.py')
        if a.acceptance_report and not Path(a.acceptance_report).is_file():
            raise SystemExit('--acceptance-report must be an existing passed campus report')
        if a.acceptance_report and json.loads(Path(a.acceptance_report).read_text()).get('passed') is not True:
            raise SystemExit('campus acceptance did not pass; formal training refused')
        if a.resume and (a.mode != 'train' or not Path(a.resume).is_file()):
            raise SystemExit('--resume requires train mode and an existing complete v2 checkpoint')
        c['DTO_KD']['UPDATE_RULE'] = 'paper_pi_softmax'
        c['DTO_KD'].update(GRADIENT_EARLY_STEPS=500, GRADIENT_EARLY_INTERVAL=25)
        c['RIGOROUS'] = {'VERSION': 2, 'SMOKE': a.smoke,
                         'SELECTION_PROTOCOL': a.selection_protocol,
                         'ACCEPTANCE_REPORT': str(Path(a.acceptance_report).resolve()) if a.acceptance_report else '',
                         'RESUME': str(Path(a.resume).resolve()) if a.resume else ''}
    elif a.resume or a.acceptance_report:
        raise SystemExit('--resume/--acceptance-report require --protocol v2')
    c['TRAIN']['ENABLE'] = a.mode == 'train'
    c['TEST']['ENABLE'] = a.mode == 'eval'
    if a.mode == 'eval':
        if not a.checkpoint or not Path(a.checkpoint).is_file():
            raise SystemExit('--checkpoint must name an existing file for --mode eval')
        c['TEST']['LOAD_PRETRAIN'] = True
        c['TEST']['PRETRAIN_MODEL_PATH'] = str(Path(a.checkpoint).resolve())
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prepare_protocol(c, output, a.mode)
    output.write_text(yaml.safe_dump(c, sort_keys=False))


if __name__ == '__main__':
    main()
