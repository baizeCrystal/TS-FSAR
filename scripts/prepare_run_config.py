#!/usr/bin/env python3
"""Generate isolated run configs; checkpoint selection uses Val, never Test."""
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
    evaluation = 'val' if mode == 'train' else 'test'
    ann = output.parent / ('meta_val_annotations' if mode == 'train' else 'meta_test_annotations')
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
    provenance.update(checkpoint_selection_split='Meta-Val', current_evaluation_split=evaluation,
                      evaluation_prompt=c['TEST']['LLM_test_json'],
                      label_prefix_alias='valN -> testN for legacy loader only' if evaluation == 'val' else None)
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
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
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
