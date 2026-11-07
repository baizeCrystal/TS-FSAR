#!/usr/bin/env python3
"""Freeze E3 from early Meta-Train shared-LSN effective KL/task norm ratios."""
import argparse
import json
from pathlib import Path
import statistics

import yaml


def calibrate(rows, max_step=500, minimum_samples=20, task_alpha=1.):
    coefficients = []
    steps = set()
    for row in rows:
        if (row['parameter_group'] != 'lsn' or row['step'] > max_step or not row.get('finite')):
            continue
        if row.get('mode') != 'dynamic':
            raise ValueError('E3 calibration requires E1 dynamic early Meta-Train probes')
        if row['step'] in steps:
            # Resume can replay an uncommitted attempt; do not silently double count.
            raise ValueError('duplicate calibration step; provide one committed attempt')
        d, t = row['raw_distill_norm'], row['raw_task_norm']
        weighted_ratio = row['weighted_norm_ratio_distill_over_task']
        if d <= 0 or t <= 0 or row['task_coefficient'] <= 0:
            continue
        steps.add(row['step'])
        coefficients.append(task_alpha * weighted_ratio * t / d)
    if len(coefficients) < minimum_samples:
        raise ValueError(f'need at least {minimum_samples} distinct early shared-LSN samples; got {len(coefficients)}')
    return {'selection_split': 'Meta-Train only', 'group': 'lsn', 'max_step': max_step,
            'samples': len(coefficients), 'steps': sorted(steps),
            'static_kl_alpha': statistics.median(coefficients),
            'definition': 'median alpha matching E1 effective KL/task norm ratio on E1 early gradient probes',
            'limitation': 'relative scale control; does not match both absolute contributions or later training trajectory'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gradients', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='E3 template, read-only')
    parser.add_argument('--output-config', type=Path, required=True)
    parser.add_argument('--output-report', type=Path, required=True)
    args = parser.parse_args()
    if args.output_config.exists() or args.output_report.exists():
        parser.error('outputs already exist; refusing overwrite')
    cfg = yaml.safe_load(args.config.read_text())
    if cfg['DTO_KD']['MODE'] != 'scale_matched':
        parser.error('requires an E3 scale_matched template')
    result = calibrate([json.loads(line) for line in args.gradients.read_text().splitlines()],
                       task_alpha=cfg['TRAIN']['VIS_ALPHA'])
    cfg['DTO_KD']['STATIC_KL_ALPHA'] = result['static_kl_alpha']
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_config.open('x') as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    with args.output_report.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
