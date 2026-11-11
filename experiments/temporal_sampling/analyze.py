"""Offline paired statistics and balanced case selection from saved forwards."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .core import (BRANCHES, margin, paired_bootstrap, read_rows, validate_output, write_json)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k] for k in z.files}
    validate_output(result)
    return result


def attribution(output):
    y = output['labels']
    a, b = output['ts_dcm'].argmax(-1) == y, output['fusion'].argmax(-1) == y
    # 0: both wrong, 1: matching wrong/fusion right,
    # 2: matching right/fusion wrong, 3: both right.
    return a.astype(int) * 2 + b.astype(int)


def summarize(pairs, repeats=5000):
    ids = np.concatenate([np.repeat(ep, 5) for ep, _, _ in pairs])
    y = np.concatenate([b['labels'] for _, b, _ in pairs])
    report = {}
    for branch in BRANCHES:
        b = np.concatenate([b[branch] for _, b, _ in pairs])
        a = np.concatenate([a[branch] for _, _, a in pairs])
        bc, ac = b.argmax(-1) == y, a.argmax(-1) == y
        quantities = {'baseline_accuracy': bc, 'accuracy': ac,
                      'accuracy_delta': ac.astype(float) - bc,
                      'prediction_change_rate': a.argmax(-1) != b.argmax(-1),
                      'correct_to_wrong_rate': bc & ~ac,
                      'wrong_to_correct_rate': ~bc & ac,
                      'baseline_margin': margin(b, y), 'margin': margin(a, y),
                      'margin_delta': margin(a, y) - margin(b, y)}
        report[branch] = {name: paired_bootstrap(values, ids, repeats=repeats)
                          for name, values in quantities.items()}
        report[branch]['correct_to_wrong_count'] = int((bc & ~ac).sum())
        report[branch]['wrong_to_correct_count'] = int((~bc & ac).sum())
    transitions = np.zeros((4, 4), dtype=int)
    r_delta, w_delta, true_mean, wrong_mean = [], [], [], []
    decomposition = []
    for ep, b, a in pairs:
        np.testing.assert_array_equal(b['labels'], a['labels'])
        for x, z in zip(attribution(b), attribution(a)):
            transitions[x, z] += 1
        dr, dw = a['R'] - b['R'], a['W'] - b['W']
        r_delta.extend(np.abs(dr).mean((1, 2, 3)))
        w_delta.extend(np.abs(dw).mean((1, 2, 3)))
        y = b['labels']
        for i, label in enumerate(y):
            rival_scores = b['ts_dcm'][i].copy()
            rival_scores[label] = -np.inf
            rival = int(rival_scores.argmax())
            true_mean.append(float(dr[i, label].mean()))
            wrong_mean.append(float(dr[i, rival].mean()))
            terms = [dr[i] * b['W'][i], b['R'][i] * dw[i], dr[i] * dw[i]]
            values = [float((t[label] - t[rival]).sum()) for t in terms]
            decomposition.append(values)
    report['attribution_state_order'] = ['both_wrong', 'matching_wrong_fusion_right', 'matching_right_fusion_wrong', 'both_right']
    report['attribution_transition_counts'] = transitions.tolist()
    report['R_absolute_change'] = paired_bootstrap(r_delta, ids, repeats=repeats)
    report['W_absolute_change'] = paired_bootstrap(w_delta, ids, repeats=repeats)
    report['true_class_R_mean_change'] = paired_bootstrap(true_mean, ids, repeats=repeats)
    report['original_rival_R_mean_change'] = paired_bootstrap(wrong_mean, ids, repeats=repeats)
    report['fixed_original_rival_margin_decomposition'] = {
        name: paired_bootstrap(np.asarray(decomposition)[:, i], ids, repeats=repeats)
        for i, name in enumerate(('R_change_with_W_fixed', 'W_change_with_R_fixed', 'interaction'))}
    report['decomposition_limit'] = 'Algebra at frame-index positions, not semantic alignment or proof of H1b'
    return report


def feature_summary(metadata, repeats):
    report = {}
    for role in ('query', 'support'):
        selected = [(eid, r) for eid, rows in metadata for r in rows if r['role'] == role]
        if not selected:
            continue
        ids = [eid for eid, _ in selected]
        report[role] = {}
        for field in ('duplicate_fraction', 'unique_frames', 'mean_abs_shift', 'rms_shift'):
            report[role][field] = paired_bootstrap([r['sampling'][field] for _, r in selected], ids, repeats=repeats)
        if all('control_rms_residual' in r['sampling'] for _, r in selected):
            for field in ('control_target_rms', 'control_rms_residual', 'control_unique_frame_residual'):
                report[role][field] = paired_bootstrap([r['sampling'][field] for _, r in selected], ids, repeats=repeats)
        for feature in ('clip', 'lsn'):
            for field in ('index_cosine_distance', 'time_interpolated_cosine_distance', 'shared_frame_cosine_distance'):
                valid = [(eid, r[feature][field]) for eid, r in selected if r[feature][field] is not None]
                report[role][feature + '_' + field] = (paired_bootstrap([v for _, v in valid], [eid for eid, _ in valid], repeats=repeats)
                                                        if valid else None)
    return report


def choose_cases(pairs, episodes, condition):
    selected = {}
    for eid, b, a in pairs:
        y = b['labels']
        bc, ac = b['fusion'].argmax(1) == y, a['fusion'].argmax(1) == y
        mc = a['ts_dcm'].argmax(1) == y
        masks = {'correct_to_wrong': bc & ~ac, 'stays_correct': bc & ac,
                 'wrong_to_correct': ~bc & ac, 'matching_right_fusion_wrong': mc & ~ac,
                 'stays_wrong': ~bc & ~ac}
        for category, mask in masks.items():
            if category not in selected and mask.any():
                qi = int(np.flatnonzero(mask)[0])
                selected[category] = {'condition': condition, 'episode': eid, 'query_index': qi,
                    'video_id': episodes[eid]['query'][qi]['video_id'], 'label': int(y[qi]),
                    'selection': 'first in fixed episode/query order; no magnitude/outcome cherry-picking within category'}
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new analysis directory')
    p.add_argument('--repeats', type=int, default=5000)
    p.add_argument('--allow-partial', action='store_true')
    a = p.parse_args()
    if a.output.exists() or a.repeats < 1:
        p.error('new output directory and positive repeats required')
    status = json.loads((a.run / 'status.json').read_text())
    if status['status'] != 'complete' and not a.allow_partial:
        p.error('Run incomplete; --allow-partial reports matched completed subsets only')
    if not (a.run / 'e0_verification.json').exists():
        p.error('Historical E0 verification did not finish; do not interpret interventions')
    manifest = json.loads((a.run / 'manifest.json').read_text())
    episodes = {ep['episode']: ep for ep in read_rows(a.run / 'episodes.jsonl')}
    baselines = {eid: load(a.run / f'e0_{eid:05d}.npz') for eid in episodes}
    report = {'run_status': status, 'manifest': manifest, 'conditions': {}, 'paired_random_control_contrasts': {}}
    report['E0'] = {}
    e0_ids = np.repeat(list(baselines), 5)
    e0_y = np.concatenate([b['labels'] for b in baselines.values()])
    for branch in BRANCHES:
        scores = np.concatenate([b[branch] for b in baselines.values()])
        report['E0'][branch] = {'accuracy': paired_bootstrap(scores.argmax(1) == e0_y, e0_ids, repeats=a.repeats),
                                'margin': paired_bootstrap(margin(scores, e0_y), e0_ids, repeats=a.repeats)}
    cases = {}
    by_condition = {}
    prediction_rows = []
    for scheme in manifest['schemes']:
        if scheme == 'e0':
            continue
        for scope in manifest['scopes']:
            condition = scheme + '_' + scope
            pairs, metadata = [], []
            for eid in episodes:
                stem = a.run / f'{condition}_{eid:05d}'
                if not stem.with_suffix('.npz').exists() or not stem.with_suffix('.json').exists():
                    continue
                out = load(stem.with_suffix('.npz'))
                meta = json.loads(stem.with_suffix('.json').read_text())
                pairs.append((eid, baselines[eid], out))
                metadata.append((eid, meta['feature_changes']))
                for branch in BRANCHES:
                    b = baselines[eid]
                    y = out['labels']
                    mb, ma = margin(b[branch], y), margin(out[branch], y)
                    pb, pa = b[branch].argmax(1), out[branch].argmax(1)
                    for qi in range(5):
                        prediction_rows.append({'condition': condition, 'episode': eid,
                            'query_index': qi, 'video_id': episodes[eid]['query'][qi]['video_id'],
                            'branch': branch, 'label': int(y[qi]), 'E0_prediction': int(pb[qi]),
                            'prediction': int(pa[qi]), 'E0_margin': float(mb[qi]), 'margin': float(ma[qi]),
                            'correct_to_wrong': bool(pb[qi] == y[qi] and pa[qi] != y[qi]),
                            'wrong_to_correct': bool(pb[qi] != y[qi] and pa[qi] == y[qi]),
                            'E0_attribution': int(attribution(b)[qi]), 'attribution': int(attribution(out)[qi])})
            if not pairs:
                continue
            by_condition[condition] = pairs
            report['conditions'][condition] = summarize(pairs, repeats=a.repeats)
            report['conditions'][condition]['features_and_sampling'] = feature_summary(metadata, a.repeats)
            cases[condition] = choose_cases(pairs, episodes, condition)
    # Contrast mild warps against their paired ordered-random controls on the
    # same episodes, never select a gamma by observed test accuracy.
    for warp, control in [('p075', 'r075'), ('p133', 'r133')]:
        for scope in manifest['scopes']:
            x, z = warp + '_' + scope, control + '_' + scope
            if x not in by_condition or z not in by_condition:
                continue
            xp = {eid: v for eid, _, v in by_condition[x]}
            zp = {eid: v for eid, _, v in by_condition[z]}
            shared = sorted(xp.keys() & zp.keys())
            contrasts = {}
            for branch in ('ts_dcm', 'fusion'):
                diffs = np.concatenate([margin(xp[e][branch], xp[e]['labels']) - margin(zp[e][branch], zp[e]['labels']) for e in shared])
                contrasts[branch] = paired_bootstrap(diffs, np.repeat(shared, 5), repeats=a.repeats)
            report['paired_random_control_contrasts'][x + '_minus_' + z] = contrasts
    report['conclusion'] = ('Real inference completed; separate A representation, B classification, and C localization. '
        'Interpolation and R/W decomposition alone cannot establish semantic preservation or H1b. '
        'Do not recommend speed normalization unless paired effects are stable beyond random controls; '
        'this intervention does not simulate all effects of real execution speed.')
    a.output.mkdir(parents=True)
    write_json(a.output / 'summary.json', report)
    write_json(a.output / 'cases.json', cases)
    if prediction_rows:
        with (a.output / 'query_predictions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    lines = ['# Temporal sampling diagnostic', '', 'Run status: ' + status['status'], '',
             '| E0 branch | Accuracy % [95% CI] |', '|---|---|']
    for branch in BRANCHES:
        b = report['E0'][branch]['accuracy']
        lines.append(f"| {branch} | {b['estimate']*100:.3f} [{b['ci95'][0]*100:.3f}, {b['ci95'][1]*100:.3f}] |")
    lines.extend(['',
             'Accuracy differences below are paired episode bootstrap estimates in percentage points.', '',
             '| Condition | Branch | E0 accuracy | Accuracy | Delta pp [95% CI] |',
             '|---|---|---:|---:|---|'])
    for condition, r in report['conditions'].items():
        for branch in BRANCHES:
            s = r[branch]
            d = s['accuracy_delta']
            lines.append(f"| {condition} | {branch} | {s['baseline_accuracy']['estimate']*100:.3f} | {s['accuracy']['estimate']*100:.3f} | {d['estimate']*100:+.3f} [{d['ci95'][0]*100:+.3f}, {d['ci95'][1]*100:+.3f}] |")
    if not report['conditions']:
        lines.extend(['', 'Only E0 was run; intervention hypotheses remain untested.'])
    lines.extend(['', report['conclusion'], '', 'Repeated videos/shared supports across episodes may retain residual dependence. '
                  'These reused Test episodes are exploratory, not an independent confirmatory holdout.'])
    (a.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(a.output / 'report.md')


if __name__ == '__main__':
    main()
