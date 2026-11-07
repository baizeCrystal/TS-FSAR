"""Strict paired episode statistics; no selection or calibration of the method."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .protocol import read_records, sha_file, write_new, canonical


def bootstrap(deltas, repetitions=10000, seed=61007):
    if repetitions < 5000:
        raise ValueError('at least 5000 episode-level paired bootstrap draws required')
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    # One vector entry = ALL queries within an episode, never query-level resampling.
    for start in range(0, repetitions, 100):
        indices = rng.integers(0,len(values),(min(100,repetitions-start),len(values)))
        draws.extend(values[indices].mean(1).tolist())
    return dict(delta_pp=float(values.mean()*100),
                ci95_pp=(np.quantile(draws,[.025,.975])*100).tolist(),
                repetitions=repetitions, unit='episode', seed=seed)


def paired_statistics(rows, branch, repetitions=10000):
    a = np.array([r['e0_'+branch]['correct'] for r in rows], dtype=bool)
    b = np.array([r['e1_'+branch]['correct'] for r in rows], dtype=bool)
    if a.shape != (len(rows),5) or b.shape != a.shape:
        raise ValueError('unbalanced/non-official query counts')
    ci = bootstrap(b.mean(1)-a.mean(1),repetitions)
    return dict(e0_accuracy_percent=float(a.mean()*100), e1_accuracy_percent=float(b.mean()*100),
                both_correct=int((a&b).sum()), both_wrong=int((~a&~b).sum()),
                corrected=int((~a&b).sum()), introduced=int((a&~b).sum()), **ci)


def validate_rows(folder):
    header = json.loads((folder/'run_identity.json').read_text())
    complete = json.loads((folder/'completion.json').read_text())
    path = folder/'paired_predictions.jsonl'
    if not complete['passed'] or sha_file(path) != complete['predictions_sha256']:
        raise ValueError('incomplete/corrupt run')
    rows = read_records(path)
    if len(rows) != header['episodes'] or len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('missing/duplicated episodes')
    for i,r in enumerate(rows):
        if r['episode_index'] != i or not r['original_forward_parity']:
            raise ValueError('episode order/parity mismatch')
        target = np.array(r['target'])
        if sorted(target.tolist()) != list(range(5)):
            raise ValueError('invalid official query labels')
        for branch in ['e0_matching','e1_matching','e0_fusion','e1_fusion']:
            score = np.array(r[branch]['scores'])
            if score.shape != (5,5) or not np.isfinite(score).all():
                raise ValueError('invalid saved score')
            pred = score.argmax(1)
            if pred.tolist() != r[branch]['prediction'] or (pred==target).tolist() != r[branch]['correct']:
                raise ValueError('saved prediction/correctness differs from scores')
    return header, complete, rows


def hard_negatives(rows, branch):
    """Predeclared hardest wrong class under E0, no threshold/episode selection.

    For E0 errors this equals its predicted confuser. Track whether E1 raises the
    true-class score above that SAME confuser, and whether final argmax is correct.
    """
    stats = dict(e0_errors=0, corrected=0, beaten_original_confuser=0, switched_to_another_error=0)
    pairs = {}
    for row in rows:
        for q,y in enumerate(row['target']):
            s0 = np.array(row['e0_'+branch]['scores'][q])
            s1 = np.array(row['e1_'+branch]['scores'][q])
            wrong = s0.copy(); wrong[y] = -np.inf
            hard = int(wrong.argmax())
            if int(s0.argmax()) == y:
                continue
            stats['e0_errors'] += 1
            correct = int(s1.argmax()) == y
            beaten = bool(s1[y] > s1[hard])
            stats['corrected'] += int(correct)
            stats['beaten_original_confuser'] += int(beaten)
            stats['switched_to_another_error'] += int(not correct and int(s1.argmax()) != hard)
            key = (row['class_order'][y],row['class_order'][hard])
            pair = pairs.setdefault(key,dict(e0_errors=0,corrected=0,beaten_original_confuser=0))
            pair['e0_errors'] += 1; pair['corrected'] += int(correct); pair['beaten_original_confuser'] += int(beaten)
    return dict(definition=hard_negatives.__doc__, **stats,
                class_pairs=[dict(true_global=k[0],confuser_global=k[1],**v) for k,v in sorted(pairs.items())])


def describe(values):
    x = np.asarray(values,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),
                min=float(x.min()),max=float(x.max()))


def analyze_folder(folder, repetitions):
    header, complete, rows = validate_rows(folder)
    result = dict(shot=header['shot'], episodes=len(rows), formal=len(rows)==10000,
                  method_disabled=header['method_disabled'], identity=header,
                  matching=paired_statistics(rows,'matching',repetitions),
                  fusion=paired_statistics(rows,'fusion',repetitions),
                  hard_negative={b:hard_negatives(rows,b) for b in ['matching','fusion']},
                  e0_episode_seconds=describe([r['e0_seconds'] for r in rows]))
    if not header['method_disabled']:
        ds = [r['diagnostics'] for r in rows]
        result.update(extra_episode_seconds=describe([d['total_additional_seconds'] for d in ds]),
                      template_fit_seconds=describe([d['fit_seconds'] for d in ds]),
                      orthogonality_error=describe([d['max_orthogonality_error'] for d in ds]),
                      rank=header['settings']['rank'],
                      min_sample_effective_rank=min(min(d['sample_effective_rank']) for d in ds),
                      ps_objective_increased_episodes=sum(d['ps_objective_increased'] for d in ds),
                      ds_objective_increased_episodes=sum(d['ds_objective_increased'] for d in ds))
    result['scale'] = {key:describe([r['scale'][key]['std'] for r in rows])
                       for key in ['e0_logits','e1_logits']}
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shot1',type=Path,required=True,help='e.g. runs/shot1/formal')
    p.add_argument('--shot5',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='fresh report directory')
    p.add_argument('--bootstrap',type=int,default=10000)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('choose a fresh output path; never overwrite analysis')
    results = [analyze_folder(path,args.bootstrap) for path in [args.shot1,args.shot5]]
    if [r['shot'] for r,_ in results] != [1,5]:
        raise ValueError('shot order mismatch')
    if results[0][0]['episodes'] != results[1][0]['episodes']:
        raise ValueError('do not mix formal and exploratory reports')
    if results[0][0]['identity']['implementation_sha256'] != results[1][0]['identity']['implementation_sha256']:
        raise ValueError('shots used different implementations')
    args.output.mkdir(parents=True)
    table = ['| 设置 | E0 Matching | E1 PS+DS Matching | E0 Fusion | E1 PS+DS Fusion |',
             '| --- | ---: | ---: | ---: | ---: |']
    report = []
    for result,rows in results:
        shot=result['shot']; m=result['matching']; f=result['fusion']
        table.append(f"| {shot}-shot | {m['e0_accuracy_percent']:.3f}% | {m['e1_accuracy_percent']:.3f}% | {f['e0_accuracy_percent']:.3f}% | {f['e1_accuracy_percent']:.3f}% |")
        write_new(args.output/f'shot{shot}_statistics.json',result)
        report.append(f"{shot}-shot: Matching Δ={m['delta_pp']:.3f} pp, 95% CI {m['ci95_pp']}; "
                      f"Fusion Δ={f['delta_pp']:.3f} pp, 95% CI {f['ci95_pp']}. "
                      f"纠正/新增错误：Matching {m['corrected']}/{m['introduced']}，Fusion {f['corrected']}/{f['introduced']}。")
        with (args.output/f'shot{shot}_prediction_changes.csv').open('w') as out:
            writer=csv.writer(out)
            writer.writerow(['episode_id','query_video','true_global','branch','e0_global','e1_global','transition'])
            for row in rows:
                for branch in ['matching','fusion']:
                    for q,y in enumerate(row['target']):
                        a=row['e0_'+branch]['prediction'][q]; b=row['e1_'+branch]['prediction'][q]
                        transition=('both_correct' if a==y and b==y else 'corrected' if b==y else
                                    'introduced' if a==y else 'both_wrong')
                        writer.writerow([row['episode_id'],row['query_video_ids'][q],row['class_order'][y],
                                         branch,row['class_order'][a],row['class_order'][b],transition])
    text='\n'.join(table)+'\n'
    (args.output/'accuracy_table.md').write_text(text)
    formal=all(r['formal'] for r,_ in results)
    (args.output/'experiment_report.md').write_text(
        ('正式固定 10,000 Episode 配对评测。' if formal else '探索结果，不能作为最终正式成绩。')
        +'\n\n'+text+'\n'+'\n\n'.join(report)
        +'\n\nCI 是 Episode 配对 percentile bootstrap；表示固定 checkpoint 下的 Episode 抽样不确定性，不包含训练种子/checkpoint 选择不确定性。'
        +'\n\nHard Negative：由 E0 分数定义最强错误候选，详见 statistics.json；没有用标签校准融合权重。'
        +'\n\n历史 best 成绩和当前固定 Manifest 的采样流不同，不要求重得完全相同的小数。结论只基于本次配对结果；没有独立确认集，也不做因果归因。\n')
    print(text)


if __name__=='__main__':
    main()
