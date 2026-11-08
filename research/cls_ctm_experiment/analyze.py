"""Episode-paired statistics for the two frozen-E0 CLS-CTM experiments."""
import argparse
import csv
import json
from pathlib import Path
from research.ps_ds_subspace_experiment.analyze import (
    validate_rows, paired_statistics, hard_negatives, describe)
from research.ps_ds_subspace_experiment.protocol import write_new


def summarize(folder, repetitions):
    header, complete, rows = validate_rows(folder)
    if header.get('method') != 'CLS-CTM-discriminative' or header.get('preflight_module'):
        raise ValueError('not a trained CLS-CTM comparison')
    result = dict(shot=header['shot'], episodes=len(rows), formal=len(rows)==10000,
        identity=header, matching=paired_statistics(rows,'matching',repetitions),
        fusion=paired_statistics(rows,'fusion',repetitions),
        hard_negative={b:hard_negatives(rows,b) for b in ('matching','fusion')},
        e0_episode_seconds=describe([r['e0_seconds'] for r in rows]),
        extra_episode_seconds=describe([r['diagnostics']['total_additional_seconds'] for r in rows]),
        gate={k:describe([r['diagnostics'][k] for r in rows]) for k in
              ('gate_min','gate_max','gate_mean','gate_std','gate_mean_absolute_deviation_from_one','nearest_gap_mean')}
              if not header['method_disabled'] else {},
        score_scale={k:describe([r['scale'][k]['std'] for r in rows])
                     for k in ('e0_logits','e1_logits')})
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shot1', type=Path)
    p.add_argument('--shot5', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('choose a fresh analysis directory')
    selected = [(shot,path) for shot,path in ((1,args.shot1),(5,args.shot5)) if path is not None]
    if not selected:
        p.error('provide --shot1 and/or --shot5')
    results = [summarize(path,args.bootstrap) for _,path in selected]
    if [r['shot'] for r,_ in results] != [shot for shot,_ in selected]:
        raise ValueError('shot order mismatch')
    if len({r['episodes'] for r,_ in results}) != 1:
        raise ValueError('do not mix exploratory and formal results')
    if len({r['identity']['implementation_sha256'] for r,_ in results}) != 1:
        raise ValueError('implementations differ')
    args.output.mkdir(parents=True)
    table = ['| 设置 | E0 Matching | E1 CLS-CTM Matching | E0 Fusion | E1 CLS-CTM Fusion |',
             '| --- | ---: | ---: | ---: | ---: |']
    details = []
    for result, rows in results:
        shot = result['shot']; m = result['matching']; f = result['fusion']
        write_new(args.output/f'shot{shot}_statistics.json',result)
        table.append(f"| {shot}-shot | {m['e0_accuracy_percent']:.3f}% | {m['e1_accuracy_percent']:.3f}% | {f['e0_accuracy_percent']:.3f}% | {f['e1_accuracy_percent']:.3f}% |")
        details.append(f"{shot}-shot: Matching Δ={m['delta_pp']:.3f} pp, 95% CI {m['ci95_pp']}; "
            f"Fusion Δ={f['delta_pp']:.3f} pp, 95% CI {f['ci95_pp']}. "
            f"纠正/新增错误：Matching {m['corrected']}/{m['introduced']}，Fusion {f['corrected']}/{f['introduced']}。 "
            f"额外秒/Episode={result['extra_episode_seconds']['mean']:.4f}。")
        with (args.output/f'shot{shot}_prediction_changes.csv').open('x') as stream:
            writer = csv.writer(stream)
            writer.writerow(['episode_id','query_video','true_global','branch','e0_global','e1_global','transition'])
            for row in rows:
                for branch in ('matching','fusion'):
                    for i,y in enumerate(row['target']):
                        a = row['e0_'+branch]['prediction'][i]
                        b = row['e1_'+branch]['prediction'][i]
                        transition = ('both_correct' if a==y and b==y else 'corrected' if b==y
                                      else 'introduced' if a==y else 'both_wrong')
                        writer.writerow([row['episode_id'],row['query_video_ids'][i],row['class_order'][y],
                            branch,row['class_order'][a],row['class_order'][b],transition])
    text = '\n'.join(table)+'\n'
    (args.output/'accuracy_table.md').write_text(text)
    formal = all(r['formal'] for r,_ in results)
    (args.output/'experiment_report.md').write_text(
        ('正式固定 10,000 Episode 配对评测。' if formal else '探索评测，不作为最终成绩。')
        +'\n\n'+text+'\n'+'\n\n'.join(details)
        +'\n\n冻结各 shot 对应的 E0，仅在 64 类 Meta-Train 上学习 CLS-CTM；固定训练预算取最终模块，不以 Test 选模块。'
        +'\n\nCLS-CTM 为 CTM 启发的 CLS 均值/方差加最近竞争类判别统计，不是官方 CTM 或 TDM 的复现。CI 按 Episode 配对重采样，不包含训练种子及历史 E0 选模不确定性。'
        +'\n\nFusion 是主要指标，Matching 为辅助指标。保持官方乘积融合公式及每个 shot 对应的固定系数。'
        +'\n\nTest 仅用于本次既定比较，不搜索通道权重、超参数或融合系数。\n')
    print(text)
    print('\n\n'.join(details))


if __name__ == '__main__':
    main()
