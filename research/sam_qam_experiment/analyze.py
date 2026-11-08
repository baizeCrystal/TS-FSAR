"""Episode-paired statistics for the two frozen-E0 SAM/QAM experiments."""
import argparse
import csv
import json
from pathlib import Path
from research.ps_ds_subspace_experiment.analyze import (
    validate_rows, paired_statistics, hard_negatives, describe)
from research.ps_ds_subspace_experiment.protocol import write_new


def summarize(folder, repetitions):
    header, complete, rows = validate_rows(folder)
    if header.get('method') != 'TDM-SAM-QAM-video' or header.get('preflight_module'):
        raise ValueError('not a trained SAM/QAM comparison')
    result = dict(shot=header['shot'], episodes=len(rows), formal=len(rows)==10000,
        identity=header, matching=paired_statistics(rows,'matching',repetitions),
        fusion=paired_statistics(rows,'fusion',repetitions),
        hard_negative={b:hard_negatives(rows,b) for b in ('matching','fusion')},
        e0_episode_seconds=describe([r['e0_seconds'] for r in rows]),
        extra_episode_seconds=describe([r['diagnostics']['total_additional_seconds'] for r in rows]),
        gate={k:describe([r['diagnostics'][k] for r in rows]) for k in
              ('gate_min','gate_max','gate_mean','gate_std','class_gate_difference','query_gate_difference')}
              if not header['method_disabled'] else {},
        score_scale={k:describe([r['scale'][k]['std'] for r in rows])
                     for k in ('e0_logits','e1_logits')})
    return result, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shot1', type=Path, required=True)
    p.add_argument('--shot5', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('choose a fresh analysis directory')
    results = [summarize(path,args.bootstrap) for path in (args.shot1,args.shot5)]
    if [r['shot'] for r,_ in results] != [1,5]:
        raise ValueError('shot order mismatch')
    if len({r['episodes'] for r,_ in results}) != 1:
        raise ValueError('do not mix exploratory and formal results')
    if len({r['identity']['implementation_sha256'] for r,_ in results}) != 1:
        raise ValueError('implementations differ')
    args.output.mkdir(parents=True)
    table = ['| 设置 | E0 Matching | E1 SAM+QAM Matching | E0 Fusion | E1 SAM+QAM Fusion |',
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
        +'\n\n冻结各 shot 对应的 E0，仅在 64 类 Meta-Train 上学习 SAM/QAM；固定训练预算取最终模块，不以 Test 选模块。'
        +'\n\nSAM/QAM 为 TDM 向视频 alpha-D 匹配的迁移，不是原版图像分类实验。CI 按 Episode 配对重采样，不包含训练种子及历史 E0 选模不确定性。'
        +'\n\nTest 仅用于本次既定比较，不搜索通道权重、超参数或融合系数。\n')
    print(text)
    print('\n\n'.join(details))


if __name__ == '__main__':
    main()
