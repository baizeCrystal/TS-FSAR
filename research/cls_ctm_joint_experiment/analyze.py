"""Pair the independently joint-trained, fusion-selected E0/E1 checkpoints."""
import argparse
import json
import numpy as np
from pathlib import Path
from research.ps_ds_subspace_experiment.protocol import sha_file,read_records,write_new
from research.ps_ds_subspace_experiment.analyze import paired_statistics,hard_negatives


def summarize(root):
    headers=[];branches=[]
    for variant in ('E0','E1'):
        base=root/variant
        complete=json.loads((base/'training_complete.json').read_text())
        if not complete['passed'] or sha_file(base/'model_best.pth')!=complete['best_checkpoint_sha256']:
            raise RuntimeError('incomplete/changed '+variant+' checkpoint')
        identity=complete['identity'];headers.append(identity)
        if identity['variant']!=variant or identity['method']!='CLS-CTM-joint' or identity['smoke']:
            raise RuntimeError('wrong training variant/protocol')
        folder=base/'formal'
        report=json.loads((folder/'completion.json').read_text())
        path=folder/'predictions.jsonl'
        if not report['passed'] or report['episodes']!=10000 or sha_file(path)!=report['predictions_sha256']:
            raise RuntimeError('incomplete/changed '+variant+' predictions')
        rows=read_records(path)
        if len(rows)!=10000:raise RuntimeError('expected 10000 episodes')
        branches.append(rows)
    for key in ('shot','seed','initialization_sha256','manifest_sha256','implementation_sha256'):
        if headers[0][key]!=headers[1][key]:raise RuntimeError('unpaired '+key)
    c0=json.loads((root/'E0/training_complete.json').read_text())
    c1=json.loads((root/'E1/training_complete.json').read_text())
    if c0['train_sequence_sha256']!=c1['train_sequence_sha256']:
        raise RuntimeError('training episode sequences differ')
    paired=[]
    for i,(a,b) in enumerate(zip(*branches)):
        for key in ('episode_id','target','pixels_sha256','class_order','query_video_ids','support_video_ids'):
            if a[key]!=b[key]:raise RuntimeError('unpaired episode '+key)
        if a['episode_index']!=i or b['episode_index']!=i:raise RuntimeError('episode order differs')
        for row in (a,b):
            if not row['native_forward_parity'] or sorted(row['target'])!=list(range(5)):
                raise RuntimeError('native parity or official Query labels invalid')
            for name in ('matching','fusion'):
                saved=row[name];scores=np.asarray(saved['scores'])
                if scores.shape!=(5,5) or not np.isfinite(scores).all():raise RuntimeError('invalid scores')
                pred=scores.argmax(1)
                if pred.tolist()!=saved['prediction'] or (pred==row['target']).tolist()!=saved['correct']:
                    raise RuntimeError('saved prediction disagrees with scores')
        r=dict(a)
        for name in ('matching','fusion'):
            r['e0_'+name]=a[name];r['e1_'+name]=b[name]
        paired.append(r)
    result=dict(shot=headers[0]['shot'],episodes=10000,matching=paired_statistics(paired,'matching'),
        fusion=paired_statistics(paired,'fusion'),
        hard_negative={n:hard_negatives(paired,n) for n in ('matching','fusion')},
        initialization_sha256=headers[0]['initialization_sha256'],
        train_sequence_sha256=c0['train_sequence_sha256'],
        selected_steps={v:c['best']['selection_forward_episode'] for v,c in [('E0',c0),('E1',c1)]})
    out=root/'analysis';out.mkdir(exist_ok=False)
    write_new(out/'statistics.json',result)
    m=result['matching'];f=result['fusion']
    table=('| 设置 | E0 Matching | E1 CLS-CTM Joint Matching | E0 Fusion | E1 CLS-CTM Joint Fusion |\n'
           '| --- | ---: | ---: | ---: | ---: |\n'
           f"| {result['shot']}-shot | {m['e0_accuracy_percent']:.3f}% | {m['e1_accuracy_percent']:.3f}% | {f['e0_accuracy_percent']:.3f}% | {f['e1_accuracy_percent']:.3f}% |\n")
    details='\n'.join(f"{name}: Δ={result[name]['delta_pp']:+.3f} pp，95% CI={result[name]['ci95_pp']}，纠正/新增={result[name]['corrected']}/{result[name]['introduced']}。" for name in ('fusion','matching'))
    (out/'experiment_report.md').write_text(table+'\n'+details+
        '\n\n双方从相同 CLIP/PCA 初始化联合训练，使用相同训练 Episode 和固定官方 Novel-Test Replay。'
        '原版全部 CE、attached global KL、教师 Adapter 及四组优化器保留。'
        '从 2500 起每 500 次在 10,000 个 Novel-Test Episodes 上以 Fusion 选模，之后重放同一 Manifest。'
        '这是按要求使用 Test 选模的比较，不是独立确认集；配对 CI 不包含训练种子和选模不确定性。\n')
    print(table+'\n'+details,flush=True)
    print('JOINT_CTM_ANALYSIS_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    summarize(p.parse_args().root)
