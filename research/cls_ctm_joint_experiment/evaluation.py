"""Full native three-branch inference on the fixed official Novel-Test replay."""
import time
import torch
import torch.nn.functional as F
from research.ps_ds_subspace_experiment.episodes import Replay,stream
from research.ps_ds_subspace_experiment.evaluate import verify_manifest_bundle
from research.ps_ds_subspace_experiment.protocol import canonical,write_new,sha_file
from .core import Observe


def evaluate(model,embeddings,args,folder,count=10000):
    records,_,audit=verify_manifest_bundle(args.manifest,args.shot,0)
    audited={r['episode_id']:r['pixels_sha256'] for r in audit['records']}
    folder.mkdir(parents=True,exist_ok=False)
    observer=Observe(model)
    training=model.training
    rng=torch.get_rng_state(),torch.cuda.get_rng_state_all()
    model.eval()
    counts={k:0 for k in ('matching','clip','side','fusion')}
    queries=0;loss=0.;started=time.perf_counter()
    path=folder/'predictions.jsonl'
    try:
        with torch.no_grad(),path.open('x') as output:
            for i,item in enumerate(stream(Replay(model.cfg,records[:count]),args.workers,0)):
                row=item['record']
                if row['episode_id'] in audited and item['pixels_sha256']!=audited[row['episode_id']]:
                    raise RuntimeError('frame preprocessing differs from audited official replay')
                support=torch.tensor([v['episode_label'] for v in row['support']],device='cuda')
                target=torch.tensor([v['episode_label'] for v in row['query']],device='cuda')
                # Official class order and support-label mapping, no Query labels in the model's scorer.
                from utils.functions import extract_class_indices
                real=torch.tensor([v['class_id'] for v in row['support']],device='cuda')
                ep_embeddings=torch.stack([torch.mean(torch.index_select(embeddings[real],0,
                    extract_class_indices(support,c)),dim=0) for c in torch.unique(support)])
                observer.clear()
                result=model(item['images'].cuda(),support,target,None,ep_embeddings)
                v=observer.value
                clip=v['clip'];side=v['side']
                cp=((clip/clip.norm(dim=-1,keepdim=True))@ep_embeddings.T*model.cfg.TEMP_CLIP).softmax(1)
                sp=((side/side.norm(dim=-1,keepdim=True))@ep_embeddings.T*model.cfg.TEMP_SIDE).softmax(1)
                fusion=(v['score'].softmax(1).pow(model.cfg.TEST.FUSE_VISION)
                        *cp.pow(model.cfg.TEST.FUSE_CLIP)*sp.pow(model.cfg.TEST.FUSE_SIDE))
                torch.testing.assert_close(F.nll_loss(torch.log(fusion+1e-8),target),result['loss'],atol=0,rtol=0)
                if float((fusion.argmax(1)==target).float().mean())!=result['acc']:
                    raise RuntimeError('native fusion parity failed')
                entry=dict(episode_index=i,episode_id=row['episode_id'],target=target.tolist(),
                    pixels_sha256=item['pixels_sha256'],class_order=row['class_order'],
                    query_video_ids=[v['video_id'] for v in row['query']],
                    support_video_ids=[v['video_id'] for v in row['support']],
                    native_forward_parity=True,gate=model.ts_dcm.last_gate)
                for name,score in [('matching',v['score']),('clip',cp),('side',sp),('fusion',fusion)]:
                    if not bool(torch.isfinite(score).all()):
                        raise FloatingPointError('nonfinite '+name)
                    pred=score.argmax(1)
                    correct=pred==target
                    counts[name]+=int(correct.sum())
                    entry[name]=dict(scores=score.tolist(),prediction=pred.tolist(),correct=correct.tolist())
                queries+=len(target);loss+=float(result['loss'])
                output.write(canonical(entry)+'\n');output.flush()
                if (i+1)%500==0:
                    print('JOINT_CTM_EVAL_PROGRESS',folder.name,i+1,'/',count,flush=True)
                observer.clear()
        report=dict(passed=True,episodes=count,queries=queries,
            accuracy_percent={k:100*v/queries for k,v in counts.items()},loss=loss/count,
            predictions_sha256=sha_file(path),manifest_sha256=sha_file(args.manifest),
            seconds=time.perf_counter()-started,precision='FP32 native official inference',
            protocol='official 24 Novel-Test classes; fixed paired replay; 5-way; 8 frames; 1 query/class')
        write_new(folder/'completion.json',report)
        return report
    finally:
        observer.close()
        model.train(training)
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
