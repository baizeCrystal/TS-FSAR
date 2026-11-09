"""Fresh joint E0/E1 training; original optimizer, CE and attached global KL."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import torch
from .common import (ROOT,configure,prepare,load_model,implementation_hash,canonical,
    sha_file,write_new,identity_file,atomic_save,output_lock,cpu_state,rng_state,
    restore_rng,preserve_prefix,train_loader)
from .core import Observe,original_optimizer
from .evaluation import evaluate


def should_evaluate(step):
    return step>=2500 and step%500==0


def base_fingerprint(model,trainable=False):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if 'ctm_generator' not in n and (p.requires_grad==trainable):
            h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def amp_step(model,optimizer,scaler,images,support,target,real,embeddings,observer,offload=True):
    optimizer.zero_grad(set_to_none=True)
    observer.clear()
    # Offload saved activations without detaching any objective/parameter path.
    context=torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
    with context,torch.cuda.amp.autocast():
        result=model(images,support,target,real,embeddings)
        loss=result['loss']
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite original total loss')
    with torch.no_grad(),torch.cuda.amp.autocast():
        components={k:float(v) for k,v in observer.losses(model,target,real,embeddings).items()}
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norms={}
    for tag in ('side','Adapter','Linear','up_linear','generator','ctm_generator'):
        gs=[p.grad for n,p in model.named_parameters() if tag in n and p.grad is not None]
        norms[tag]=float(torch.stack([g.float().square().sum() for g in gs]).sum().sqrt()) if gs else 0.
    old=scaler.get_scale()
    # Caller retains the original pre-optimizer Test-selection ordering.
    return loss,result,components,norms,old


def run(args):
    model,embeddings,bank,report,raw=load_model(args,args.variant)
    optimizer,groups=original_optimizer(model,model.cfg)
    total=10000
    scaler=torch.cuda.amp.GradScaler(init_scale=128.)
    from utils.functions import cosine_lr
    schedule=cosine_lr(optimizer,0,total)
    frozen=base_fingerprint(model)
    initial_trainable=base_fingerprint(model,True)
    seed_rng=bank['rng']
    del bank
    torch.cuda.empty_cache()
    identity=dict(method='CLS-CTM-joint',variant=args.variant,shot=args.shot,seed=0,
        implementation_sha256=implementation_hash(),initialization_sha256=report['sha256'],
        manifest_sha256=sha_file(args.manifest),effective_config=raw,optimizer_groups=groups,
        steps=total,amp_initial_scale=128.,activation_offload='all saved tensors to CPU',
        selection='official Novel-Test fusion best; steps 2500,3000,...,10000; before optimizer step',
        smoke=args.smoke)
    identity_file(args.output/'identity.json',identity)
    identity_file(args.output/'optimizer_groups.json',dict(groups=groups))
    start=0;updates=0;best=None;elapsed=0.;sequence=hashlib.sha256()
    final=args.output/'training_complete.json'
    if final.exists():
        done=json.loads(final.read_text())
        if sha_file(args.output/'model_best.pth')!=done['best_checkpoint_sha256']:
            raise RuntimeError('completed checkpoint changed')
        print('JOINT_CTM_ALREADY_COMPLETE',args.output,flush=True);return
    latest=args.output/'resume_latest.pth'
    if args.resume:
        s=torch.load(latest,map_location='cpu',weights_only=True)
        if s['identity']!=identity:raise RuntimeError('resume identity mismatch')
        model.load_state_dict(s['model'],strict=True)
        optimizer.load_state_dict(s['optimizer']);scaler.load_state_dict(s['scaler'])
        start=s['step'];updates=s['successful_updates'];best=s['best'];elapsed=s['elapsed_seconds']
        for name in ('train_trace.jsonl','train_episodes.jsonl'):preserve_prefix(args.output/name,start)
        for line in (args.output/'train_episodes.jsonl').read_text().splitlines():
            sequence.update(json.loads(line)['episode_id'].encode())
        if sequence.hexdigest()!=s['train_sequence_sha256']:raise RuntimeError('resume episode prefix changed')
        restore_rng(s['rng']);del s
    else:
        if latest.exists() or (args.output/'train_trace.jsonl').exists():
            raise RuntimeError('existing output; use --resume')
        restore_rng(seed_rng)
    model.train();observer=Observe(model);started=time.perf_counter();window=[]
    count=4 if args.smoke else total
    cfg=model.cfg
    loader=train_loader(cfg,0,start,count-start,args.workers)
    try:
        with (args.output/'train_trace.jsonl').open('a') as trace,(args.output/'train_episodes.jsonl').open('a') as ledger:
            for step,batch in enumerate(loader,start=start+1):
                observer.clear();torch.cuda.reset_peak_memory_stats()
                images=torch.cat((batch['support_set'],batch['target_set'])).cuda()
                support=batch['support_labels'].long().cuda();target=batch['target_labels'].long().cuda()
                real=torch.cat((batch['real_support_labels'],batch['real_target_labels'])).long().cuda()
                retry=0
                while True:
                    loss,result,components,norms,scale=amp_step(model,optimizer,scaler,images,
                        support,target,real,embeddings['train_embeddings'],observer)
                    training_gate=dict(model.ts_dcm.last_gate)
                    # Exact original chronology: validate after backward, BEFORE this update.
                    if not args.smoke and should_evaluate(step):
                        folder=args.output/f'selection/step{step:05d}'
                        if folder.exists():
                            raise RuntimeError('selection evidence exists past resume point; preserve and inspect '+str(folder))
                        metrics=evaluate(model,embeddings['test_embeddings'],args,folder)
                        selection=dict(metrics,selection_forward_episode=step,
                            successful_updates_before_validation=updates,checkpoint_selection_split='Novel-Test')
                        print('TEST_SELECTION',canonical(selection),flush=True)
                        if best is None or metrics['accuracy_percent']['fusion']>best['accuracy_percent']['fusion']:
                            best=dict(selection,predictions_directory=str(folder))
                            atomic_save(dict(identity=identity,model=cpu_state(model),selection=best),args.output/'model_best.pth')
                    scaler.step(optimizer);scaler.update()
                    effective=scaler.get_scale()>=scale
                    if effective:updates+=1
                    if not args.smoke or effective:break
                    retry+=1
                    if retry>=12:raise RuntimeError('GPU smoke has no finite effective update after AMP backoff')
                optimizer.zero_grad(set_to_none=True)
                record=dict(batch['episode_record'],step=step)
                sequence.update(record['episode_id'].encode())
                row=dict(step=step,losses=dict(components,total=float(loss.detach())),matching_accuracy=result['acc'],
                    learning_rates=[g['lr'] for g in optimizer.param_groups],effective_update=effective,
                    successful_updates=updates,amp_scale=scaler.get_scale(),gradient_norms=norms,
                    gate=training_gate,train_episode_id=record['episode_id'],
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
                trace.write(canonical(row)+'\n');trace.flush();ledger.write(canonical(record)+'\n');ledger.flush()
                window.append(row['losses']['total'])
                if step%100==0 or args.smoke:
                    print('JOINT_CTM_TRAIN',canonical(dict(row,mean_total_loss=sum(window)/len(window),
                        elapsed_seconds=elapsed+time.perf_counter()-started)),flush=True);window.clear()
                observer.clear()
                del loss,result,images,support,target,real,batch
                schedule(step-1)  # released loop calls scheduler(cur_iter) AFTER the optimizer
                # Same numbers, built-in floats: weights_only can restore them safely.
                for group in optimizer.param_groups:group['lr']=float(group['lr'])
                if step%500==0 or step==count:
                    atomic_save(dict(identity=identity,model=cpu_state(model),optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict(),step=step,successful_updates=updates,best=best,
                        rng=rng_state(),elapsed_seconds=elapsed+time.perf_counter()-started,
                        train_sequence_sha256=sequence.hexdigest()),latest)
    finally:
        observer.close()
    if frozen!=base_fingerprint(model):raise RuntimeError('frozen CLIP weights changed')
    if initial_trainable==base_fingerprint(model,True):raise RuntimeError('original trainable model did not update')
    if args.smoke:
        if updates<4:raise RuntimeError('smoke must establish four effective joint updates')
        required=('side','Adapter','Linear','up_linear','generator') + (('ctm_generator',) if args.variant=='E1' else ())
        if any(not 0 < norms[k] < float('inf') for k in required):raise RuntimeError('missing/nonfinite joint gradient: '+str(norms))
        metrics=evaluate(model,embeddings['test_embeddings'],args,args.output/'smoke_evaluation',3)
        write_new(args.output/'smoke_passed.json',dict(passed=True,successful_updates=updates,
            frozen_clip_unchanged=True,gradient_norms=norms,evaluation=metrics))
        print('JOINT_CTM_SMOKE_PASSED',args.output,flush=True);return
    selected=torch.load(args.output/'model_best.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'],strict=True)
    evaluation=evaluate(model,embeddings['test_embeddings'],args,args.output/'formal',10000)
    if evaluation['accuracy_percent']!=best['accuracy_percent']:
        raise RuntimeError('saved selected checkpoint does not replay selection accuracy exactly')
    write_new(final,dict(passed=True,identity=identity,best=best,formal=evaluation,
        successful_updates=updates,train_sequence_sha256=sequence.hexdigest(),
        frozen_clip_unchanged=True,best_checkpoint_sha256=sha_file(args.output/'model_best.pth')))
    print('JOINT_CTM_TRAINING_COMPLETE',final,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('prepare','train'),default='train')
    p.add_argument('--variant',choices=('E0','E1'),default='E1')
    p.add_argument('--shot',type=int,choices=(1,5),required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--bank',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    for n in ('data_root','manifest','bank','output'):
        if getattr(args,n) is not None:setattr(args,n,getattr(args,n).resolve())
    os.chdir(ROOT)
    if args.workers<0:p.error('workers must be nonnegative')
    if args.phase=='train' and args.bank is None:p.error('--bank required for training')
    args.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output).free<8*1024**3:raise RuntimeError('need at least 8 GiB free on /data')
    with output_lock(args.output):
        prepare(args) if args.phase=='prepare' else run(args)


if __name__=='__main__':main()
