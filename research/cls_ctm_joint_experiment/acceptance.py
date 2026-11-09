"""Real GPU identity/gradient acceptance before either formal training run."""
import gc
import json
import torch
from .common import load_model,train_loader,restore_rng,write_new
from .core import Observe,original_optimizer
from .train import amp_step


def run(args):
    destination=args.output/'acceptance.json'
    if destination.exists():raise RuntimeError('acceptance already exists; do not overwrite')
    reference=None
    for variant in ('E0','E1'):
        model,embeddings,bank,report,raw=load_model(args,variant)
        optimizer,groups=original_optimizer(model,model.cfg)
        batch=next(iter(train_loader(model.cfg,0,0,1,args.workers)))
        images=torch.cat((batch['support_set'],batch['target_set'])).cuda()
        support=batch['support_labels'].long().cuda();target=batch['target_labels'].long().cuda()
        real=torch.cat((batch['real_support_labels'],batch['real_target_labels'])).long().cuda()
        model.train();observer=Observe(model)
        scale=128. if reference is None else reference['scale']
        for retry in range(12):
            scaler=torch.cuda.amp.GradScaler(init_scale=scale)
            restore_rng(bank['rng'])
            loss,result,components,norms,_=amp_step(model,optimizer,scaler,images,
                support,target,real,embeddings['train_embeddings'],observer)
            finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            if finite:break
            if reference is not None:raise RuntimeError('E1 has nonfinite gradients at accepted E0 AMP scale')
            scale/=2
        else:raise RuntimeError('no finite real reference gradients after AMP backoff')
        values=dict(loss=loss.detach().cpu(),score=observer.value['score'].detach().cpu(),
            adc=observer.value['adc'].detach().cpu(),cls=observer.value['cls'].detach().cpu(),
            student=observer.value['student'].detach().cpu(),clip=observer.value['clip'].detach().cpu(),
            side=observer.value['side'].detach().cpu())
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()
               if p.grad is not None and 'ctm_generator' not in n}
        if reference is None:
            reference=dict(values=values,gradients=grads,scale=scale,episode_id=batch['episode_record']['episode_id'],
                           rng=bank['rng'],initialization_sha256=report['sha256'])
        else:
            if batch['episode_record']['episode_id']!=reference['episode_id']:
                raise RuntimeError('acceptance episode mismatch')
            for name in values:
                torch.testing.assert_close(values[name],reference['values'][name],rtol=0,atol=0)
            if grads.keys()!=reference['gradients'].keys():raise RuntimeError('gradient coverage differs')
            for name in grads:
                torch.testing.assert_close(grads[name],reference['gradients'][name],rtol=3e-3,atol=3e-5,msg=name)
            if norms['ctm_generator']<=0:raise RuntimeError('new output projector has no gradient')
            before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
            scaler.step(optimizer);scaler.update()
            changed={tag:any(not torch.equal(before[n],p.detach().cpu()) for n,p in model.named_parameters()
                        if n in before and tag in n)
                     for tag in ('side','Adapter','Linear','up_linear','generator','ctm_generator')}
            if not all(changed.values()):raise RuntimeError('joint update missing: '+str(changed))
            write_new(destination,dict(passed=True,episode_id=reference['episode_id'],
                initialization_sha256=report['sha256'],finite_common_amp_scale=scale,
                native_loss_and_auxiliary_inputs_exact=True,existing_gradients_close=True,
                all_joint_groups_updated=changed,activation_offload_verified=True))
        observer.close()
        del model,embeddings,bank,optimizer,observer,batch,images,support,target,real,loss,result,grads,values,scaler
        gc.collect();torch.cuda.empty_cache()
    print('JOINT_CTM_ACCEPTANCE_PASSED',destination,flush=True)


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from .common import ROOT,output_lock
    import os
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('data-root','manifest','bank','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--shot',type=int,choices=(1,5),required=True)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();os.chdir(ROOT)
    for n in ('data_root','manifest','bank','output'):setattr(args,n,getattr(args,n).resolve())
    with output_lock(args.output):run(args)
