"""Native full-loss/gradient tests with small synthetic features; no CLIP download."""
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from trainers.ts_fsar import TS_FSAR,TS_DCM
from utils.functions import DistillKL,cosine_lr
from research.cls_ctm_joint_experiment.core import JointModel,JointDCM,Observe,original_optimizer
from research.cls_ctm_joint_experiment.train import should_evaluate
from research.cls_ctm_joint_experiment.launch import allocations


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.side=nn.Linear(12,5*8)
        self.clip=nn.Linear(12,6)
        self.Adapter=nn.Linear(6,6)
        self.up_ln=nn.LayerNorm(8);self.up_linear=nn.Linear(8,6,bias=False)
        self.Linear=nn.Linear(36,7,bias=False);self.dropout=nn.Dropout(0.)

    def forward(self,x):
        flat=x.flatten(1)
        return self.side(flat).reshape(len(x),5,8),self.clip(flat)


def make(shot,enabled):
    torch.manual_seed(42)
    cfg=NS(CTM_JOINT=NS(ENABLE=enabled),DATA=NS(NUM_INPUT_FRAMES=2),
        TRAIN=NS(RANDOM_SEED=0,WAY=3,SHOT=shot,QUERY_PER_CLASS=2,QUERY_PER_CLASS_TEST=1,
            DC_ALPHA=.4,VIS_ALPHA=1.,CLS_ALPHA=2.,SIDE_CLS_ALPHA=.1,KD_ALPHA=.1,KD_T=4),
        SOLVER=NS(LSN_LR=.00025,LSNHEAD_LR=.00025,GENERATOR_LR=.00025,
                  ADAPTER_ALPHA=1.,WEIGHT_DECAY=.1,OPTIM_METHOD='adamw'),
        TEST=NS(FUSE_CLIP=.2,FUSE_VISION=.4,FUSE_SIDE=.4),TEMP_CLIP=5.,TEMP_SIDE=5.)
    model=JointModel.__new__(JointModel);nn.Module.__init__(model)
    model.cfg=cfg;model.model=Encoder();model.ts_dcm=JointDCM(cfg,8)
    model.ce=nn.CrossEntropyLoss();model.kl=DistillKL(4)
    original_optimizer(model,cfg)
    g=torch.Generator().manual_seed(17)
    images=torch.randn((3*shot+6)*2,3,2,2,generator=g)
    support=torch.arange(3).repeat_interleave(shot)
    query=torch.arange(3).repeat_interleave(2)
    real=torch.arange(3*shot+6)%7
    text=F.normalize(torch.randn(7,6,generator=g),dim=-1)
    return model,(images,support,query,real,text)


class JointTests(unittest.TestCase):
    def test_zero_gate_preserves_native_all_losses_and_existing_gradients_both_shots(self):
        for shot in (1,5):
            e0,args=make(shot,False);e1,_=make(shot,True)
            # Added construction neither changes existing weights nor consumes RNG.
            missing,unexpected=e1.load_state_dict(e0.state_dict(),strict=False)
            self.assertTrue(all('ctm_generator' in n for n in missing));self.assertFalse(unexpected)
            o0=Observe(e0);o1=Observe(e1)
            a=e0(*args);b=e1(*args)
            self.assertTrue(torch.equal(a['loss'],b['loss']))
            for key in ('score','adc','cls','student','clip','side','kl'):
                self.assertTrue(torch.equal(o0.value[key],o1.value[key]),key)
            a['loss'].backward();b['loss'].backward()
            for n,p in e0.named_parameters():
                q=dict(e1.named_parameters())[n]
                if p.grad is None:self.assertIsNone(q.grad)
                else:torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-6,msg=n)
            self.assertGreater(float(e1.ts_dcm.ctm_generator.projector[-1].weight.grad.norm()),0.)
            o0.close();o1.close()

    def test_nonuniform_gate_keeps_global_ce_kl_raw_inputs_and_original_tsm(self):
        e0,args=make(5,False);e1,_=make(5,True)
        e1.load_state_dict(e0.state_dict(),strict=False)
        with torch.no_grad():e1.ts_dcm.ctm_generator.projector[-1].bias.copy_(torch.linspace(-1.,1.,8))
        o0=Observe(e0);o1=Observe(e1);conditions=[]
        hs=[m.ts_dcm.generator.register_forward_pre_hook(lambda m,a:conditions.append(a[0].clone())) for m in (e0,e1)]
        e0(*args);e1(*args)
        for k in ('adc','cls','student','clip','side','kl'):
            self.assertTrue(torch.equal(o0.value[k],o1.value[k]),k)
        self.assertTrue(torch.equal(*conditions));self.assertFalse(torch.equal(o0.value['score'],o1.value['score']))
        for h in hs:h.remove()
        o0.close();o1.close()

    def test_original_total_and_attached_teacher_kl_and_joint_parameter_updates(self):
        e1,args=make(1,True);o=Observe(e1)
        optimizer,groups=original_optimizer(e1,e1.cfg)
        result=e1(*args)
        values=o.losses(e1,args[2],args[3],args[4]);c=e1.cfg.TRAIN
        expected=c.VIS_ALPHA*values['task']+c.CLS_ALPHA*values['student_ce']+c.SIDE_CLS_ALPHA*values['side_ce']+c.KD_ALPHA*(values['kl']+values['teacher_ce'])
        torch.testing.assert_close(result['loss'],expected,rtol=2e-6,atol=2e-6)
        teacher=torch.autograd.grad(values['kl'],tuple(e1.model.Adapter.parameters()),retain_graph=True)
        self.assertGreater(sum(float(g.norm()) for g in teacher),0.)
        before={n:p.detach().clone() for n,p in e1.named_parameters()}
        result['loss'].backward();optimizer.step()
        for tag in ('side','Adapter','up_linear','Linear','generator','ctm_generator'):
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in e1.named_parameters() if tag in n),tag)
        self.assertTrue(all(torch.equal(p,before[n]) for n,p in e1.named_parameters() if not p.requires_grad))
        self.assertEqual(len(groups),4)
        gen=next(g for g in groups if 'ts_dcm.generator.weight' in g['names'])
        self.assertTrue(all(n in gen['names'] for n in before if 'ctm_generator' in n))
        self.assertEqual(optimizer.defaults['eps'],1e-6)
        o.close()

    def test_e0_and_switch_off_use_actual_original_dcm_and_clean_hooks(self):
        for shot in (1,5):
            model,args=make(shot,True)
            raw=model.model(args[0])[0]
            native=TS_DCM(model.cfg,8);native.load_state_dict({'generator.weight':model.ts_dcm.generator.weight})
            model.ts_dcm.enabled=False
            expected=native(raw,args[1],3,shot,2)
            actual=model.ts_dcm(raw,args[1],3,shot,2)
            for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b))
            model.ts_dcm.enabled=True
            model.ts_dcm(raw,args[1],3,shot,2)
            self.assertEqual(len(model.ts_dcm.generator._forward_pre_hooks),0)

    @unittest.skipUnless(torch.cuda.is_available(),'campus CUDA check')
    def test_cuda_offload_keeps_joint_amp_gradients_and_updates(self):
        from research.cls_ctm_joint_experiment.train import amp_step
        for shot in (1,5):
            a,args=make(shot,True);a.cuda();b=copy.deepcopy(a)
            args=tuple(t.cuda() for t in args)
            oa,_=original_optimizer(a,a.cfg);ob,_=original_optimizer(b,b.cfg)
            va=Observe(a);vb=Observe(b)
            sa=torch.cuda.amp.GradScaler(init_scale=1.)
            sb=torch.cuda.amp.GradScaler(init_scale=1.)
            for _ in range(3):
                la,_,_,_,_=amp_step(a,oa,sa,*args,va,offload=False)
                lb,_,_,_,_=amp_step(b,ob,sb,*args,vb,offload=True)
                torch.testing.assert_close(la,lb,rtol=0,atol=0)
                for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
                    if p.grad is not None:
                        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=n)
                sa.step(oa);sa.update();sb.step(ob);sb.update()
                for p,q in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(p,q,rtol=0,atol=0)
            va.close();vb.close()

    def test_actual_official_per_shot_configs_and_loss_weights_are_retained(self):
        import tempfile
        from pathlib import Path
        from research.cls_ctm_joint_experiment.common import configure
        with tempfile.TemporaryDirectory() as temp:
            for shot,lr,kd in ((1,.00025,.1),(5,.0003,.01)):
                cfg,raw,_=configure(shot,Path(temp),Path(temp),2,'E1',skip_pca=True)
                self.assertEqual(cfg.SOLVER.LSN_LR,lr)
                self.assertEqual(cfg.SOLVER.LSNHEAD_LR,lr)
                self.assertEqual(cfg.SOLVER.GENERATOR_LR,lr)
                self.assertEqual(cfg.TRAIN.KD_ALPHA,kd)
                self.assertEqual(cfg.TRAIN.SIDE_CLS_ALPHA,kd)
                self.assertEqual(cfg.TRAIN.CLS_ALPHA,2.)
                self.assertEqual(cfg.TRAIN.QUERY_PER_CLASS,5)
                self.assertEqual(cfg.TRAIN.QUERY_PER_CLASS_TEST,1)
                self.assertEqual(len(cfg.TEST.CLASS_NAME),24)
                self.assertEqual(cfg.TRAIN.SHOT,shot)
                self.assertTrue(cfg.CTM_JOINT.ENABLE)

    def test_full_joint_checkpoint_resumes_optimizer_and_rng_with_weights_only(self):
        import tempfile
        from pathlib import Path
        from research.cls_ctm_joint_experiment.common import atomic_save,rng_state,restore_rng
        a,args=make(1,True);oa,_=original_optimizer(a,a.cfg)
        a(*args)['loss'].backward();oa.step();oa.zero_grad(set_to_none=True)
        schedule=cosine_lr(oa,0,10000);schedule(1)
        for group in oa.param_groups:group['lr']=float(group['lr'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'resume.pth'
            atomic_save(dict(model=a.state_dict(),optimizer=oa.state_dict(),rng=rng_state()),path)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            restore_rng(saved['rng']);expected=torch.rand(5)
            restore_rng(saved['rng']);self.assertTrue(torch.equal(expected,torch.rand(5)))
            b,_=make(1,True);ob,_=original_optimizer(b,b.cfg)
            b.load_state_dict(saved['model']);ob.load_state_dict(saved['optimizer'])
            a(*args)['loss'].backward();b(*args)['loss'].backward();oa.step();ob.step()
            for pa,pb in zip(a.parameters(),b.parameters()):
                torch.testing.assert_close(pa,pb,rtol=0,atol=0)

    def test_training_schedule_and_allocations(self):
        self.assertEqual([i for i in range(1,10001) if should_evaluate(i)],list(range(2500,10001,500)))
        self.assertEqual(allocations('1','4'),[(1,4)])
        self.assertEqual(allocations('1,5','4,5'),[(1,4),(5,5)])
        model,_=make(1,False);optimizer,_=original_optimizer(model,model.cfg)
        schedule=cosine_lr(optimizer,0,10000)
        initial=optimizer.param_groups[0]['lr'];used=[]
        for step in range(1,4):used.append(optimizer.param_groups[0]['lr']);schedule(step-1)
        self.assertEqual(used[:2],[initial,initial]);self.assertLess(used[2],initial)


if __name__=='__main__':unittest.main()
