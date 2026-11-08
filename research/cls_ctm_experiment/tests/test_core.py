import copy
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from research.cls_ctm_experiment.module import CLSCTM, Settings, cls_statistics
from research.cls_ctm_experiment.matching import matching, original_tsm


def episode(shot=1, channels=8, frames=2, patches=6, way=3):
    gen = torch.Generator().manual_seed(71+shot)
    features = torch.randn((way*shot+way)*frames, patches+1, channels, generator=gen)
    labels = torch.arange(way).repeat_interleave(shot)
    from trainers.ts_fsar import TS_DCM
    cfg = SimpleNamespace(DATA=SimpleNamespace(NUM_INPUT_FRAMES=frames),
                          TRAIN=SimpleNamespace(DC_ALPHA=.4))
    model = TS_DCM(cfg, channels).eval().requires_grad_(False)
    return model, features, labels


def activate(module):
    with torch.no_grad():
        module.projector[-1].weight.normal_(0, .03)


class CLSCTMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_statistics_match_literal_cls_and_nearest_competitor_reference(self):
        labels = torch.tensor([2,0,1,0,2,1])
        x = torch.randn(6,8,7)
        mean, var, gap = cls_statistics(x, labels, 3)
        u = torch.stack([x[labels == c].mean(1).mean(0) for c in range(3)])
        torch.testing.assert_close(mean, u.mean(0))
        torch.testing.assert_close(var, u.var(0, unbiased=False))
        reference = []
        for c in range(3):
            other = [j for j in range(3) if j != c]
            similarities = [F.cosine_similarity(u[c],u[j],dim=0) for j in other]
            nearest = other[int(torch.stack(similarities).argmax())]
            reference.append((u[c]-u[nearest]).square())
        torch.testing.assert_close(gap, torch.stack(reference).mean(0))

    def test_single_shot_nonzero_discrimination_identical_classes_and_ties(self):
        x = torch.tensor([[[1.,0.,0.]], [[.8,.2,0.]], [[0.,0.,1.]]])
        _, var, gap = cls_statistics(x, torch.arange(3), 3)
        self.assertGreater(float(var.sum()),0.)
        self.assertGreater(float(gap.sum()),0.)
        _, v, d = cls_statistics(x[:1].expand(3,-1,-1), torch.arange(3), 3)
        torch.testing.assert_close(v,torch.zeros_like(v))
        torch.testing.assert_close(d,torch.zeros_like(d))
        # All cosines tie at zero: average all nearest other classes.
        x = torch.eye(3)[:,None]
        _,_,d = cls_statistics(x,torch.arange(3),3)
        torch.testing.assert_close(d,torch.full((3,),2./3.))

    def test_support_frame_permutation_class_relabel_and_shared_identity_gate(self):
        module = CLSCTM(Settings(channels=8))
        x, labels = torch.randn(6,8,8), torch.tensor([0,1,2,0,1,2])
        gate, _ = module(x,labels,3)
        self.assertTrue(torch.equal(gate,torch.ones(8)))
        activate(module)
        gate,_ = module(x,labels,3)
        permutation = torch.tensor([5,3,1,0,4,2])
        other,_ = module(x[permutation].flip(1), (labels[permutation]+1)%3, 3)
        torch.testing.assert_close(gate,other)
        self.assertAlmostEqual(float(gate.mean()),1.,places=6)
        self.assertGreaterEqual(float(gate.min()),.9-1e-6)

    def test_no_query_or_patch_can_enter_gate_and_tsm_stays_unweighted(self):
        model,x,labels = episode()
        module = CLSCTM(Settings(channels=8)); activate(module)
        seen=[]
        hook=module.register_forward_hook(lambda m,a,out:seen.append(out[0].detach().clone()))
        condition=[]
        h=model.generator.register_forward_pre_hook(lambda m,a: condition.append(a[0].clone()))
        matching(model,x,labels,3,1,1,module)
        y=x.clone(); y[6:]*=17.; y[:6,1:]*=-9.
        matching(model,y,labels,3,1,1,module)
        self.assertTrue(torch.equal(seen[0],seen[1]))
        with torch.no_grad():
            model(x,labels,3,1,1)
        self.assertTrue(torch.equal(condition[0],condition[2]))
        hook.remove(); h.remove()

    def test_disabled_and_neutral_paths_preserve_actual_original_dcm_both_shots(self):
        for shot in (1,5):
            model,x,labels = episode(shot)
            with torch.no_grad():
                expected = model(x,labels,3,shot,1)[0]
                disabled,_ = matching(model,x,labels,3,shot,1,enabled=False)
                neutral,_ = matching(model,x,labels,3,shot,1,CLSCTM(Settings(channels=8)),
                                     feature_frames_per_chunk=3)
            self.assertTrue(torch.equal(disabled,expected))
            torch.testing.assert_close(neutral,expected,rtol=1e-5,atol=1e-5)

    def test_nonuniform_gate_uses_original_alpha_d_and_original_tsm(self):
        from utils.functions import cos_sim, extract_class_indices
        for shot in (1,5):
            model,x,labels=episode(shot)
            gate=torch.linspace(.9,1.1,8)
            actual,_=matching(model,x,labels,3,shot,1,gate_override=gate,feature_frames_per_chunk=3)
            z=x*gate
            p=F.cosine_similarity(z[:,0,None],z[:,1:],dim=-1)[...,None]*z[:,1:]
            a=model.alpha_dc(p.transpose(1,2))
            s=a[:3*shot*2].reshape(3*shot,2,-1)
            proto=torch.stack([s.index_select(0,extract_class_indices(labels,c)).mean(0)
                               for c in torch.unique(labels)])
            r=cos_sim(a[3*shot*2:],proto.flatten(0,1)).reshape(3,2,3,2).permute(0,2,1,3)
            expected=(r.reshape(3,3,4)*original_tsm(model,x,3,shot,1)).sum(-1)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)

    def test_recomputed_chunks_preserve_backward_and_optimizer_update(self):
        model,x,labels=episode(5)
        a=CLSCTM(Settings(channels=8)); activate(a); b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=.00025)
        ob=torch.optim.AdamW(b.parameters(),lr=.00025)
        sa,_=matching(model,x,labels,3,5,1,a,feature_frames_per_chunk=1000)
        sb,_=matching(model,x,labels,3,5,1,b,feature_frames_per_chunk=3,checkpoint_chunks=True)
        torch.testing.assert_close(sa,sb,rtol=1e-5,atol=1e-5)
        F.cross_entropy(sa,torch.arange(3)).backward()
        F.cross_entropy(sb,torch.arange(3)).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.isfinite(pa.grad).all())
            torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-4,atol=3e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        oa.step(); ob.step()
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa,pb,rtol=2e-4,atol=2e-5)

    def test_identity_initialization_learns_then_discriminative_input_receives_gradient(self):
        model,x,labels=episode()
        module=CLSCTM(Settings(channels=8))
        opt=torch.optim.AdamW(module.parameters(),lr=.001)
        for step in range(2):
            opt.zero_grad(set_to_none=True)
            score,_=matching(model,x,labels,3,1,1,module)
            F.cross_entropy(score,torch.arange(3)).backward()
            if step == 1:
                self.assertGreater(float(module.projector[0].weight.grad[:,16:].norm()),0.)
            opt.step()

    def test_real_eight_frame_256_channel_196_patch_backward(self):
        model,x,labels=episode(channels=256,frames=8,patches=196,way=2)
        module=CLSCTM()
        score,detail=matching(model,x,labels,2,1,1,module,checkpoint_chunks=True)
        F.cross_entropy(score,torch.arange(2)).backward()
        self.assertEqual(tuple(score.shape),(2,2))
        self.assertEqual(detail['gate_shape'],[256])
        self.assertTrue(torch.isfinite(score).all())
        self.assertGreater(float(module.projector[-1].weight.grad.norm()),0.)

    def test_rejects_nonfinite_inputs_and_unbalanced_labels(self):
        x=torch.randn(4,2,8)
        with self.assertRaises(ValueError):
            cls_statistics(x,torch.tensor([0,0,0,1]),2)
        x[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):
            cls_statistics(x,torch.tensor([0,0,1,1]),2)


if __name__=='__main__':
    unittest.main()
