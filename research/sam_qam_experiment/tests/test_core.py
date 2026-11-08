import copy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from research.sam_qam_experiment.module import SAMQAM, Settings, representative_scores
from research.sam_qam_experiment.matching import matching


def dcm(channels=8, frames=2):
    from trainers.ts_fsar import TS_DCM
    cfg = SimpleNamespace(DATA=SimpleNamespace(NUM_INPUT_FRAMES=frames),
                          TRAIN=SimpleNamespace(DC_ALPHA=.4))
    return TS_DCM(cfg, channels).eval().requires_grad_(False)


def episode(shot=1, way=3, queries=1, frames=2, channels=8, patches=6):
    g = torch.Generator().manual_seed(121+shot)
    features = torch.randn((way*shot+way*queries)*frames, patches+1, channels, generator=g)
    labels = torch.arange(way).repeat_interleave(shot)
    return features, labels


def oracle_inputs():
    support = ((torch.arange(3*2*64*12).reshape(3,2,64,12) % 37)-18).float()/11
    query = ((torch.arange(3*64*12).reshape(3,64,12) % 29)-14).float()/9
    return support, query


def deterministic_block(block, index):
    with torch.no_grad():
        for offset, layer in enumerate((block.linear1, block.linear2)):
            values = torch.arange(layer.weight.numel()).reshape_as(layer.weight).float()
            layer.weight.copy_(torch.sin(values*.019+index+offset)*.025)
        block.bn.running_mean.copy_(torch.linspace(-.2,.2,block.bn.num_features))
        block.bn.running_var.copy_(torch.linspace(.7,1.3,block.bn.num_features))
        block.bn.weight.copy_(torch.linspace(.8,1.2,block.bn.num_features))
        block.bn.bias.copy_(torch.linspace(-.1,.1,block.bn.num_features))


class SAMQAMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_representativeness_matches_literal_paper_distances(self):
        s = torch.randn(6,3,4,5)
        q = torch.randn(3,3,4,5)
        labels = torch.tensor([2,0,1,0,2,1])
        intra, inter, qi = representative_scores(s,labels,q,3)
        proto = torch.stack([s[labels == c].mean(0) for c in range(3)])
        maps = proto.mean(2)
        for c in range(3):
            for ch in range(4):
                own = ((proto[c,:,ch]-maps[c])**2).mean()
                other = min(((proto[c,:,ch]-maps[j])**2).mean() for j in range(3) if j != c)
                torch.testing.assert_close(intra[c,ch],own)
                torch.testing.assert_close(inter[c,ch],other)
        torch.testing.assert_close(qi,(q-q.mean(2,keepdim=True)).square().mean((1,3)))

    def test_pinned_authors_implementation_golden_weights(self):
        fixture = json.loads((Path(__file__).parent/'author_oracle.json').read_text())
        module = SAMQAM(Settings(channels=64,training_noise=0)).eval()
        for i,name in enumerate(('sam_intra','sam_inter','qam')):
            deterministic_block(getattr(module,name),i)
        s,q = oracle_inputs()
        support = s.reshape(6,64,3,4).permute(0,2,1,3)
        query = q.reshape(3,64,3,4).permute(0,2,1,3)
        with torch.no_grad():
            gate = module(support,torch.arange(3).repeat_interleave(2),query,3)
        # Authors' output is [class,query,channel], ours [query,class,channel].
        expected = torch.tensor(fixture['expected_class_query_channel']).transpose(0,1)
        torch.testing.assert_close(gate,expected,rtol=2e-6,atol=2e-6)

    def test_disabled_path_is_bitwise_original(self):
        for shot in (1,5):
            model = dcm()
            features,labels = episode(shot)
            with torch.no_grad():
                expected = model(features,labels,3,shot,1)[0]
                actual,_ = matching(model,features,labels,3,shot,1,enabled=False)
            self.assertTrue(torch.equal(expected,actual))

    def test_neutral_gates_preserve_original_alpha_d_and_tsm(self):
        for shot in (1,5):
            model = dcm()
            features,labels = episode(shot)
            with torch.no_grad():
                expected = model(features,labels,3,shot,1)[0]
                actual,_ = matching(model,features,labels,3,shot,1,
                    gate_override=torch.ones(3,3,8),query_chunk=2)
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)

    def test_checkpointed_chunks_preserve_scores_gradients_and_adam_update(self):
        model = dcm()
        features,labels = episode(shot=5)
        a = SAMQAM(Settings(channels=8,training_noise=0)).train()
        b = copy.deepcopy(a)
        opta = torch.optim.AdamW(a.parameters(),lr=.00025)
        optb = torch.optim.AdamW(b.parameters(),lr=.00025)
        sa,_ = matching(model,features,labels,3,5,1,a,query_chunk=3)
        sb,_ = matching(model,features,labels,3,5,1,b,query_chunk=1,checkpoint_chunks=True)
        torch.testing.assert_close(sa,sb,rtol=1e-5,atol=1e-5)
        F.cross_entropy(sa,torch.arange(3)).backward()
        F.cross_entropy(sb,torch.arange(3)).backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb)
            self.assertIsNotNone(pa.grad)
            torch.testing.assert_close(pa.grad,pb.grad,rtol=3e-4,atol=3e-5)
        for name in ('sam_intra','sam_inter','qam'):
            self.assertEqual(int(getattr(b,name).bn.num_batches_tracked),1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        opta.step(); optb.step()
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa,pb,rtol=2e-4,atol=2e-5)

    def test_support_permutation_and_query_batch_independence_at_test(self):
        module = SAMQAM(Settings(channels=8,training_noise=0)).eval()
        support = torch.randn(6,2,8,6)
        query = torch.randn(3,2,8,6)
        labels = torch.tensor([0,1,2,0,1,2])
        with torch.no_grad():
            original = module(support,labels,query,3)
            permutation = torch.tensor([5,3,1,0,4,2])
            permuted = module(support[permutation],labels[permutation],query,3)
            alone = module(support,labels,query[:1],3)
        torch.testing.assert_close(original,permuted)
        torch.testing.assert_close(original[:1],alone,rtol=1e-5,atol=1e-6)

    def test_training_noise_is_bounded_and_absent_in_eval(self):
        module = SAMQAM(Settings(channels=8)).train()
        s = torch.randn(3,2,8,6)
        q = torch.randn(3,2,8,6)
        labels = torch.arange(3)
        gates = module(s,labels,q,3)
        self.assertGreaterEqual(float(gates.min()),0.)
        self.assertLessEqual(float(gates.max()),2.)
        module.eval()
        with torch.no_grad():
            a = module(s,labels,q,3); b = module(s,labels,q,3)
        self.assertTrue(torch.equal(a,b))

    def test_real_256_channel_196_patch_eight_frame_forward_backward(self):
        # Actual alpha-D dimensions; only reduce way/query count for CPU cost.
        model = dcm(channels=256,frames=8)
        features,labels = episode(shot=1,way=2,frames=8,channels=256,patches=196)
        module = SAMQAM(Settings(training_noise=0)).train()
        score,detail = matching(model,features,labels,2,1,1,module,checkpoint_chunks=True)
        F.cross_entropy(score,torch.arange(2)).backward()
        self.assertEqual(tuple(score.shape),(2,2))
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in module.parameters()))
        self.assertEqual(detail['gate_shape'],[2,2,256])


if __name__ == '__main__':
    unittest.main()
