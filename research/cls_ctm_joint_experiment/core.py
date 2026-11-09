"""Restore the historical metric-CTM intervention and all original objectives.

Only matching uses gated tokens. Global ADC CE/KL and side-text CE receive
ungated features; the TSM condition uses ungated CLS, but learns jointly.
"""
from unittest.mock import patch
import torch
import torch.nn.functional as F
import trainers.ts_fsar as production
from trainers.ts_fsar import TS_DCM, TS_FSAR
from research.cls_ctm_experiment.module import CLSCTM, Settings
from research.ps_ds_subspace_experiment.matching import validate_original


class JointDCM(TS_DCM):
    def __init__(self, cfg, hidden_dim):
        super().__init__(cfg, hidden_dim)
        validate_original(self)
        self.enabled = bool(cfg.CTM_JOINT.ENABLE)
        self.last_gate = {}
        if self.enabled:
            # Preserve the original model/head and dropout RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg.TRAIN.RANDOM_SEED)
                self.ctm_generator = CLSCTM(Settings(channels=hidden_dim))

    def forward(self, features, labels, way, shot, queries):
        self.last_gate = {}
        if not self.enabled:
            return super().forward(features, labels, way, shot, queries)
        t = self.cfg.DATA.NUM_INPUT_FRAMES
        raw_cls, patch_tokens = features[:,0], features[:,1:]
        support = raw_cls[:way*shot*t].reshape(way*shot,t,-1)
        gate, detail = self.ctm_generator(support, labels, way)
        weighted = F.cosine_similarity(raw_cls[:,None], patch_tokens, dim=-1)[...,None]*patch_tokens
        raw_adc = self.alpha_dc(weighted.transpose(1,2))
        # Match the historical AMP product/cast; auxiliary paths retain raw data.
        enhanced = (features.float()*gate[None,None]).to(features.dtype)
        s = support.mean(1).mean(0).reshape(1,1,-1).repeat(way*queries,way,1)
        q = raw_cls[way*shot*t:].reshape(way*queries,t,-1).mean(1)[:,None].repeat(1,way,1)
        condition = s+q
        hook = self.generator.register_forward_pre_hook(lambda m,a: (condition,)+a[1:],prepend=True)
        try:
            score, _, _ = super().forward(enhanced, labels, way, shot, queries)
        finally:
            hook.remove()
        g = gate.detach()
        self.last_gate = dict(detail, gate_min=float(g.min()),gate_max=float(g.max()),
            gate_mean=float(g.mean()),gate_mean_absolute_deviation_from_one=float((g-1).abs().mean()))
        return score, raw_adc, raw_cls


class JointModel(TS_FSAR):
    def __init__(self, cfg, vision):
        # Scoped construction only; no persistent mutation to original modules.
        with patch.object(production,'TS_DCM',JointDCM):
            super().__init__(cfg,vision)


def original_optimizer(model, cfg):
    from utils.optimizer import construct_optimizer
    optimizer = construct_optimizer(model,cfg)
    names = {id(p):n for n,p in model.named_parameters()}
    groups = [dict(lr=g['lr'],weight_decay=g.get('weight_decay'),
                   names=[names[id(p)] for p in g['params']]) for g in optimizer.param_groups]
    added = [n for n,p in model.named_parameters() if 'ctm_generator' in n]
    if added:
        group = next(g for g in groups if 'ts_dcm.generator.weight' in g['names'])
        if not set(added) <= set(group['names']):
            raise RuntimeError('CLS-CTM must belong to the original generator optimizer group')
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for g in optimizer.param_groups for p in g['params']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise RuntimeError('optimizer coverage mismatch')
    return optimizer,groups


class Observe:
    """Observe native objective inputs/outputs without replacing any formula."""
    def __init__(self, model):
        self.value = {}
        def dcm(m,a,o): self.value.update(score=o[0],adc=o[1],cls=o[2])
        def linear(m,a,o): self.value['student']=o
        def kl(m,a,o): self.value['kl']=o
        def adapter(m,a,o): self.value['clip']=o.mean(1)
        def side(m,a,o): self.value['side']=o.mean(1)
        self.handles=[model.ts_dcm.register_forward_hook(dcm),
            model.model.Linear.register_forward_hook(linear),model.kl.register_forward_hook(kl),
            model.model.Adapter.register_forward_hook(adapter),model.model.up_linear.register_forward_hook(side)]

    def losses(self, model, target, real, embeddings):
        v = self.value
        clip = F.normalize(v['clip'],dim=-1)@embeddings.T*model.cfg.TEMP_CLIP
        side = F.normalize(v['side'],dim=-1)@embeddings.T*model.cfg.TEMP_SIDE
        return dict(task=F.cross_entropy(v['score'],target),
            student_ce=F.cross_entropy(v['student'],real),
            teacher_ce=F.cross_entropy(clip,real),side_ce=F.cross_entropy(side,real),kl=v['kl'])

    def clear(self): self.value.clear()

    def close(self):
        for h in self.handles: h.remove()
        self.clear()
