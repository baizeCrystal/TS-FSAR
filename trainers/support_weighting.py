"""Parameter-free, video-CLS support aggregation for the original TSM generator."""
import csv
import json
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import utils.logging as logging


def class_prototypes(support, labels):
    classes, inverse = labels.unique(sorted=True, return_inverse=True)
    total = support.new_zeros(len(classes), support.shape[-1]).index_add(0, inverse, support)
    count = torch.bincount(inverse, minlength=len(classes)).to(support.dtype)
    return total / count[:, None]


def discriminate(support, strength, eps):
    normalized = F.normalize(support, dim=-1, eps=eps)
    similarity = normalized @ normalized.T
    n = len(support)
    similarity = similarity.masked_fill(torch.eye(n, dtype=torch.bool, device=support.device), 0.)
    if n == 1:
        delta = torch.zeros_like(support)
    else:
        delta = (similarity.sum(1, keepdim=True)*support - similarity @ support)/(n-1)
    return support + strength*delta, similarity, delta


class SupportWeightingStatistics:
    """Buffer on GPU; copy/write one small batch per logging interval."""
    names = ('support_interclass_cosine', 'query_support_cosine_mean',
             'alpha_b1_entropy', 'alpha_b2_entropy', 'alpha_b1_max', 'alpha_b2_max',
             'alpha_b1_l1_from_uniform', 'alpha_b2_l1_from_uniform',
             'support_relative_norm_change', 'prototype_b1_b2_cosine',
             'weighted_aggregate_relative_change', 'uniform_diff_cancellation_relative',
             'accuracy')

    def __init__(self, period, directory, mode):
        self.period=period; self.directory=Path(directory); self.mode=mode
        self.pending={}; self.total={}; self.blocks={}; self.last_phase=None

    @torch.no_grad()
    def update(self, diagnostics, prediction, labels, training):
        phase='train' if training else 'eval'
        if self.last_phase != phase:
            if self.last_phase is not None: self.flush(self.last_phase)
            self.blocks[phase]=self.blocks.get(phase,0)+1
            self.last_phase=phase
        self.total[phase]=self.total.get(phase,0)+1
        values=torch.cat((diagnostics['metrics'],(prediction==labels).float().mean()[None],
                          diagnostics['query_similarity_mean'],diagnostics['alpha_b1_mean'],
                          diagnostics['alpha_b2_mean'],diagnostics['alpha_b1_first'],diagnostics['alpha_b2_first']))
        self.pending.setdefault(phase,[]).append((self.total[phase],self.blocks[phase],values.detach()))
        if len(self.pending[phase])>=self.period: self.flush(phase)

    def flush(self, phase):
        batch=self.pending.pop(phase,[])
        if not batch:return
        values=torch.stack([item[2] for item in batch]).float().cpu()
        n=(values.shape[1]-len(self.names))//5
        rows=[]
        for (episode,block,_),value in zip(batch,values.tolist()):
            row=dict(zip(self.names,value[:len(self.names)]))
            row.update(episode_total=episode,phase_block=block,phase=phase,mode=self.mode)
            for i,key in enumerate(('query_support_cosine_per_class','alpha_b1_mean_per_class',
                                    'alpha_b2_mean_per_class','alpha_b1_first_query','alpha_b2_first_query')):
                start=len(self.names)+i*n;row[key]=json.dumps(value[start:start+n])
            rows.append(row)
        self.directory.mkdir(parents=True,exist_ok=True)
        path=self.directory/f'support_weighting_{phase}.csv'
        with path.open('a',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            if stream.tell()==0:writer.writeheader()
            writer.writerows(rows)
        means=values[:,:len(self.names)].mean(0).tolist()
        report=dict(zip(self.names,means));report.update(mode=self.mode,phase=phase,
            window_episodes=len(batch),episodes_total=batch[-1][0],phase_block=batch[-1][1])
        # Confusion groups use measured episode similarity, never labels to construct weights.
        bins=[(-1.,0.),(0.,.25),(.25,.5),(.5,.75),(.75,1.000001)]
        report['accuracy_by_support_similarity']={}
        for lo,hi in bins:
            keep=(values[:,0]>=lo)&(values[:,0]<hi)
            if keep.any():report['accuracy_by_support_similarity'][f'[{lo},{min(hi,1.)})']={
                'episodes':int(keep.sum()),'accuracy':float(values[keep,len(self.names)-1].mean())}
        logging.get_logger(__name__).info('[TSM_SUPPORT_STATS] %s',json.dumps(report,sort_keys=True))

    def flush_all(self):
        for phase in list(self.pending):self.flush(phase)


class DiscriminativeSupportAggregation(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.mode=getattr(cfg,'TSM_SUPPORT_AGGREGATION','mean')
        if self.mode not in ('query_softmax','team_diff_query_softmax'):
            raise ValueError('This module requires query_softmax or team_diff_query_softmax')
        self.temperature=float(getattr(cfg,'TSM_SUPPORT_TEMPERATURE',.2))
        self.strength=float(getattr(cfg,'TSM_SUPPORT_DIFF_LAMBDA',.5))
        self.eps=float(getattr(cfg,'TSM_SUPPORT_EPS',1e-8))
        if not math.isfinite(self.temperature) or self.temperature<=0:raise ValueError('TSM_SUPPORT_TEMPERATURE must be finite and positive')
        if not math.isfinite(self.strength) or self.strength<0:raise ValueError('TSM_SUPPORT_DIFF_LAMBDA must be finite and nonnegative')
        if not math.isfinite(self.eps) or self.eps<=0:raise ValueError('TSM_SUPPORT_EPS must be finite and positive')
        period=int(getattr(cfg,'TSM_SUPPORT_LOG_PERIOD',0))
        if period<0:raise ValueError('TSM_SUPPORT_LOG_PERIOD must be nonnegative')
        self.statistics=SupportWeightingStatistics(period,cfg.OUTPUT_DIR,self.mode) if period else None
        self.last_diagnostics=None

    def forward(self,query,support,support_labels,return_details=False):
        # FP32 cosine/softmax and difference accumulation even under AMP.
        # These are all differentiable; no prediction-path tensor is detached.
        with torch.autocast(device_type=query.device.type,enabled=False):
            q=query.float();s=class_prototypes(support.float(),support_labels)
            disc,similarity,delta=discriminate(s,self.strength,self.eps)
            qn=F.normalize(q,dim=-1,eps=self.eps)
            r1=qn@F.normalize(s,dim=-1,eps=self.eps).T
            r2=qn@F.normalize(disc,dim=-1,eps=self.eps).T
            a1=F.softmax(r1/self.temperature,dim=1)
            a2=F.softmax(r2/self.temperature,dim=1)
            weighted1=a1@s;weighted2=a2@disc
            p1=q+weighted1;p2=q+weighted2
            result=p1 if self.mode=='query_softmax' else p2
            if self.statistics is not None:
                self._record(q,s,disc,similarity,delta,r1,a1,a2,weighted1,weighted2,p1,p2)
        # Keep the baseline's final-vector dtype and magnitude convention: no L2 normalization.
        result=result.to(query.dtype)
        if return_details:
            return result,{'support':s,'enhanced':disc,'similarity':similarity,'delta':delta,
                'raw_query_similarity':r1,'enhanced_query_similarity':r2,'alpha_b1':a1,'alpha_b2':a2,
                'weighted_b1':weighted1,'weighted_b2':weighted2,'prototype_b1':p1,'prototype_b2':p2}
        return result

    @torch.no_grad()
    def _record(self,q,s,disc,similarity,delta,r1,a1,a2,w1,w2,p1,p2):
        n=len(s);tiny=self.eps
        ent=lambda a:-(a*a.clamp_min(torch.finfo(a.dtype).tiny).log()).sum(1).mean()
        offdiag=similarity.sum()/max(n*(n-1),1)
        metrics=torch.stack((offdiag,r1.mean(),ent(a1),ent(a2),a1.max(1).values.mean(),
            a2.max(1).values.mean(),(a1-1/n).abs().sum(1).mean(),(a2-1/n).abs().sum(1).mean(),
            ((disc.norm(dim=-1)-s.norm(dim=-1))/s.norm(dim=-1).clamp_min(tiny)).mean(),
            F.cosine_similarity(p1,p2,dim=-1,eps=tiny).mean(),
            ((w2-w1).norm(dim=-1)/w1.norm(dim=-1).clamp_min(tiny)).mean(),
            (self.strength*delta.mean(0)).norm()/s.mean(0).norm().clamp_min(tiny)))
        self.last_diagnostics={'metrics':metrics.detach(),'query_similarity_mean':r1.mean(0).detach(),
            'alpha_b1_mean':a1.mean(0).detach(),'alpha_b2_mean':a2.mean(0).detach(),
            'alpha_b1_first':a1[0].detach(),'alpha_b2_first':a2[0].detach()}

    def record_accuracy(self,prediction,labels,training):
        if self.statistics is not None:
            self.statistics.update(self.last_diagnostics,prediction,labels,training)
