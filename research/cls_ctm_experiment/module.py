"""CTM-inspired CLS statistics with nearest competing-class discrimination.

This is our video adaptation, not the original CTM or TDM SAM/QAM. No query
features/labels enter statistics; the output is ONE shared episode gate.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Settings:
    channels: int = 256
    residual_strength: float = 0.1
    eps: float = 1e-8


def cls_statistics(support_cls, labels, way, eps=1e-8):
    """[support_video, frame, channel] -> mean, population variance, gap.

Find each class's closest OTHER prototype by full-vector cosine. Average its
channel-wise squared gaps to that competitor, then average across classes.
Exact cosine ties share equal weight, preserving class permutation symmetry.
The gap measures class separation, not within-class variance/reliability.
In particular, 1-shot makes no claim to estimate cross-video intra-class
variance, and temporal coordinates are averaged, not treated as extra shots.
"""
    if support_cls.ndim != 3 or not support_cls.is_floating_point():
        raise ValueError('support CLS must be floating [video,frame,channel]')
    if way < 2 or labels.shape != (len(support_cls),):
        raise ValueError('invalid way or support labels')
    if labels.device != support_cls.device or labels.dtype != torch.long:
        raise ValueError('support labels must be long and on the feature device')
    if not torch.equal(torch.unique(labels), torch.arange(way, device=labels.device)):
        raise ValueError('support labels must be contiguous 0..way-1')
    counts = torch.stack([(labels == c).sum() for c in range(way)])
    if not bool((counts == counts[0]).all()) or not bool((counts > 0).all()):
        raise ValueError('balanced nonempty support classes required')
    if not bool(torch.isfinite(support_cls).all()):
        raise FloatingPointError('nonfinite support CLS')
    video = support_cls.mean(1)
    prototypes = torch.stack([video[labels == c].mean(0) for c in range(way)])
    mean = prototypes.mean(0)
    variance = prototypes.var(0, unbiased=False)
    normalized = F.normalize(prototypes, dim=-1, eps=eps)
    cosine = normalized @ normalized.T
    cosine = cosine.masked_fill(torch.eye(way, device=cosine.device, dtype=torch.bool), -torch.inf)
    nearest = cosine == cosine.max(1, keepdim=True).values
    competitors = nearest.to(prototypes.dtype) / nearest.sum(1, keepdim=True)
    gaps = (prototypes[:, None] - prototypes[None]).square()
    gap = (gaps * competitors[..., None]).sum(1).mean(0)
    return mean, variance, gap


class CLSCTM(nn.Module):
    def __init__(self, settings=Settings()):
        super().__init__()
        if (settings.channels < 1 or not math.isfinite(settings.residual_strength)
                or not 0 <= settings.residual_strength <= 1
                or not math.isfinite(settings.eps) or settings.eps <= 0):
            raise ValueError('invalid CLS-CTM settings')
        self.settings = settings
        c = settings.channels
        self.projector = nn.Sequential(nn.Linear(3*c, c), nn.GELU(), nn.Linear(c, c))
        # Same identity start as the successful historical CTM adaptation.
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def forward(self, support_cls, labels, way):
        if support_cls.shape[-1] != self.settings.channels:
            raise ValueError('CLS channel dimension differs from settings')
        with torch.autocast(device_type=support_cls.device.type, enabled=False):
            mean, variance, gap = cls_statistics(support_cls.float(), labels, way, self.settings.eps)
            summary = torch.cat((mean, variance, gap))
            logits = self.projector(summary)
            mask = self.settings.channels * logits.softmax(-1)
            gate = 1. + self.settings.residual_strength * (mask - 1.)
            if not bool(torch.isfinite(gate).all()):
                raise FloatingPointError('nonfinite episode gate')
        return gate, dict(cls_mean_norm=float(mean.detach().norm()),
                          cross_class_variance_mean=float(variance.detach().mean()),
                          nearest_gap_mean=float(gap.detach().mean()),
                          summary_dimension=3*self.settings.channels)
