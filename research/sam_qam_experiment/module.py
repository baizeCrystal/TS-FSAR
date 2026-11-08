"""Independent implementation of TDM, CVPR 2022, Eq. 2--12 / Table 1.

Video adaptation: keep the eight frame coordinates, form shot-mean prototypes,
and average squared distances over (time, patch). One channel gate is shared
over time. No query labels, query-to-query attention or episode-time optimizer.
Unlike a scalar temporal gate, channel scaling changes alpha-D geometry.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class Settings:
    channels: int = 256
    alpha: float = 0.5
    beta: float = 0.5
    training_noise: float = 0.2


class WeightBlock(nn.Module):
    """Paper Table 1; bias-free linear layers match the authors' implementation."""
    def __init__(self, channels):
        super().__init__()
        self.linear1 = nn.Linear(channels, 2 * channels, bias=False)
        self.bn = nn.BatchNorm1d(2 * channels)
        self.linear2 = nn.Linear(2 * channels, channels, bias=False)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, scores):
        return 1. + torch.tanh(self.linear2(torch.relu(self.bn(self.linear1(scores)))))


def representative_scores(support, labels, query, way):
    """[Ns,T,C,P], [Ns], [Nq,T,C,P] -> paper intra/inter/query scores.

The paper's intra is distance to the channel-mean spatial response, NOT
within-class sample variance. Inter is the MIN distance to another class's
channel-mean response. Original signed LSN features are kept, without ReLU.
"""
    if support.ndim != 4 or query.ndim != 4 or support.shape[1:] != query.shape[1:]:
        raise ValueError('expected support/query [video,time,channel,patch]')
    if labels.shape != (len(support),) or way < 2:
        raise ValueError('invalid support labels/way')
    if not torch.equal(torch.unique(labels), torch.arange(way, device=labels.device)):
        raise ValueError('support labels must be contiguous 0..way-1')
    counts = [(labels == c).sum().item() for c in range(way)]
    if len(set(counts)) != 1:
        raise ValueError('balanced support classes required')
    proto = torch.stack([support[labels == c].mean(0) for c in range(way)])
    # Retain time and patch as separate coordinates until the final reduction.
    mean_map = proto.mean(2)
    distances = torch.stack([
        (proto - mean_map[j][None, :, None, :]).square().mean((1, 3))
        for j in range(way)], dim=1)  # [own_class, reference_class, channel]
    indices = torch.arange(way, device=labels.device)
    intra = distances[indices, indices]
    diagonal = torch.eye(way, device=labels.device, dtype=torch.bool)[..., None]
    inter = distances.masked_fill(diagonal, float('inf')).amin(1)
    query_intra = (query - query.mean(2, keepdim=True)).square().mean((1, 3))
    return intra, inter, query_intra


class SAMQAM(nn.Module):
    def __init__(self, settings=Settings()):
        super().__init__()
        if not (0 <= settings.alpha <= 1 and 0 <= settings.beta <= 1
                and 0 <= settings.training_noise < 1):
            raise ValueError('invalid fixed TDM settings')
        self.settings = settings
        self.sam_intra = WeightBlock(settings.channels)
        self.sam_inter = WeightBlock(settings.channels)
        self.qam = WeightBlock(settings.channels)

    def forward(self, support, labels, query, way):
        if support.shape[2] != self.settings.channels:
            raise ValueError('channel dimension differs from configured TDM')
        # Keep small distance statistics / BN in FP32 on A10.
        with torch.autocast(device_type=support.device.type, enabled=False):
            intra, inter, query_intra = representative_scores(
                support.float(), labels, query.float(), way)
            a, b = self.settings.alpha, self.settings.beta
            ws = a * self.sam_intra(intra) + (1. - a) * self.sam_inter(inter)
            wq = self.qam(query_intra)
            gates = b * ws[None] + (1. - b) * wq[:, None]
            if self.training and self.settings.training_noise:
                # Paper Sec. 4.2 / official add_noise: one draw per gate,
                # before matching checkpoint recomputation, never during test.
                gates = (gates + (2. * torch.rand_like(gates) - 1.)
                         * self.settings.training_noise).clamp(0., 2.)
        return gates
