"""Optional signed TSM residual; inputs contain no query labels or text."""
import torch
from torch import nn
from utils.functions import cos_sim


def support_condition(support_adc, support_labels, class_order):
    """[support,T,D] -> [class,4,T]; original cosine denominator is norm product+.01.

    The experiment is explicitly 1-shot. Never silently average videos before
    computing support-to-support correlations if called with another shot.
    """
    if len(class_order) < 2:
        raise ValueError('Support condition needs at least two classes')
    frames = []
    for label in class_order:
        selected = support_adc[support_labels == label]
        if len(selected) != 1:
            raise ValueError('TSM residual support condition requires one video per class')
        frames.append(selected[0])
    a = torch.stack(frames)
    n, t, d = a.shape
    r = cos_sim(a.reshape(n*t, d), a.reshape(n*t, d))
    r = r.reshape(n, t, n, t).permute(0, 2, 1, 3)
    # Preserve j; reduce the other video's time u before symmetric class k.
    off_diagonal = ~torch.eye(n, device=a.device, dtype=torch.bool)
    other = r[off_diagonal].reshape(n, n-1, t, t)
    m, v = other.mean(-1), other.amax(-1)
    return torch.stack((m.mean(1), m.amax(1), v.mean(1), v.amax(1)), dim=1)


class TSMResidual(nn.Module):
    def __init__(self, mode, conditioned):
        super().__init__()
        self.conditioned = bool(conditioned)
        self.mode = mode
        if mode == 'pointwise':
            layers = [nn.Conv2d(6, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 1),
                      nn.GELU(), nn.Conv2d(64, 1, 1)]
        elif mode == 'neighborhood':
            layers = [nn.Conv2d(6, 32, 1), nn.GELU(), nn.Conv2d(32, 16, 3, padding=1),
                      nn.GELU(), nn.Conv2d(16, 1, 1)]
        else:
            raise ValueError('Unknown residual mode: '+mode)
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, correlation, original_weight, condition=None):
        q, n, tq, ts = correlation.shape
        weight = original_weight.expand(q, n, tq, ts)
        if self.conditioned:
            if condition is None or condition.shape != (n, 4, ts):
                raise ValueError('Condition must be [class,4,support_time]')
            bc = condition[None, :, :, None, :].expand(q, n, 4, tq, ts)
        else:
            bc = correlation.new_zeros((q, n, 4, tq, ts))
        channels = torch.cat((correlation.unsqueeze(2), weight.unsqueeze(2), bc), dim=2)
        return self.network(channels.reshape(q*n, 6, tq, ts)).reshape(q, n, tq, ts)
