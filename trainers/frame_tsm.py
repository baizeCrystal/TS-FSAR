"""Frame residual weights in TS-FSAR's [query, support, query frame, support frame] layout."""
import math
import json
import torch
from torch import nn
import utils.logging as logging


class EMatchStatistics:
    """Detached, query-weighted diagnostics; one device-to-host transfer per log interval."""
    names = ('alpha_max', 'alpha_entropy', 'alpha_top1_top2_gap',
             'initial_prediction_correct', 'initial_score_margin')

    def __init__(self, period, temperature, selection):
        self.period = period
        self.temperature = temperature
        self.selection = selection
        self.pending = {}

    @torch.no_grad()
    def update(self, initial, support_labels, query_labels, training):
        # Labels enter only this analysis, after matching and enhancement finished.
        score = initial.detach().float()
        prob = torch.softmax(score / self.temperature, dim=1)
        k = min(2, score.shape[1])
        top_prob = prob.topk(k, dim=1).values
        top_score = score.topk(k, dim=1).values
        prediction = support_labels[score.argmax(1)]
        values = torch.stack((top_prob[:, 0],
            -(prob * prob.clamp_min(torch.finfo(prob.dtype).tiny).log()).sum(1),
            top_prob[:, 0] - top_prob[:, -1],
            (prediction == query_labels).float(),
            top_score[:, 0] - top_score[:, -1]), dim=1).sum(0)
        phase = 'train' if training else 'eval'
        row = self.pending.setdefault(phase, {'sum': torch.zeros_like(values), 'queries': 0, 'episodes': 0})
        row['sum'].add_(values)
        row['queries'] += score.shape[0]
        row['episodes'] += 1
        if row['episodes'] >= self.period:
            report = dict(zip(self.names, (row['sum'] / row['queries']).cpu().tolist()))
            report.update(phase=phase, selection=self.selection,
                          episodes=row['episodes'], queries=row['queries'])
            logging.get_logger(__name__).info('[EMATCH_STATS] %s', json.dumps(report, sort_keys=True))
            del self.pending[phase]


class FrameMatchingGenerator(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.residual_scale = float(getattr(cfg, 'FRAME_RESIDUAL_SCALE', 1.0))
        self.temperature = float(getattr(cfg, 'EMATCH_TEMPERATURE', 1.0))
        self.detach_weight = getattr(cfg, 'EMATCH_DETACH_WEIGHT', True)
        self.selection = getattr(cfg, 'EMATCH_SELECTION', 'hard')
        if self.selection not in ('hard', 'soft'):
            raise ValueError('EMATCH_SELECTION must be hard or soft')
        if self.selection == 'soft' and not self.detach_weight:
            raise ValueError('Soft E-Match requires EMATCH_DETACH_WEIGHT=true')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('EMATCH_TEMPERATURE must be finite and positive')
        if not math.isfinite(self.residual_scale):
            raise ValueError('FRAME_RESIDUAL_SCALE must be finite')
        hidden = int(getattr(cfg, 'FRAME_MLP_HIDDEN_DIM', 64))
        self.mlp = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        period = int(getattr(cfg, 'EMATCH_LOG_PERIOD', 0))
        if period < 0:
            raise ValueError('EMATCH_LOG_PERIOD must be nonnegative')
        self.statistics = EMatchStatistics(period, self.temperature, self.selection) if period else None

    def enhance(self, query, support, initial):
        prob = torch.softmax(initial / self.temperature, dim=1)
        if self.selection == 'hard':
            mask = torch.zeros_like(prob).scatter_(1, initial.argmax(1, keepdim=True), 1)
            alpha = prob * mask
        else:
            alpha = prob
        if self.detach_weight:
            alpha = alpha.detach()
        return support.unsqueeze(0) + alpha[:, :, None, None] * query[:, None], alpha

    def forward(self, query, support, initial=None):
        # No temporal pooling. Chunk queries to bound the pair-concatenation memory.
        enhanced = support.unsqueeze(0).expand(query.shape[0], -1, -1, -1)
        if initial is not None:
            enhanced, _ = self.enhance(query, support, initial)
        result = []
        for q, s in zip(query, enhanced):
            q_pair = q[None, :, None, :].expand(s.shape[0], -1, s.shape[1], -1)
            s_pair = s[:, None, :, :].expand(-1, q.shape[0], -1, -1)
            result.append(self.mlp(torch.cat((s_pair, q_pair, s_pair * q_pair), -1)).squeeze(-1))
        return torch.stack(result)
