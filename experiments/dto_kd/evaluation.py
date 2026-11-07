"""Four branch predictions and original fusion, for either declared RNG policy."""
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.functions import extract_class_indices
from .episodes import unpack, official_stream, number_stream_episode
from .state import EpisodeLedger, append_json


def evaluate(data, model, embeddings, cfg, split, step, device):
    directory = Path(cfg.OUTPUT_DIR) / 'predictions'
    directory.mkdir(exist_ok=True)
    # Never overwrite an earlier attempt, including after an interrupted resume.
    path = directory / f'{split}_step{step:05d}_{time.time_ns()}.jsonl'
    # Official evaluations resample on every iterator; never reuse fixed-bank IDs.
    ledger = EpisodeLedger(path.with_suffix('.episodes.jsonl') if official_stream(cfg) else
                           Path(cfg.OUTPUT_DIR) / f'{split}_episodes.jsonl')
    captures = {}
    hooks = [model.ts_dcm.register_forward_hook(lambda m, i, o: captures.update(score=o[0])),
             model.model.Adapter.register_forward_hook(lambda m, i, o: captures.update(clip=o)),
             model.model.up_linear.register_forward_hook(lambda m, i, o: captures.update(side=o))]
    correct = dict.fromkeys(('matching', 'clip', 'side', 'fusion'), 0)
    total, loss_sum = 0, 0.
    fusion_episode_accs = []
    model.eval()
    try:
        with torch.no_grad():
            for episode_index, batch in enumerate(data):
                images, support, query, _, tensors, record = unpack(batch, device)
                record = number_stream_episode(record, episode_index)
                labels = tensors['real_support_labels'].long()
                text = torch.stack([embeddings[labels].index_select(
                    0, extract_class_indices(support, c)).mean(0) for c in torch.unique(support)])
                output = model(images, support, query, None, text)
                matching = captures['score'].softmax(1)
                clip = captures['clip'].mean(1)
                clip = clip / clip.norm(dim=-1, keepdim=True)
                side = captures['side'].mean(1)
                side = side / side.norm(dim=-1, keepdim=True)
                # Preserve the original feature normalization and probability
                # product, with the released shot-specific fusion coefficients.
                clip_prob = (clip @ text.t() * cfg.TEMP_CLIP).softmax(1)
                side_prob = (side @ text.t() * cfg.TEMP_SIDE).softmax(1)
                fused = (matching.pow(cfg.TEST.FUSE_VISION) * clip_prob.pow(cfg.TEST.FUSE_CLIP)
                         * side_prob.pow(cfg.TEST.FUSE_SIDE))
                branches = {'matching': matching, 'clip': clip_prob, 'side': side_prob, 'fusion': fused}
                predictions = {k: v.argmax(1) for k, v in branches.items()}
                fusion_acc = predictions['fusion'].eq(query).float().mean().item()
                if not math.isclose(fusion_acc, output['acc'], abs_tol=1e-7):
                    raise RuntimeError('branch reconstruction changed original fusion predictions')
                fusion_episode_accs.append(float(output['acc']))
                for key, pred in predictions.items():
                    correct[key] += pred.eq(query).sum().item()
                total += len(query)
                loss_sum += float(output['loss'])
                ledger.observe(record)
                append_json(path, {'episode': record['episode'], 'episode_sha256': record['episode_sha256'],
                                   'labels': query.cpu().tolist(),
                                   'predictions': {k: v.cpu().tolist() for k, v in predictions.items()},
                                   'scores': {k: v.float().cpu().tolist() for k, v in branches.items()}})
                captures.clear()
                del images, support, query, tensors, output, matching, clip, side, fused, branches
                period = int(getattr(cfg.TEST, 'PROGRESS_LOG_PERIOD', 1000))
                if period > 0 and (episode_index + 1) % period == 0:
                    print('EVAL_PROGRESS', json.dumps({'split': split, 'step': step,
                                                      'episodes': episode_index + 1,
                                                      'total': len(data)}), flush=True)
        count = len(data)
        if total == 0 or count == 0:
            raise RuntimeError('empty evaluation')
        accuracies = {k: 100. * v / total for k, v in correct.items()}
        if official_stream(cfg):
            # Exactly the official main.test reduction, including float rounding.
            accuracies['fusion'] = float(np.mean(fusion_episode_accs) * 100.)
        return {'split': split, 'selection_forward_episode': step, 'episodes': count,
                'queries': total, 'accuracy_percent': accuracies,
                'loss': loss_sum / count, 'predictions': str(path),
                'episode_sequence_sha256': ledger.sequence_hash(count)}
    finally:
        for hook in hooks:
            hook.remove()
        captures.clear()
        model.clear_dto_cache()
