"""Fixed episodes and four branch predictions using the original fusion rule."""
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from utils.functions import extract_class_indices
from .episodes import unpack
from .state import EpisodeLedger, append_json


def evaluate(data, model, embeddings, cfg, split, step, device):
    directory = Path(cfg.OUTPUT_DIR) / 'predictions'
    directory.mkdir(exist_ok=True)
    ledger = EpisodeLedger(Path(cfg.OUTPUT_DIR) / f'{split}_episodes.jsonl')
    # Never overwrite an earlier attempt, including after an interrupted resume.
    path = directory / f'{split}_step{step:05d}_{time.time_ns()}.jsonl'
    captures = {}
    hooks = [model.ts_dcm.register_forward_hook(lambda m, i, o: captures.update(score=o[0])),
             model.model.Adapter.register_forward_hook(lambda m, i, o: captures.update(clip=o)),
             model.model.up_linear.register_forward_hook(lambda m, i, o: captures.update(side=o))]
    correct = dict.fromkeys(('matching', 'clip', 'side', 'fusion'), 0)
    total, loss_sum = 0, 0.
    model.eval()
    try:
        with torch.no_grad():
            for batch in data:
                images, support, query, _, tensors, record = unpack(batch, device)
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
        count = len(data)
        if total == 0 or count == 0:
            raise RuntimeError('empty evaluation')
        return {'split': split, 'selection_forward_episode': step, 'episodes': count,
                'queries': total, 'accuracy_percent': {k: 100. * v / total for k, v in correct.items()},
                'loss': loss_sum / count, 'predictions': str(path),
                'episode_sequence_sha256': ledger.sequence_hash(count)}
    finally:
        for hook in hooks:
            hook.remove()
        captures.clear()
        model.clear_dto_cache()
