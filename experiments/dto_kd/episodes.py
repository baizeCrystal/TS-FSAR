"""Deterministic episode RNG and actual video/frame provenance for protocol v2."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from datasets.video_few_shot import Video_few_shot


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class AuditedEpisodes(Video_few_shot):
    """Preserve the original sampler/transforms; isolate their per-episode RNG.

    The RNG organization changes from the released stream; every v2 group,
    including its fresh E0, must use this wrapper. No cached old run is paired.
    """
    def __init__(self, cfg, split, evaluation_split='val'):
        self.protocol_split = 'train' if split == 'train' else evaluation_split
        self.episode_order_observer = self._observe_order
        super().__init__(cfg, split)

    def _decode_video(self, sample_info, index, **kwargs):
        self._current_video = str(Path(sample_info['path']).relative_to(self.data_root_dir))
        return super()._decode_video(sample_info, index, **kwargs)

    def _get_video_frames_list(self, *args, **kwargs):
        frames = super()._get_video_frames_list(*args, **kwargs)
        self._reads.append({'video': self._current_video, 'frames': [int(x) for x in frames]})
        return frames

    def get_seq(self, *args, **kwargs):
        start = len(self._reads)
        video, index = super().get_seq(*args, **kwargs)
        self._tensor_records[id(video)] = self._reads[start:]
        return video, index

    def _observe_order(self, support, query):
        def records(items):
            return [{'episode_label': int(label), 'class_index': int(real),
                     'reads': self._tensor_records[id(video)]}
                    for video, label, real in items]
        self._ordered = {'support': records(support), 'query': records(query)}

    def __getitem__(self, index):
        seed = int(digest({'version': 2, 'seed': self.cfg.TRAIN.RANDOM_SEED,
                          'shot': self.cfg.TRAIN.SHOT, 'split': self.protocol_split,
                          'episode': int(index)})[:8], 16)
        previous = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self._reads, self._tensor_records, self._ordered = [], {}, None
        try:
            random.seed(seed)
            np.random.seed(seed)
            # CPU default generator only; never initialize CUDA in workers.
            torch.random.default_generator.manual_seed(seed)
            batch = super().__getitem__(index)
            if self._ordered is None:
                raise RuntimeError('original sampler did not report episode ordering')
            record = {'version': 2, 'split': self.protocol_split, 'episode': int(index),
                      'seed': seed, **self._ordered}
            # Include an exact tensor fingerprint to verify augmentation output,
            # not just the sampled IDs/frames. This is computed on CPU workers.
            h = hashlib.sha256()
            for name in ('support_set', 'target_set'):
                h.update(batch[name].contiguous().numpy().tobytes())
            record['input_sha256'] = h.hexdigest()
            record['episode_sha256'] = digest(record)
            batch['episode_record'] = json.dumps(record, separators=(',', ':'))
            return batch
        finally:
            random.setstate(previous[0])
            np.random.set_state(previous[1])
            torch.set_rng_state(previous[2])
            self._reads, self._tensor_records, self._ordered = [], {}, None


class EpisodeRange(Sampler):
    def __init__(self, start, stop):
        self.start, self.stop = start, stop

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self):
        return self.stop - self.start


def loader(cfg, split, evaluation_split='val', start=0):
    if cfg.TRAIN.BATCH_SIZE != 1 or cfg.TEST.BATCH_SIZE != 1 or cfg.TRAIN.BATCH_SIZE_PER_TASK != 1:
        raise ValueError('protocol v2 requires the released one-episode update')
    count = cfg.TRAIN.NUM_TRAIN_TASKS if split == 'train' else cfg.TRAIN.NUM_TEST_TASKS
    dataset = AuditedEpisodes(cfg, split, evaluation_split)
    generator = torch.Generator().manual_seed(17001 + cfg.TRAIN.RANDOM_SEED)
    return DataLoader(dataset, batch_size=1, sampler=EpisodeRange(start, count),
                      num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                      pin_memory=cfg.DATA_LOADER.PIN_MEMORY, generator=generator)


def unpack(batch, device):
    record = json.loads(batch['episode_record'][0])
    tensors = {key: value[0].to(device, non_blocking=True) for key, value in batch.items()
               if key != 'episode_record'}
    support_labels = tensors['support_labels'].long()
    query_labels = tensors['target_labels'].long()
    real = torch.cat((tensors['real_support_labels'], tensors['real_target_labels'])).long()
    images = torch.cat((tensors['support_set'], tensors['target_set']))
    return images, support_labels, query_labels, real, tensors, record
