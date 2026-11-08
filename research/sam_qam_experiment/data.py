"""Original Meta-Train episode sampling/augmentation, deterministic by index.

Only Meta-Train is used for learning the new module. The per-index RNG policy is
an adaptation engineering choice, not the released training worker RNG stream.
"""
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.video_few_shot import Video_few_shot
from research.ps_ds_subspace_experiment.protocol import ROOT, sha_file, digest


def training_config(shot, data_root, output, workers):
    import yaml
    from utils.config import Config
    from .common import specification
    for name, expected in specification()['official_train_files'].items():
        if sha_file(ROOT/name) != expected:
            raise RuntimeError('official training configuration changed: '+name)
    raw = yaml.safe_load((ROOT/'configs/base.yaml').read_text())
    override = yaml.safe_load((ROOT/f'configs/ssv2_small/Metatrain_{shot}shot.yaml').read_text())
    raw = Config(load=False, cfg_dict={})._merge_cfg_from_base(raw, override)
    raw['DATA'].update(DATA_ROOT_DIR=str(data_root), ANNO_DIR=str(ROOT/'configs/ssv2_small'),
                       FAIL_ON_DECODE_ERROR=True, DECODE_THREADS=1)
    raw['OUTPUT_DIR'] = raw['OUTPUT_DIR_SAVE'] = str(output)
    raw['DATA_LOADER']['NUM_WORKERS'] = workers
    if (raw['TRAIN']['WAY'], raw['TRAIN']['SHOT'], raw['TRAIN']['QUERY_PER_CLASS'],
            raw['DATA']['NUM_INPUT_FRAMES']) != (5, shot, 5, 8):
        raise RuntimeError('released training episode geometry changed')
    if (output/'train_few_shot.txt').exists():
        raise RuntimeError('refuse shadow training split')
    return Config(load=False, cfg_dict=raw), raw


class TrainEpisodes(Video_few_shot):
    def __init__(self, cfg, seed, start, count):
        super().__init__(cfg, 'train')
        self.seed, self.start, self.count = seed, start, count
        if set(self.split_few_shot.gt_a_list) != set(range(64)):
            raise ValueError('expected official 64 Base Train classes')

    def __len__(self):
        return self.count

    def _get_video_frames_list(self, *args, **kwargs):
        indices = super()._get_video_frames_list(*args, **kwargs)
        self.current_indices = [int(i) for i in indices]
        return indices

    def get_seq(self, label, idx=-1):
        frames, global_index = super().get_seq(label, idx)
        self.drawn.append(dict(class_id=int(label),
            video_id=self.split_few_shot.videos[global_index],
            frame_indices=list(self.current_indices)))
        return frames, global_index

    def __getitem__(self, index):
        episode = self.start+index
        state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            seed = (self.seed + 1000003*episode) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            self.drawn = []
            batch = super().__getitem__(episode)
            row = dict(split='train', episode_index=episode,
                       class_order=batch['batch_class_list'].long().tolist(),
                       support_labels=batch['support_labels'].long().tolist(),
                       query_labels=batch['target_labels'].long().tolist(),
                       video_draw_order=self.drawn)
            row['episode_id'] = digest(row)
            batch['episode_record'] = row
            return batch
        finally:
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])


def train_loader(cfg, seed, start, count, workers):
    from research.ps_ds_subspace_experiment.episodes import identity
    dataset = TrainEpisodes(cfg, seed, start, count)
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=workers,
        collate_fn=identity, pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
        **({'prefetch_factor': 1} if workers else {}))
