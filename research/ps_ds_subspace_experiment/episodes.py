"""Capture the ACTUAL original test sampler; strict video/frame/pixel replay.

No alternative episode sampling implementation. Only adds observations around
Video_few_shot.__getitem__/get_seq/_get_video_frames_list. Decode failures abort.
"""
import hashlib
import json
import torch
from torch.utils.data import DataLoader
from datasets.video_few_shot import Video_few_shot
from .protocol import digest


def pixel_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def identity(item):
    return item


class Capture(Video_few_shot):
    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        indices = super()._get_video_frames_list(vid_length, vid_fps, clip_idx, random_sample)
        self._sample = dict(source_num_frames=int(vid_length), source_fps=float(vid_fps),
                            frame_indices=[int(i) for i in indices])
        return indices

    def get_seq(self, label, idx=-1):
        frames, global_index = super().get_seq(label, idx)
        record = dict(self._sample, class_id=int(label),
                      video_id=self.split_few_shot.videos[global_index],
                      pixel_sha256=pixel_hash(frames))
        self._videos.append(record)
        return frames, global_index

    def __getitem__(self, index):
        self._videos = []
        task = super().__getitem__(index)
        row = dict(schema=1, split='test', shot=self.cfg.TRAIN.SHOT_TEST,
                   episode_index=index, class_order=task['batch_class_list'].long().tolist())
        by_pixels = {}
        for video in self._videos:
            key = (video['class_id'], video['pixel_sha256'])
            if key in by_pixels:
                raise ValueError('ambiguous pixel-identical videos in capture; cannot infer shuffle safely')
            by_pixels[key] = video
        for role, stem in [('support', 'support'), ('query', 'target')]:
            row[role] = []
            frames = task[stem+'_set'].reshape(-1, 8, 3, 224, 224)
            for v, label, real in zip(frames, task[stem+'_labels'], task['real_'+stem+'_labels']):
                key = (int(real), pixel_hash(v))
                row[role].append(dict(by_pixels.pop(key), episode_label=int(label)))
        assert not by_pixels
        row['pixels_sha256'] = pixel_hash(torch.cat((task['support_set'], task['target_set'])))
        row['episode_id'] = digest(row)
        # Send only metadata out of worker; parent need not retain decoded images.
        return row


class Replay(Video_few_shot):
    def __init__(self, cfg, records):
        super().__init__(cfg, 'test')
        self.records = records
        self.lookup = {}
        c = self.split_few_shot
        for label in c.get_unique_classes():
            paths = [v for v, y in zip(c.videos, c.gt_a_list) if y == label]
            self.lookup.update({(label, path): i for i, path in enumerate(paths)})

    def __len__(self):
        return len(self.records)

    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        if int(vid_length) != self.current['source_num_frames']:
            raise RuntimeError('video length changed')
        return torch.tensor(self.current['frame_indices'], dtype=torch.long)

    def __getitem__(self, index):
        row = self.records[index]
        images = []
        for video in row['support']+row['query']:
            self.current = video
            frames, _ = self.get_seq(video['class_id'], self.lookup[(video['class_id'], video['video_id'])])
            if video.get('pixel_sha256') and pixel_hash(frames) != video['pixel_sha256']:
                raise RuntimeError('video pixel replay mismatch: '+video['video_id'])
            images.append(frames)
        images = torch.cat(images)
        pixels = pixel_hash(images)
        if row.get('pixels_sha256') and pixels != row['pixels_sha256']:
            raise RuntimeError('episode pixel replay mismatch')
        return dict(images=images, record=row, pixels_sha256=pixels)


class Plan(Video_few_shot):
    """Fast manifest using original __getitem__, original random frame sampler.

    Test center resize/crop/normalize consumes no RNG. Replacing only get_seq's
    video decoding with ID tokens therefore preserves the sampler exactly.
    This equivalence is audited against Capture (actual decode/preprocessing).
    Pixels are hashed at GPU replay; E0/E1 share the same tensor/encoder output.
    """
    def __init__(self, cfg, metadata):
        super().__init__(cfg, 'test')
        self.metadata = metadata

    def get_seq(self, label, idx=-1):
        path, global_index = self.split_few_shot.get_rand_vid(label, idx)
        source = self.metadata[path]
        frames = self._get_video_frames_list(source['source_num_frames'],source['source_fps'],
                     self._spatial_temporal_index[global_index],random_sample=False)
        self._videos[global_index] = dict(class_id=int(label),video_id=path,
             frame_indices=[int(i) for i in frames],**source)
        return torch.full((8,1),global_index,dtype=torch.int64), global_index

    def __getitem__(self, index):
        self._videos = {}
        task = super().__getitem__(index)
        row = dict(schema=1,split='test',shot=self.cfg.TRAIN.SHOT_TEST,
                   episode_index=index,class_order=task['batch_class_list'].long().tolist())
        for role,stem in [('support','support'),('query','target')]:
            row[role] = []
            ids = task[stem+'_set'].reshape(-1,8)[:,0].tolist()
            for i,label in zip(ids,task[stem+'_labels']):
                row[role].append(dict(self._videos[i],episode_label=int(label)))
        row['episode_id'] = digest(row)
        return row


def stream(dataset, workers, seed):
    # Matches official test batch_size=1, shuffle=False, no worker_init_fn;
    # batch_size=None only avoids an irrelevant outer tensor dimension.
    # Dedicated generator fixes worker seeds independently of encoder creation.
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=workers,
                      collate_fn=identity, pin_memory=False,
                      generator=torch.Generator().manual_seed(seed))
