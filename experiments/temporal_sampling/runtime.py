"""GPU runtime; all imports which can initialize CUDA live behind explicit CLI."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .core import BRANCHES, cache_key, digest, file_hash


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in value.items()})


def preprocess(frames, cfg):
    # Reuse the exact repository test transform, including its square bilinear
    # resize (not torchvision aspect-ratio resize) and controlled center crop.
    import torch
    from torchvision.transforms import Compose
    import torchvision.transforms._transforms_video as video_transforms
    from datasets.augmentations import KineticsResizedCropFewshot
    size = cfg.DATA.TEST_SCALE
    size = list(size) if isinstance(size, (tuple, list)) else [size, size]
    transform = Compose([video_transforms.ToTensorVideo(), KineticsResizedCropFewshot(
        short_side_range=size, crop_size=cfg.DATA.TEST_CROP_SIZE, idx=0),
        video_transforms.NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD, inplace=True)])
    return transform(torch.from_numpy(np.array(frames, copy=True))).permute(1, 0, 2, 3).contiguous()


def decode(root, video, indices):
    from decord import VideoReader
    vr = VideoReader(str(Path(root) / (str(video['video_id']) + '.mp4')), num_threads=1)
    if len(vr) != video['source_num_frames']:
        raise ValueError('Video length changed: ' + str(video['video_id']))
    return vr.get_batch(indices).asnumpy()


class FeatureCache:
    """Bounded LRU memory and optional capped disk cache of complete sequences.

    No eviction/deletion of existing disk files. On budget exhaustion compute
    and retain a small memory LRU only. Scheme+actual indices+model identity
    prevent any original-video feature cache from bypassing resampling.
    """
    def __init__(self, directory, identity, memory_mb=256, disk_mb=8192):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.memory = OrderedDict()
        self.used = 0
        self.budget = memory_mb * 1024 ** 2
        self.disk_budget = disk_mb * 1024 ** 2
        self.disk_used = sum(p.stat().st_size for p in self.directory.glob('*.npz'))
        self.hits = self.misses = 0

    def key(self, v, sampling):
        return cache_key(v, sampling, self.identity)

    def get(self, key):
        if key in self.memory:
            self.memory.move_to_end(key)
            self.hits += 1
            return self.memory[key]
        path = self.directory / (key + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                result = {k: z[k] for k in z.files}
            self.put(key, result, disk=False)
            self.hits += 1
            return result
        self.misses += 1
        return None

    def put(self, key, features, disk=True):
        size = sum(v.nbytes for v in features.values())
        if key not in self.memory:
            while self.memory and self.used + size > self.budget:
                _, removed = self.memory.popitem(last=False)
                self.used -= sum(v.nbytes for v in removed.values())
            if size <= self.budget:
                self.memory[key] = features
                self.used += size
        path = self.directory / (key + '.npz')
        if disk and not path.exists() and self.disk_used + size + 4096 <= self.disk_budget:
            # Uncompressed arrays bound write cost; only this experiment's new
            # cache directory is used. No external cache is read or modified.
            with path.open('xb') as stream:
                np.savez(stream, **features)
            self.disk_used += path.stat().st_size


class Encoder:
    def __init__(self, config, checkpoint, model_dir, video_root, cache_dir, disk_mb=8192):
        import torch
        import transformers
        import torchvision
        import decord
        from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
        from .frozen_ts_fsar import TS_FSAR
        from utils.functions import build_embeddings
        self.torch = torch
        self.device = 'cuda'
        self.cfg = namespace(config)
        self.cfg.TEST.ENABLE, self.cfg.TRAIN.ENABLE = True, False
        if self.cfg.DATA.NUM_INPUT_FRAMES != 8:
            raise ValueError('Eight-frame checkpoint required')
        torch.manual_seed(20261111)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        vision = CLIPVisionModelWithProjection.from_pretrained(str(model_dir), local_files_only=True)
        self.model = TS_FSAR(self.cfg, vision).cuda().eval().requires_grad_(False)
        del vision
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        del state
        text = CLIPTextModelWithProjection.from_pretrained(str(model_dir), local_files_only=True).cuda().eval()
        tokenizer = CLIPTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        names = self.cfg.TEST.CLASS_NAME
        import json
        prompts = json.loads(Path(self.cfg.TEST.LLM_test_json).read_text())
        if any(n.replace('_', ' ') not in prompts for n in names):
            raise ValueError('Prompt class coverage incomplete')
        self.text = build_embeddings(names, self.cfg.TEST.LLM_test_json, text, tokenizer)
        if self.text.shape != (len(names), 512):
            raise ValueError('Text embedding shape mismatch')
        del text, tokenizer
        self.root = video_root
        identity = {'checkpoint': file_hash(checkpoint), 'config': config,
                    'sources': {p: file_hash(p) for p in (
                        'trainers/model.py', 'utils/functions.py', 'datasets/augmentations.py',
                        'experiments/temporal_sampling/runtime.py', 'experiments/temporal_sampling/frozen_ts_fsar.py')},
                    'model_assets': {p.name: file_hash(p) for p in sorted(Path(model_dir).iterdir())
                                     if p.suffix in ('.json', '.bin', '.safetensors')},
                    'versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                                 'torchvision': torchvision.__version__, 'decord': decord.__version__,
                                 'numpy': np.__version__, 'cuda': torch.version.cuda},
                    'gpu': torch.cuda.get_device_name(), 'dtype': 'fp32', 'TF32': False,
                    'video_root': str(Path(video_root).resolve())}
        self.identity = identity
        self.cache = FeatureCache(cache_dir, identity, disk_mb=disk_mb)
        self.native_check = None

    def encode_tensor(self, images):
        torch = self.torch
        import torch.nn.functional as F
        with torch.inference_mode():
            features, clip = self.model.model(images.cuda())
            cls, patch = features[:, 0], features[:, 1:]
            weighted = F.cosine_similarity(cls[:, None], patch, dim=-1)[..., None] * patch
            adc = self.model.ts_dcm.alpha_dc(weighted.transpose(1, 2))
        n = len(images) // 8
        chunks = [{k: v[i*8:(i+1)*8].cpu().numpy().copy()
                   for k, v in {'clip': clip, 'lsn': cls, 'adc': adc}.items()} for i in range(n)]
        return chunks, features, clip

    def baseline(self, ep):
        """Full episode encoding preserves historical GEMM batch dimensions."""
        import hashlib
        from .core import sample_indices
        videos = ep['support'] + ep['query']
        images = self.torch.cat([preprocess(decode(self.root, v, v['frame_indices']), self.cfg) for v in videos])
        pixels = hashlib.sha256(images.numpy().tobytes()).hexdigest()
        chunks, features, clip = self.encode_tensor(images)
        if self.native_check is None:
            self.verify_native(ep, chunks, features, clip, images)
        for v, chunk in zip(videos, chunks):
            self.cache.put(self.cache.key(v, sample_indices(v)), chunk)
        return chunks, pixels

    def changed(self, videos, samplings, batch_videos, check_deadline):
        result = [None] * len(videos)
        missing = []
        for i, (v, sampling) in enumerate(zip(videos, samplings)):
            result[i] = self.cache.get(self.cache.key(v, sampling))
            if result[i] is None:
                missing.append(i)
        # Batch whole videos, never shuffle or mix a temporal group's frames.
        for start in range(0, len(missing), batch_videos):
            check_deadline()
            indices = missing[start:start+batch_videos]
            images = self.torch.cat([preprocess(decode(self.root, videos[i], samplings[i]['indices']), self.cfg) for i in indices])
            chunks, _, _ = self.encode_tensor(images)
            for i, chunk in zip(indices, chunks):
                self.cache.put(self.cache.key(videos[i], samplings[i]), chunk)
                result[i] = chunk
        return result

    def score(self, ep, chunks):
        torch = self.torch
        from utils.functions import cos_sim
        from .core import validate_episode
        with torch.inference_mode():
            shot = self.cfg.TRAIN.SHOT
            ns = 5 * shot
            labels = torch.tensor([v['episode_label'] for v in ep['support']], device=self.device)
            a = torch.as_tensor(np.stack([v['adc'] for v in chunks]), device=self.device)
            cls = torch.as_tensor(np.stack([v['lsn'] for v in chunks]), device=self.device)
            clip = torch.as_tensor(np.stack([v['clip'] for v in chunks]), device=self.device)
            proto = torch.stack([a[:ns][labels == c].mean(0) for c in labels.unique()])
            r = cos_sim(a[ns:].reshape(-1, a.shape[-1]), proto.reshape(-1, a.shape[-1]))
            r = r.reshape(5, 8, 5, 8).permute(0, 2, 1, 3).contiguous()
            task_query = cls[ns:].mean(1)[:, None].repeat(1, 5, 1)
            task_support = cls[:ns].mean(1).mean(0)[None, None].repeat(5, 5, 1)
            w = self.model.ts_dcm.generator(task_query + task_support).reshape(5, 5, 8, 8)
            score = (r.reshape(5, 5, 64) * w.reshape(5, 5, 64)).sum(-1)
            text = self.text[validate_episode(ep, shot)]
            qc = self.model.model.Adapter(clip[ns:]).mean(1)
            qc = qc / qc.norm(dim=-1, keepdim=True)
            adapter_logits = qc @ text.t() * self.cfg.TEMP_CLIP
            ql = self.model.model.up_linear(self.model.model.up_ln(cls[ns:])).mean(1)
            ql = ql / ql.norm(dim=-1, keepdim=True)
            lsn_logits = ql @ text.t() * self.cfg.TEMP_SIDE
            pv, pc, ps = [v.softmax(1) for v in (score, adapter_logits, lsn_logits)]
            fusion = pv.pow(self.cfg.TEST.FUSE_VISION) * pc.pow(self.cfg.TEST.FUSE_CLIP) * ps.pow(self.cfg.TEST.FUSE_SIDE)
            output = {'ts_dcm': score, 'adapter_text': adapter_logits, 'lsn_text': lsn_logits,
                      'fusion': fusion, 'R': r, 'W': w,
                      'adapter_probability': pc, 'lsn_probability': ps, 'vision_probability': pv}
            out = {k: v.cpu().numpy().copy() for k, v in output.items()}
            out['labels'] = np.array([v['episode_label'] for v in ep['query']], dtype=np.int64)
            return out

    def verify_native(self, ep, chunks, features, clip, images):
        """Compare cached readout to actual unchanged TS_FSAR.forward."""
        from unittest.mock import patch
        torch = self.torch
        from .core import validate_episode
        labels = torch.tensor([v['episode_label'] for v in ep['support']], device=self.device)
        targets = torch.tensor([v['episode_label'] for v in ep['query']], device=self.device)
        with torch.inference_mode():
            original_score, _, _ = self.model.ts_dcm(features, labels, 5, self.cfg.TRAIN.SHOT, 1)
            ours = self.score(ep, chunks)
            np.testing.assert_allclose(ours['ts_dcm'], original_score.cpu(), atol=3e-5, rtol=2e-5)
            # Reuse only these just-encoded tensors for an equation check, not
            # as a pixel intervention. Native model.forward sees all labels.
            with patch.object(self.model.model, 'forward', return_value=(features, clip)):
                native = self.model(images.to(self.device), labels, targets,
                                    class_embeddings=self.text[validate_episode(ep, self.cfg.TRAIN.SHOT)])
            accuracy = (ours['fusion'].argmax(1) == targets.cpu().numpy()).mean()
            loss = -np.log(ours['fusion'][np.arange(5), targets.cpu().numpy()] + 1e-8).mean()
            if abs(accuracy - native['acc']) > 1e-6 or abs(loss - native['loss'].item()) > 3e-5:
                raise RuntimeError('Native fusion parity check failed')
        self.native_check = {'status': 'passed', 'score_max_error': float(np.max(np.abs(ours['ts_dcm'] - original_score.cpu().numpy()))),
                             'native_accuracy': native['acc'], 'native_loss': native['loss'].item()}
