"""Locked official configuration/checkpoints and small, atomic audit utilities."""
import copy
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read_records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        return [json.loads(line) for line in f if line.strip()]


def source_audit():
    from trainers.ts_fsar import Alpha_DC, TS_DCM, TS_FSAR
    from trainers.model import LSN, build_model
    from utils.functions import cos_sim, build_embeddings
    from datasets.video_few_shot import Video_few_shot
    from datasets.base_dataset import BaseVideoDataset
    from datasets.augmentations import KineticsResizedCropFewshot
    objects = [Alpha_DC, TS_DCM, TS_FSAR, LSN, build_model, cos_sim, build_embeddings,
               Video_few_shot, BaseVideoDataset, KineticsResizedCropFewshot]
    return {o.__name__: hashlib.sha256(inspect.getsource(o).encode()).hexdigest() for o in objects}


def load_config(shot, data_root, output, workers=4):
    import yaml
    from utils.config import Config
    raw = yaml.safe_load((ROOT/'configs/base.yaml').read_text())
    official = yaml.safe_load((ROOT/f'configs/ssv2_small/Metatest_{shot}shot.yaml').read_text())
    merger = Config(load=False, cfg_dict={})
    raw = merger._merge_cfg_from_base(raw, official)
    lock = json.loads((HERE/'configs/locked.json').read_text())
    for rel, expected in lock['official_files'].items():
        if sha_file(ROOT/rel) != expected:
            raise RuntimeError('official file differs from audited source: '+rel)
    if source_audit() != lock['source_objects']:
        raise RuntimeError('original model/loader functions differ from audited source; do not silently continue')
    # Only operational paths, decode failure policy and loader concurrency change.
    raw['DATA']['DATA_ROOT_DIR'] = str(Path(data_root).resolve())
    raw['DATA']['ANNO_DIR'] = str(ROOT/'configs/ssv2_small')
    raw['DATA']['FAIL_ON_DECODE_ERROR'] = True
    # Decord auto-threading (0) hits FFmpeg EAGAIN on this CPU host. This is
    # decoder concurrency only, not frame indices or image transformations.
    raw['DATA']['DECODE_THREADS'] = 1
    raw['OUTPUT_DIR'] = raw['OUTPUT_DIR_SAVE'] = str(Path(output).resolve())
    raw['DATA_LOADER']['NUM_WORKERS'] = workers
    raw['TRAIN']['LLM_train_json'] = str(ROOT/raw['TRAIN']['LLM_train_json'])
    raw['TEST']['LLM_test_json'] = str(ROOT/raw['TEST']['LLM_test_json'])
    assert raw['TRAIN']['ENABLE'] is False and raw['TEST']['ENABLE'] is True
    assert (raw['TRAIN']['WAY'], raw['TRAIN']['SHOT'], raw['TRAIN']['SHOT_TEST'],
            raw['TRAIN']['QUERY_PER_CLASS_TEST'], raw['DATA']['NUM_INPUT_FRAMES'],
            raw['TRAIN']['NUM_TEST_TASKS']) == (5, shot, shot, 1, 8, 10000)
    assert raw['DATA']['TEST_CENTER_CROP'] is True
    assert raw['DATA']['SAMPLING_MODE'] == 'interval_based'
    for split, count in [('train', 64), ('val', 12), ('test', 24)]:
        lines = (ROOT/f'configs/ssv2_small/{split}_few_shot.txt').read_text().splitlines()
        ids = {int(line.split('/')[0][len(split):]) for line in lines}
        assert ids == set(range(count))
    # The original dataset prioritizes OUTPUT_DIR/test_few_shot.txt. Forbid a shadow split.
    if (Path(output)/'test_few_shot.txt').exists():
        raise RuntimeError('output contains shadow test split')
    return Config(load=False, cfg_dict=raw), raw, lock


def check_manifest(records, shot):
    if len(records) != 10000:
        raise ValueError('fixed manifest must contain exactly 10000 official test episodes')
    for i, row in enumerate(records):
        body = dict(row)
        token = body.pop('episode_id')
        if token != digest(body) or row['episode_index'] != i or row['shot'] != shot:
            raise ValueError('episode identity/order/shot mismatch')
        if row['split'] != 'test' or len(row['class_order']) != 5:
            raise ValueError('non-official split/way')
        if len(set(row['class_order'])) != 5 or not set(row['class_order']) <= set(range(24)):
            raise ValueError('invalid novel classes')
        for role, per_class in [('support', shot), ('query', 1)]:
            videos = row[role]
            labels = [v['episode_label'] for v in videos]
            if len(videos) != 5*per_class or any(labels.count(c) != per_class for c in range(5)):
                raise ValueError('invalid official support/query counts')
            for v in videos:
                if v['class_id'] != row['class_order'][v['episode_label']]:
                    raise ValueError('label mapping differs')
                if len(v['frame_indices']) != 8 or any(j < 0 or j >= v['source_num_frames'] for j in v['frame_indices']):
                    raise ValueError('invalid frame sample')
        ids = [v['video_id'] for v in row['support']+row['query']]
        if len(ids) != len(set(ids)):
            raise ValueError('support/query overlap or duplicate videos')


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
