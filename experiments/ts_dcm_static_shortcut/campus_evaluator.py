"""Frozen TS-FSAR evaluator hook for ``run_counterfactual``.

Configure it with ``TS_FSAR_STATIC_CONFIG`` and
``TS_FSAR_STATIC_CHECKPOINT``.  The module is imported once by the runner and
keeps one frozen model in memory.  It intentionally bypasses the repository's
episode sampler only to feed the exact episode and transformed pixels supplied
by the runner; the TS-FSAR forward equations remain unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


_STATE = None


def _initialise():
    import torch
    import yaml
    from utils.config import Config
    from trainers.model import build_model
    from trainers.ts_fsar import TS_FSAR
    from utils.functions import build_embeddings

    cfg_path = os.environ.get("TS_FSAR_STATIC_CONFIG")
    checkpoint = os.environ.get("TS_FSAR_STATIC_CHECKPOINT")
    if not cfg_path or not checkpoint:
        raise RuntimeError("set TS_FSAR_STATIC_CONFIG and TS_FSAR_STATIC_CHECKPOINT")
    payload = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    cfg = Config(load=False, cfg_dict=payload)
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    image_clip, text_clip, tokenizer = build_model(cfg)
    model = TS_FSAR(cfg, image_clip).cuda().eval()
    state = torch.load(checkpoint, map_location="cuda")
    model.load_state_dict(state, strict=True)
    class_embeddings = build_embeddings(cfg.TEST.CLASS_NAME, cfg.TEST.LLM_test_json, text_clip, tokenizer)
    from torchvision.transforms import Compose
    import torchvision.transforms._transforms_video as transforms
    from datasets.augmentations import KineticsResizedCropFewshot
    resize = KineticsResizedCropFewshot(
        short_side_range=[cfg.DATA.TEST_SCALE, cfg.DATA.TEST_SCALE],
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        idx=cfg.DATA.TEST_CENTER_CROP,
    )
    preprocess = Compose([
        transforms.ToTensorVideo(), resize,
        transforms.NormalizeVideo(mean=cfg.DATA.MEAN, std=cfg.DATA.STD, inplace=True),
    ])
    return {"torch": torch, "cfg": cfg, "model": model, "class_embeddings": class_embeddings,
            "preprocess": preprocess, "class_names": list(cfg.TEST.CLASS_NAME)}


def _decode(path: str, frame_indices):
    from decord import VideoReader
    vr = VideoReader(path, num_threads=1)
    return np.asarray(vr.get_batch(np.asarray(frame_indices, dtype=np.int64)).asnumpy(), dtype=np.uint8)


def _prepare_native(frames, preprocess, torch):
    # preprocess returns [C,T,H,W]; the model consumes [T,C,H,W] after the
    # episode concatenation below.
    return preprocess(torch.from_numpy(np.asarray(frames)))


def _branch_forward(state, support, queries, context_labels, target_labels, text_embedding):
    torch = state["torch"]
    model = state["model"]
    cfg = state["cfg"]
    images = torch.cat([support, queries], dim=0).cuda(non_blocking=True)
    with torch.no_grad():
        lsn_features, clip_cls = model.model(images.view(-1, *images.shape[2:]))
        score, adc, lsn_cls = model.ts_dcm(
            lsn_features, context_labels, cfg.TRAIN.WAY, cfg.TRAIN.SHOT, cfg.TRAIN.QUERY_PER_CLASS_TEST
        )
        vision = torch.softmax(score, dim=1)
        num_support = cfg.TRAIN.WAY * cfg.TRAIN.SHOT * cfg.DATA.NUM_INPUT_FRAMES
        qn = queries.shape[0]
        feat_clip = model.model.Adapter(clip_cls[num_support:].reshape(qn, cfg.DATA.NUM_INPUT_FRAMES, -1)).mean(1)
        feat_clip = feat_clip / feat_clip.norm(dim=-1, keepdim=True)
        adapter_logits = feat_clip @ text_embedding.t() * cfg.TEMP_CLIP
        adapter = torch.softmax(adapter_logits, dim=1)
        feat_lsn = model.model.up_linear(model.model.up_ln(lsn_cls[num_support:].reshape(qn, cfg.DATA.NUM_INPUT_FRAMES, -1))).mean(1)
        feat_lsn = feat_lsn / feat_lsn.norm(dim=-1, keepdim=True)
        lsn_logits = feat_lsn @ text_embedding.t() * cfg.TEMP_SIDE
        lsn_text = torch.softmax(lsn_logits, dim=1)
        fusion = vision.pow(cfg.TEST.FUSE_VISION) * adapter.pow(cfg.TEST.FUSE_CLIP) * lsn_text.pow(cfg.TEST.FUSE_SIDE)
        # Reconstruct the exact query-specific generator readout for logging.
        lsn_cls_all = lsn_cls
        query_cls = lsn_cls_all[num_support:]
        support_cls = lsn_cls_all[:num_support]
        task_query = query_cls.reshape(qn, cfg.DATA.NUM_INPUT_FRAMES, -1).mean(1)
        task_query = task_query[:, None, :].repeat(1, cfg.TRAIN.WAY, 1)
        task_support = support_cls.reshape(cfg.TRAIN.WAY * cfg.TRAIN.SHOT, cfg.DATA.NUM_INPUT_FRAMES, -1).mean(1).mean(0)
        task_support = task_support[None, None, :].repeat(qn, cfg.TRAIN.WAY, 1)
        tsm = model.ts_dcm.generator(task_support + task_query).reshape(qn, cfg.TRAIN.WAY, cfg.DATA.NUM_INPUT_FRAMES, cfg.DATA.NUM_INPUT_FRAMES)
        # Reconstruct the class-prototype alpha-D correlation in the same order.
        support_adc = adc[:num_support].reshape(cfg.TRAIN.WAY * cfg.TRAIN.SHOT, cfg.DATA.NUM_INPUT_FRAMES, -1)
        proto = torch.stack([support_adc[context_labels == c].mean(0) for c in torch.unique(context_labels)])
        from utils.functions import cos_sim
        ad = cos_sim(adc[num_support:], proto.reshape(-1, proto.shape[-1]))
        ad = ad.reshape(qn, cfg.TRAIN.WAY, cfg.DATA.NUM_INPUT_FRAMES, cfg.DATA.NUM_INPUT_FRAMES)
    return {
        "ts_dcm": score.detach().cpu().numpy(), "ts_dcm_probability": vision.detach().cpu().numpy(),
        "adapter_text": adapter_logits.detach().cpu().numpy(), "adapter_text_probability": adapter.detach().cpu().numpy(),
        "lsn_text": lsn_logits.detach().cpu().numpy(), "lsn_text_probability": lsn_text.detach().cpu().numpy(),
        "fusion": fusion.detach().cpu().numpy(), "alpha_d": ad.detach().cpu().numpy(), "tsm": tsm.detach().cpu().numpy(),
    }


def evaluate(*, episode: Dict[str, Any], condition_paths: Dict[str, Dict[str, str]], output_dir: str) -> Dict[str, Any]:
    """Evaluator hook called once per fixed episode by the guarded runner."""
    global _STATE
    if _STATE is None:
        _STATE = _initialise()
    torch, cfg, preprocess = _STATE["torch"], _STATE["cfg"], _STATE["preprocess"]
    root = Path(os.environ.get("TS_FSAR_VIDEO_ROOT", cfg.DATA.DATA_ROOT_DIR))
    names = _STATE["class_names"]
    name_to_idx = {name: i for i, name in enumerate(names)}
    try:
        text_indices = [name_to_idx[n] for n in episode["class_order_names"]]
    except KeyError as exc:
        raise RuntimeError(f"episode class name is absent from TEST.CLASS_NAME: {exc}") from exc
    text_embedding = _STATE["class_embeddings"][text_indices].cuda()
    supports = []
    for item in episode["support"]:
        frames = _decode(str(root / f"{item['video_id']}.mp4"), item["frame_indices"])
        supports.append(_prepare_native(frames, preprocess, torch).permute(1, 0, 2, 3))
    support = torch.stack(supports)
    context_labels = torch.tensor(episode["support_labels"], dtype=torch.long, device="cuda").repeat_interleave(cfg.DATA.NUM_INPUT_FRAMES)
    labels = torch.tensor(episode["target_labels"], dtype=torch.long, device="cuda")
    rows = []
    for condition in ("E0", "E1_background_blur", "E2_background_texture", "E3_foreground_blur"):
        queries = []
        query_ids = []
        for query in episode["query"]:
            query_ids.append(str(query["video_id"]))
            with np.load(condition_paths[str(query["video_id"])][condition], allow_pickle=False) as data:
                frames = data["frames"]
            queries.append(_prepare_native(frames, preprocess, torch).permute(1, 0, 2, 3))
        query_tensor = torch.stack(queries)
        out = _branch_forward(_STATE, support, query_tensor, context_labels, labels, text_embedding)
        for i, query_id in enumerate(query_ids):
            record = {"episode_id": int(episode["episode"]), "query_id": query_id, "condition": condition,
                      "label": int(labels[i].item())}
            for branch, scores in (("ts_dcm", out["ts_dcm"]), ("adapter_text", out["adapter_text"]),
                                   ("lsn_text", out["lsn_text"]), ("fusion", out["fusion"])):
                vec = np.asarray(scores[i], dtype=float)
                record[f"{branch}_scores"] = vec.tolist()
                record[f"{branch}_prediction"] = int(vec.argmax())
                wrong = np.delete(vec, int(labels[i].item()))
                record[f"{branch}_margin"] = float(vec[int(labels[i].item())] - wrong.max())
            record["alpha_d"] = np.asarray(out["alpha_d"][i], dtype=float).tolist()
            record["tsm"] = np.asarray(out["tsm"][i], dtype=float).tolist()
            rows.append(record)
    return {"rows": rows, "feature_cache_used": False, "checkpoint_strict": True}

