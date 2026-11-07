"""Training loop adapter with AMP-safe DTO tracing and checkpoint sidecars."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch.cuda.amp.autocast_mode import autocast

from .gradient_utils import activation_context, loss_gradients
from .controller import gradient_metrics


def train_epoch(train_loader, val_loader, model, train_embeddings, test_embeddings, optimizer, lr_scheduler, train_meter, cfg):
    import main as baseline

    model.train()
    train_meter.iter_tic()
    best_acc, best_iter = -float("inf"), None
    dto = getattr(cfg, "DTO_KD", None)
    sample_interval = int(getattr(dto, "GRADIENT_SAMPLE_INTERVAL", 0))
    offload = str(getattr(dto, "ACTIVATION_OFFLOAD", "diagnostics"))
    for cur_iter, task_dict in enumerate(train_loader):
        if cur_iter >= cfg.TRAIN.NUM_TRAIN_TASKS:
            break
        for key in task_dict:
            task_dict[key] = task_dict[key][0]
        context_imgs = task_dict["support_set"].squeeze(0).cuda(non_blocking=True)
        context_labels = task_dict["support_labels"].squeeze(0).long().cuda(non_blocking=True)
        target_imgs = task_dict["target_set"].squeeze(0).cuda(non_blocking=True)
        target_labels = task_dict["target_labels"].squeeze(0).long().cuda(non_blocking=True)
        context_real = task_dict["real_support_labels"].squeeze(0).long().cuda(non_blocking=True)
        target_real = task_dict["real_target_labels"].squeeze(0).long().cuda(non_blocking=True)
        images = torch.cat((context_imgs, target_imgs), dim=0)
        real_labels = torch.cat((context_real, target_real), dim=0)
        sampled = bool(sample_interval and (cur_iter + 1) % sample_interval == 0
                       and getattr(model, "dto_mode", "off") != "off")
        with activation_context(offload, sampled), autocast():
            result = model(images, context_labels, target_labels, real_labels, train_embeddings)
            loss = result["loss"]
        if sampled:
            task, distill, _, _, _ = model._components(target_labels)
            params = [p for p in model.parameters() if p.requires_grad]
            gradients = loss_gradients({"distill": distill, "task": task}, params)
            metrics = gradient_metrics(gradients["distill"], gradients["task"])
            del gradients, task, distill, params
            trace = Path(cfg.OUTPUT_DIR) / "dto_kd_gradients.csv"
            trace.parent.mkdir(parents=True, exist_ok=True)
            fields = ["step", *metrics.keys()]
            with trace.open("a") as stream:
                if trace.stat().st_size == 0:
                    stream.write(",".join(fields) + "\n")
                stream.write(",".join(str({"step": cur_iter + 1, **metrics}[k]) for k in fields) + "\n")
        baseline.scaler.scale(loss).backward()
        if hasattr(model, "clear_dto_cache"):
            model.clear_dto_cache()
        if ((cur_iter + 1) >= int(getattr(cfg.TRAIN, "EVAL_START_EPISODE", 1))
                and (cur_iter + 1) % cfg.TRAIN.VAL_FRE_ITER == 0):
            model.eval()
            acc, _ = baseline.test(val_loader, model, test_embeddings, cfg)
            model.train()
            if not math.isfinite(acc):
                raise RuntimeError("non-finite validation accuracy")
            if acc > best_acc:
                best_acc, best_iter = acc, cur_iter
                path = os.path.join(cfg.OUTPUT_DIR_SAVE, "model_best.pth")
                torch.save(model.state_dict(), path + ".tmp")
                os.replace(path + ".tmp", path)
            best = {
                "best_accuracy_percent": float(best_acc), "best_iteration": best_iter,
                "best_episode": best_iter + 1, "latest_accuracy_percent": float(acc),
                "latest_iteration": cur_iter,
                "checkpoint": os.path.join(cfg.OUTPUT_DIR_SAVE, "model_best.pth"),
            }
            Path(cfg.OUTPUT_DIR_SAVE).mkdir(parents=True, exist_ok=True)
            Path(cfg.OUTPUT_DIR_SAVE, "best_result.json").write_text(json.dumps(best, indent=2) + "\n")
        if (cur_iter + 1) % cfg.TRAIN.BATCH_SIZE_PER_TASK == 0:
            baseline.scaler.step(optimizer)
            baseline.scaler.update()
            optimizer.zero_grad()
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("non-finite DTO-KD loss")
        train_meter.update_stats(top1_err=100.0 - result["acc"] * 100.0,
                                 loss=float(loss.detach()), lr=optimizer.param_groups[0]["lr"], mb_size=1)
        train_meter.log_iter_stats(cur_iter)
        train_meter.iter_tic()
        lr_scheduler(cur_iter)
        # Do not keep the preceding episode's inputs/graph roots alive while
        # allocating the next episode. Optimizer accumulation is unaffected.
        del result, loss, images, real_labels, task_dict
        del context_imgs, context_labels, target_imgs, target_labels, context_real, target_real
    train_meter.log_epoch_stats()
    train_meter.reset()
