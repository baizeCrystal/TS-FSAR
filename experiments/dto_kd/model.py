"""Adapter that exposes the existing TS-FSAR loss components to DTO-KD."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from trainers.ts_fsar import TS_FSAR

from .controller import DTOController


class DTOTSFSAR(TS_FSAR):
    """TS-FSAR with optional E1/E2/E3 objective coordination.

    Hooks capture the already-computed DCM score, CE terms, and GLAC KL term.
    The forward graph is otherwise the original graph.  This keeps teacher
    Adapter gradients and all auxiliary CE terms intact.
    """

    def __init__(self, cfg, image_clip):
        super().__init__(cfg, image_clip)
        dto = getattr(cfg, "DTO_KD", None)
        self.dto_mode = getattr(dto, "MODE", "off") if dto is not None else "off"
        self.dto_controller = DTOController(
            eta=float(getattr(dto, "ETA", 0.01)),
            init_distill=float(getattr(dto, "INIT_DISTILL_WEIGHT", 0.5)),
            eps=float(getattr(dto, "EPS", 1e-8)),
            logit_clip=float(getattr(dto, "LOGIT_CLIP", 20.0)),
        ) if self.dto_mode == "dynamic" else None
        self.dto_static_kl_alpha = float(getattr(dto, "STATIC_KL_ALPHA", cfg.TRAIN.KD_ALPHA)) if dto is not None else float(cfg.TRAIN.KD_ALPHA)
        self.dto_trace_interval = int(getattr(dto, "TRACE_INTERVAL", 100)) if dto is not None else 0
        self.dto_step = 0
        self.dto_last_metrics: Dict[str, float] = {}
        self._dto_ce_values = []
        self._dto_kl_value = None
        self._dto_score = None
        self._dto_task_loss = None
        self.ce.register_forward_hook(self._capture_ce)
        self.kl.register_forward_hook(self._capture_kl)
        self.ts_dcm.register_forward_hook(self._capture_dcm)

    def _capture_ce(self, module, inputs, output):
        if self.training:
            self._dto_ce_values.append(output)

    def _capture_kl(self, module, inputs, output):
        if self.training:
            self._dto_kl_value = output

    def _capture_dcm(self, module, inputs, output):
        if self.training:
            self._dto_score = output[0]

    def _components(self, target_labels):
        if self._dto_score is None or len(self._dto_ce_values) < 3 or self._dto_kl_value is None:
            raise RuntimeError("DTO-KD hooks did not capture the expected TS-FSAR losses")
        if self._dto_task_loss is None:
            self._dto_task_loss = F.cross_entropy(self._dto_score, target_labels.long())
        task = self._dto_task_loss
        # TS_FSAR calls CE in this fixed order: student global, teacher text,
        # and LSN text.  The order is part of the existing forward contract.
        student_ce, teacher_ce, lsn_ce = self._dto_ce_values[:3]
        return task, self._dto_kl_value, student_ce, teacher_ce, lsn_ce

    def clear_dto_cache(self):
        """Release episode graph references once diagnostics/backward finish."""
        self._dto_ce_values = []
        self._dto_kl_value = None
        self._dto_score = None
        self._dto_task_loss = None

    def forward(self, images, context_labels, target_labels, real_labels=None, class_embeddings=None):
        self.clear_dto_cache()
        output = super().forward(images, context_labels, target_labels, real_labels, class_embeddings)
        if not self.training or self.dto_mode == "off":
            return output
        task, distill, student_ce, teacher_ce, lsn_ce = self._components(target_labels)
        base_vis = float(self.cfg.TRAIN.VIS_ALPHA)
        base_kd = float(self.cfg.TRAIN.KD_ALPHA)
        # Preserve all auxiliary supervision.  Only DCM CE and GLAC KL are
        # replaced by the selected coordination rule.
        auxiliary = (
            float(self.cfg.TRAIN.CLS_ALPHA) * student_ce
            + float(self.cfg.TRAIN.SIDE_CLS_ALPHA) * lsn_ce
            + base_kd * teacher_ce
        )
        if self.dto_mode == "dynamic":
            coordinated = self.dto_controller.weighted_log_objective(distill, task)
            p_dist, p_task = self.dto_controller.weights
            state = self.dto_controller.update(distill, task)
            self.dto_last_metrics = {
                "pi_distill": p_dist,
                "pi_task": p_task,
                "next_pi_distill": state.pi_distill,
                "next_pi_task": state.pi_task,
                "task_loss": float(task.detach()),
                "distill_loss": float(distill.detach()),
                "dto_objective": float(state.objective),
            }
        elif self.dto_mode == "static_kl":
            coordinated = base_vis * task + self.dto_static_kl_alpha * distill
            self.dto_last_metrics = {
                "static_kl_alpha": self.dto_static_kl_alpha,
                "task_loss": float(task.detach()),
                "distill_loss": float(distill.detach()),
            }
        elif self.dto_mode == "scale_matched":
            # E3's fixed alpha is supplied by a meta-train-only calibration
            # pass; this mode intentionally performs no per-step adaptation.
            coordinated = base_vis * task + self.dto_static_kl_alpha * distill
            self.dto_last_metrics = {
                "scale_matched_kl_alpha": self.dto_static_kl_alpha,
                "task_loss": float(task.detach()),
                "distill_loss": float(distill.detach()),
            }
        elif self.dto_mode == "episode_aware":
            raise NotImplementedError("E4 is reserved and intentionally not implemented")
        else:
            raise ValueError(f"unknown DTO_KD.MODE: {self.dto_mode}")
        output["loss"] = auxiliary + coordinated
        self.dto_step += 1
        self._append_trace()
        return output

    def _append_trace(self):
        interval = self.dto_trace_interval
        if not interval or self.dto_step % interval:
            return
        path = Path(self.cfg.OUTPUT_DIR) / "dto_kd_trace.csv"
        row = {"step": self.dto_step, "mode": self.dto_mode, **self.dto_last_metrics}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def dto_state_dict(self):
        return self.dto_controller.state_dict() if self.dto_controller is not None else None
