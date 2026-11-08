"""Adapter that exposes the existing TS-FSAR loss components to DTO-KD."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from trainers.ts_fsar import TS_FSAR

from .controller import DTOController
from .exact import log_loss, shared_lsn_weights, task_anchored_kl_weight


class DTOTSFSAR(TS_FSAR):
    """TS-FSAR with optional E1/E2/E3 objective coordination.

    Hooks capture the already-computed DCM score, CE terms, and GLAC KL term.
    Legacy configurations preserve the original graph. Explicit new variants
    stop only the KL target gradient; teacher CE still trains its Adapter.
    """

    def __init__(self, cfg, image_clip):
        super().__init__(cfg, image_clip)
        dto = getattr(cfg, "DTO_KD", None)
        self.dto_mode = getattr(dto, "MODE", "off") if dto is not None else "off"
        self.dto_detach_teacher = bool(getattr(dto, 'DETACH_TEACHER', False))
        if self.dto_mode in ('safe_dynamic', 'paper_exact', 'paper_amortized', 'task_anchored') and not self.dto_detach_teacher:
            raise ValueError(self.dto_mode + ' requires DETACH_TEACHER=true')
        self.dto_controller = DTOController(
            eta=float(getattr(dto, "ETA", 0.01)),
            init_distill=float(getattr(dto, "INIT_DISTILL_WEIGHT", 0.5)),
            eps=float(getattr(dto, "EPS", 1e-8)),
            logit_clip=float(getattr(dto, "LOGIT_CLIP", 20.0)),
            update_rule=str(getattr(dto, "UPDATE_RULE", "legacy_logits")),
        ) if self.dto_mode in ("dynamic", "safe_dynamic", "paper_amortized") else None
        if self.dto_mode == 'paper_amortized' and self.dto_controller.update_rule != 'paper_pi_softmax':
            raise ValueError('paper_amortized requires the direct paper_pi_softmax update')
        self.dto_static_kl_alpha = float(getattr(dto, "STATIC_KL_ALPHA", cfg.TRAIN.KD_ALPHA)) if dto is not None else float(cfg.TRAIN.KD_ALPHA)
        self.dto_trace_interval = int(getattr(dto, "TRACE_INTERVAL", 100)) if dto is not None else 0
        self.dto_step = 0
        self.dto_last_metrics: Dict[str, float] = {}
        self._dto_ce_values = []
        self._dto_kl_value = None
        self._dto_score = None
        self._dto_task_loss = None
        self._dto_supervised_loss = None
        self._dto_real_labels = None
        self.dto_response_metrics = {}
        if self.dto_detach_teacher:
            self.kl.register_forward_pre_hook(self._stop_kl_target)
        self.ce.register_forward_hook(self._capture_ce)
        self.kl.register_forward_hook(self._capture_kl)
        self.ts_dcm.register_forward_hook(self._capture_dcm)

    @staticmethod
    def _stop_kl_target(module, inputs):
        return (inputs[0], inputs[1].detach())

    def _capture_ce(self, module, inputs, output):
        if self.training:
            self._dto_ce_values.append(output)

    def _capture_kl(self, module, inputs, output):
        if self.training:
            self._dto_kl_value = output
            # Detached, tiny global logits only. No extra model forward or RNG.
            with torch.no_grad():
                for name, logits in zip(('student', 'teacher'), inputs):
                    logs = F.log_softmax(logits.detach().float(), dim=1)
                    probs = logs.exp()
                    self.dto_response_metrics[name + '_entropy'] = float(-(probs * logs).sum(1).mean())
                    self.dto_response_metrics[name + '_max_probability'] = float(probs.max(1).values.mean())
                    if self._dto_real_labels is not None:
                        self.dto_response_metrics[name + '_global_accuracy'] = float(
                            (logits.argmax(1) == self._dto_real_labels).float().mean())

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
        self._dto_supervised_loss = None
        self._dto_real_labels = None

    def forward(self, images, context_labels, target_labels, real_labels=None, class_embeddings=None):
        self.clear_dto_cache()
        self.dto_response_metrics = {}
        self._dto_real_labels = real_labels
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
        if self.dto_mode == 'safe_dynamic':
            # New FSAR adaptation, not an exact reproduction of DTO-KD:
            # all original label supervision forms ONE supervised objective.
            supervised = base_vis * task.float() + auxiliary.float()
            self._dto_supervised_loss = supervised
            # Roundoff can make a mathematically nonnegative KL slightly < 0.
            if not bool(torch.isfinite(distill.detach())) or float(distill.detach()) < -1e-6:
                raise FloatingPointError('invalid KL in safe_dynamic')
            positive_distill = distill.float().clamp_min(0.)
            log_d = torch.log1p(base_kd * positive_distill)
            log_t = torch.log1p(supervised)
            p_dist, p_task = self.dto_controller.weights
            coordinated = p_dist * log_d + p_task * log_t
            state = self.dto_controller.update_from_logs(log_d, log_t)
            sup_weight = p_task / (1. + float(supervised.detach()))
            self.dto_last_metrics = {
                'pi_distill': p_dist, 'pi_task': p_task,
                'next_pi_distill': state.pi_distill, 'next_pi_task': state.pi_task,
                'task_loss': float(task.detach()), 'distill_loss': float(distill.detach()),
                'supervised_loss': float(supervised.detach()),
                'task_coefficient': base_vis * sup_weight,
                'auxiliary_coefficient': sup_weight,
                'distill_coefficient': p_dist * base_kd / (1. + base_kd * float(positive_distill.detach()))
                                       if float(distill.detach()) >= 0. else 0.,
                'distill_coefficient_upper_bound': base_kd,
                'dto_objective': state.objective,
            }
        elif self.dto_mode in ('paper_exact', 'paper_amortized'):
            # Preserve the declared KL/DCM objectives and ALL auxiliary CE at
            # their original strengths. No log1p, regrouping or KL alpha cap.
            eps = float(self.cfg.DTO_KD.EPS)
            log_d, log_t = log_loss(distill, eps), log_loss(task, eps)
            if self.dto_mode == 'paper_exact':
                solver = shared_lsn_weights(self, distill, task, eps)
                p_dist, p_task = solver['pi_distill'], solver['pi_task']
            else:
                solver = {'solver': 'paper-Eq18-amortized', 'solver_seconds': 0.}
                p_dist, p_task = self.dto_controller.weights
                state = self.dto_controller.update_from_logs(log_d, log_t)
                solver.update(next_pi_distill=state.pi_distill, next_pi_task=state.pi_task)
            coordinated = p_dist * log_d + p_task * log_t
            d, t = float(distill.detach()), float(task.detach())
            self.dto_last_metrics = {
                **solver, 'pi_distill': p_dist, 'pi_task': p_task,
                'task_loss': t, 'distill_loss': d,
                'task_coefficient': p_task / t if t >= eps else 0.,
                'distill_coefficient': p_dist / d if d >= eps else 0.,
                'auxiliary_coefficient': 1., 'log_loss_floor': eps,
                'distill_log_floor_active': d < eps, 'task_log_floor_active': t < eps,
            }
        elif self.dto_mode == 'task_anchored':
            # TS-FSAR adaptation: DCM matching CE is primary. Keep its
            # original coefficient and attenuate only KL on conflict/scale.
            kappa = float(getattr(self.cfg.DTO_KD, 'TASK_ANCHOR_KAPPA', 0.0))
            rho = float(getattr(self.cfg.DTO_KD, 'TASK_ANCHOR_RHO', 1.0))
            solver = task_anchored_kl_weight(
                self, distill, task, base_vis, base_kd,
                float(self.cfg.DTO_KD.EPS), kappa=kappa, rho=rho)
            coordinated = base_vis * task + solver['effective_kl_coefficient'] * distill
            self.dto_last_metrics = {
                **solver,
                'task_loss': float(task.detach()),
                'distill_loss': float(distill.detach()),
                'auxiliary_coefficient': 1.,
            }
        elif self.dto_mode == "dynamic":
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
        output["loss"] = coordinated if self.dto_mode == 'safe_dynamic' else auxiliary + coordinated
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
