"""Numerically stable DTO-KD weight updates.

DTO-KD uses gradients of log losses and chooses simplex weights by minimising
the squared norm of their convex combination.  The exact closed form is used
when both gradients are available (``closed_form_weights``).  The training
adapter uses the paper's amortized update, Eq. (18), so a batch needs one
backward pass rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


def _finite_scalar(value: torch.Tensor, eps: float, *, detach: bool = True) -> torch.Tensor:
    value = value.detach().float().reshape(()) if detach else value.float().reshape(())
    if not torch.isfinite(value.detach()):
        raise FloatingPointError("DTO-KD received a non-finite loss")
    return value.clamp_min(eps)


def closed_form_weights(
    grad_dist: torch.Tensor,
    grad_task: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the unconstrained two-objective simplex solution.

    For ``J=[g_dist,g_task]``, DTO-KD solves ``argmin_{pi in simplex}
    ||J pi||^2``.  The line segment is clipped to the simplex when the
    stationary point lies outside it.  The returned weights are detached.
    """
    g1 = grad_dist.reshape(-1).float()
    g2 = grad_task.reshape(-1).float()
    g11 = torch.dot(g1, g1)
    g22 = torch.dot(g2, g2)
    g12 = torch.dot(g1, g2)
    denom = g11 + g22 - 2.0 * g12
    if not torch.isfinite(denom) or float(denom.abs()) <= eps:
        return torch.tensor(0.5, device=g1.device), torch.tensor(0.5, device=g1.device)
    p1 = ((g22 - g12) / denom).clamp(0.0, 1.0)
    p2 = 1.0 - p1
    return p1.detach(), p2.detach()


@dataclass
class DTOState:
    step: int
    pi_distill: float
    pi_task: float
    log_distill: float
    log_task: float
    objective: float
    finite: bool = True


class DTOController:
    """Amortized DTO-KD simplex controller.

    ``weights`` are used for the current batch.  ``update`` applies Eq. (18)
    to produce weights for the next batch.  The controller is deliberately
    outside the model optimizer, so its state is saved explicitly in a
    checkpoint sidecar and does not add trainable model parameters.
    """

    def __init__(
        self,
        eta: float = 0.01,
        init_distill: float = 0.5,
        eps: float = 1e-8,
        logit_clip: float = 20.0,
    ) -> None:
        if eta < 0 or eps <= 0 or init_distill <= 0 or init_distill >= 1:
            raise ValueError("invalid DTO-KD controller hyperparameters")
        self.eta = float(eta)
        self.eps = float(eps)
        self.logit_clip = float(logit_clip)
        self._logits = torch.tensor(
            [torch.log(torch.tensor(init_distill / (1.0 - init_distill))), 0.0],
            dtype=torch.float64,
        )
        self.step_index = 0
        self.history = []

    def _weights(self) -> torch.Tensor:
        return torch.softmax(self._logits.clamp(-self.logit_clip, self.logit_clip), dim=0)

    @property
    def weights(self) -> Tuple[float, float]:
        p = self._weights()
        return float(p[0]), float(p[1])

    def weighted_log_objective(self, distill_loss: torch.Tensor, task_loss: torch.Tensor) -> torch.Tensor:
        """Compute ``pi_distill log L_distill + pi_task log L_task``."""
        p = self._weights().to(device=distill_loss.device, dtype=distill_loss.dtype)
        ld = _finite_scalar(distill_loss, self.eps, detach=False).to(distill_loss.device)
        lt = _finite_scalar(task_loss, self.eps, detach=False).to(task_loss.device)
        return p[0] * torch.log(ld) + p[1] * torch.log(lt)

    def update(self, distill_loss: torch.Tensor, task_loss: torch.Tensor) -> DTOState:
        ld = _finite_scalar(distill_loss, self.eps)
        lt = _finite_scalar(task_loss, self.eps)
        p = self._weights()
        logs = torch.stack([torch.log(ld), torch.log(lt)]).to(torch.float64)
        scalar = torch.dot(p, logs)
        # d/d pi 1/2 (pi dot log L)^2, followed by the softmax Jacobian.
        grad_pi = scalar * logs
        grad_logits = p * (grad_pi - torch.dot(p, grad_pi))
        if self.eta:
            self._logits = (self._logits - self.eta * grad_logits).clamp(
                -self.logit_clip, self.logit_clip
            )
        self.step_index += 1
        next_p = self._weights()
        state = DTOState(
            step=self.step_index,
            pi_distill=float(next_p[0]),
            pi_task=float(next_p[1]),
            log_distill=float(logs[0]),
            log_task=float(logs[1]),
            objective=float(0.5 * scalar.square()),
        )
        self.history.append(state.__dict__.copy())
        return state

    def state_dict(self) -> Dict[str, object]:
        return {
            "eta": self.eta,
            "eps": self.eps,
            "logit_clip": self.logit_clip,
            "logits": self._logits.clone(),
            "step_index": self.step_index,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.eta = float(state["eta"])
        self.eps = float(state["eps"])
        self.logit_clip = float(state["logit_clip"])
        self._logits = torch.as_tensor(state["logits"], dtype=torch.float64).clone()
        self.step_index = int(state["step_index"])


def flatten_gradients(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    values = [p.grad.detach().float().reshape(-1) for p in parameters if p.grad is not None]
    if not values:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(values)


def gradient_metrics(grad_dist: torch.Tensor, grad_task: torch.Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Return conflict, projection, and dominance metrics used in reports."""
    a, b = grad_dist.float().reshape(-1), grad_task.float().reshape(-1)
    na, nb = a.norm(), b.norm()
    dot = torch.dot(a, b)
    cosine = dot / (na * nb).clamp_min(eps)
    return {
        "distill_norm": float(na),
        "task_norm": float(nb),
        "cosine": float(cosine),
        "conflict": float(cosine < 0),
        "relative_projection_distill_on_task": float(dot / nb.square().clamp_min(eps)),
        "norm_ratio_distill_over_task": float(na / nb.clamp_min(eps)),
    }
