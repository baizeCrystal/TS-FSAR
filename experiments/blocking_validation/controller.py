"""Small, deterministic controllers for regulating only the GLAC KL term."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence

import torch


@dataclass(frozen=True)
class KLControlConfig:
    """Configuration shared by A and B.

    ``attenuation`` is the retained fraction of KL on an intervention step.
    A value of 0.5 therefore halves KL and a value of 1.0 is an identity
    controller.  ``interference_threshold`` applies to the relative KL
    projection ``dot(g_kl, g_match) / ||g_match||^2``.
    """

    mode: str = "original"
    attenuation: float = 0.5
    interference_threshold: float = 0.0
    frequency: Optional[float] = None
    seed: int = 0

    def validate(self) -> "KLControlConfig":
        if self.mode not in {"original", "selective", "random", "uniform"}:
            raise ValueError(f"unknown KL control mode: {self.mode}")
        if not 0.0 <= self.attenuation <= 1.0:
            raise ValueError("attenuation must be in [0, 1]")
        if self.frequency is not None and not 0.0 <= self.frequency <= 1.0:
            raise ValueError("frequency must be in [0, 1]")
        return self


@dataclass(frozen=True)
class KLDecision:
    step: int
    weight: float
    selected: bool
    reason: str
    relative_projection: Optional[float] = None


class KLWeightController:
    """Deterministic KL schedule with an auditable decision record.

    ``plan`` is preferred for B: it first derives the selective mask from
    Meta-Train interference values, then uses exactly the same number of
    intervention steps for random regulation.  Uniform regulation spreads the
    same total KL budget over every step.  No controller has trainable state.
    """

    def __init__(self, config: KLControlConfig):
        self.config = config.validate()
        self.decisions: List[KLDecision] = []

    @property
    def config_dict(self):
        return asdict(self.config)

    def _weight(self, selected: bool) -> float:
        return self.config.attenuation if selected else 1.0

    def decide(self, step: int, relative_projection: Optional[float] = None,
               *, selected: Optional[bool] = None) -> KLDecision:
        mode = self.config.mode
        if mode == "original":
            chosen, reason = False, "original"
        elif selected is not None:
            chosen, reason = bool(selected), mode
        elif mode == "selective":
            value = float("inf") if relative_projection is None else relative_projection
            chosen, reason = value < self.config.interference_threshold, "projection"
        elif mode == "uniform":
            chosen, reason = True, "uniform"
        else:
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed + int(step))
            p = 0.0 if self.config.frequency is None else self.config.frequency
            chosen = bool(torch.rand((), generator=generator).item() < p)
            reason = "random"
        decision = KLDecision(int(step), self._weight(chosen), chosen, reason,
                              None if relative_projection is None else float(relative_projection))
        self.decisions.append(decision)
        return decision

    def plan(self, relative_projections: Sequence[float]) -> List[KLDecision]:
        """Create a fixed B schedule from Meta-Train-only diagnostics."""
        values = [float(v) for v in relative_projections]
        n = len(values)
        if self.config.mode == "original" or n == 0:
            selected = [False] * n
        elif self.config.mode == "selective":
            selected = [v < self.config.interference_threshold for v in values]
        elif self.config.mode == "random":
            k = round((self.config.frequency if self.config.frequency is not None else
                       sum(v < self.config.interference_threshold for v in values) / n) * n)
            k = max(0, min(n, k))
            g = torch.Generator(device="cpu").manual_seed(self.config.seed)
            selected = [False] * n
            if k:
                for i in torch.randperm(n, generator=g)[:k].tolist():
                    selected[i] = True
        else:  # uniform: equal total retained-KL budget as selected schedule
            k = sum(v < self.config.interference_threshold for v in values)
            selected = [True] * n
            if n:
                retained = (n - k + k * self.config.attenuation) / n
                # Encode the uniform weight directly below.
                decisions = [KLDecision(i, retained, True, "uniform", values[i])
                             for i in range(n)]
                self.decisions.extend(decisions)
                return decisions
        decisions = [self.decide(i, values[i], selected=selected[i]) for i in range(n)]
        return decisions

    @staticmethod
    def weights(decisions: Iterable[KLDecision]) -> List[float]:
        return [float(d.weight) for d in decisions]
