"""Serializable configuration for the blocking-validation runners."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Mapping
import json

from .controller import KLControlConfig


@dataclass
class BlockingExperimentConfig:
    dataset: str = "ssv2_small"
    shot: int = 1
    seed: int = 0
    output_dir: str = "output/blocking_validation"
    device: str = "cpu"
    max_steps: int = 100
    checkpoint_every: int = 25
    kl_temperature: float = 4.0
    kl: KLControlConfig = field(default_factory=KLControlConfig)

    def validate(self):
        if self.shot not in (1, 5):
            raise ValueError("shot must be 1 or 5")
        self.kl.validate()
        return self

    def to_dict(self):
        result = asdict(self)
        result["kl"] = asdict(self.kl)
        return result


def load_config(path) -> BlockingExperimentConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON instead") from exc
        data = yaml.safe_load(text)
    data = dict(data or {})
    kl = KLControlConfig(**dict(data.pop("kl", {})))
    return BlockingExperimentConfig(kl=kl, **data).validate()
