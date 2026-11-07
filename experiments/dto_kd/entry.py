"""CLI entry point for E1/E2/E3 without changing the baseline entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def main():
    import main as training
    from .gradient_utils import activation_context
    from .model import DTOTSFSAR
    from .train import train_epoch

    cfg = training.Config()
    dto = getattr(cfg, "DTO_KD", None)
    enabled = getattr(dto, "MODE", "off") != "off"
    offload = str(getattr(dto, "ACTIVATION_OFFLOAD", "diagnostics"))
    if enabled:
        training.TS_FSAR = DTOTSFSAR
        training.train_epoch = train_epoch
    # E0 keeps its original model and training loop. Full activation offload
    # is an optional storage policy, equally available to every experiment.
    with activation_context(offload if not enabled and offload == "all" else "off"):
        training.main(cfg)


if __name__ == "__main__":
    main()
