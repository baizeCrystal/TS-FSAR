"""CLI entry point for E1/E2/E3 without changing the baseline entry point."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def main():
    import main as training
    from .model import DTOTSFSAR
    from .train import train_epoch

    training.TS_FSAR = DTOTSFSAR
    training.train_epoch = train_epoch
    training.main(training.Config())


if __name__ == "__main__":
    main()
