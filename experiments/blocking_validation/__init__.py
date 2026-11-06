"""Causal validation utilities for the FSAR GLAC blocking hypothesis.

The package deliberately does not import the production model at module import
time.  Production training can inject :class:`TS_FSAR` and its episode loader,
while the same utilities remain usable with tiny CPU models in tests.
"""

from .controller import KLControlConfig, KLDecision, KLWeightController
from .metrics import gradient_metrics
from .counterfactual import counterfactual_update
from .protocol import episode_sequence_hash, assert_split_disjoint

__all__ = [
    "KLControlConfig", "KLDecision", "KLWeightController",
    "gradient_metrics", "counterfactual_update", "episode_sequence_hash",
    "assert_split_disjoint",
]
