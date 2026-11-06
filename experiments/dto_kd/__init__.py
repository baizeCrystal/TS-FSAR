"""DTO-KD coordination for the TS-FSAR training objective.

The implementation follows the two-objective, amortized formulation in
DTO-KD (ICLR 2026).  It is an adapter around the existing TS-FSAR forward
pass: model architecture, data sampling, and inference remain unchanged.
"""

from .controller import DTOController
from .model import DTOTSFSAR

__all__ = ["DTOController", "DTOTSFSAR"]
