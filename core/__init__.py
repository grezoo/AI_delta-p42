"""
ΔP-42 Sovereign Cognitive Core (Gate 1)
Package export and version definition.
"""

from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot, NEED_NAMES
from core.theta_engine import DEG2RAD, ThetaEngine

__version__ = "0.1.0-gate1"

__all__ = [
    "__version__",
    "DEG2RAD",
    "NEED_NAMES",
    "MaslowIntegrator",
    "MaslowSnapshot",
    "ThetaEngine",
]
