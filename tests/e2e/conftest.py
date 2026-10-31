"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - E2E Test Fixtures & Configuration
Authoritative reference constants, tolerances, and progressive module availability detectors.
"""

from __future__ import annotations

import math
import sys
from typing import Any, Dict, Generator, Optional, Tuple
import pytest
import numpy as np

# SI Radians conversion constant
DEG2RAD: float = np.pi / 180.0

# Canonical Physical Constants & Thresholds
THETA_0_DEG: float = 15.0
THETA_GATE_DEG: float = 18.0
THETA_BOOTSTRAP_DEG: float = 15.5
DELTA_THETA_SAV_DEG: float = 0.35

THETA_0_RAD: float = THETA_0_DEG * DEG2RAD
THETA_GATE_RAD: float = THETA_GATE_DEG * DEG2RAD
THETA_BOOTSTRAP_RAD: float = THETA_BOOTSTRAP_DEG * DEG2RAD
DELTA_THETA_SAV_RAD: float = DELTA_THETA_SAV_DEG * DEG2RAD

DEFAULT_KV: float = 0.50
DEFAULT_K: float = 0.80
DEFAULT_ALPHA: float = 0.50
DEFAULT_GAMMA: float = 0.40
DEFAULT_DT_THETA: float = 0.02  # 50 Hz

DEFAULT_LAMBDA_BASE: float = 0.1386  # s^-1 (tau = 7.215 s)
DEFAULT_LAMBDA_WORK: float = 0.20
DEFAULT_ETA_GAIN: float = 2.0
DEFAULT_DT_MASLOW: float = 0.10  # 10 Hz

DEFAULT_EPSILON_THALAMUS: float = 0.15
DEFAULT_KAPPA_AGC: float = 0.50

# DVP-01 Quantitative Acceptance Boundaries
DVP01_PERIOD_MIN_S: float = 1.0
DVP01_PERIOD_MAX_S: float = 5.0
DVP01_PEAK_CV_MAX: float = 0.05  # 5.0%
DVP01_IGNITION_MAX_S: float = 10.0
DVP01_R_INIT_MIN: float = 0.85  # 85%

# Progressive Module Import Checks
try:
    from core.theta_engine import ThetaEngine
    from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot, NEED_NAMES
    HAS_M1 = True
except ImportError:
    HAS_M1 = False
    ThetaEngine = None
    MaslowIntegrator = None
    MaslowSnapshot = None
    NEED_NAMES = None

try:
    from core.thalamus_agc import ThalamusAGC, ThalamusOutput
    from core.basal_ganglia_gate import BasalGangliaGate, GateDecision
    HAS_M2 = True
except ImportError:
    HAS_M2 = False
    ThalamusAGC = None
    ThalamusOutput = None
    BasalGangliaGate = None
    GateDecision = None

try:
    from core.gate1_verification import Gate1Verifier, DVP01Results
    HAS_M3 = True
except ImportError:
    HAS_M3 = False
    Gate1Verifier = None
    DVP01Results = None


@pytest.fixture
def theta_engine():
    """Default Theta Engine instance with canonical parameters."""
    if not HAS_M1:
        pytest.skip("Milestone 1 modules not yet available")
    return ThetaEngine()


@pytest.fixture
def maslow_integrator():
    """Default Maslow Integrator instance with Day 0 precharge."""
    if not HAS_M1:
        pytest.skip("Milestone 1 modules not yet available")
    return MaslowIntegrator()


@pytest.fixture
def coupled_system():
    """Coupled Theta Engine + Maslow Integrator subcortical core."""
    if not HAS_M1:
        pytest.skip("Milestone 1 modules not yet available")
    engine = ThetaEngine()
    integrator = MaslowIntegrator()
    return engine, integrator
