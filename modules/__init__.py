"""
ΔP-42 Sovereign Cognitive Architecture - Modules Package (Gate 2).
Exports fast subcortical and cortical modules for the cognitive architecture:
- AmygdalaOverride, AmygdalaOutput (LeDoux Dual Pathway & STDP Associative Conditioning)
- CerebellumEngine, CerebellarState (20 Hz Motor Smoothing & Purkinje LTD Plasticity)
"""

from __future__ import annotations

from modules.amygdala_override import AmygdalaOutput, AmygdalaOverride
from modules.cerebellum_engine import CerebellarState, CerebellumEngine
from modules.pfc_executive import GoalCandidate, GoalStructure, PFCExecutive
from modules.sensorimotor_cortex import (
    ProprioceptiveSnapshot,
    ProprioceptiveState,
    SensorimotorCortex,
)

__all__ = [
    "AmygdalaOutput",
    "AmygdalaOverride",
    "CerebellarState",
    "CerebellumEngine",
    "GoalCandidate",
    "GoalStructure",
    "PFCExecutive",
    "ProprioceptiveSnapshot",
    "ProprioceptiveState",
    "SensorimotorCortex",
]

