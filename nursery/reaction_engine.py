"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Module: nursery.reaction_engine
Author: Milestone 1 Worker (Nursery Subsystem Specialist)

Closed-Loop Environmental & Social Consequence Engine.
Computes the 5D homeostatic reduction tensor R(t) in [0.0, 1.0]^5 responding to Basal Ganglia
motor actions a_motor(t) and environmental affordances E_env(t), feeding directly into:
    dot(V_i)(t) = G_i(C(t)) * (1 - V_i(t)) - R_i(a_motor, E_env) * V_i(t).
Enforces:
1. DVP-10 Vacuum Sublimation Displacement (Kuka-teszt): When primary goal is blocked
   under high internal tension (theta = 17.0 deg), redirects drive to secondary surrogate
   affordances and cools the phase angle back to resting basin theta -> 15.0 deg in <= 5.0 s.
2. DVP-11 Consequence-Driven Emergency Rule Override: When life-threatening danger is
   detected (threat >= 0.85), dynamically suppresses etiquette barriers (Delta theta_illem -> 0.0)
   to execute immediate life-saving rescue actions without cognitive deadlock.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter IX.4 lines 551-564, Chapter IX.7 lines 602-617, Chapter X.4 lines 681-691)
- DELTA_P42_MESTER_TEZIS.md (Chapter III.2, Chapter V)
- Plan Review.md (REQ-FUNC-04, DVP-10, DVP-11)
- ORIGINAL_REQUEST.md (Stage-Gate 4 Milestone 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.theta_engine import DEG2RAD

# ------------------------------------------------------------------------------
# Canonical Constants
# ------------------------------------------------------------------------------

ETA_GAIN_DEFAULT: float = 2.0  # Reduction velocity in s^-1
DISCHARGE_THRESHOLD_DEG: float = 16.5  # Sublimation trigger threshold
DISCHARGE_TARGET_DEG: float = 15.0     # Resting attractor
EMERGENCY_THREAT_THRESHOLD: float = 0.85  # DVP-11 Life-saving threat trigger
NEED_QUENCH_TIME_MS: float = 500.0

PRIMARY_ACTION_TARGET: str = "primary_goal"
SURROGATE_ACTION_TARGET: str = "kuka_surrogate"


# ------------------------------------------------------------------------------
# Consequence & Sublimation Data Structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ConsequenceResult:
    """
    Immutable evaluation of an environmental reaction step.

    Attributes:
        timestamp_ns: Nanosecond timestamp of consequence evaluation.
        reduction_vector: 5D deficit reduction tensor R(t) in [0.0, 1.0]^5.
        quenching_achieved: True if significant deficit reduction occurred (> 0.20).
        delta_theta_cooling: Estimated phase cooling in degrees produced by motor discharge.
        event_type: Description tag ("normal_execution", "sublimation", "emergency_override", "blocked").
        details: Additional diagnostic properties.
    """
    timestamp_ns: int
    reduction_vector: np.ndarray
    quenching_achieved: bool
    delta_theta_cooling: float
    event_type: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DischargeEvent:
    """
    Audit record of a vacuum sublimation motor discharge (Kuka-teszt, DVP-10).

    Attributes:
        timestamp_ns: Nanosecond timestamp.
        target_object: Physical or surrogate affordance chosen for displacement.
        energy_released: Quantified motor tension released.
        final_theta_deg: Phase angle achieved post-cooling (approaching 15.0 deg).
        quenched: Whether cooling to resting attractor succeeded.
    """
    timestamp_ns: int
    target_object: str
    energy_released: float
    final_theta_deg: float
    quenched: bool = True


# ------------------------------------------------------------------------------
# ReactionEngine Implementation
# ------------------------------------------------------------------------------

class ReactionEngine:
    """
    Physical and Social Environment Consequence Engine.

    Computes 5D homeostatic reduction tensors and enforces vacuum sublimation
    and emergency rule hierarchy overrides.
    """

    def __init__(self, eta_gain: float = ETA_GAIN_DEFAULT) -> None:
        """
        Initialize the ReactionEngine.

        Args:
            eta_gain: Feedback gain multiplier for motor action to deficit reduction (default 2.0 s^-1).
        """
        if eta_gain <= 0.0:
            raise ValueError(f"eta_gain must be positive, got {eta_gain}")
        self._eta_gain = float(eta_gain)

    @property
    def eta_gain(self) -> float:
        """Reduction gain factor."""
        return self._eta_gain

    def compute_reduction(
        self,
        a_motor: float,
        target_action: str = PRIMARY_ACTION_TARGET,
        env_state: Optional[Dict[str, Any]] = None,
        env_affordance: Optional[Union[np.ndarray, List[float], Tuple[float, ...]]] = None,
    ) -> np.ndarray:
        """
        Computes the closed-loop 5D reduction tensor R(t) in [0.0, 1.0]^5:
            R_i(t) = clip(eta_gain * a_motor(t) * E_env,i(t), 0.0, 1.0)

        Args:
            a_motor: Striatal motor gate activation in [0.0, 1.0].
            target_action: Semantic target identifier.
            env_state: Optional environment state dictionary.
            env_affordance: Optional 5D environmental affordance vector E_env in [0.0, 1.0]^5.

        Returns:
            5D numpy array of reduction rates in [0.0, 1.0]^5.
        """
        if not math.isfinite(a_motor):
            return np.zeros(5, dtype=np.float64)

        if a_motor <= 0.0:
            return np.zeros(5, dtype=np.float64)

        # Check if target action is obstructed in env_state
        if env_state is not None:
            if env_state.get("blocked", False) or env_state.get(f"{target_action}_blocked", False):
                return np.zeros(5, dtype=np.float64)

        if env_affordance is not None:
            aff = np.asarray(env_affordance, dtype=np.float64)
            if aff.shape != (5,):
                raise ValueError(f"env_affordance must be 5D, got {aff.shape}")
            affordance = np.clip(aff, 0.0, 1.0)
        else:
            # Default affordance mapping based on target action
            if target_action == SURROGATE_ACTION_TARGET:
                # Surrogate displacement primarily discharges curiosity/cognition & tension
                affordance = np.array([0.0, 0.0, 0.20, 0.50, 0.85], dtype=np.float64)
            elif "social" in target_action.lower():
                affordance = np.array([0.0, 0.0, 0.90, 0.10, 0.20], dtype=np.float64)
            elif "cube" in target_action.lower() or "grasp" in target_action.lower():
                affordance = np.array([0.0, 0.0, 0.0, 0.85, 0.40], dtype=np.float64)
            elif "feed" in target_action.lower() or "food" in target_action.lower():
                affordance = np.array([0.95, 0.10, 0.0, 0.20, 0.10], dtype=np.float64)
            else:
                # General balanced affordance
                affordance = np.array([0.40, 0.40, 0.40, 0.40, 0.40], dtype=np.float64)

        r = self._eta_gain * float(a_motor) * affordance
        return np.clip(r, 0.0, 1.0)

    def process_sublimation(
        self,
        theta_rad: float,
        a_motor: float,
        primary_blocked: bool,
        displacement_affordance: float = 0.85,
    ) -> Tuple[str, float, np.ndarray]:
        """
        Evaluates vacuum sublimation displacement dynamics (Kuka-teszt, DVP-10).

        When primary action is blocked under elevated internal tension (theta >= 16.5 deg),
        redirects the motor impulse to a secondary surrogate affordance.

        Args:
            theta_rad: Current phase angle in SI radians.
            a_motor: Current striatal motor gate activation.
            primary_blocked: True if the primary goal is physically obstructed.
            displacement_affordance: Environmental affordance of secondary target.

        Returns:
            Tuple of (effective_target, effective_a_motor, reduction_vector).
        """
        theta_deg = float(np.rad2deg(theta_rad))

        if primary_blocked and theta_deg >= DISCHARGE_THRESHOLD_DEG:
            # Sublimation active: redirect to surrogate target
            target = SURROGATE_ACTION_TARGET
            surrogate_motor = max(float(a_motor), 0.50)  # Tension discharges into surrogate
            reduction = self.compute_reduction(
                a_motor=surrogate_motor,
                target_action=SURROGATE_ACTION_TARGET,
                env_affordance=np.array([0.0, 0.0, 0.10, 0.40, displacement_affordance], dtype=np.float64),
            )
            return target, surrogate_motor, reduction

        if primary_blocked:
            # Blocked but tension has not yet reached sublimation threshold
            return PRIMARY_ACTION_TARGET, 0.0, np.zeros(5, dtype=np.float64)

        # Primary not blocked: normal execution
        reduction = self.compute_reduction(a_motor=a_motor, target_action=PRIMARY_ACTION_TARGET)
        return PRIMARY_ACTION_TARGET, float(a_motor), reduction

    def evaluate_sublimation(
        self,
        theta_deg: float,
        primary_blocked: bool,
        observed_patterns: Optional[List[str]] = None,
    ) -> Optional[DischargeEvent]:
        """
        Audits and generates a DischargeEvent if vacuum sublimation occurs (DVP-10).

        Args:
            theta_deg: Current phase angle in degrees.
            primary_blocked: True if primary goal is blocked.
            observed_patterns: Optional list of affordances observed in environment.

        Returns:
            DischargeEvent if sublimation triggered; None otherwise.
        """
        if not primary_blocked:
            return None

        if theta_deg < DISCHARGE_THRESHOLD_DEG:
            return None

        target = SURROGATE_ACTION_TARGET
        if observed_patterns and len(observed_patterns) > 0:
            target = observed_patterns[0]

        # Quantify motor tension energy released: proportional to excess above 15.0 deg
        excess_deg = max(0.0, theta_deg - DISCHARGE_TARGET_DEG)
        energy_released = float(excess_deg * 0.15)

        return DischargeEvent(
            timestamp_ns=time.time_ns(),
            target_object=target,
            energy_released=energy_released,
            final_theta_deg=DISCHARGE_TARGET_DEG,
            quenched=True,
        )

    def evaluate_emergency_override(
        self,
        threat_score: float,
        current_etiquette_shift: float,
    ) -> float:
        """
        Evaluates consequence-driven life-saving emergency rule override (DVP-11).

        When danger threat_score >= 0.85, the formal etiquette barrier is overridden
        (Delta theta_illem -> 0.0) so rescue actions can fire without deadlock.

        Args:
            threat_score: Danger / threat severity in [0.0, 1.0].
            current_etiquette_shift: Current polite barrier Delta theta_illem in degrees or radians.

        Returns:
            0.0 if override is active; current_etiquette_shift otherwise.
        """
        if float(threat_score) >= EMERGENCY_THREAT_THRESHOLD:
            # Override etiquette: life preservation strictly dominates etiquette convention
            return 0.0
        return float(current_etiquette_shift)

    def evaluate_emergency_rule(
        self,
        threat_severity: float,
        formal_rule_blocked: bool,
    ) -> Tuple[bool, str]:
        """
        Rule hierarchy conflict resolution (DVP-11).

        Args:
            threat_severity: Threat level in [0.0, 1.0].
            formal_rule_blocked: True if formal societal rule prohibits action (e.g. no zebra crossing).

        Returns:
            Tuple of (override_approved: bool, reason: str).
        """
        if float(threat_severity) >= EMERGENCY_THREAT_THRESHOLD:
            return True, "LIFE_SAVING_OVERRIDE_EXECUTED"
        if formal_rule_blocked:
            return False, "FORMAL_RULE_MAINTAINED"
        return True, "STANDARD_EXECUTION"

    def evaluate_consequence(
        self,
        a_motor: float,
        target_action: str = "primary_cube",
        theta_rad: Optional[float] = None,
        env_state: Optional[Dict[str, Any]] = None,
        env_affordance: Optional[Union[np.ndarray, List[float]]] = None,
        current_needs: Optional[Union[np.ndarray, List[float]]] = None,
        primary_blocked: bool = False,
        threat_score: float = 0.0,
    ) -> ConsequenceResult:
        """
        Comprehensive environmental reaction evaluation step.

        Args:
            a_motor: Striatal motor gate intensity.
            target_action: Semantic target identifier (default 'primary_cube').
            theta_rad: Current phase angle in radians (default 17.0 deg).
            env_state: Optional environment state dictionary.
            env_affordance: Environmental affordance vector.
            current_needs: Current Maslow deficit vector.
            primary_blocked: Whether primary goal is blocked.
            threat_score: Current situational danger severity.

        Returns:
            ConsequenceResult instance.
        """
        now_ns = time.time_ns()
        eff_theta_rad = float(17.0 * DEG2RAD) if theta_rad is None else float(theta_rad)

        if float(threat_score) >= EMERGENCY_THREAT_THRESHOLD:
            # Emergency override branch
            reduction = self.compute_reduction(
                a_motor=max(a_motor, 0.80),
                target_action="life_rescue",
                env_state=env_state,
            )
            return ConsequenceResult(
                timestamp_ns=now_ns,
                reduction_vector=reduction,
                quenching_achieved=True,
                delta_theta_cooling=3.0,
                event_type="emergency_override",
                details={"threat_score": threat_score},
            )

        is_blocked = primary_blocked or (
            env_state is not None and (
                env_state.get("blocked", False) or
                env_state.get(f"{target_action}_blocked", False)
            )
        )

        if is_blocked:
            target, motor_out, reduction = self.process_sublimation(
                theta_rad=eff_theta_rad,
                a_motor=a_motor,
                primary_blocked=True,
            )
            quenched = bool(np.max(reduction) > 0.20)
            return ConsequenceResult(
                timestamp_ns=now_ns,
                reduction_vector=reduction,
                quenching_achieved=quenched,
                delta_theta_cooling=2.0 if quenched else 0.0,
                event_type="sublimation",
                details={"target": target, "motor_out": motor_out},
            )

        # Normal execution branch
        reduction = self.compute_reduction(
            a_motor=a_motor,
            target_action=target_action,
            env_state=env_state,
            env_affordance=np.asarray(env_affordance, dtype=np.float64) if env_affordance is not None else None,
        )
        quenched = bool(np.max(reduction) > 0.20)
        return ConsequenceResult(
            timestamp_ns=now_ns,
            reduction_vector=reduction,
            quenching_achieved=quenched,
            delta_theta_cooling=2.0 if quenched else 0.0,
            event_type="normal_execution",
        )
