"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_reaction_engine.py
Author: Stage-Gate 4 Milestone 2 Specialist

Verifies:
1. Closed-loop 5D reduction tensor R(t) in [0.0, 1.0]^5 calculation with feedback gain eta_gain = 2.0.
2. DVP-10 Vacuum sublimation displacement (Kuka-teszt):
   Primary goal blocked under elevated tension (theta >= 16.5 deg) redirects drive to surrogate affordance,
   emitting DischargeEvent and cooling the phase oscillator to resting attractor theta -> 15.0 deg.
3. DVP-11 Consequence-driven life-saving emergency rule override:
   Threat severity >= 0.85 overrides societal etiquette barriers (Delta theta_illem -> 0.0) without deadlock.
4. Affordance vector mappings, obstruction handling, and edge case resilience (NaN, non-positive motor drive).
5. Immutable ConsequenceResult and DischargeEvent audit records.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.theta_engine import DEG2RAD
from nursery.reaction_engine import (
    DISCHARGE_TARGET_DEG,
    DISCHARGE_THRESHOLD_DEG,
    EMERGENCY_THREAT_THRESHOLD,
    ETA_GAIN_DEFAULT,
    PRIMARY_ACTION_TARGET,
    SURROGATE_ACTION_TARGET,
    ConsequenceResult,
    DischargeEvent,
    ReactionEngine,
)


# ==============================================================================
# 1. CONSTANTS & INITIALIZATION
# ==============================================================================

def test_reaction_engine_constants() -> None:
    assert ETA_GAIN_DEFAULT == 2.0
    assert DISCHARGE_THRESHOLD_DEG == 16.5
    assert DISCHARGE_TARGET_DEG == 15.0
    assert EMERGENCY_THREAT_THRESHOLD == 0.85
    assert PRIMARY_ACTION_TARGET == "primary_goal"
    assert SURROGATE_ACTION_TARGET == "kuka_surrogate"


def test_reaction_engine_init_and_validation() -> None:
    engine = ReactionEngine(eta_gain=2.0)
    assert engine.eta_gain == 2.0

    with pytest.raises(ValueError, match="eta_gain must be positive"):
        ReactionEngine(eta_gain=0.0)

    with pytest.raises(ValueError, match="eta_gain must be positive"):
        ReactionEngine(eta_gain=-1.5)


# ==============================================================================
# 2. 5D REDUCTION TENSOR R(t) COMPUTATION
# ==============================================================================

def test_compute_reduction_zero_and_negative_motor() -> None:
    engine = ReactionEngine()

    # Zero motor drive
    r_zero = engine.compute_reduction(a_motor=0.0)
    np.testing.assert_allclose(r_zero, np.zeros(5))

    # Negative motor drive
    r_neg = engine.compute_reduction(a_motor=-0.5)
    np.testing.assert_allclose(r_neg, np.zeros(5))

    # NaN / Inf motor drive
    r_nan = engine.compute_reduction(a_motor=float("nan"))
    np.testing.assert_allclose(r_nan, np.zeros(5))


def test_compute_reduction_affordances() -> None:
    engine = ReactionEngine(eta_gain=2.0)

    # 1. Custom affordance vector
    custom_aff = np.array([0.5, 0.0, 0.2, 0.1, 0.8], dtype=np.float64)
    # R = clip(2.0 * 0.5 * [0.5, 0.0, 0.2, 0.1, 0.8], 0, 1) = [0.5, 0.0, 0.2, 0.1, 0.8]
    r_custom = engine.compute_reduction(a_motor=0.5, env_affordance=custom_aff)
    np.testing.assert_allclose(r_custom, custom_aff)

    # 2. Clamping at upper bound 1.0
    # a_motor = 1.0, eta = 2.0 -> 2.0 * [0.8, ...] = [1.6, ...] -> clamped to 1.0
    r_clamped = engine.compute_reduction(a_motor=1.0, env_affordance=np.ones(5) * 0.8)
    np.testing.assert_allclose(r_clamped, np.ones(5))

    # 3. Surrogate target default affordance
    r_surrogate = engine.compute_reduction(a_motor=0.5, target_action=SURROGATE_ACTION_TARGET)
    assert r_surrogate[4] > r_surrogate[0]  # Cognition/tension reduction dominates

    # 4. Blocked target in env_state produces zero reduction
    r_blocked = engine.compute_reduction(
        a_motor=0.8,
        target_action="eat_apple",
        env_state={"blocked": True},
    )
    np.testing.assert_allclose(r_blocked, np.zeros(5))

    r_action_blocked = engine.compute_reduction(
        a_motor=0.8,
        target_action="open_door",
        env_state={"open_door_blocked": True},
    )
    np.testing.assert_allclose(r_action_blocked, np.zeros(5))


# ==============================================================================
# 3. DVP-10 VACUUM SUBLIMATION DISPLACEMENT (KUKA-TESZT)
# ==============================================================================

def test_process_sublimation_active_discharge() -> None:
    engine = ReactionEngine()

    # High internal tension: theta = 17.0 deg (above 16.5 deg threshold)
    theta_rad = 17.0 * DEG2RAD
    a_motor = 0.2  # Low initial motor

    # When primary goal is blocked, sublimation activates:
    # 1. Target redirects to 'kuka_surrogate'
    # 2. Motor drive boosts to at least 0.50
    # 3. Reduction tensor discharges tension
    target, eff_motor, reduction = engine.process_sublimation(
        theta_rad=theta_rad,
        a_motor=a_motor,
        primary_blocked=True,
    )
    assert target == SURROGATE_ACTION_TARGET
    assert eff_motor >= 0.50
    assert reduction[4] > 0.50  # 5th component discharged


def test_process_sublimation_below_threshold() -> None:
    engine = ReactionEngine()

    # Lower tension: theta = 16.0 deg (below 16.5 deg threshold)
    theta_rad = 16.0 * DEG2RAD
    target, eff_motor, reduction = engine.process_sublimation(
        theta_rad=theta_rad,
        a_motor=0.5,
        primary_blocked=True,
    )
    # Not yet sublimating: target remains primary, action blocked (0.0 motor)
    assert target == PRIMARY_ACTION_TARGET
    assert eff_motor == 0.0
    np.testing.assert_allclose(reduction, np.zeros(5))


def test_process_sublimation_unblocked() -> None:
    engine = ReactionEngine()

    # Unblocked: standard execution
    theta_rad = 17.0 * DEG2RAD
    target, eff_motor, reduction = engine.process_sublimation(
        theta_rad=theta_rad,
        a_motor=0.7,
        primary_blocked=False,
    )
    assert target == PRIMARY_ACTION_TARGET
    assert eff_motor == 0.7
    assert np.any(reduction > 0.0)


def test_evaluate_sublimation_audit_event() -> None:
    engine = ReactionEngine()

    # Active sublimation at 17.5 deg
    event = engine.evaluate_sublimation(
        theta_deg=17.5,
        primary_blocked=True,
        observed_patterns=["toy_bin", "wooden_block"],
    )
    assert event is not None
    assert isinstance(event, DischargeEvent)
    assert event.target_object == "toy_bin"
    assert event.final_theta_deg == 15.0
    assert event.quenched is True
    assert event.energy_released > 0.0

    # No sublimation if primary is not blocked
    assert engine.evaluate_sublimation(theta_deg=17.5, primary_blocked=False) is None

    # No sublimation if theta < 16.5 deg
    assert engine.evaluate_sublimation(theta_deg=16.0, primary_blocked=True) is None


# ==============================================================================
# 4. DVP-11 LIFE-SAVING EMERGENCY RULE OVERRIDE
# ==============================================================================

def test_evaluate_emergency_override() -> None:
    engine = ReactionEngine()

    current_etiquette = 4.5  # Polite barrier of 4.5 degrees

    # Threat >= 0.85 -> overrides etiquette to 0.0
    res_override = engine.evaluate_emergency_override(
        threat_score=0.90,
        current_etiquette_shift=current_etiquette,
    )
    assert res_override == 0.0

    # Threat < 0.85 -> etiquette maintained
    res_normal = engine.evaluate_emergency_override(
        threat_score=0.40,
        current_etiquette_shift=current_etiquette,
    )
    assert res_normal == current_etiquette


def test_evaluate_emergency_rule_hierarchy() -> None:
    engine = ReactionEngine()

    # Life-saving override dominates formal prohibitions
    override_ok, reason = engine.evaluate_emergency_rule(
        threat_severity=0.95,
        formal_rule_blocked=True,
    )
    assert override_ok is True
    assert reason == "LIFE_SAVING_OVERRIDE_EXECUTED"

    # Moderate threat without emergency -> formal rule respected
    rule_blocked, reason_blocked = engine.evaluate_emergency_rule(
        threat_severity=0.60,
        formal_rule_blocked=True,
    )
    assert rule_blocked is False
    assert reason_blocked == "FORMAL_RULE_MAINTAINED"

    # No threat and no formal prohibition -> standard execution
    std_ok, std_reason = engine.evaluate_emergency_rule(
        threat_severity=0.10,
        formal_rule_blocked=False,
    )
    assert std_ok is True
    assert std_reason == "STANDARD_EXECUTION"


# ==============================================================================
# 5. COMPREHENSIVE CONSEQUENCE EVALUATION STEP
# ==============================================================================

def test_evaluate_consequence_branches() -> None:
    engine = ReactionEngine()

    # Branch 1: Emergency override
    res_emerg = engine.evaluate_consequence(
        a_motor=0.3,
        threat_score=0.92,
    )
    assert res_emerg.event_type == "emergency_override"
    assert res_emerg.quenching_achieved is True
    assert res_emerg.delta_theta_cooling == 3.0

    # Branch 2: Sublimation displacement
    res_sub = engine.evaluate_consequence(
        a_motor=0.4,
        theta_rad=17.2 * DEG2RAD,
        primary_blocked=True,
    )
    assert res_sub.event_type == "sublimation"
    assert res_sub.details["target"] == SURROGATE_ACTION_TARGET

    # Branch 3: Normal execution
    res_norm = engine.evaluate_consequence(
        a_motor=0.6,
        primary_blocked=False,
    )
    assert res_norm.event_type == "normal_execution"
    assert isinstance(res_norm, ConsequenceResult)
    assert res_norm.quenching_achieved is True
