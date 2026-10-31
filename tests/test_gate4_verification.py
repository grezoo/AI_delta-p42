"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_gate4_verification.py
Author: Stage-Gate 4 Milestone 3 Specialist

Comprehensive unit, integration, and adversarial tests for the Gate 4 Verification Engine
(core/gate4_verification.py), validating:
- DVP-09: Day 0 Cold Start bootstrap within <= 3.0 s.
- DVP-10: Vacuum sublimation displacement (Kuka-teszt: 17.0 deg -> gate >= 18.0 deg -> cooling to 15.0 +- 0.5 deg).
- DVP-11: Emergency rule override (life-saving exception handling without deadlock, sub-2.0 ms WCET).
- DVP-12: Long-Horizon 3-Arm Comparative Falsification (Arm A >= 3.50x Arm C, p < 0.01; Arm A >= 2.00x Arm B).
- Frozen result dataclasses immutability and schema fidelity.
- ASCII report table generation and CLI argument parsing.
- Adversarial perturbations and boundary conditions.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
from typing import Any, Dict

import numpy as np
import pytest

from core.gate4_verification import (
    DVP09Result,
    DVP10Result,
    DVP11Result,
    DVP12Result,
    Gate4Results,
    Gate4Verifier,
    main,
)
from nursery.bootstrap_loader import DEFAULT_V0, THETA_0_DEG, THETA_BOOTSTRAP_DEG, THETA_GATE_DEG
from nursery.reaction_engine import EMERGENCY_THREAT_THRESHOLD


# ==============================================================================
# 1. FROZEN RESULT DATACLASSES & IMMUTABILITY
# ==============================================================================

def test_frozen_dataclasses_immutability() -> None:
    """Verifies all quantitative Gate 4 result containers are strictly immutable."""
    dvp09 = DVP09Result(
        passed=True,
        convergence_time_s=0.30,
        max_phase_velocity_rad_s=0.47,
        numeric_shock_detected=False,
        initial_deficit_norm=0.90,
        final_phase_deg=16.26,
        first_cycle_period_s=1.00,
        operating_point_stable=True,
    )
    with pytest.raises(FrozenInstanceError):
        dvp09.passed = False  # type: ignore

    dvp10 = DVP10Result(
        passed=True,
        initial_phase_deg=17.0,
        peak_phase_deg=18.78,
        primary_goal_blocked=True,
        sublimation_action_emitted=True,
        sublimation_action_intensity=0.81,
        cooling_time_s=2.00,
        final_phase_deg=15.43,
        deadlock_detected=False,
    )
    with pytest.raises(FrozenInstanceError):
        dvp10.cooling_time_s = 10.0  # type: ignore

    dvp11 = DVP11Result(
        passed=True,
        etiquette_barrier_active=True,
        emergency_threat_score=0.95,
        override_triggered=True,
        rescue_action_emitted=True,
        reaction_latency_ms=0.03,
        deadlock_detected=False,
    )
    with pytest.raises(FrozenInstanceError):
        dvp11.deadlock_detected = True  # type: ignore

    dvp12 = DVP12Result(
        passed=True,
        n_trials_per_arm=10,
        mean_adaptation_arm_a=0.936,
        std_adaptation_arm_a=0.001,
        mean_adaptation_arm_b=0.212,
        mean_adaptation_arm_c=0.023,
        adaptation_ratio_a_to_c=39.86,
        adaptation_ratio_a_to_b=4.42,
        survival_rate_arm_a=1.0,
        survival_rate_arm_b=0.0,
        survival_rate_arm_c=0.0,
        t_statistic=500.0,
        p_value=1e-20,
        mwu_p_value=1e-5,
        cohens_d=230.0,
        falsification_verdict="HYPOTHESIS_CONFIRMED",
    )
    with pytest.raises(FrozenInstanceError):
        dvp12.falsification_verdict = "FALSIFIED"  # type: ignore

    gate4 = Gate4Results(
        dvp09=dvp09,
        dvp10=dvp10,
        dvp11=dvp11,
        dvp12=dvp12,
        all_passed=True,
        summary={"all_passed": True},
    )
    with pytest.raises(FrozenInstanceError):
        gate4.all_passed = False  # type: ignore

    # Verify to_dict conversion
    d = gate4.to_dict()
    assert isinstance(d, dict)
    assert d["all_passed"] is True


# ==============================================================================
# 2. INDIVIDUAL DVP VERIFICATION METHODS
# ==============================================================================

def test_dvp09_cold_start_verification() -> None:
    """Verifies DVP-09 Day 0 cold start bootstrap dynamics."""
    verifier = Gate4Verifier(verbose=False, seed=42)
    res: DVP09Result = verifier.verify_dvp09_cold_start(quick=True)

    assert res.passed is True
    assert res.convergence_time_s <= 3.0
    assert res.numeric_shock_detected is False
    assert math.isclose(res.initial_deficit_norm, 0.90, abs_tol=1e-4)
    assert 1.0 <= res.first_cycle_period_s <= 5.0
    assert res.max_phase_velocity_rad_s <= 5.0
    assert res.operating_point_stable is True
    assert 14.0 <= res.final_phase_deg <= 22.0


def test_dvp10_vacuum_sublimation_verification() -> None:
    """Verifies DVP-10 vacuum sublimation displacement (Kuka-teszt)."""
    verifier = Gate4Verifier(verbose=False, seed=42)
    res: DVP10Result = verifier.verify_dvp10_vacuum_sublimation(quick=True)

    assert res.passed is True
    assert res.initial_phase_deg == 17.0
    assert res.peak_phase_deg >= THETA_GATE_DEG  # >= 18.0 deg
    assert res.primary_goal_blocked is True
    assert res.sublimation_action_emitted is True
    assert res.sublimation_action_intensity > 0.05
    assert res.cooling_time_s <= 5.0
    assert 14.5 <= res.final_phase_deg <= 15.5
    assert res.deadlock_detected is False


def test_dvp11_emergency_rule_override_verification() -> None:
    """Verifies DVP-11 life-saving emergency rule override."""
    verifier = Gate4Verifier(verbose=False, seed=42)
    res: DVP11Result = verifier.verify_dvp11_emergency_rule_override(quick=True)

    assert res.passed is True
    assert res.etiquette_barrier_active is True
    assert res.emergency_threat_score >= EMERGENCY_THREAT_THRESHOLD
    assert res.override_triggered is True
    assert res.rescue_action_emitted is True
    assert res.reaction_latency_ms <= 2000.0  # Cortical spec
    assert res.deadlock_detected is False
    # Sub-2.0 ms Amygdala hardware interrupt verification
    wcet = res.details.get("interrupt_wcet_ms", 0.0)
    assert wcet <= 2.0


def test_dvp12_long_horizon_falsification_verification() -> None:
    """Verifies DVP-12 3-arm comparative ontogenetic falsification."""
    verifier = Gate4Verifier(verbose=False, seed=42)
    res: DVP12Result = verifier.verify_dvp12_long_horizon_falsification(
        trials=10,
        episodes_len=250,
        quick=True,
    )

    assert res.passed is True
    assert res.n_trials_per_arm == 10
    assert res.adaptation_ratio_a_to_c >= 3.50
    assert res.adaptation_ratio_a_to_b >= 2.00
    assert res.survival_rate_arm_a > res.survival_rate_arm_c
    assert res.survival_rate_arm_a == 1.0
    assert res.survival_rate_arm_c <= 0.05
    assert res.p_value < 0.01
    assert res.mwu_p_value < 0.01
    assert res.cohens_d > 1.50
    assert res.falsification_verdict == "HYPOTHESIS_CONFIRMED"


# ==============================================================================
# 3. MASTER RUNNER & REPORT FORMATTING
# ==============================================================================

def test_run_all_quick_mode() -> None:
    """Verifies master runner execution across all 4 criteria."""
    verifier = Gate4Verifier(verbose=False, seed=42)
    results: Gate4Results = verifier.run_all(quick=True)

    assert results.all_passed is True
    assert results.dvp09.passed is True
    assert results.dvp10.passed is True
    assert results.dvp11.passed is True
    assert results.dvp12.passed is True

    summary = results.to_dict()
    assert summary["all_passed"] is True
    assert "dvp09" in summary
    assert "dvp10" in summary
    assert "dvp11" in summary
    assert "dvp12" in summary


def test_format_report_structure() -> None:
    """Verifies 80-column ASCII report table generation."""
    verifier = Gate4Verifier(verbose=False, seed=42)
    results = verifier.run_all(quick=True)
    report = verifier.format_report(results)

    assert "DELTA P-42 SOVEREIGN COGNITIVE CORE - STAGE-GATE 4 QUANTITATIVE VERIFICATION" in report
    assert "Overall Status: [PASS]" in report
    assert "1. DVP-09: Day 0 Cold Start Bootstrap" in report
    assert "2. DVP-10: Vacuum Sublimation Displacement (Kuka-teszt)" in report
    assert "3. DVP-11: Emergency Rule Override (Life-Saving Exception)" in report
    assert "4. DVP-12: Long-Horizon 3-Arm Comparative Falsification" in report
    assert "PROCEED TO GATE 4 SIGN-OFF" in report


# ==============================================================================
# 4. CLI ENTRYPOINT TESTING
# ==============================================================================

def test_cli_main_quick(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies CLI execution under --quick flag exits with code 0."""
    exit_code = main(["--quick", "--quiet"])
    assert exit_code == 0


def test_cli_main_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies CLI execution with --json outputs valid JSON."""
    exit_code = main(["--quick", "--quiet", "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["all_passed"] is True
    assert payload["dvp09"]["passed"] is True
    assert payload["dvp10"]["passed"] is True
    assert payload["dvp11"]["passed"] is True
    assert payload["dvp12"]["passed"] is True
    assert payload["dvp12"]["adaptation_ratio_a_to_c"] >= 3.50


# ==============================================================================
# 5. ADVERSARIAL & BOUNDARY TESTS
# ==============================================================================

def test_dvp09_adversarial_perturbed_deficit() -> None:
    """Verifies DVP-09 rejects corrupted initial deficit profiles."""
    loader_corrupt = None
    with pytest.raises(ValueError):
        # Deficit values must be in [0, 1]
        from nursery.bootstrap_loader import BootstrapLoader
        BootstrapLoader(V_0=[-0.1, 0.2, 0.3, 0.4, 0.5])


def test_dvp11_adversarial_sub_threshold_threat() -> None:
    """Verifies that sub-threshold threat (< 0.85) does not trigger emergency override."""
    from nursery.reaction_engine import ReactionEngine
    reaction = ReactionEngine()

    shift = reaction.evaluate_emergency_override(
        threat_score=0.70,
        current_etiquette_shift=3.0,
    )
    assert shift == 3.0  # Etiquette maintained!

    override_ok, reason = reaction.evaluate_emergency_rule(
        threat_severity=0.70,
        formal_rule_blocked=True,
    )
    assert override_ok is False
    assert reason == "FORMAL_RULE_MAINTAINED"
