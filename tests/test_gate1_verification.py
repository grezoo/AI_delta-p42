"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Unit Test Suite: DVP-01 Verification Engine
Comprehensive unit and integration test suite for core/gate1_verification.py.

Verifies:
1. DVP01Results dataclass structure, fields, and telemetry contract.
2. Gate1Verifier configuration, peak detection, and transient rejection.
3. Criterion 1: Closed Limit Cycle Formation & Stability (10,000 steps @ 50 Hz).
4. Criterion 2: Idle State Phase Tipping & Motor Ignition (<= 10.0 s).
5. Criterion 3: Autonomous Initiation Rate R_init (>= 0.85 over 100 stochastic trials).
6. Unified verification execution, report generator, and CLI exit code behavior.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.1, III.2, III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.1, I.5, II.1, X.2)
- Plan Review.md (DVP-01 Quantitative Protocol)
- ORIGINAL_REQUEST.md (Requirements R1, R4)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import sys
from typing import List

import numpy as np
import pytest

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.basal_ganglia_gate import BasalGangliaGate
from core.gate1_verification import DVP01Results, Gate1Verifier, main
from core.theta_engine import DEG2RAD


# ==============================================================================
# SECTION 1: DVP01Results Dataclass & Telemetry Contract Tests
# ==============================================================================

class TestDVP01ResultsContract:
    """Verifies DVP01Results dataclass attributes, defaults, and invariants."""

    def test_dvp01_results_frozen_immutability(self):
        """Verifies that DVP01Results is immutable and raises FrozenInstanceError on attribute mutation."""
        results = DVP01Results(
            criterion_1_passed=True,
            t_cycle_mean=1.60,
            t_cycle_valid_range=(1.0, 5.0),
            peak_amp_mean=0.32,
            peak_amp_std=0.01,
            peak_amp_cv=0.03,
            peak_amp_cv_threshold=0.05,
            cycles_detected=50,
            criterion_2_passed=True,
            t_ignition=4.38,
            t_ignition_threshold=10.0,
            criterion_3_passed=True,
            n_trials=100,
            n_success=100,
            r_init=1.00,
            r_init_threshold=0.85,
            all_passed=True,
            telemetry_summary={},
        )
        with pytest.raises(FrozenInstanceError):
            results.all_passed = False

        with pytest.raises(FrozenInstanceError):
            results.criterion_1_passed = False

        with pytest.raises(FrozenInstanceError):
            results.t_cycle_mean = 999.0

    def test_dvp01_results_valid_instantiation(self):
        """Verifies full instantiation of DVP01Results with canonical values."""
        results = DVP01Results(
            criterion_1_passed=True,
            t_cycle_mean=1.60,
            t_cycle_valid_range=(1.0, 5.0),
            peak_amp_mean=0.3228,
            peak_amp_std=0.001,
            peak_amp_cv=0.0031,
            peak_amp_cv_threshold=0.05,
            cycles_detected=118,
            criterion_2_passed=True,
            t_ignition=4.38,
            t_ignition_threshold=10.0,
            criterion_3_passed=True,
            n_trials=100,
            n_success=100,
            r_init=1.00,
            r_init_threshold=0.85,
            all_passed=True,
            telemetry_summary={"test_key": "test_val"},
        )

        assert results.criterion_1_passed is True
        assert results.criterion_2_passed is True
        assert results.criterion_3_passed is True
        assert results.all_passed is True
        assert results.t_cycle_mean == 1.60
        assert results.t_cycle_valid_range == (1.0, 5.0)
        assert results.cycles_detected == 118
        assert results.t_ignition == 4.38
        assert results.r_init == 1.00
        assert results.telemetry_summary["test_key"] == "test_val"

    def test_dvp01_results_partial_failure_flag(self):
        """Verifies that if any criterion fails, all_passed reflects overall failure."""
        results = DVP01Results(
            criterion_1_passed=True,
            t_cycle_mean=1.60,
            t_cycle_valid_range=(1.0, 5.0),
            peak_amp_mean=0.32,
            peak_amp_std=0.01,
            peak_amp_cv=0.03,
            peak_amp_cv_threshold=0.05,
            cycles_detected=50,
            criterion_2_passed=False,  # Failed criterion 2
            t_ignition=12.5,
            t_ignition_threshold=10.0,
            criterion_3_passed=True,
            n_trials=100,
            n_success=95,
            r_init=0.95,
            r_init_threshold=0.85,
            all_passed=False,
            telemetry_summary={},
        )

        assert results.criterion_1_passed is True
        assert results.criterion_2_passed is False
        assert results.criterion_3_passed is True
        assert results.all_passed is False


# ==============================================================================
# SECTION 2: Gate1Verifier Configuration & Peak Detector Tests
# ==============================================================================

class TestGate1VerifierConfiguration:
    """Verifies default parameters and helper algorithms of Gate1Verifier."""

    def test_default_configuration_attributes(self):
        """Verifies default physical and simulation thresholds."""
        verifier = Gate1Verifier(verbose=False)
        assert verifier.transient_steps == 500  # 10.0 s settling at 50 Hz
        assert math.isclose(verifier.noise_sigma, 0.01, rel_tol=1e-6)
        assert verifier.n_trials == 100
        assert verifier.verbose is False

    def test_custom_configuration_attributes(self):
        """Verifies custom configuration initialization."""
        verifier = Gate1Verifier(
            verbose=True,
            transient_steps=300,
            noise_sigma=0.02,
            n_trials=50,
        )
        assert verifier.transient_steps == 300
        assert math.isclose(verifier.noise_sigma, 0.02, rel_tol=1e-6)
        assert verifier.n_trials == 50
        assert verifier.verbose is True

    @pytest.mark.parametrize("invalid_n_trials", [0, -1, -100])
    def test_invalid_n_trials_raises_value_error(self, invalid_n_trials: int):
        """Verifies that n_trials <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="n_trials must be positive"):
            Gate1Verifier(n_trials=invalid_n_trials)

    @pytest.mark.parametrize("invalid_dt", [0.0, -0.02, -1.0])
    def test_invalid_dt_raises_value_error(self, invalid_dt: float):
        """Verifies that dt <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="dt must be positive"):
            Gate1Verifier(dt=invalid_dt)

    @pytest.mark.parametrize("invalid_timeout", [0.0, -5.0, -10.0])
    def test_invalid_t_ignition_timeout_raises_value_error(self, invalid_timeout: float):
        """Verifies that t_ignition_timeout <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="t_ignition_timeout must be positive"):
            Gate1Verifier(t_ignition_timeout=invalid_timeout)

    @pytest.mark.parametrize("invalid_steps", [0, -100, -10000])
    def test_invalid_steps_criterion1_raises_value_error(self, invalid_steps: int):
        """Verifies that steps_criterion1 <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="steps_criterion1 must be positive"):
            Gate1Verifier(steps_criterion1=invalid_steps)

    def test_invalid_transient_steps_raises_value_error(self):
        """Verifies that negative transient_steps raises ValueError."""
        with pytest.raises(ValueError, match="transient_steps must be non-negative"):
            Gate1Verifier(transient_steps=-1)

    @pytest.mark.parametrize("invalid_sigma", [-0.01, -1.0, float("nan"), float("inf")])
    def test_invalid_noise_sigma_raises_value_error(self, invalid_sigma: float):
        """Verifies that negative or non-finite noise_sigma raises ValueError."""
        with pytest.raises(ValueError, match="noise_sigma must be non-negative and finite"):
            Gate1Verifier(noise_sigma=invalid_sigma)

    def test_criterion_3_zero_trials_raises_value_error(self):
        """Verifies that calling verify_criterion_3 with n_trials=0 raises ValueError without division by zero."""
        verifier = Gate1Verifier(verbose=False)
        with pytest.raises(ValueError, match="n_trials must be positive"):
            verifier.verify_criterion_3_autonomous_initiation(n_trials=0)

    def test_detect_peaks_synthetic_sinusoid(self):
        """Verifies derivative zero-crossing peak detection on synthetic sinusoidal signal."""
        t = np.linspace(0, 4 * np.pi, 200)
        sig = np.sin(t)

        # In interval [0, 4*pi], sin(t) has 2 peaks: at t = pi/2 and t = 5*pi/2
        peaks = Gate1Verifier.detect_peaks(sig, min_threshold=0.5)
        assert len(peaks) == 2

        # Check values at detected peaks are close to 1.0
        for p in peaks:
            assert sig[p] > 0.99

    def test_detect_peaks_boundary_and_degenerate_signals(self):
        """Verifies peak detector behavior on short, empty, or flat arrays."""
        assert Gate1Verifier.detect_peaks(np.array([])) == []
        assert Gate1Verifier.detect_peaks(np.array([1.0, 2.0])) == []
        assert Gate1Verifier.detect_peaks(np.array([1.0, 1.0, 1.0, 1.0])) == []
        assert Gate1Verifier.detect_peaks(np.array([1.0, 2.0, 1.0]), min_threshold=5.0) == []

    def test_detect_peaks_threshold_filtering(self):
        """Verifies that sub-threshold local maxima are filtered out."""
        # 3 peaks of heights 0.5, 1.5, 0.8
        sig = np.array([0.0, 0.5, 0.1, 1.5, 0.2, 0.8, 0.0])
        # With threshold 1.0, only the 1.5 peak should be detected
        peaks_filtered = Gate1Verifier.detect_peaks(sig, min_threshold=1.0)
        assert len(peaks_filtered) == 1
        assert sig[peaks_filtered[0]] == 1.5


# ==============================================================================
# SECTION 3: Criterion 1 Limit Cycle Formation & Stability Tests
# ==============================================================================

class TestCriterion1LimitCycle:
    """Verifies DVP-01 Criterion 1: Closed Limit Cycle Formation & Stability."""

    def test_criterion_1_full_10000_step_simulation(self):
        """Runs the full 10,000-step (200.0 s @ 50 Hz) simulation and validates all thresholds."""
        verifier = Gate1Verifier(verbose=False)
        passed, t_cycle_mean, peak_amp_cv, peak_amp_mean, cycles = (
            verifier.verify_criterion_1_limit_cycle(sim_duration_s=200.0, dt=0.02)
        )

        assert passed is True
        # Mean cycle period strictly bounded between 1.0 and 5.0 seconds
        assert 1.0 <= t_cycle_mean <= 5.0
        # Peak amplitude CV strictly <= 5.0%
        assert peak_amp_cv <= 0.05
        # Completed cycles >= 35
        assert cycles >= 35

    def test_criterion_1_orbit_period_consistency(self):
        """Verifies that the non-equilibrium limit cycle period is highly stable."""
        verifier = Gate1Verifier(verbose=False)
        passed, t_cycle_mean, peak_amp_cv, _, cycles = (
            verifier.verify_criterion_1_limit_cycle()
        )
        assert passed is True
        # Canonical coupled relaxation period is ~1.60 s
        assert math.isclose(t_cycle_mean, 1.60, abs_tol=0.20)
        assert peak_amp_cv < 0.01  # Ultra-stable orbit CV < 1%

    def test_criterion_1_transient_discard_effect(self):
        """Verifies that initial 500 transient steps are discarded."""
        verifier = Gate1Verifier(verbose=False, transient_steps=500)
        assert verifier.transient_steps == 500
        # Over 20s (1000 steps), discarding 500 steps leaves 10s of steady state
        _, t_mean, cv, peak_mean, cycles = verifier.verify_criterion_1_limit_cycle(
            sim_duration_s=20.0, dt=0.02
        )
        assert cycles >= 3
        assert 1.0 <= t_mean <= 5.0


# ==============================================================================
# SECTION 4: Criterion 2 Idle State Phase Tipping & Motor Ignition Tests
# ==============================================================================

class TestCriterion2IdleIgnition:
    """Verifies DVP-01 Criterion 2: Idle State Phase Tipping & Motor Ignition <= 10.0 s."""

    def test_criterion_2_day0_precharge_ignition_within_10s(self):
        """Validates that basal drive tips phase past 18.0 deg and fires a_motor in <= 10.0 s."""
        verifier = Gate1Verifier(verbose=False)
        passed, t_ignition = verifier.verify_criterion_2_idle_ignition()

        assert passed is True
        assert 0.0 < t_ignition <= 10.0
        # Analytical window verification [2.0, 7.0] s
        assert 2.0 <= t_ignition <= 7.0

    def test_criterion_2_zero_prompt_idle_condition(self):
        """Validates ignition under idle conditions without external excitation."""
        verifier = Gate1Verifier(verbose=False)
        passed, t_ign = verifier.verify_criterion_2_idle_ignition(max_duration_s=10.0)
        assert passed is True
        assert t_ign < 10.0

    def test_criterion_2_custom_timeout_detection(self):
        """Verifies that insufficient duration correctly flags failure if ignition has not occurred."""
        verifier = Gate1Verifier(verbose=False)
        # If max_duration_s is shorter than ignition time (e.g. 1.0 s), returns passed = False
        passed, t_ign = verifier.verify_criterion_2_idle_ignition(max_duration_s=1.0)
        assert passed is False
        assert t_ign == 1.0


# ==============================================================================
# SECTION 5: Criterion 3 Autonomous Initiation Rate (R_init) Tests
# ==============================================================================

class TestCriterion3AutonomousInitiation:
    """Verifies DVP-01 Criterion 3: Autonomous Initiation Rate R_init over N=100 Trials."""

    def test_criterion_3_100_monte_carlo_trials(self):
        """Executes 100 stochastic trials and verifies R_init >= 0.85 (85%)."""
        verifier = Gate1Verifier(verbose=False, n_trials=100)
        passed, n_success, r_init = verifier.verify_criterion_3_autonomous_initiation(n_trials=100)

        assert passed is True
        assert n_success >= 85
        assert r_init >= 0.85
        assert n_success + (100 - n_success) == 100

    def test_criterion_3_deterministic_reproducibility(self):
        """Verifies that repeated runs with deterministic PRNG seeds yield identical R_init."""
        verifier = Gate1Verifier(verbose=False)
        p1, s1, r1 = verifier.verify_criterion_3_autonomous_initiation(n_trials=20)
        p2, s2, r2 = verifier.verify_criterion_3_autonomous_initiation(n_trials=20)

        assert p1 == p2 == True
        assert s1 == s2
        assert math.isclose(r1, r2, rel_tol=1e-9)

    def test_criterion_3_stochastic_noise_sigma(self):
        """Verifies canonical noise parameter sigma = 0.01 rad/s^0.5."""
        verifier = Gate1Verifier(verbose=False)
        assert math.isclose(verifier.noise_sigma, 0.01, rel_tol=1e-6)


# ==============================================================================
# SECTION 6: Master Execution, Report Generator & CLI Entry Point Tests
# ==============================================================================

class TestUnifiedEngineAndCLI:
    """Verifies run_all_checks, format_report, and main CLI execution."""

    def test_run_all_checks_e2e_approval(self):
        """Verifies that run_all_checks executes all 3 criteria and produces all_passed=True."""
        verifier = Gate1Verifier(verbose=False)
        results = verifier.run_all_checks()

        assert results.all_passed is True
        assert results.criterion_1_passed is True
        assert results.criterion_2_passed is True
        assert results.criterion_3_passed is True

        # Check telemetry contents
        telemetry = results.telemetry_summary
        assert "t_cycle_mean" in telemetry
        assert "peak_amp_cv" in telemetry
        assert "t_ignition" in telemetry
        assert "r_init" in telemetry
        assert "cycles_detected" in telemetry
        assert telemetry["all_passed"] is True

    def test_format_report_string_contents(self):
        """Verifies that format_report produces a well-structured report containing DVP-01."""
        verifier = Gate1Verifier(verbose=False)
        report = verifier.format_report()

        assert "DVP-01" in report
        assert "Criterion 1" in report
        assert "Criterion 2" in report
        assert "Criterion 3" in report
        assert "[PASS]" in report
        assert "GATE 1 APPROVED" in report

    def test_main_cli_returns_exit_code_zero(self, monkeypatch):
        """Verifies that main() executes cleanly and returns exit code 0."""
        exit_code = main()
        assert exit_code == 0

    def test_main_cli_returns_exit_code_one_on_failure(self, monkeypatch):
        """Verifies that main() returns exit code 1 when any criterion fails."""
        failing_results = DVP01Results(
            criterion_1_passed=False,
            t_cycle_mean=0.0,
            t_cycle_valid_range=(1.0, 5.0),
            peak_amp_mean=0.0,
            peak_amp_std=0.0,
            peak_amp_cv=1.0,
            peak_amp_cv_threshold=0.05,
            cycles_detected=0,
            criterion_2_passed=False,
            t_ignition=10.0,
            t_ignition_threshold=10.0,
            criterion_3_passed=False,
            n_trials=100,
            n_success=0,
            r_init=0.0,
            r_init_threshold=0.85,
            all_passed=False,
            telemetry_summary={},
        )
        monkeypatch.setattr(Gate1Verifier, "run_all_checks", lambda self: failing_results)
        exit_code = main()
        assert exit_code == 1


# ==============================================================================
# SECTION 7: Basal Ganglia Gate Latch Quench Drive Tests (Remediation Task 2)
# ==============================================================================

class TestBasalGangliaGateLatchQuench:
    """Verifies BasalGangliaGate latch_quench_drive functionality (Task 2)."""

    def test_default_unlatched_monotonic_behavior(self):
        """Verifies default latch_quench_drive=False preserves canonical monotonic onset."""
        gate = BasalGangliaGate(latch_quench_drive=False)
        d_18_1 = gate.evaluate(theta_rad=18.1 * DEG2RAD)
        d_18_2 = gate.evaluate(theta_rad=18.2 * DEG2RAD)
        assert 0.0 < d_18_1.a_motor < d_18_2.a_motor
        assert math.isclose(d_18_1.a_motor, 0.1419, abs_tol=1e-2)

    def test_latched_quench_drive_active_firing(self):
        """Verifies latch_quench_drive=True enforces a_motor = max(instant_a, quench_drive) once firing."""
        gate = BasalGangliaGate(latch_quench_drive=True)
        d_18_1 = gate.evaluate(theta_rad=18.1 * DEG2RAD)
        assert d_18_1.a_motor >= 0.05
        assert gate._firing is True
        # On subsequent evaluation while firing, quench drive (0.70) is enforced
        d_18_2 = gate.evaluate(theta_rad=18.2 * DEG2RAD)
        assert d_18_2.a_motor == max(d_18_2.a_motor, 0.70)
        assert d_18_2.a_motor == 0.70
