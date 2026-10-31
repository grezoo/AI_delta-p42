"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_bootstrap_loader.py
Author: Stage-Gate 4 Milestone 2 Specialist

Verifies:
1. Day 0 calibrated deficit vector V_0 = [0.15, 0.10, 0.25, 0.10, 0.30]^T invariants.
2. Initial symmetry-breaking phase perturbation kick theta(0) = 15.5 deg (0.270526 rad).
3. Soft sigmoid AGC ramp-up gain g_boot(t) = 1 / (1 + exp(-2(t - 1.5))) for t in [0, 3.0 s].
4. Subcortical engine coupling and bootstrap initialization (ThetaEngine, MaslowIntegrator, ThalamusAGC).
5. DVP-09 limit-cycle convergence verification, refractory peak detection, and zero numeric shock.
6. End-to-end autonomous bootstrap simulation (run_bootstrap_simulation) achieving 100% pass verdict.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.maslow_integrator import MaslowIntegrator
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import DEG2RAD, ThetaEngine
from nursery.bootstrap_loader import (
    DEFAULT_V0,
    RAMP_UP_DURATION_S,
    RAMP_UP_SIGMOID_K,
    RAMP_UP_SIGMOID_T0,
    THETA_0_DEG,
    THETA_0_RAD,
    THETA_BOOTSTRAP_DEG,
    THETA_BOOTSTRAP_RAD,
    THETA_GATE_DEG,
    THETA_GATE_RAD,
    V0_CANONICAL,
    BootstrapLoader,
    BootstrapProfile,
    ConvergenceResult,
)


# ==============================================================================
# 1. CANONICAL CONSTANTS & INITIAL CONDITIONS
# ==============================================================================

def test_bootstrap_constants() -> None:
    expected_v0 = np.array([0.15, 0.10, 0.25, 0.10, 0.30], dtype=np.float64)
    np.testing.assert_allclose(DEFAULT_V0, expected_v0)
    np.testing.assert_allclose(V0_CANONICAL, expected_v0)
    assert np.isclose(np.sum(DEFAULT_V0), 0.90)  # L1 norm = 0.90

    assert THETA_0_DEG == 15.0
    assert pytest.approx(THETA_0_RAD, rel=1e-6) == 15.0 * DEG2RAD
    assert THETA_BOOTSTRAP_DEG == 15.5
    assert pytest.approx(THETA_BOOTSTRAP_RAD, rel=1e-6) == 15.5 * DEG2RAD
    assert THETA_GATE_DEG == 18.0
    assert pytest.approx(THETA_GATE_RAD, rel=1e-6) == 18.0 * DEG2RAD

    assert RAMP_UP_DURATION_S == 3.0
    assert RAMP_UP_SIGMOID_K == 2.0
    assert RAMP_UP_SIGMOID_T0 == 1.5


# ==============================================================================
# 2. BOOTSTRAP PROFILE DATA STRUCTURE & LOADER VALIDATION
# ==============================================================================

def test_bootstrap_profile_validation() -> None:
    valid_v0 = np.array([0.15, 0.10, 0.25, 0.10, 0.30], dtype=np.float64)
    prof = BootstrapProfile(
        V_0=valid_v0,
        theta_0_rad=THETA_BOOTSTRAP_RAD,
        theta_0_deg=THETA_BOOTSTRAP_DEG,
    )
    assert prof.ramp_duration_s == 3.0
    np.testing.assert_allclose(prof.V_0, valid_v0)

    # Invalid V_0 shape
    with pytest.raises(ValueError, match="V_0 must be a 5-element 1D array"):
        BootstrapProfile(
            V_0=np.array([0.1, 0.2, 0.3]),
            theta_0_rad=THETA_BOOTSTRAP_RAD,
            theta_0_deg=THETA_BOOTSTRAP_DEG,
        )

    # Invalid V_0 bounds (> 1.0 or < 0.0)
    with pytest.raises(ValueError, match="must be in"):
        BootstrapProfile(
            V_0=np.array([0.1, 1.2, 0.3, 0.1, 0.2]),
            theta_0_rad=THETA_BOOTSTRAP_RAD,
            theta_0_deg=THETA_BOOTSTRAP_DEG,
        )


def test_bootstrap_loader_initialization_and_validation() -> None:
    loader = BootstrapLoader()
    np.testing.assert_allclose(loader.V_0, DEFAULT_V0)
    np.testing.assert_allclose(loader.get_initial_deficit(), DEFAULT_V0)
    assert loader.theta_bootstrap_deg == 15.5
    assert loader.get_initial_phase_deg() == 15.5
    assert pytest.approx(loader.theta_bootstrap_rad, rel=1e-6) == THETA_BOOTSTRAP_RAD
    assert pytest.approx(loader.get_initial_phase(), rel=1e-6) == THETA_BOOTSTRAP_RAD

    # Custom valid V_0
    custom_v0 = [0.2, 0.2, 0.2, 0.2, 0.2]
    loader_custom = BootstrapLoader(V_0=custom_v0, theta_0_deg=16.0, ramp_duration_s=2.5)
    np.testing.assert_allclose(loader_custom.V_0, np.array(custom_v0))
    assert loader_custom.theta_bootstrap_deg == 16.0

    # Symmetry breaking check: theta_0_deg must be > 15.0 deg
    with pytest.raises(ValueError, match="must exceed resting attractor"):
        BootstrapLoader(theta_0_deg=15.0)

    with pytest.raises(ValueError, match="must exceed resting attractor"):
        BootstrapLoader(theta_0_deg=14.0)

    # Invalid V_0 inputs
    with pytest.raises(ValueError, match="must be a 5-element array"):
        BootstrapLoader(V_0=[0.1, 0.2])

    with pytest.raises(ValueError, match="must be in"):
        BootstrapLoader(V_0=[-0.1, 0.2, 0.2, 0.2, 0.2])


# ==============================================================================
# 3. SOFT SIGMOID AGC RAMP-UP MATHEMATICAL INTEGRITY
# ==============================================================================

def test_soft_sigmoid_gain_curve_evaluation() -> None:
    loader = BootstrapLoader()

    # t <= 0: 1 / (1 + exp(3)) ~= 0.04742587
    g_0 = loader.get_boot_gain(0.0)
    expected_g0 = 1.0 / (1.0 + math.exp(3.0))
    assert pytest.approx(g_0, rel=1e-6) == expected_g0
    assert loader.evaluate_boot_gain(0.0) == g_0

    # Negative time should return t=0 baseline
    assert loader.get_boot_gain(-1.0) == g_0

    # t = 1.5 s: midpoint where 2*(1.5 - 1.5) = 0 -> 1 / (1 + exp(0)) = 0.500
    g_mid = loader.get_boot_gain(1.5)
    assert pytest.approx(g_mid, rel=1e-6) == 0.50

    # t >= 3.0 s: saturated to 1.0
    assert loader.get_boot_gain(3.0) == 1.0
    assert loader.get_boot_gain(5.0) == 1.0
    assert loader.get_boot_gain(100.0) == 1.0

    # Intermediate monotonicity check
    times = np.linspace(0.0, 3.0, 50)
    gains = [loader.get_boot_gain(t) for t in times]
    assert np.all(np.diff(gains) >= 0.0)  # Monotonically non-decreasing
    assert gains[0] < 0.05
    assert gains[-1] == 1.0


# ==============================================================================
# 4. SUBCORTICAL ENGINE COUPLING & INITIALIZATION
# ==============================================================================

def test_apply_bootstrap_to_engines() -> None:
    loader = BootstrapLoader()
    theta_engine = ThetaEngine(dt=0.02)
    maslow = MaslowIntegrator(dt=0.10)
    thalamus_agc = ThalamusAGC()

    # Apply bootstrap Day 0 injection
    loader.apply_bootstrap(theta_engine, maslow, thalamus_agc)

    # Check theta engine perturbed to 15.5 deg
    assert pytest.approx(theta_engine.theta_deg, abs=1e-5) == 15.5
    assert pytest.approx(theta_engine.theta_rad, abs=1e-5) == THETA_BOOTSTRAP_RAD
    assert theta_engine.dot_theta == 0.0

    # Check maslow integrator initialized with V_0
    np.testing.assert_allclose(maslow.vector, DEFAULT_V0)
    assert pytest.approx(maslow.get_snapshot().l1_norm, rel=1e-5) == 0.90


# ==============================================================================
# 5. DVP-09 LIMIT-CYCLE CONVERGENCE & VERIFICATION
# ==============================================================================

def test_verify_convergence_insufficient_history() -> None:
    loader = BootstrapLoader()
    res = loader.verify_convergence([])
    assert res.passed is False
    assert res.numeric_shock is True
    assert "Insufficient trajectory history" in res.details.get("error", "")

    res_short = loader.verify_convergence([(0.0, 0.26, 0.0)] * 5)
    assert res_short.passed is False


def test_verify_convergence_numeric_shock_detection() -> None:
    loader = BootstrapLoader()

    # Case 1: Trajectory with NaN
    traj_nan = [(i * 0.02, 0.26, 0.0) for i in range(50)]
    traj_nan[25] = (0.50, float("nan"), 0.0)
    res_nan = loader.verify_convergence(traj_nan)
    assert res_nan.passed is False
    assert res_nan.numeric_shock is True

    # Case 2: Trajectory with derivative spike (> 5.0 rad/s)
    traj_spike = [(i * 0.02, 0.26, 0.0) for i in range(50)]
    traj_spike[30] = (0.60, 0.28, 6.5)  # Spike
    res_spike = loader.verify_convergence(traj_spike)
    assert res_spike.passed is False
    assert res_spike.numeric_shock is True

    # Case 3: Runaway angle (> 25 deg = 0.436 rad)
    traj_runaway = [(i * 0.02, 0.50, 0.0) for i in range(50)]  # 0.50 rad ~= 28.6 deg
    res_runaway = loader.verify_convergence(traj_runaway)
    assert res_runaway.passed is False
    assert res_runaway.numeric_shock is True


def test_verify_convergence_synthetic_stable_orbit() -> None:
    loader = BootstrapLoader()
    # Synthesize clean limit-cycle orbit:
    # Period T = 2.0 s, peaks at 18.0 deg (0.314 rad), min at 15.0 deg (0.2618 rad)
    dt = 0.02
    duration = 8.0
    steps = int(duration / dt)

    traj = []
    for s in range(steps):
        t = s * dt
        # theta(t) = 16.5 deg + 1.5 deg * sin(2*pi*t / 2.0)
        center_rad = 16.5 * DEG2RAD
        amp_rad = 1.5 * DEG2RAD
        theta = center_rad + amp_rad * math.sin(2.0 * math.pi * t / 2.0)
        dot_theta = (2.0 * math.pi / 2.0) * amp_rad * math.cos(2.0 * math.pi * t / 2.0)
        traj.append((t, theta, dot_theta))

    res = loader.verify_convergence(traj)
    assert res.numeric_shock is False
    assert res.stable is True
    assert pytest.approx(res.first_cycle_period_s, abs=0.10) == 2.0
    assert res.convergence_time_s <= 3.0
    assert res.peak_cv <= 0.05
    assert res.passed is True


# ==============================================================================
# 6. END-TO-END AUTONOMOUS BOOTSTRAP SIMULATION (DVP-09)
# ==============================================================================

def test_run_bootstrap_simulation_dvp09_pass() -> None:
    """
    Executes the full closed-loop bootstrap simulation and confirms:
    - Convergence time <= 3.0 s
    - Orbit period in [1.0, 5.0] s
    - Peak amplitude CV <= 0.05
    - Zero numeric shock
    - Stable limit-cycle orbit
    - Overall VERDICT: PASS
    """
    loader = BootstrapLoader()
    result: ConvergenceResult = loader.run_bootstrap_simulation(duration_s=6.0, dt=0.02)

    assert result.passed is True, f"DVP-09 bootstrap convergence failed: {result}"
    assert result.numeric_shock is False
    assert result.convergence_time_s <= 3.0
    assert 1.0 <= result.first_cycle_period_s <= 5.0
    assert result.peak_cv <= 0.05
    assert result.stable is True
    assert result.details["num_peaks_detected"] >= 2
    assert result.details["max_velocity_rad_s"] <= 5.0
    assert result.details["max_phase_deg"] <= 25.0
