"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Tier 2: Boundary & Corner Cases Test Suite
Rigorous stress testing of extreme values, empty/zero inputs, division by zero edge cases,
phase wrap-around discipline, and immutability invariants.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from tests.e2e.conftest import (
    DEG2RAD,
    THETA_0_DEG,
    THETA_GATE_DEG,
    THETA_BOOTSTRAP_DEG,
    THETA_0_RAD,
    THETA_GATE_RAD,
    THETA_BOOTSTRAP_RAD,
    DEFAULT_KV,
    DEFAULT_K,
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_BASE,
    DEFAULT_LAMBDA_WORK,
    DEFAULT_ETA_GAIN,
    HAS_M1,
    HAS_M2,
)

if HAS_M1:
    from core.theta_engine import ThetaEngine
    from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot

if HAS_M2:
    from core.thalamus_agc import ThalamusAGC
    from core.basal_ganglia_gate import BasalGangliaGate


# ==============================================================================
# TIER 2: EMPTY & ZERO INPUTS
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestTier2ZeroInputs:
    """Stress tests on empty, zero, and vanishing inputs."""

    def test_tier2_zero_deficiency_vector(self, theta_engine):
        """Zero deficiency ||V||_1 = 0 produces zero Maslow driving force."""
        theta_engine.reset(theta_deg=THETA_0_DEG)
        _, _, dot_th = theta_engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert math.isclose(dot_th, 0.0, abs_tol=1e-12)

    def test_tier2_zero_sensory_dissonance(self, theta_engine):
        """Zero dissonance delta = 0 produces zero forward drive."""
        theta_engine.reset(theta_deg=THETA_0_DEG)
        _, _, dot_th = theta_engine.step(V_l1=1.0, delta=0.0, a_motor=0.0)
        expected = theta_engine.kv * 1.0
        assert math.isclose(dot_th, expected, rel_tol=1e-9)

    def test_tier2_zero_motor_activation(self, theta_engine, maslow_integrator):
        """Zero motor activation a_motor = 0 produces zero quenching in both engines."""
        theta_engine.reset(theta_deg=THETA_0_DEG)
        _, _, dot_th = theta_engine.step(V_l1=1.0, a_motor=0.0)
        assert math.isclose(dot_th, theta_engine.kv * 1.0, rel_tol=1e-9)

        snap = maslow_integrator.step(C=0.0, a_motor=0.0, E_env=1.0)
        # Needs grow because R_i = eta * 0 * 1 = 0
        assert snap.l1_norm > 0.90

    def test_tier2_zero_cognitive_load(self, maslow_integrator):
        """Zero cognitive load C = 0 results in growth driven solely by lambda_base."""
        integrator = MaslowIntegrator(initial_vector=[0.0]*5)
        snap = integrator.step(C=0.0, a_motor=0.0)
        expected_val = 1.0 - math.exp(-DEFAULT_LAMBDA_BASE * integrator.dt)
        np.testing.assert_allclose(snap.vector, np.full(5, expected_val), rtol=1e-5)

    def test_tier2_zero_environmental_reward(self, maslow_integrator):
        """Zero reward E_env = 0 prevents motor action from satisfying needs."""
        integrator = MaslowIntegrator(initial_vector=[0.5]*5)
        snap = integrator.step(C=0.0, a_motor=1.0, E_env=0.0)
        # With E_env=0, R_i = 0, so needs continue rising despite motor action
        assert np.all(snap.vector > 0.5)

    def test_tier2_zero_dt_rejected(self):
        """Non-positive dt <= 0 raises ValueError in both engines."""
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            ThetaEngine(dt=0.0)
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            ThetaEngine(dt=-0.02)
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            MaslowIntegrator(dt=0.0)
        with pytest.raises(ValueError, match="Time step dt must be positive"):
            MaslowIntegrator(dt=-0.1)


# ==============================================================================
# TIER 2: EXTREME VALUES & SATURATION
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestTier2ExtremeValues:
    """Stress tests on upper bounds, extreme gains, and large step sizes."""

    def test_tier2_extreme_maslow_vector_saturation(self, maslow_integrator):
        """Fully saturated need vector [1.0]*5 (L1 = 5.0) remains strictly bounded in [0, 1]."""
        integrator = MaslowIntegrator(initial_vector=[1.0]*5)
        for _ in range(50):
            snap = integrator.step(C=1.0, a_motor=0.0)
            assert np.all(snap.vector <= 1.0)
            assert np.all(snap.vector >= 0.0)
            assert snap.l1_norm <= 5.0

    def test_tier2_extreme_time_step_stability(self):
        """Massive time steps (dt = 1000.0 s) evaluate without NaN/Inf or overflow."""
        integrator = MaslowIntegrator(dt=1000.0, initial_vector=[0.2]*5)
        snap = integrator.step(C=1.0, a_motor=0.5, E_env=1.0)
        assert np.all(np.isfinite(snap.vector))
        assert np.all(snap.vector >= 0.0)
        assert np.all(snap.vector <= 1.0)

    def test_tier2_extreme_motor_action_saturation(self, theta_engine):
        """Maximum motor action a_motor = 1.0 exerts maximum allowable braking."""
        theta_engine.reset(theta_deg=THETA_0_DEG)
        _, _, dot_th = theta_engine.step(V_l1=0.0, a_motor=1.0)
        assert math.isclose(dot_th, -theta_engine.gamma, rel_tol=1e-9)

    def test_tier2_extreme_dissonance_saturation(self, theta_engine):
        """Maximum dissonance delta = 1.0 exerts maximum allowable forward drive."""
        theta_engine.reset(theta_deg=THETA_0_DEG)
        _, _, dot_th = theta_engine.step(V_l1=0.0, delta=1.0)
        assert math.isclose(dot_th, theta_engine.alpha, rel_tol=1e-9)

    def test_tier2_extreme_phase_angle_continuity(self, theta_engine):
        """Extreme phase angle (theta = 100.0 rad ~ 5729 deg) maintains trigonometric stability."""
        theta_engine.reset(theta_deg=5729.57795)
        th_rad, _, dot_th = theta_engine.step(V_l1=1.0)
        assert math.isfinite(th_rad)
        assert math.isfinite(dot_th)


# ==============================================================================
# TIER 2: DIVISION BY ZERO & NUMERICAL SINGULARITIES
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestTier2DivisionByZero:
    """Verification that mathematical edge singularities are guarded."""

    def test_tier2_maslow_vanishing_rates(self):
        """When total_lambda = G + R = 0, division by zero is safely bypassed."""
        integrator = MaslowIntegrator(lambda_base=0.0, lambda_work=0.0, eta_gain=0.0, initial_vector=[0.3]*5)
        snap = integrator.step(C=0.0, a_motor=0.0)
        # Preserves state perfectly
        np.testing.assert_allclose(snap.vector, np.full(5, 0.3), rtol=1e-9)

    def test_tier2_maslow_invalid_negative_rates(self):
        """Negative rate constants raise ValueError."""
        with pytest.raises(ValueError):
            MaslowIntegrator(lambda_base=-0.1)
        with pytest.raises(ValueError):
            MaslowIntegrator(lambda_work=-0.1)
        with pytest.raises(ValueError):
            MaslowIntegrator(eta_gain=-0.1)

    def test_tier2_theta_invalid_negative_gains(self):
        """Negative gains in ThetaEngine raise ValueError."""
        with pytest.raises(ValueError):
            ThetaEngine(kv=-0.5)
        with pytest.raises(ValueError):
            ThetaEngine(K=-0.8)
        with pytest.raises(ValueError):
            ThetaEngine(alpha=-0.5)
        with pytest.raises(ValueError):
            ThetaEngine(gamma=-0.4)

    @pytest.mark.skipif(not HAS_M2, reason="M2 modules not available")
    def test_tier2_thalamus_agc_zero_denominator_safe(self):
        """Zero Maslow vector in AGC denominator produces g_AGC = 1.0 without singularity."""
        th = ThalamusAGC(kappa_agc=0.5)
        g = th.compute_agc_gain(np.zeros(5), sim_time_s=10.0)
        assert math.isclose(g, 1.0, rel_tol=1e-6)

    @pytest.mark.skipif(not HAS_M2, reason="M2 modules not available")
    def test_tier2_basal_ganglia_sigmoid_overflow_protection(self):
        """Extreme phase angle does not cause exp() overflow due to argument clamping."""
        gate = BasalGangliaGate()
        # theta = 1,000,000 rad
        d_pos = gate.evaluate(theta_rad=1_000_000.0)
        assert math.isclose(d_pos.a_motor, 1.0, rel_tol=1e-3)
        # theta = -1,000,000 rad
        d_neg = gate.evaluate(theta_rad=-1_000_000.0)
        assert math.isclose(d_neg.a_motor, 0.0, abs_tol=1e-6)


# ==============================================================================
# TIER 2: PHASE WRAP-AROUND & ANGULAR DISCIPLINE
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestTier2PhaseDiscipline:
    """Verification of phase angle continuity and anti-symmetric restoring force."""

    def test_tier2_phase_continuous_evolution(self, theta_engine):
        """Phase advances smoothly without discontinuous modulo reduction around the operating band."""
        theta_engine.reset(theta_deg=17.5)
        # Step through 18.0 deg gate
        angles = []
        for _ in range(50):
            th, _, _ = theta_engine.step(V_l1=1.0)
            angles.append(th)
        # Differences should all be small positive increments (< 0.05 rad)
        diffs = np.diff(angles)
        assert np.all(diffs > 0.0)
        assert np.all(diffs < 0.05)

    def test_tier2_phase_anti_symmetric_kuramoto(self, theta_engine):
        """Kuramoto spring is anti-symmetric around theta_0: F(theta_0 + d) == -F(theta_0 - d)."""
        theta_engine.reset(theta_deg=THETA_0_DEG + 2.0)
        _, _, dot_plus = theta_engine.step(V_l1=0.0)
        theta_engine.reset(theta_deg=THETA_0_DEG - 2.0)
        _, _, dot_minus = theta_engine.step(V_l1=0.0)
        assert math.isclose(dot_plus, -dot_minus, rel_tol=1e-9)

    def test_tier2_phase_negative_angle_restoring(self, theta_engine):
        """Starting at negative angle produces positive restoring force toward theta_0."""
        theta_engine.reset(theta_deg=-10.0)
        _, _, dot_th = theta_engine.step(V_l1=0.0)
        assert dot_th > 0.0


# ==============================================================================
# TIER 2: SNAPSHOT IMMUTABILITY & MEMORY INVARIANTS
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestTier2ImmutabilityInvariants:
    """Verification of immutable data contracts across modules."""

    def test_tier2_snapshot_frozen_attributes(self, maslow_integrator):
        """Setting attributes on emitted MaslowSnapshot raises FrozenInstanceError."""
        snap = maslow_integrator.get_snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.dominant_name = "tampered"
        with pytest.raises((AttributeError, TypeError)):
            snap.timestamp_ns = 12345

    def test_tier2_snapshot_array_buffer_read_only(self, maslow_integrator):
        """Mutating the vector buffer in-place raises ValueError (read-only buffer)."""
        snap = maslow_integrator.get_snapshot()
        assert snap.vector.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            snap.vector[2] = 0.999

    def test_tier2_snapshot_vector_copy_isolation(self, maslow_integrator):
        """Advancing the integrator does not mutate previously emitted snapshot data."""
        snap1 = maslow_integrator.get_snapshot()
        v1_original = snap1.vector.copy()
        # Advance integrator
        maslow_integrator.step(C=1.0, a_motor=0.0)
        snap2 = maslow_integrator.get_snapshot()
        # snap1 remains completely untouched
        np.testing.assert_allclose(snap1.vector, v1_original, rtol=1e-12)
        assert snap2.l1_norm > snap1.l1_norm
