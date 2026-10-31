"""
Unit tests for core.theta_engine (ThetaEngine).
Validates 50 Hz coupled phase oscillator, SI radian architecture,
Kuramoto homeostatic spring restoration, multi-rate latching, and physical invariants.
"""

import math
import numpy as np
import pytest

from core.maslow_integrator import MaslowSnapshot
from core.theta_engine import DEG2RAD, ThetaEngine


class TestThetaEngine:
    """Test suite for ThetaEngine."""

    def test_initialization_defaults_and_constants(self):
        """Verify default parameters, physical constants, and SI radian architecture."""
        engine = ThetaEngine()

        assert engine.dt == pytest.approx(0.02, rel=1e-6)
        assert engine.kv == pytest.approx(0.50, rel=1e-6)
        assert engine.K == pytest.approx(0.80, rel=1e-6)
        assert engine.alpha == pytest.approx(0.50, rel=1e-6)
        assert engine.gamma == pytest.approx(0.40, rel=1e-6)

        # SI Radian conversions
        assert DEG2RAD == pytest.approx(np.pi / 180.0, rel=1e-9)
        assert engine.theta_0_deg == pytest.approx(15.0, rel=1e-6)
        assert engine.theta_0_rad == pytest.approx(15.0 * DEG2RAD, rel=1e-9)
        assert engine.theta_0_rad == pytest.approx(0.2617993877991494, rel=1e-6)

        assert engine.theta_gate_deg == pytest.approx(18.0, rel=1e-6)
        assert engine.theta_gate_rad == pytest.approx(18.0 * DEG2RAD, rel=1e-9)
        assert engine.theta_gate_rad == pytest.approx(0.3141592653589793, rel=1e-6)

        assert engine.theta_bootstrap_deg == pytest.approx(15.5, rel=1e-6)
        assert engine.theta_bootstrap_rad == pytest.approx(15.5 * DEG2RAD, rel=1e-9)
        assert engine.theta_bootstrap_rad == pytest.approx(0.2705260340591211, rel=1e-6)

        assert engine.delta_theta_sav_deg == pytest.approx(0.35, rel=1e-6)
        assert engine.delta_theta_sav_rad == pytest.approx(0.35 * DEG2RAD, rel=1e-9)
        assert engine.delta_theta_sav_rad == pytest.approx(0.0061086523819801535, rel=1e-6)

        # Day 0 Bootstrap state
        assert engine.theta_rad == pytest.approx(engine.theta_bootstrap_rad, rel=1e-9)
        assert engine.theta_deg == pytest.approx(15.5, rel=1e-6)
        assert engine.dot_theta == pytest.approx(0.0, rel=1e-6)
        assert engine.time_s == pytest.approx(0.0, rel=1e-6)

    def test_step_return_tuple_and_types(self):
        """Verify step returns (theta_rad, theta_deg, dot_theta) with correct types."""
        engine = ThetaEngine()
        res = engine.step(V_l1=0.5, delta=0.1, a_motor=0.0)

        assert isinstance(res, tuple)
        assert len(res) == 3

        th_rad, th_deg, dot_th = res
        assert isinstance(th_rad, float)
        assert isinstance(th_deg, float)
        assert isinstance(dot_th, float)

        assert th_deg == pytest.approx(th_rad / DEG2RAD, rel=1e-9)
        assert engine.theta_rad == pytest.approx(th_rad, rel=1e-9)
        assert engine.theta_deg == pytest.approx(th_deg, rel=1e-9)
        assert engine.dot_theta == pytest.approx(dot_th, rel=1e-9)
        assert engine.time_s == pytest.approx(0.02, rel=1e-6)

    def test_kuramoto_spring_restoration(self):
        """Verify -K * sin(theta - theta_0) homeostatic spring pulls back to resting attractor."""
        engine = ThetaEngine()

        # At theta = theta_0 (15.0 deg), spring torque must be identically 0
        engine.reset(theta_deg=15.0)
        _, _, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert dot_th == pytest.approx(0.0, abs=1e-9)

        # Above resting angle (e.g., 16.0 deg), spring torque must be negative (restoring)
        engine.reset(theta_deg=16.0)
        expected_torque = -0.80 * math.sin((16.0 - 15.0) * DEG2RAD)
        _, _, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert dot_th == pytest.approx(expected_torque, rel=1e-6)
        assert dot_th < 0.0

        # Below resting angle (e.g., 14.0 deg), spring torque must be positive (pushing up)
        engine.reset(theta_deg=14.0)
        expected_torque = -0.80 * math.sin((14.0 - 15.0) * DEG2RAD)
        _, _, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert dot_th == pytest.approx(expected_torque, rel=1e-6)
        assert dot_th > 0.0

        # Max torque at 90 deg displacement is bounded by K
        engine.reset(theta_deg=15.0 + 90.0)
        _, _, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert dot_th == pytest.approx(-0.80, rel=1e-6)

    def test_maslow_coupling_acceleration(self):
        """Verify kv * ||V||_1 forward phase acceleration."""
        engine = ThetaEngine()
        engine.reset(theta_deg=15.0)  # zero Kuramoto spring

        # V_l1 = 0.50 -> dot_theta = 0.50 * 0.50 = 0.25 rad/s
        _, _, dot_th = engine.step(V_l1=0.50, delta=0.0, a_motor=0.0)
        assert dot_th == pytest.approx(0.25, rel=1e-6)

        # V_l1 = 0.80 -> dot_theta = 0.50 * 0.80 = 0.40 rad/s
        engine.reset(theta_deg=15.0)
        _, _, dot_th = engine.step(V_l1=0.80, delta=0.0, a_motor=0.0)
        assert dot_th == pytest.approx(0.40, rel=1e-6)

    def test_thalamic_dissonance_forward_coupling(self):
        """Verify alpha * delta forward excitation."""
        engine = ThetaEngine()
        engine.reset(theta_deg=15.0)

        # delta = 1.0 -> dot_theta = 0.50 * 1.0 = 0.50 rad/s
        _, _, dot_th = engine.step(V_l1=0.0, delta=1.0, a_motor=0.0)
        assert dot_th == pytest.approx(0.50, rel=1e-6)

        # delta = 0.20 -> dot_theta = 0.50 * 0.20 = 0.10 rad/s
        engine.reset(theta_deg=15.0)
        _, _, dot_th = engine.step(V_l1=0.0, delta=0.20, a_motor=0.0)
        assert dot_th == pytest.approx(0.10, rel=1e-6)

    def test_motor_quenching_braking(self):
        """Verify -gamma * a_motor negative feedback phase cooling."""
        engine = ThetaEngine()
        engine.reset(theta_deg=15.0)

        # a_motor = 1.0 -> dot_theta = -0.40 * 1.0 = -0.40 rad/s
        _, _, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=1.0)
        assert dot_th == pytest.approx(-0.40, rel=1e-6)

    def test_idle_spontaneous_breakthrough_threshold(self):
        """
        Verify that idle Maslow deficiency (||V||_1 = 0.80) easily overcomes
        maximum Kuramoto barrier at the 18.0 deg motor gate.
        """
        engine = ThetaEngine()
        engine.reset(theta_deg=18.0)

        # At theta = 18.0 deg (Delta theta = +3.0 deg from 15.0 deg):
        # Kuramoto restoring force: -0.80 * sin(3 deg) = -0.0418687 rad/s
        # Maslow forward drive: 0.50 * 0.80 = 0.40 rad/s
        # Net velocity: 0.40 - 0.0418687 = 0.35813 rad/s > 0
        _, _, dot_th = engine.step(V_l1=0.80, delta=0.0, a_motor=0.0)
        assert dot_th > 0.35
        assert dot_th == pytest.approx(0.50 * 0.80 - 0.80 * math.sin(3.0 * DEG2RAD), rel=1e-5)

        # Run forward simulation from Day 0 bootstrap (15.5 deg):
        # Should cross 18.0 deg threshold in well under 1.0 s
        engine.reset()
        assert engine.theta_deg == pytest.approx(15.5, rel=1e-6)
        steps_to_gate = 0
        while engine.theta_deg < 18.0 and steps_to_gate < 500:
            engine.step(V_l1=0.80, delta=0.0, a_motor=0.0)
            steps_to_gate += 1

        # At ~0.36 rad/s, delta_theta = 2.5 deg = 0.0436 rad, requires ~0.12 s (6 steps of 20 ms)
        time_to_gate = steps_to_gate * engine.dt
        assert time_to_gate < 0.50
        assert engine.theta_deg >= 18.0

    def test_multi_rate_sync_latching(self):
        """Verify latching of 10 Hz MaslowSnapshot for 50 Hz sub-stepping."""
        engine = ThetaEngine()

        vec = np.array([0.10, 0.20, 0.15, 0.15, 0.20], dtype=np.float64)  # sum = 0.80
        snap = MaslowSnapshot(timestamp_ns=1000, vector=vec)

        # Latch snapshot
        engine.latch_maslow(snap)

        # Perform 5 sub-steps at 50 Hz (100 ms total) without re-passing V_l1
        engine.reset(theta_deg=15.0)
        engine.latch_maslow(snap)
        for _ in range(5):
            _, _, dot_th = engine.step(delta=0.0, a_motor=0.0)
            # Velocity should reflect V_l1 = 0.80 forward drive
            assert dot_th > 0.35

        assert engine.time_s == pytest.approx(0.10, rel=1e-6)

        # Also verify step accepts MaslowSnapshot directly
        engine.reset(theta_deg=15.0)
        _, _, dot_th = engine.step(V_l1=snap)
        assert dot_th == pytest.approx(0.40, rel=1e-6)

        # Also verify step accepts numpy array directly
        engine.reset(theta_deg=15.0)
        _, _, dot_th = engine.step(V_l1=vec)
        assert dot_th == pytest.approx(0.40, rel=1e-6)

    def test_invalid_inputs_and_edge_cases(self):
        """Verify robust error handling and input validation."""
        # Invalid dt
        with pytest.raises(ValueError):
            ThetaEngine(dt=0.0)
        with pytest.raises(ValueError):
            ThetaEngine(dt=-0.02)
        with pytest.raises(ValueError):
            ThetaEngine(dt=float("nan"))

        engine = ThetaEngine()

        # Stepping with no V_l1 and no latched value
        with pytest.raises(ValueError):
            engine.step()

        # Negative V_l1
        with pytest.raises(ValueError):
            engine.step(V_l1=-0.5)

        # Non-finite V_l1
        with pytest.raises(ValueError):
            engine.step(V_l1=float("nan"))

        # Negative delta
        with pytest.raises(ValueError):
            engine.step(V_l1=0.5, delta=-0.1)

        # Out-of-bounds delta
        with pytest.raises(ValueError):
            engine.step(V_l1=0.5, delta=1.5)

        # Negative a_motor
        with pytest.raises(ValueError):
            engine.step(V_l1=0.5, a_motor=-0.1)

        # Out-of-bounds a_motor
        with pytest.raises(ValueError):
            engine.step(V_l1=0.5, a_motor=1.5)

        # Unsupported type
        with pytest.raises(TypeError):
            engine.step(V_l1="invalid_string")  # type: ignore

    def test_phase_continuity_and_long_run(self):
        """Verify numerical stability and continuity over thousands of steps."""
        engine = ThetaEngine()
        engine.reset()

        for step_idx in range(1000):
            # Alternating drive and quenching
            v_val = 0.80 if step_idx % 100 < 50 else 0.10
            a_val = 1.0 if engine.theta_deg >= 18.0 else 0.0
            th_rad, th_deg, dot_th = engine.step(V_l1=v_val, delta=0.05, a_motor=a_val)

            assert math.isfinite(th_rad)
            assert math.isfinite(th_deg)
            assert math.isfinite(dot_th)

    def test_maslow_theta_coupled_co_simulation(self):
        """
        Verify multi-rate coupling: 10 Hz MaslowIntegrator coupled to 50 Hz ThetaEngine
        under zero-input idle conditions (C=0, a_motor=0, delta=0).
        Proves autonomous phase breakthrough (theta >= 18.0 deg) occurs in <= 10.0 s.
        """
        from core.maslow_integrator import MaslowIntegrator

        # Start with Day 0 baseline pre-charge
        maslow = MaslowIntegrator()
        theta = ThetaEngine()

        t_sim = 0.0
        crossed_threshold = False
        crossing_time = None

        # Simulate 10 seconds (500 steps at 50 Hz, 100 steps at 10 Hz)
        for step_50hz in range(500):
            # Every 5 steps of 50 Hz (100 ms), update Maslow Integrator
            if step_50hz % 5 == 0:
                snap = maslow.step(C=0.0, a_motor=0.0)
                theta.latch_maslow(snap)

            # Advance 50 Hz Theta Engine
            th_rad, th_deg, dot_th = theta.step(delta=0.0, a_motor=0.0)
            t_sim += theta.dt

            if th_deg >= 18.0 and not crossed_threshold:
                crossed_threshold = True
                crossing_time = t_sim

        assert crossed_threshold, "Phase failed to cross 18.0 deg threshold in 10 s idle"
        assert crossing_time is not None
        assert crossing_time <= 10.0, f"Crossed at {crossing_time} s, expected <= 10.0 s"
        # Since Day 0 pre-charge has ||V||_1 = 0.90, forward drive kv * 0.90 = 0.45 rad/s,
        # ignition occurs rapidly in < 1.0 s
        assert crossing_time < 2.0

