"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Unit Test Suite: Basal Ganglia Gate Module
Tests bipolar Striatum Go/No-Go gating, continuous C0 jerk-free transition,
biological safety interlocks (Sleep & Panic locks), social etiquette modulation,
and Cerebellar Purkinje LTD plasticity interface contracts.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.5, I.6, I.7, VII.2)
- Plan Review.md (REQ-GATE-01..03, DVP-04, DVP-14)
- megvalósitási terv.md (Section 3.3)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.basal_ganglia_gate import (
    DEG2RAD,
    THETA_GATE_DEG,
    THETA_GATE_RAD,
    DELTA_THETA_SAV_DEG,
    DELTA_THETA_SAV_RAD,
    THETA_0_DEG,
    THETA_0_RAD,
    GateDecision,
    BasalGangliaGate,
    PurkinjeLTDInterface,
)
from core.maslow_integrator import MaslowIntegrator
from core.theta_engine import ThetaEngine


class TestC0ContinuousMotorGate:
    """Verification of continuous C0 jerk-free striatal action gating (Feature 18)."""

    def test_canonical_angular_constants(self):
        """Verifies exact radian conversions for physical parameters."""
        gate = BasalGangliaGate()
        assert math.isclose(gate.theta_gate_deg, 18.0, rel_tol=1e-7)
        assert math.isclose(gate.theta_gate_rad, 18.0 * DEG2RAD, rel_tol=1e-7)
        assert math.isclose(gate.delta_theta_sav_deg, 0.35, rel_tol=1e-7)
        assert math.isclose(gate.delta_theta_sav_rad, 0.35 * DEG2RAD, rel_tol=1e-7)

    def test_exact_c0_continuity_at_threshold(self):
        """At theta == theta_gate, a_motor starts strictly at 0.0000 (zero torque jump)."""
        gate = BasalGangliaGate()
        decision = gate.evaluate(theta_rad=THETA_GATE_RAD)
        assert math.isclose(decision.a_motor, 0.0, abs_tol=1e-12)
        assert decision.is_locked is False
        assert decision.go_winner is True

    def test_subthreshold_quiescence(self):
        """For all theta < theta_gate, motor output is strictly 0.0000 (DVP-04 zero false positive)."""
        gate = BasalGangliaGate()
        for deg in [0.0, 5.0, 10.0, 15.0, 17.0, 17.9, 17.99]:
            decision = gate.evaluate(theta_rad=deg * DEG2RAD)
            assert decision.a_motor == 0.0

    def test_smooth_monotonic_onset_above_threshold(self):
        """Motor throughput increases monotonically and smoothly above threshold."""
        gate = BasalGangliaGate()
        d_18_1 = gate.evaluate(theta_rad=18.1 * DEG2RAD)
        d_18_2 = gate.evaluate(theta_rad=18.2 * DEG2RAD)
        d_18_5 = gate.evaluate(theta_rad=18.5 * DEG2RAD)
        d_19_0 = gate.evaluate(theta_rad=19.0 * DEG2RAD)

        assert 0.0 < d_18_1.a_motor < d_18_2.a_motor < d_18_5.a_motor < d_19_0.a_motor <= 1.0
        # Check specific expected numerical progress values from canonical spec
        assert math.isclose(d_18_1.a_motor, 0.1419, abs_tol=1e-2)
        assert math.isclose(d_18_5.a_motor, 0.6133, abs_tol=1e-2)

    def test_extreme_overflow_underflow_protection(self):
        """Extreme phase angles do not overflow float exponentiation due to [-10, 10] clipping."""
        gate = BasalGangliaGate()
        d_huge = gate.evaluate(theta_rad=1e6)
        assert math.isclose(d_huge.a_motor, 1.0, rel_tol=1e-3)
        d_neg = gate.evaluate(theta_rad=-1e6)
        assert math.isclose(d_neg.a_motor, 0.0, abs_tol=1e-12)

    def test_social_etiquette_threshold_shift(self):
        """Social etiquette shift Delta_theta_etiquette postpones motor release during partner speech."""
        gate = BasalGangliaGate()
        # Normal operation: 18.5 deg triggers action
        normal_dec = gate.evaluate(theta_rad=18.5 * DEG2RAD)
        assert normal_dec.a_motor > 0.50

        # Etiquette shifted by +1.0 deg (effective threshold = 19.0 deg): action held back
        shifted_dec = gate.evaluate(theta_rad=18.5 * DEG2RAD, delta_etiquette_deg=1.0)
        assert shifted_dec.a_motor == 0.0

    def test_striatal_go_vs_nogo_competition(self):
        """Action is suppressed if No-Go pathway dominates striatal competition."""
        gate = BasalGangliaGate()
        # Go wins
        dec_go = gate.evaluate(theta_rad=19.0 * DEG2RAD, go_signal=1.0, nogo_signal=0.2)
        assert dec_go.a_motor > 0.8
        assert dec_go.go_winner is True

        # NoGo wins
        dec_nogo = gate.evaluate(theta_rad=19.0 * DEG2RAD, go_signal=0.3, nogo_signal=0.8)
        assert dec_nogo.a_motor == 0.0
        assert dec_nogo.go_winner is False


class TestBiologicalSafetyInterlocks:
    """Verification of hardware-level biological interlocks (Feature 19)."""

    def test_sleep_motor_paralysis(self):
        """Sleep lock unconditionally zeroes motor throughput regardless of phase angle."""
        gate = BasalGangliaGate()
        for test_angle_deg in [18.5, 20.0, 45.0, 90.0, 360.0]:
            dec = gate.evaluate(theta_rad=test_angle_deg * DEG2RAD, sleep_lock=True)
            assert dec.a_motor == 0.0
            assert dec.is_locked is True

    def test_amygdala_panic_emergency_brake(self):
        """Amygdalar panic alarm immediately pulls down motor drive to zero."""
        gate = BasalGangliaGate()
        dec = gate.evaluate(theta_rad=20.0 * DEG2RAD, panic_amy=True)
        assert dec.a_motor == 0.0
        assert dec.is_locked is True

    def test_simultaneous_locks(self):
        """Both sleep and panic locks active simultaneously result in zero motor throughput."""
        gate = BasalGangliaGate()
        dec = gate.evaluate(theta_rad=25.0 * DEG2RAD, sleep_lock=True, panic_amy=True)
        assert dec.a_motor == 0.0
        assert dec.is_locked is True

    def test_interlock_release_restores_activity(self):
        """Clearing all interlocks restores normal continuous motor gating throughput."""
        gate = BasalGangliaGate()
        dec_locked = gate.evaluate(theta_rad=18.5 * DEG2RAD, sleep_lock=True)
        assert dec_locked.a_motor == 0.0
        dec_unlocked = gate.evaluate(theta_rad=18.5 * DEG2RAD, sleep_lock=False)
        assert dec_unlocked.a_motor > 0.50
        assert dec_unlocked.is_locked is False


class TestCerebellarPurkinjeLTD:
    """Verification of Cerebellar Spiking Reservoir & Purkinje LTD Plasticity (Feature 22)."""

    def test_purkinje_initialization(self):
        """Purkinje interface initializes with specified weight and cadence."""
        p = PurkinjeLTDInterface(initial_weight=0.5, learning_rate=0.01, cadence_hz=20.0)
        assert math.isclose(p.current_weight, 0.5, rel_tol=1e-7)
        assert math.isclose(p.cadence_hz, 20.0, rel_tol=1e-7)
        assert p.n_granule == 10000

    def test_ltd_weight_depression_dynamics(self):
        """Climbing fiber error depresses synaptic weight according to Delta_W = -eta * error * pf."""
        p = PurkinjeLTDInterface(initial_weight=0.6, learning_rate=0.05)
        # Error signal = 2.0, parallel fiber = 1.0 -> Delta_W = -0.05 * 2.0 = -0.10
        updated = p.adapt(error_signal=2.0, parallel_fiber=1.0)
        assert math.isclose(updated, 0.50, rel_tol=1e-6)
        assert math.isclose(p.current_weight, 0.50, rel_tol=1e-6)

    def test_zero_error_signal_preserves_weights(self):
        """Zero motor tracking error produces strictly zero synaptic weight change."""
        p = PurkinjeLTDInterface(initial_weight=0.42)
        p.adapt(error_signal=0.0)
        assert math.isclose(p.current_weight, 0.42, abs_tol=1e-12)

    def test_weight_lower_and_upper_clamping(self):
        """Synaptic weights remain bounded within [0.0, 1.0] under extreme learning steps."""
        p = PurkinjeLTDInterface(initial_weight=0.05, learning_rate=1.0)
        p.adapt(error_signal=5.0)  # Extreme depression
        assert p.current_weight == 0.0

        p_high = PurkinjeLTDInterface(initial_weight=0.95, learning_rate=1.0)
        p_high.adapt(error_signal=-5.0)  # Potentiation
        assert p_high.current_weight == 1.0


class TestClosedLimitCycleWithGate:
    """Verification of continuous closed-loop limit cycle orbit generated with Basal Ganglia Gate (DVP-01)."""

    def test_closed_limit_cycle_multi_cycle_stability(self):
        """Simulates 10,000 steps at 50 Hz, verifying >= 35 cycles, period in [1, 5] s, and peak CV <= 0.05."""
        engine = ThetaEngine()
        integrator = MaslowIntegrator()
        gate = BasalGangliaGate()

        dt = engine.dt
        total_steps = 10000  # 200.0 s
        transient_steps = 500  # 10.0 s settling

        trajectory_th = []
        timestamps = []
        current_motor = 0.0

        for step in range(total_steps):
            t_s = step * dt

            # 10 Hz Maslow cadence
            if step % 5 == 0:
                snap = integrator.step(C=0.0, a_motor=current_motor, E_env=1.0)
                engine.latch_maslow(snap)

            # 50 Hz Theta step
            th_rad, th_deg, dot_th = engine.step(delta=0.0, a_motor=current_motor)

            # Gate evaluation
            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            if step >= transient_steps:
                trajectory_th.append(th_rad)
                timestamps.append(t_s)

        th_arr = np.array(trajectory_th)
        t_arr = np.array(timestamps)

        # Peak detection: local maxima above threshold
        peaks_idx = []
        for i in range(1, len(th_arr) - 1):
            if th_arr[i] > th_arr[i - 1] and th_arr[i] >= th_arr[i + 1] and th_arr[i] > THETA_GATE_RAD:
                peaks_idx.append(i)

        assert len(peaks_idx) >= 35, f"Expected >= 35 limit cycle peaks, got {len(peaks_idx)}"

        peak_times = t_arr[peaks_idx]
        peak_amps = th_arr[peaks_idx]

        periods = np.diff(peak_times)
        mean_period = float(np.mean(periods))
        amp_mean = float(np.mean(peak_amps))
        amp_std = float(np.std(peak_amps))
        amp_cv = amp_std / amp_mean if amp_mean > 0 else 1.0

        assert 1.0 <= mean_period <= 5.0, f"Limit cycle period {mean_period:.3f} s outside [1.0, 5.0] s"
        assert amp_cv <= 0.05, f"Peak amplitude relative standard deviation {amp_cv:.4f} exceeds 0.05"
