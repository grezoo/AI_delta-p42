"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Tier 3: Cross-Feature Interactions Test Suite
Verifies coupled dynamical feedback loops:
- Theta Engine + Maslow Integrator closed-loop pacing
- Quenching feedback dynamics (motor phase braking & environmental need satisfaction)
- Thalamus AGC dynamic gain damping under rising Maslow need tension
- Biological sleep & panic interlock overrides under high drive
- Multi-rate synchronization across 10 Hz and 50 Hz boundaries
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
    DEFAULT_KV,
    DEFAULT_K,
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA_BASE,
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
# TIER 3: COUPLED THETA-MASLOW SUBCORTICAL INTERACTIONS
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestTier3ThetaMaslowCoupling:
    """Interactions between 50 Hz Theta Engine and 10 Hz Maslow Integrator."""

    def test_tier3_basal_need_drives_phase_advancement(self, theta_engine, maslow_integrator):
        """Basal entropy accumulation in Maslow integrator autonomously accelerates phase velocity."""
        # Start at rest angle theta_0
        theta_engine.reset(theta_deg=THETA_0_DEG)
        # Advance Maslow in idle for 1.0 s (10 steps)
        for _ in range(10):
            snap = maslow_integrator.step(C=0.0, a_motor=0.0)
        # Latch into Theta Engine
        theta_engine.latch_maslow(snap)
        _, _, dot_th = theta_engine.step()
        # Maslow drive should be strictly positive
        assert dot_th > 0.0
        assert math.isclose(dot_th, theta_engine.kv * snap.l1_norm, rel_tol=1e-6)

    def test_tier3_multirate_cadence_50hz_10hz(self, theta_engine, maslow_integrator):
        """10 Hz Maslow integrator syncs with 50 Hz Theta engine (1 Maslow step per 5 Theta steps)."""
        theta_engine.reset(theta_deg=THETA_BOOTSTRAP_DEG)
        maslow_integrator.reset()

        trajectory_theta = []
        for maslow_step in range(20):  # 2.0 seconds total
            snap = maslow_integrator.step(C=0.0, a_motor=0.0)
            theta_engine.latch_maslow(snap)
            for theta_substep in range(5):
                th_rad, _, _ = theta_engine.step()
                trajectory_theta.append(th_rad)

        assert len(trajectory_theta) == 100
        # Phase should advance continuously
        assert trajectory_theta[-1] > trajectory_theta[0]

    def test_tier3_quenching_feedback_loop(self, theta_engine, maslow_integrator):
        """Motor activation triggers both phase deceleration and environmental need discharge."""
        # 1. Drive system to elevated need
        for _ in range(10):
            snap = maslow_integrator.step(C=0.0, a_motor=0.0)
        elevated_l1 = snap.l1_norm

        # 2. Fire motor action a_motor = 1.0 with reward E_env = 1.0
        theta_engine.reset(theta_deg=18.5)
        quenched_snap = maslow_integrator.step(C=0.0, a_motor=1.0, E_env=1.0)
        theta_engine.latch_maslow(quenched_snap)
        _, _, dot_th = theta_engine.step(a_motor=1.0)

        # Need collapsed
        assert quenched_snap.l1_norm < elevated_l1
        # Phase velocity braked by motor quenching
        assert dot_th < theta_engine.kv * elevated_l1


# ==============================================================================
# TIER 3: FULL 4-MODULE CLOSED-LOOP INTERACTIONS
# ==============================================================================

@pytest.mark.skipif(not (HAS_M1 and HAS_M2), reason="M1 and M2 modules required")
class TestTier3FullClosedLoopCoupling:
    """Interactions coupling ThetaEngine + MaslowIntegrator + ThalamusAGC + BasalGangliaGate."""

    def test_tier3_closed_loop_single_cycle(self):
        """A complete cycle: Need growth -> Phase tipping -> Gate firing -> Quenching reset."""
        engine = ThetaEngine()
        integrator = MaslowIntegrator()
        thalamus = ThalamusAGC()
        gate = BasalGangliaGate()

        # Step forward until motor gate triggers
        motor_fired = False
        steps = 0
        max_steps = 500  # 10.0 s

        while steps < max_steps and not motor_fired:
            # Maslow 10 Hz cadence (every 5 steps)
            if steps % 5 == 0:
                snap = integrator.step(C=0.0, a_motor=0.0)
                engine.latch_maslow(snap)

            # Thalamus 50 Hz
            th_out = thalamus.process(x_ext=np.zeros(64), c_pred=np.zeros(64))

            # Theta step
            th_rad, th_deg, dot_th = engine.step(delta=th_out.delta, a_motor=0.0)

            # Gate evaluation
            decision = gate.evaluate(theta_rad=th_rad)
            if decision.a_motor > 0.05:
                motor_fired = True
                # Trigger quenching
                integrator.step(C=0.0, a_motor=decision.a_motor, E_env=1.0)
                _, _, dot_quenched = engine.step(a_motor=decision.a_motor)
                assert dot_quenched < dot_th  # Velocity braked
            steps += 1

        assert motor_fired is True
        assert steps <= 500  # Within 10.0 s

    def test_tier3_dynamic_agc_protects_against_excitotoxic_burst(self):
        """Rising need tension drives down AGC gain, attenuating external dissonance throughput."""
        integrator = MaslowIntegrator(initial_vector=[0.8]*5)
        thalamus = ThalamusAGC(kappa_agc=0.5)

        snap = integrator.get_snapshot()  # L1 = 4.0
        g_saturated = thalamus.compute_agc_gain(snap.vector, sim_time_s=5.0)

        # Baseline with zero needs
        g_baseline = thalamus.compute_agc_gain(np.zeros(5), sim_time_s=5.0)

        assert g_saturated < g_baseline
        # Dampening protects against unbounded acceleration
        assert g_saturated <= 0.35

    def test_tier3_sleep_lock_overrides_high_arousal(self):
        """High need tension and strong sensory dissonance cannot breach sleep motor lock."""
        engine = ThetaEngine()
        gate = BasalGangliaGate()

        # Simulate high drive: theta pushed to 25.0 deg
        engine.reset(theta_deg=25.0)
        assert engine.theta_deg > 18.0

        decision = gate.evaluate(theta_rad=engine.theta_rad, sleep_lock=True)
        assert decision.a_motor == 0.0
        assert decision.is_locked is True

    def test_tier3_compute_bypass_during_low_dissonance_orbit(self):
        """Subcortical loop maintains rhythmic phase pacing while Thalamus stays in 0-Watt bypass."""
        engine = ThetaEngine()
        integrator = MaslowIntegrator()
        thalamus = ThalamusAGC()

        pings = 0
        for step in range(50):
            if step % 5 == 0:
                snap = integrator.step(C=0.0, a_motor=0.0)
                engine.latch_maslow(snap)

            # Predictable sensory input
            th_out = thalamus.process(x_ext=np.zeros(64), c_pred=np.zeros(64))
            if th_out.emit_ping:
                pings += 1
            engine.step(delta=th_out.delta)

        # Zero pings emitted, pure low-power standby
        assert pings == 0
        # Phase advanced autonomously
        assert engine.theta_rad > THETA_BOOTSTRAP_RAD
