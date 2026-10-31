"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Tier 4: Real-World Scenarios Test Suite
High-fidelity simulations of end-to-end cognitive dynamics:
- Scenario 1: Spontaneous Idle Ignition Without External Prompt (CO2 dead-zone falsification)
- Scenario 2: Closed Limit Cycle Orbit over 200 s (10,000 steps @ 50 Hz, CV <= 0.05)
- Scenario 3: Sleep Lock Override During Vivid Dreaming (Nightmare/REM emulation)
- Scenario 4: Cortical LLM Compute Bypass Transition (Sudden Step Disturbance)
- Scenario 5: 100-Run Stochastic Autonomous Initiation (R_init >= 0.85 Monte Carlo)
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
    DVP01_PERIOD_MIN_S,
    DVP01_PERIOD_MAX_S,
    DVP01_PEAK_CV_MAX,
    DVP01_IGNITION_MAX_S,
    DVP01_R_INIT_MIN,
    HAS_M1,
    HAS_M2,
    HAS_M3,
)

if HAS_M1:
    from core.theta_engine import ThetaEngine
    from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot

if HAS_M2:
    from core.thalamus_agc import ThalamusAGC
    from core.basal_ganglia_gate import BasalGangliaGate

if HAS_M3:
    from core.gate1_verification import Gate1Verifier


# ==============================================================================
# SCENARIO 1: SPONTANEOUS IDLE IGNITION WITHOUT PROMPT
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestScenario1IdleIgnition:
    """Scenario 1: Autonomous phase tipping and ignition without external stimulation."""

    def test_scenario1_spontaneous_idle_tipping_within_10s(self, theta_engine, maslow_integrator):
        """Validates that basal entropy buildup tips phase past 18.0 deg in <= 10.0 s."""
        # Initial cold start
        theta_engine.reset(theta_deg=THETA_BOOTSTRAP_DEG)
        maslow_integrator.reset()

        dt_theta = theta_engine.dt  # 0.02 s (50 Hz)
        dt_maslow = maslow_integrator.dt  # 0.10 s (10 Hz)

        t_sim = 0.0
        crossed_threshold = False
        crossing_time = None

        max_duration_s = 10.0
        steps = int(max_duration_s / dt_theta)

        for step in range(steps):
            # 10 Hz Maslow update every 5 steps
            if step % 5 == 0:
                # Idle: C=0.0, a_motor=0.0, E_env=0.0
                snap = maslow_integrator.step(C=0.0, a_motor=0.0, E_env=0.0)
                theta_engine.latch_maslow(snap)

            # Advance 50 Hz Theta Engine
            th_rad, th_deg, dot_th = theta_engine.step(delta=0.0, a_motor=0.0)
            t_sim += dt_theta

            if th_deg >= THETA_GATE_DEG and not crossed_threshold:
                crossed_threshold = True
                crossing_time = t_sim
                break

        assert crossed_threshold is True
        assert crossing_time is not None
        assert crossing_time <= DVP01_IGNITION_MAX_S
        # Confirms non-zero autonomous drive (falsifying prompt-dependent dormancy)
        assert crossing_time < 5.0


# ==============================================================================
# SCENARIO 2: CLOSED LIMIT CYCLE ORBIT (200 s / 10,000 STEPS)
# ==============================================================================

@pytest.mark.skipif(not (HAS_M1 and HAS_M2), reason="M1 and M2 modules required")
class TestScenario2ClosedLimitCycleOrbit:
    """Scenario 2: Long-term limit cycle stability over 200.0 s physical time."""

    def test_scenario2_closed_limit_cycle_stability(self):
        """Simulates 10,000 steps at 50 Hz; verifies periodic orbit and CV <= 0.05."""
        engine = ThetaEngine()
        integrator = MaslowIntegrator()
        thalamus = ThalamusAGC()
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

            # Thalamus 50 Hz
            th_out = thalamus.process(x_ext=np.zeros(64), c_pred=np.zeros(64))

            # Theta step
            th_rad, th_deg, dot_th = engine.step(delta=th_out.delta, a_motor=current_motor)

            # Gate evaluation
            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            if step >= transient_steps:
                trajectory_th.append(th_rad)
                timestamps.append(t_s)

        th_arr = np.array(trajectory_th)
        t_arr = np.array(timestamps)

        # Peak detection: simple numerical zero crossing of derivative
        peaks_idx = []
        for i in range(1, len(th_arr) - 1):
            if th_arr[i] > th_arr[i - 1] and th_arr[i] >= th_arr[i + 1] and th_arr[i] > THETA_GATE_RAD:
                peaks_idx.append(i)

        assert len(peaks_idx) >= 35, f"Expected >= 35 cycles, found {len(peaks_idx)}"

        peak_times = t_arr[peaks_idx]
        peak_amps = th_arr[peaks_idx]

        periods = np.diff(peak_times)
        mean_period = float(np.mean(periods))
        amp_mean = float(np.mean(peak_amps))
        amp_std = float(np.std(peak_amps))
        amp_cv = amp_std / amp_mean if amp_mean > 0 else 1.0

        assert DVP01_PERIOD_MIN_S <= mean_period <= DVP01_PERIOD_MAX_S
        assert amp_cv <= DVP01_PEAK_CV_MAX


# ==============================================================================
# SCENARIO 3: SLEEP LOCK OVERRIDE DURING VIVID DREAMING
# ==============================================================================

@pytest.mark.skipif(not (HAS_M1 and HAS_M2), reason="M1 and M2 modules required")
class TestScenario3SleepLockOverride:
    """Scenario 3: Motor paralysis during REM/NREM dreaming under extreme arousal."""

    def test_scenario3_sleep_lock_dream_paralysis(self):
        """Simulates high internal dream excitation; validates absolute zero motor output."""
        engine = ThetaEngine()
        integrator = MaslowIntegrator()
        gate = BasalGangliaGate()

        # Step 1: Simulate dream with extreme internal need tension
        integrator.reset(initial_vector=[0.95, 0.90, 0.92, 0.88, 0.99])
        snap = integrator.get_snapshot()
        engine.latch_maslow(snap)

        # Drive phase angle into high excitation
        for _ in range(50):
            th_rad, _, _ = engine.step(delta=0.8, a_motor=0.0)

        # Phase is well above gate threshold (e.g. > 20 deg)
        assert engine.theta_deg > 18.0

        # Step 2: Evaluate with Sleep Lock active (REM dreaming)
        decision_sleeping = gate.evaluate(theta_rad=engine.theta_rad, sleep_lock=True)
        assert decision_sleeping.a_motor == 0.0
        assert decision_sleeping.is_locked is True

        # Step 3: Wake-up event (Sleep Lock released)
        decision_awake = gate.evaluate(theta_rad=engine.theta_rad, sleep_lock=False)
        assert decision_awake.a_motor > 0.5
        assert decision_awake.is_locked is False


# ==============================================================================
# SCENARIO 4: CORTICAL COMPUTE BYPASS TRANSITION
# ==============================================================================

@pytest.mark.skipif(not HAS_M2, reason="M2 modules not available")
class TestScenario4ComputeBypassTransition:
    """Scenario 4: Standby compute bypass and sudden step disturbance interrupt."""

    def test_scenario4_step_disturbance_wakes_cortical_pipeline(self):
        """Quiescent standby -> Step disturbance -> Instant Ping interrupt <= 20 ms."""
        thalamus = ThalamusAGC(d_sens=64, epsilon=0.15)

        # Phase 1: Steady predictable world (0-Watt standby bypass)
        x_predictable = np.zeros(64)
        c_predictable = np.zeros(64)
        for _ in range(20):
            out = thalamus.process(x_ext=x_predictable, c_pred=c_predictable)
            assert out.bypass_active is True
            assert out.emit_ping is False

        # Phase 2: Step disturbance (unexpected sensory event at t = 20)
        x_surprise = np.full(64, 0.4)  # Large discrepancy
        out_surprise = thalamus.process(x_ext=x_surprise, c_pred=c_predictable)

        # Instantaneous Ping interrupt emitted in <= 20 ms (single step)
        assert out_surprise.emit_ping is True
        assert out_surprise.bypass_active is False
        assert out_surprise.delta >= 0.15


# ==============================================================================
# SCENARIO 5: 100-RUN STOCHASTIC AUTONOMOUS INITIATION (R_init >= 0.85)
# ==============================================================================

@pytest.mark.skipif(not HAS_M1, reason="M1 modules not available")
class TestScenario5StochasticInitiationMonteCarlo:
    """Scenario 5: 100 Monte Carlo stochastic runs verifying autonomous drive."""

    def test_scenario5_monte_carlo_r_init(self):
        """100 independent trials under randomized perturbations; verifies R_init >= 0.85."""
        n_trials = 100
        n_success = 0
        max_duration_s = 10.0
        dt = 0.02
        max_steps = int(max_duration_s / dt)
        noise_sigma = 0.01  # rad/s^0.5

        rng = np.random.RandomState(42000)

        for trial in range(n_trials):
            # Sample perturbed initial need vector
            v_init = rng.uniform(0.10, 0.25, size=5)
            # Sample perturbed initial phase angle around bootstrap
            theta_init_deg = rng.normal(15.5, 0.3)

            engine = ThetaEngine(theta_bootstrap_deg=theta_init_deg)
            integrator = MaslowIntegrator(initial_vector=v_init)

            ignited = False
            for step in range(max_steps):
                if step % 5 == 0:
                    snap = integrator.step(C=0.0, a_motor=0.0, E_env=0.0)
                    engine.latch_maslow(snap)

                # Inject Brownian noise perturbation
                noise = rng.normal(0.0, noise_sigma) * math.sqrt(dt)
                th_rad, th_deg, _ = engine.step(delta=0.0, a_motor=0.0)
                # Apply noise to phase
                engine._theta += noise

                if th_deg >= THETA_GATE_DEG:
                    ignited = True
                    break

            if ignited:
                n_success += 1

        r_init = n_success / n_trials
        assert r_init >= DVP01_R_INIT_MIN, f"R_init was {r_init:.2f}, expected >= 0.85"
