"""
Unit and DVP-14 Acceptance Test Suite for Cerebellum Engine Module (Gate 2).
Covers:
1. Exactly 10,000 granule cells reservoir dimensionality.
2. Guaranteed >= 90.0% sparsity across 500 diverse input patterns.
3. 20 Hz deterministic cadence with motor timing jitter < 0.5 ms over 1000 cycles.
4. Purkinje LTD online learning achieving >= 85.0% error reduction under load disturbance (DVP-14).
5. Second-order critically damped filter: monotonic step response with 0.00% overshoot.
6. Amygdala panic hardware interlock immediate clamp to 0.0.
7. Dataclass immutability and reset behavior.
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest

from modules.cerebellum_engine import (
    CADENCE_HZ,
    DT_DEFAULT,
    MODULE_AMYGDALA,
    MODULE_CEREBELLUM,
    MODULE_ID,
    N_GRANULE_CANONICAL,
    CerebellarState,
    CerebellumEngine,
)


class TestCerebellarReservoirGeometry:
    """Verifies reservoir cell count and 90% sparsity enforcement."""

    def test_canonical_constants_and_module_ids(self) -> None:
        assert MODULE_ID == 7
        assert MODULE_CEREBELLUM == 7
        assert MODULE_AMYGDALA == 6
        assert CADENCE_HZ == 20.0
        assert DT_DEFAULT == 0.050
        assert N_GRANULE_CANONICAL == 10000

    def test_granule_cell_count_is_strictly_10000(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16, n_granule=10000)
        assert engine.n_granule == 10000
        assert engine.W_mossy.shape == (10000, 24)
        assert engine.b_mossy.shape == (10000,)
        assert engine.W_purkinje.shape == (8, 10000)

        # Forward output must have exactly 10,000 elements
        u_in = np.ones(24)
        x_gran = engine.forward_granule(u_in)
        assert x_gran.shape == (10000,)

    def test_guaranteed_sparsity_over_500_diverse_inputs(self) -> None:
        """
        DVP-14 Requirement:
        For 100% of tested input vectors, Granule Cell Sparsity >= 90.0% (<= 1000 active units).
        """
        engine = CerebellumEngine(d_act=8, d_sens=16, n_granule=10000, sparsity_target=0.90)
        rng = np.random.RandomState(42)

        # 1. Zero input
        x_zero = engine.forward_granule(np.zeros(24))
        zeros_ratio = np.count_nonzero(x_zero == 0.0) / 10000.0
        assert zeros_ratio >= 0.900

        # 2. Extreme magnitude input jump
        x_extreme = engine.forward_granule(np.ones(24) * 100.0)
        zeros_extreme = np.count_nonzero(x_extreme == 0.0) / 10000.0
        assert zeros_extreme >= 0.900

        # 3. 500 diverse random vectors (uniform, normal, sparse spikes)
        for i in range(500):
            if i % 3 == 0:
                u_test = rng.randn(24) * 5.0
            elif i % 3 == 1:
                u_test = rng.uniform(-10.0, 10.0, size=24)
            else:
                u_test = np.zeros(24)
                u_test[rng.randint(0, 24)] = rng.uniform(1.0, 50.0)

            x_out = engine.forward_granule(u_test)
            active_units = int(np.count_nonzero(x_out > 0.0))
            sparsity = (10000 - active_units) / 10000.0

            assert active_units <= 1000, f"Trial {i}: active units {active_units} > 1000"
            assert sparsity >= 0.900, f"Trial {i}: sparsity {sparsity:.4f} < 0.900"


class TestSecondOrderSmoothingAndOvershoot:
    """Verifies critical damping (zeta=1.0), zero overshoot, and smooth trajectory."""

    def test_critically_damped_unit_step_zero_overshoot(self) -> None:
        """
        Acceptance test: A step input a_target = 1.0 into a critically damped
        oscillator MUST be strictly monotonic and exhibit exactly 0.00% overshoot.
        """
        engine = CerebellumEngine(
            d_act=1,
            d_sens=1,
            damping_ratio_zeta=1.0,
            natural_freq_omega=15.0,
        )

        a_step = np.array([1.0])
        s_prop = np.array([0.0])

        positions: List[float] = []

        # Execute 50 steps at 20 Hz (2.5 seconds total)
        for _ in range(50):
            state = engine.step(a_target=a_step, s_proprio=s_prop)
            pos = float(state.y_trajectory[0])
            positions.append(pos)

        # 1. Final position converges to target 1.0
        assert pytest.approx(positions[-1], abs=1e-3) == 1.0

        # 2. Strict monotonicity: position never decreases during step rise
        for k in range(len(positions) - 1):
            assert positions[k + 1] >= positions[k] - 1e-9, f"Non-monotonic at step {k}: {positions[k]} -> {positions[k+1]}"

        # 3. Exactly 0.00% overshoot: maximum position must never exceed 1.0
        max_pos = max(positions)
        overshoot = max(0.0, max_pos - 1.0)
        assert overshoot == 0.0, f"Mechanical overshoot detected: {overshoot * 100:.4f}% > 0.0%"


class TestPurkinjeLTDOnlineLearningDVP14:
    """Verifies DVP-14 dynamic inverse model learning: >= 85% tracking error reduction."""

    def test_purkinje_ltd_error_reduction_dvp14(self) -> None:
        """
        DVP-14 Verification Protocol:
        - Couple Cerebellum to a simulated plant with an unknown load/disturbance torque.
        - Initial tracking error e_0 is recorded without compensation.
        - Execute 100 steps of online Purkinje LTD learning.
        - Final tracking error e_100 is recorded.
        - Pass Criteria: (e_0 - e_100) / e_0 >= 85.0%.
        """
        engine = CerebellumEngine(
            d_act=1,
            d_sens=2,
            n_granule=10000,
            eta_ltd=0.025,
            w_purkinje_max=5.0,
            seed=42,
        )

        # Target command: hold constant position
        a_target = np.array([1.0])
        s_proprio = np.array([0.5, 0.0])

        # Simulated physical plant:
        # Plant output is: s_actual = a_target - disturbance + gain * u_purkinje
        # With a significant disturbance of 0.80, without compensation s_actual = 0.20
        # Initial error e_0 = 1.0 - 0.20 = 0.80
        disturbance = 0.80
        plant_gain = 1.0

        errors: List[float] = []

        # Step 0: Record initial uncompensated error
        state_0 = engine.step(
            a_target=a_target,
            s_proprio=s_proprio,
            s_actual=np.array([a_target[0] - disturbance]),
        )
        e_0 = abs(float(state_0.tracking_error[0]))
        errors.append(e_0)

        # Run 100 online adaptation cycles
        for step_idx in range(1, 101):
            # Current feedforward Purkinje output from previous adaptation
            u_in = np.concatenate([a_target, s_proprio])
            x_gran = engine.forward_granule(u_in)
            u_purk = float((engine.W_purkinje @ x_gran)[0])

            # Simulated physical plant responds to target and cerebellar boost
            s_actual_current = np.array([a_target[0] - disturbance + plant_gain * u_purk])

            state = engine.step(
                a_target=a_target,
                s_proprio=s_proprio,
                s_actual=s_actual_current,
            )
            e_step = abs(float(state.tracking_error[0]))
            errors.append(e_step)

        e_100 = errors[-1]
        reduction_ratio = (e_0 - e_100) / e_0

        assert reduction_ratio >= 0.850, (
            f"DVP-14 failure: tracking error reduction {reduction_ratio * 100:.2f}% "
            f"is below required 85.0% threshold (e_0={e_0:.4f}, e_100={e_100:.4f})"
        )


class TestMotorTimingJitterAndSafetyInterlock:
    """Verifies DVP-14 jitter constraint (< 0.5 ms) and Amygdala panic clamp."""

    def test_motor_timing_jitter_dvp14(self) -> None:
        """
        Verifies timing jitter standard deviation sigma(Delta t) < 0.5 ms over 1000 cycles.
        Uses simulated periodic timestamps with sub-millisecond Gaussian variance.
        """
        engine = CerebellumEngine(d_act=4, d_sens=8, cadence_hz=20.0)
        nominal_period_ns = int(50.0 * 1_000_000)  # 50 ms = 50,000,000 ns

        rng = np.random.RandomState(42)
        base_time_ns = 1_000_000_000

        # Inject 1000 cycles with low OS jitter (jitter std ~ 0.15 ms)
        for i in range(1000):
            # Small jitter ~ 0.1 ms standard deviation
            jitter_ns = int(rng.normal(0.0, 0.15) * 1_000_000)
            t_cycle = base_time_ns + i * nominal_period_ns + jitter_ns

            engine.step(
                a_target=np.zeros(4),
                s_proprio=np.zeros(8),
                timestamp_ns=t_cycle,
            )

        metrics = engine.get_performance_metrics()
        assert metrics["dvp14_jitter_pass"] == 1.0
        assert metrics["jitter_std_ms"] < 0.5, f"Timing jitter std {metrics['jitter_std_ms']:.4f} ms >= 0.5 ms"

    def test_amygdala_panic_hardware_interlock_clamp(self) -> None:
        """
        Verifies that when panic_amy == True, trajectory is immediately clamped
        to zero within <= 2.0 ms, regardless of previous high-speed motion.
        """
        engine = CerebellumEngine(d_act=4, d_sens=8)

        # Drive robot into active motion
        for _ in range(3):
            engine.step(a_target=np.ones(4) * 10.0, s_proprio=np.ones(8) * 5.0)

        assert np.linalg.norm(engine.y_pos) > 1.0
        assert np.linalg.norm(engine.y_vel) > 1.0

        # Fire emergency Amygdala override
        t0 = time.perf_counter_ns()
        state_panic = engine.step(
            a_target=np.ones(4) * 10.0,
            s_proprio=np.ones(8) * 5.0,
            panic_amy=True,
        )
        t_clamp = (time.perf_counter_ns() - t0) / 1_000_000.0

        # 1. Output trajectory, velocity, acceleration must be strictly 0.0
        assert np.all(state_panic.y_trajectory == 0.0)
        assert np.all(state_panic.y_velocity == 0.0)
        assert np.all(state_panic.y_acceleration == 0.0)
        assert state_panic.is_interlocked is True

        # 2. Execution latency must be well within 2.0 ms
        assert t_clamp < 2.0, f"Interlock clamp latency {t_clamp:.4f} ms > 2.0 ms"

    def test_reset_state_clears_dynamical_registers(self) -> None:
        engine = CerebellumEngine(d_act=2, d_sens=4)
        engine.step(a_target=np.ones(2), s_proprio=np.zeros(4))
        assert np.linalg.norm(engine.y_pos) > 0.0

        engine.reset_state()
        assert np.all(engine.y_pos == 0.0)
        assert np.all(engine.y_vel == 0.0)
        assert np.all(engine.y_acc == 0.0)
        assert np.all(engine.W_purkinje == 0.0)
