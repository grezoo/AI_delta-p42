"""
Empirical Adversarial Stress Test Suite for Cerebellum Engine (DVP-14).
Challenger 2 Verification for Milestone 2.

Covers:
1. Granular reservoir sparsity: 1,000 adversarial inputs (zeros, ones, extremes 1e6,
   Gaussian noise, Cauchy, sparse spikes). Strict sparsity >= 90.0% verification.
2. Timing jitter: 2,000 cycles at 20 Hz. Verification that sigma(Delta t) < 0.5 ms.
3. Critically damped step response: Large steps (+100.0, -100.0, extreme reversals).
   Strict 0.00% mechanical overshoot and monotonicity verification.
4. Purkinje LTD stability: Exploding error signals (+1e5, -1e5, 1e9).
   Strict weight bounding in [-1.0, 1.0] and absence of numerical divergence.
5. Panic override: Immediate clamp to 0.0 within <= 2.0 ms under high-velocity motion.
6. Pathological input bounds and robustness.
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
    N_GRANULE_CANONICAL,
    CerebellarState,
    CerebellumEngine,
)


class TestAdversarialGranuleSparsity:
    """
    Stress-tests the 10,000-cell granule reservoir across 1,000 adversarial inputs.
    Verifies that sparsity is strictly >= 90.0% (<= 1,000 active units) in 100.0% of cases.
    """

    def test_adversarial_sparsity_1000_inputs(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16, n_granule=10000, sparsity_target=0.90)
        rng = np.random.RandomState(42)

        inputs: List[np.ndarray] = []

        # 1. Edge-case constant vectors
        inputs.append(np.zeros(24))
        inputs.append(np.ones(24))
        inputs.append(-np.ones(24))
        inputs.append(np.ones(24) * 0.5)
        inputs.append(np.ones(24) * -0.5)
        inputs.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(24)]))

        # 2. Extreme magnitude inputs (+1e6, -1e6, mixed)
        inputs.append(np.ones(24) * 1e6)
        inputs.append(-np.ones(24) * 1e6)
        inputs.append(np.array([1e6 if i % 2 == 0 else -1e6 for i in range(24)]))
        inputs.append(np.ones(24) * 1e-9)
        inputs.append(-np.ones(24) * 1e-9)

        # 3. Sparse single spikes (varying amplitudes across all 24 input channels)
        for idx in range(24):
            for amp in [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4, 1e6, -1e6]:
                sp = np.zeros(24)
                sp[idx] = amp
                inputs.append(sp)

        # 4. Multi-spike sparse vectors (2 or 3 active channels)
        for _ in range(150):
            sp = np.zeros(24)
            n_active = rng.randint(2, 4)
            indices = rng.choice(24, size=n_active, replace=False)
            sp[indices] = rng.uniform(-1e5, 1e5, size=n_active)
            inputs.append(sp)

        # 5. Gaussian noise at multiple variance scales (1e-6 to 1e6)
        scales = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1e4, 1e6]
        for scale in scales:
            for _ in range(30):
                inputs.append(rng.randn(24) * scale)

        # 6. Heavy-tailed Cauchy noise (outliers and extreme spikes)
        for _ in range(100):
            inputs.append(rng.standard_cauchy(24) * 50.0)

        # 7. Uniform noise across wide dynamic range
        for _ in range(100):
            inputs.append(rng.uniform(-1e6, 1e6, size=24))

        # Fill to exactly 1,000 adversarial inputs
        while len(inputs) < 1000:
            inputs.append(rng.randn(24) * rng.uniform(0.1, 1e5))

        test_pool = inputs[:1000]
        assert len(test_pool) == 1000

        min_sparsity = 1.0
        max_active = 0
        violations = 0

        for idx, u_vec in enumerate(test_pool):
            x_gran = engine.forward_granule(u_vec)

            # Assert shape
            assert x_gran.shape == (10000,), f"Trial {idx}: wrong shape {x_gran.shape}"

            # Assert finite and non-negative
            assert np.all(np.isfinite(x_gran)), f"Trial {idx}: non-finite output in granule activations"
            assert np.all(x_gran >= 0.0), f"Trial {idx}: negative activation found"

            # Count zeros and active units
            zeros_count = int(np.count_nonzero(x_gran == 0.0))
            active_count = int(np.count_nonzero(x_gran > 0.0))
            sparsity = zeros_count / 10000.0

            if sparsity < min_sparsity:
                min_sparsity = sparsity
            if active_count > max_active:
                max_active = active_count

            if sparsity < 0.900 or active_count > 1000:
                violations += 1

        assert violations == 0, f"DVP-14 Granular Sparsity failed in {violations}/1000 trials!"
        assert min_sparsity >= 0.900, f"Minimum sparsity {min_sparsity * 100:.2f}% < 90.0%"
        assert max_active <= 1000, f"Maximum active units {max_active} > 1000"


class TestAdversarialTimingJitter:
    """
    Stress-tests the 20 Hz deterministic cadence over 2,000 cycles.
    Verifies cycle timing standard deviation sigma(Delta t) < 0.5 ms.
    """

    def test_2000_cycles_timing_jitter_dvp14(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16, cadence_hz=20.0)
        nominal_period_ns = int(50.0 * 1_000_000)  # 50 ms in ns
        base_time_ns = 1_000_000_000
        rng = np.random.RandomState(42)

        latencies_ms: List[float] = []

        # Execute 2,000 cycles with Gaussian bus jitter (sigma = 0.18 ms)
        for i in range(2000):
            # Injected jitter with std dev ~ 0.18 ms
            jitter_ns = int(rng.normal(0.0, 0.18) * 1_000_000)
            t_cycle = base_time_ns + i * nominal_period_ns + jitter_ns

            t_start = time.perf_counter_ns()
            state = engine.step(
                a_target=np.ones(8) * 0.1,
                s_proprio=np.zeros(16),
                timestamp_ns=t_cycle,
            )
            t_stop = time.perf_counter_ns()
            latencies_ms.append((t_stop - t_start) / 1_000_000.0)

            assert state.granule_sparsity >= 0.900

        metrics = engine.get_performance_metrics()

        # DVP-14 Acceptance Criteria
        assert metrics["dvp14_jitter_pass"] == 1.0
        assert metrics["jitter_std_ms"] < 0.50, (
            f"Timing jitter std {metrics['jitter_std_ms']:.4f} ms >= 0.50 ms threshold"
        )

        # Computational speed: execution of step() should be well within 50 ms budget
        mean_step_lat = float(np.mean(latencies_ms))
        assert mean_step_lat < 2.0, f"Average step computation {mean_step_lat:.3f} ms exceeds 2.0 ms"

    def test_jitter_detector_rejects_excessive_jitter(self) -> None:
        """Adversarial oracle check: excessive jitter (sigma >= 0.5 ms) must trigger failure flag."""
        engine = CerebellumEngine(d_act=4, d_sens=8, cadence_hz=20.0)
        nominal_period_ns = int(50.0 * 1_000_000)
        base_time_ns = 1_000_000_000
        rng = np.random.RandomState(99)

        # Inject high jitter (sigma = 0.85 ms > 0.50 ms)
        for i in range(200):
            jitter_ns = int(rng.normal(0.0, 0.85) * 1_000_000)
            t_cycle = base_time_ns + i * nominal_period_ns + jitter_ns
            engine.step(a_target=np.zeros(4), s_proprio=np.zeros(8), timestamp_ns=t_cycle)

        metrics = engine.get_performance_metrics()
        assert metrics["jitter_std_ms"] >= 0.50
        assert metrics["dvp14_jitter_pass"] == 0.0, "Jitter detector failed to flag excessive jitter!"


class TestAdversarialCriticallyDampedStepResponse:
    """
    Stress-tests the second-order smoothing filter with large step commands (+100.0, -100.0).
    Verifies that mechanical overshoot is strictly 0.00% and transitions are monotonic.
    """

    def test_large_positive_step_overshoot_zero(self) -> None:
        """Step command a_target = +100.0 starting from 0.0."""
        engine = CerebellumEngine(d_act=8, d_sens=16, damping_ratio_zeta=1.0, natural_freq_omega=15.0)
        a_step = np.ones(8) * 100.0
        s_prop = np.zeros(16)

        positions: List[np.ndarray] = []
        for _ in range(80):
            state = engine.step(a_target=a_step, s_proprio=s_prop)
            positions.append(state.y_trajectory.copy())

        pos_arr = np.array(positions)  # (80, 8)

        # 1. Monotonicity: difference between consecutive steps must be non-negative
        diffs = np.diff(pos_arr, axis=0)
        assert np.all(diffs >= -1e-12), "Trajectory is non-monotonic on positive step rise!"

        # 2. Strict 0.00% mechanical overshoot: position must never exceed target 100.0
        max_pos = float(np.max(pos_arr))
        overshoot = max(0.0, max_pos - 100.0)
        overshoot_pct = (overshoot / 100.0) * 100.0
        assert overshoot_pct == 0.00, f"Mechanical overshoot detected: {overshoot_pct:.6f}% > 0.00%"

        # 3. Final convergence
        assert np.allclose(pos_arr[-1], 100.0, atol=1e-3)

    def test_large_negative_step_overshoot_zero(self) -> None:
        """Step command a_target = -100.0 starting from 0.0."""
        engine = CerebellumEngine(d_act=8, d_sens=16, damping_ratio_zeta=1.0, natural_freq_omega=15.0)
        a_step = -np.ones(8) * 100.0
        s_prop = np.zeros(16)

        positions: List[np.ndarray] = []
        for _ in range(80):
            state = engine.step(a_target=a_step, s_proprio=s_prop)
            positions.append(state.y_trajectory.copy())

        pos_arr = np.array(positions)

        # 1. Monotonicity: difference between consecutive steps must be non-positive
        diffs = np.diff(pos_arr, axis=0)
        assert np.all(diffs <= 1e-12), "Trajectory is non-monotonic on negative step fall!"

        # 2. Strict 0.00% mechanical overshoot: position must never undershoot -100.0
        min_pos = float(np.min(pos_arr))
        overshoot = max(0.0, -100.0 - min_pos)
        overshoot_pct = (overshoot / 100.0) * 100.0
        assert overshoot_pct == 0.00, f"Mechanical undershoot/overshoot detected: {overshoot_pct:.6f}% > 0.00%"

        # 3. Final convergence
        assert np.allclose(pos_arr[-1], -100.0, atol=1e-3)

    def test_full_range_step_reversal_overshoot_zero(self) -> None:
        """Step reversal from +100.0 to -100.0 and back to +100.0."""
        engine = CerebellumEngine(d_act=4, d_sens=8, damping_ratio_zeta=1.0, natural_freq_omega=15.0)

        # Stabilize at +100.0
        for _ in range(60):
            engine.step(a_target=np.ones(4) * 100.0, s_proprio=np.zeros(8))
        assert np.allclose(engine.y_pos, 100.0, atol=1e-3)

        # Reverse to -100.0
        rev_positions: List[np.ndarray] = []
        for _ in range(60):
            st = engine.step(a_target=-np.ones(4) * 100.0, s_proprio=np.zeros(8))
            rev_positions.append(st.y_trajectory.copy())

        rev_arr = np.array(rev_positions)
        min_rev = float(np.min(rev_arr))
        undershoot = max(0.0, -100.0 - min_rev)
        assert undershoot == 0.00, f"Overshoot during reversal: {undershoot:.6f}"
        diffs = np.diff(rev_arr, axis=0)
        assert np.all(diffs <= 1e-12), "Reversal trajectory non-monotonic!"


class TestAdversarialPurkinjeLTDStability:
    """
    Stress-tests Purkinje LTD plasticity against exploding error signals.
    Verifies that weights remain strictly bounded in [-1.0, 1.0] and do not diverge.
    """

    def test_exploding_positive_error_signals_1e5(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16, w_purkinje_max=1.0)
        rng = np.random.RandomState(42)
        u_in = rng.randn(24)
        x_gran = engine.forward_granule(u_in)

        # Inject exploding error of +1e5 for 100 cycles
        for _ in range(100):
            delta_w_mean = engine.adapt_purkinje(np.ones(8) * 1e5, x_gran)
            assert np.all(np.isfinite(engine.W_purkinje))
            assert np.all(engine.W_purkinje >= -1.0)
            assert np.all(engine.W_purkinje <= 1.0)
            assert np.isfinite(delta_w_mean)

        # Synaptic weights must saturate cleanly at boundary
        assert np.isclose(float(np.min(engine.W_purkinje)), -1.0, atol=1e-4)

    def test_exploding_negative_error_signals_1e5(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16, w_purkinje_max=1.0)
        rng = np.random.RandomState(42)
        u_in = rng.randn(24)
        x_gran = engine.forward_granule(u_in)

        # Inject exploding error of -1e5 for 100 cycles
        for _ in range(100):
            delta_w_mean = engine.adapt_purkinje(-np.ones(8) * 1e5, x_gran)
            assert np.all(np.isfinite(engine.W_purkinje))
            assert np.all(engine.W_purkinje >= -1.0)
            assert np.all(engine.W_purkinje <= 1.0)
            assert np.isfinite(delta_w_mean)

        assert np.isclose(float(np.max(engine.W_purkinje)), 1.0, atol=1e-4)

    def test_extreme_alternating_error_bursts_1e9(self) -> None:
        """Adversarial check: alternating extreme error bursts (+1e9, -1e9)."""
        engine = CerebellumEngine(d_act=8, d_sens=16, w_purkinje_max=1.0)
        rng = np.random.RandomState(77)

        for step in range(200):
            dyn_x = engine.forward_granule(rng.randn(24) * 50.0)
            sign = 1.0 if step % 2 == 0 else -1.0
            huge_err = rng.randn(8) * 1e9 * sign

            engine.adapt_purkinje(huge_err, dyn_x)
            assert np.all(np.isfinite(engine.W_purkinje))
            assert np.all(engine.W_purkinje >= -1.0)
            assert np.all(engine.W_purkinje <= 1.0)

    def test_zero_error_signal_zero_weight_update(self) -> None:
        engine = CerebellumEngine(d_act=4, d_sens=8)
        u_in = np.ones(12)
        x_gran = engine.forward_granule(u_in)

        # Pre-set known weights
        engine.W_purkinje = np.ones((4, 10000)) * 0.25
        dw = engine.adapt_purkinje(np.zeros(4), x_gran)

        assert dw == 0.0
        assert np.all(engine.W_purkinje == 0.25)


class TestAdversarialPanicOverride:
    """
    Stress-tests Amygdala panic hardware interlock clamp.
    Verifies immediate zero clamp within <= 2.0 ms regardless of high-speed motion.
    """

    def test_panic_clamp_latency_and_zero_trajectory(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16)

        latencies_ms: List[float] = []

        # 500 trials of high-velocity motion abruptly clamped by panic override
        for _ in range(500):
            # Prime engine with high velocity and acceleration
            engine.y_pos = np.ones(8) * 100.0
            engine.y_vel = np.ones(8) * 500.0
            engine.y_acc = np.ones(8) * 2000.0

            t0 = time.perf_counter_ns()
            state = engine.step(
                a_target=np.ones(8) * 100.0,
                s_proprio=np.ones(16) * 10.0,
                panic_amy=True,
            )
            t1 = time.perf_counter_ns()

            lat_ms = (t1 - t0) / 1_000_000.0
            latencies_ms.append(lat_ms)

            # Strict 0.0 clamping
            assert np.all(state.y_trajectory == 0.0)
            assert np.all(state.y_velocity == 0.0)
            assert np.all(state.y_acceleration == 0.0)
            assert state.is_interlocked is True
            assert state.granule_sparsity == 1.0
            assert np.all(state.purkinje_correction == 0.0)
            assert np.all(state.tracking_error == 0.0)

        max_lat = max(latencies_ms)
        mean_lat = float(np.mean(latencies_ms))

        # Must execute within <= 2.0 ms WCET ceiling
        assert max_lat <= 2.0, f"Max panic clamp latency {max_lat:.4f} ms exceeded 2.0 ms!"
        assert mean_lat < 0.1, f"Mean panic clamp latency {mean_lat:.4f} ms is too high"

    def test_recovery_from_panic_interlock_is_smooth(self) -> None:
        """When panic_amy turns False after a clamp, motion must resume smoothly without NaN/glitch."""
        engine = CerebellumEngine(d_act=4, d_sens=8)

        # 1. Drive into panic
        engine.step(a_target=np.ones(4) * 5.0, s_proprio=np.zeros(8), panic_amy=True)
        assert np.all(engine.y_pos == 0.0)

        # 2. Release panic and apply target
        st1 = engine.step(a_target=np.ones(4) * 10.0, s_proprio=np.zeros(8), panic_amy=False)
        assert st1.is_interlocked is False
        assert np.all(np.isfinite(st1.y_trajectory))
        # Initial step from 0 towards 10 must be positive and bounded
        assert np.all(st1.y_trajectory > 0.0)
        assert np.all(st1.y_trajectory < 10.0)


class TestPathologicalInputRobustness:
    """Tests handling of NaNs, Infs, and dimensional mismatches."""

    def test_non_finite_target_and_proprioception_fallback(self) -> None:
        engine = CerebellumEngine(d_act=4, d_sens=8)

        # Inject NaNs and Infinities into targets and proprioception
        dirty_target = np.array([np.nan, np.inf, -np.inf, 2.0])
        dirty_proprio = np.array([np.nan] * 8)

        state = engine.step(a_target=dirty_target, s_proprio=dirty_proprio)

        assert np.all(np.isfinite(state.y_trajectory))
        assert np.all(np.isfinite(state.y_velocity))
        assert np.all(np.isfinite(state.y_acceleration))
        assert state.granule_sparsity >= 0.900

    def test_dimension_mismatch_auto_padding(self) -> None:
        engine = CerebellumEngine(d_act=8, d_sens=16)

        # Target with 4 elements instead of 8, proprio with 10 instead of 16
        short_target = np.array([1.0, 2.0, 3.0, 4.0])
        short_proprio = np.array([0.5] * 10)

        state = engine.step(a_target=short_target, s_proprio=short_proprio)

        assert state.y_trajectory.shape == (8,)
        assert state.purkinje_correction.shape == (8,)
        assert state.tracking_error.shape == (8,)
