"""
Adversarial Stress Test Suite for Thalamus AGC Module (core/thalamus_agc.py).
Targeted verification by challenger_m2_1 for Milestone 2 Gate 1.

Mandatory Verification Dimensions:
1. Dimensional Invariance across d_sens in [1, 2, 64, 512, 4096] with uniform Delta = 0.15.
2. AGC Saturation across ||V||_1 in [0.0, 5.0, 10.0] and extreme norms [100.0, 1e6, 1e15].
3. Compute Bypass FSM boundary transitions around epsilon = 0.15 (0.14999 vs 0.15001).
4. Extended Adversarial Stress: CDC scale (d=65536), sleep override, immutability, shape mismatch.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.thalamus_agc import (
    ComputeState,
    ThalamusOutput,
    ThalamusAGC,
)


class TestAdversarialDimensionalInvariance:
    """Stress-testing the dimension-independent RMS dissonance formulation."""

    @pytest.mark.parametrize("d", [1, 2, 64, 512, 4096])
    def test_mandatory_dispatch_dimensional_invariance(self, d: int):
        """
        Dispatch Requirement 1:
        Test dissonance across d_sens in [1, 2, 64, 512, 4096] with uniform error Delta = 0.15;
        confirm delta(t) == 0.15 across all dimensions within 1e-6.
        """
        th = ThalamusAGC(d_sens=d)
        delta_val = 0.15
        x_ext = np.full(d, delta_val, dtype=np.float64)
        c_pred = np.zeros(d, dtype=np.float64)

        delta = th.compute_dissonance(x_ext, c_pred)

        # Strict empirical verification: within 1e-6
        diff = abs(delta - delta_val)
        assert diff <= 1e-6, f"Failed at d={d}: delta={delta}, diff={diff}"
        assert math.isclose(delta, delta_val, abs_tol=1e-6)

    def test_extreme_scale_cdc_dimension(self):
        """Stress-test at d = 65536 (CDC ring buffer slot capacity dimension)."""
        d = 65536
        th = ThalamusAGC(d_sens=d)
        delta_val = 0.15
        x_ext = np.full(d, delta_val, dtype=np.float64)
        c_pred = np.zeros(d, dtype=np.float64)

        delta = th.compute_dissonance(x_ext, c_pred)
        assert math.isclose(delta, delta_val, abs_tol=1e-6)

    def test_alternating_sign_perturbation(self):
        """Test with alternating +/- Delta signs; RMS should still be strictly Delta."""
        for d in [2, 64, 512, 4096]:
            th = ThalamusAGC(d_sens=d)
            signs = np.array([1.0, -1.0] * (d // 2), dtype=np.float64)
            x_ext = signs * 0.15
            c_pred = np.zeros(d, dtype=np.float64)
            delta = th.compute_dissonance(x_ext, c_pred)
            assert math.isclose(delta, 0.15, abs_tol=1e-6)

    def test_non_uniform_perturbation_with_known_rms(self):
        """
        Adversarial non-uniform distribution:
        Half elements at 0.10, half elements at sqrt(2 * 0.15^2 - 0.10^2) = sqrt(0.045 - 0.01) = sqrt(0.035) ~= 0.187082869.
        Target RMS = sqrt( (0.10^2 + 0.035) / 2 ) = sqrt( 0.045 / 2 ) = 0.15.
        """
        for d in [2, 64, 512, 4096]:
            th = ThalamusAGC(d_sens=d)
            v1 = 0.10
            v2 = math.sqrt(2.0 * (0.15 ** 2) - (v1 ** 2))
            half = d // 2
            x_ext = np.concatenate([np.full(half, v1), np.full(half, v2)])
            c_pred = np.zeros(d, dtype=np.float64)
            delta = th.compute_dissonance(x_ext, c_pred)
            assert math.isclose(delta, 0.15, abs_tol=1e-6)


class TestAdversarialAGCSaturation:
    """Stress-testing Dynamic Automatic Gain Control (AGC) saturation and bounds."""

    @pytest.mark.parametrize("l1_target,expected_raw", [
        (0.0, 1.0000000000),      # 1 / (1 + 0.5 * 0.0) = 1.0
        (5.0, 1.0 / 3.5),          # 1 / (1 + 0.5 * 5.0) = 0.2857142857
        (10.0, 1.0 / 6.0),         # 1 / (1 + 0.5 * 10.0) = 0.1666666667
    ])
    def test_mandatory_dispatch_agc_saturation(self, l1_target: float, expected_raw: float):
        """
        Dispatch Requirement 2:
        Test ||V||_1 in [0.0, 5.0, 10.0] and verify g_AGC remains bounded in [0.10, 1.00]
        without division by zero.
        """
        th = ThalamusAGC(kappa_agc=0.50)
        # Construct 5D vector with exact L1 norm
        v_vector = np.full(5, l1_target / 5.0, dtype=np.float64)
        assert math.isclose(np.sum(np.abs(v_vector)), l1_target, abs_tol=1e-12)

        # 1. Pure steady-state AGC (sim_time_s = None)
        g_pure = th.compute_agc_gain(v_vector)
        assert 0.10 <= g_pure <= 1.00
        assert math.isclose(g_pure, expected_raw, rel_tol=1e-6)

        # 2. Post-bootstrap steady-state AGC (sim_time_s = 10.0 s > 3.0 s)
        g_post_boot = th.compute_agc_gain(v_vector, sim_time_s=10.0)
        assert 0.10 <= g_post_boot <= 1.00
        assert math.isclose(g_post_boot, expected_raw, rel_tol=1e-6)

    @pytest.mark.parametrize("l1_extreme", [100.0, 1000.0, 1e6, 1e15])
    def test_extreme_norm_lower_bound_clamping(self, l1_extreme: float):
        """Massive L1 norms must clamp strictly at floor 0.10 without underflow or div-by-zero."""
        th = ThalamusAGC(kappa_agc=0.50)
        v_vector = np.full(5, l1_extreme / 5.0, dtype=np.float64)

        g_pure = th.compute_agc_gain(v_vector)
        assert math.isclose(g_pure, 0.10, abs_tol=1e-9)
        assert 0.10 <= g_pure <= 1.00

        g_post = th.compute_agc_gain(v_vector, sim_time_s=10.0)
        assert math.isclose(g_post, 0.10, abs_tol=1e-9)

    def test_zero_maslow_norm_upper_bound(self):
        """Zero vector must yield exactly 1.00, never exceeding 1.00."""
        th = ThalamusAGC(kappa_agc=0.50)
        g = th.compute_agc_gain(np.zeros(5), sim_time_s=10.0)
        assert g == 1.00

    def test_negative_values_in_maslow_vector(self):
        """Negative components must be handled via L1 norm (abs sum) without inversion."""
        th = ThalamusAGC(kappa_agc=0.50)
        v_neg = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        # ||V||_1 should be 5.0
        g = th.compute_agc_gain(v_neg, sim_time_s=10.0)
        assert math.isclose(g, 1.0 / 3.5, rel_tol=1e-6)

    def test_empty_maslow_vector(self):
        """Empty vector ||V||_1 == 0 yields 1.00 without error."""
        th = ThalamusAGC(kappa_agc=0.50)
        g = th.compute_agc_gain(np.array([]), sim_time_s=10.0)
        assert g == 1.00

    def test_day0_bootstrap_progression(self):
        """Day 0 soft bootstrap smooth ramp verification."""
        th = ThalamusAGC()
        # Verify monotonicity and key checkpoints
        t_samples = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
        g_boots = [th.compute_bootstrap_gain(t) for t in t_samples]
        for i in range(len(g_boots) - 1):
            assert g_boots[i] <= g_boots[i + 1]

        assert math.isclose(th.compute_bootstrap_gain(1.5), 0.50, abs_tol=1e-4)
        assert math.isclose(th.compute_bootstrap_gain(3.0), 1.00, abs_tol=1e-6)
        assert th.compute_bootstrap_gain(4.0) == 1.00


class TestAdversarialComputeBypassFSM:
    """Stress-testing the Cortical Compute Bypass FSM boundary transitions."""

    def test_mandatory_dispatch_boundary_transitions(self):
        """
        Dispatch Requirement 3:
        Test boundary transitions around epsilon = 0.15 (e.g. 0.14999 vs 0.15001),
        verify emit_ping and bypass_active states.
        """
        epsilon = 0.15
        th = ThalamusAGC(d_sens=1, epsilon=epsilon)

        # Subthreshold: 0.14999
        out_sub = th.process(x_ext=np.array([0.14999]), c_pred=np.array([0.0]))
        assert out_sub.delta < epsilon
        assert out_sub.emit_ping is False, f"Expected emit_ping False at delta={out_sub.delta}"
        assert out_sub.bypass_active is True, f"Expected bypass_active True at delta={out_sub.delta}"
        assert out_sub.compute_state == ComputeState.COMPUTE_BYPASS

        # Suprathreshold: 0.15001
        out_sup = th.process(x_ext=np.array([0.15001]), c_pred=np.array([0.0]))
        assert out_sup.delta > epsilon
        assert out_sup.emit_ping is True, f"Expected emit_ping True at delta={out_sup.delta}"
        assert out_sup.bypass_active is False, f"Expected bypass_active False at delta={out_sup.delta}"
        assert out_sup.compute_state == ComputeState.COMPUTE_ACTIVE

    @pytest.mark.parametrize("delta_in,expected_ping,expected_bypass,expected_state", [
        (0.149990, False, True, ComputeState.COMPUTE_BYPASS),
        (0.149999, False, True, ComputeState.COMPUTE_BYPASS),
        (0.150000, True, False, ComputeState.COMPUTE_ACTIVE),   # Exact boundary (delta >= epsilon)
        (0.150001, True, False, ComputeState.COMPUTE_ACTIVE),
        (0.150010, True, False, ComputeState.COMPUTE_ACTIVE),
    ])
    def test_micro_boundary_sweep_epsilon_015(
        self,
        delta_in: float,
        expected_ping: bool,
        expected_bypass: bool,
        expected_state: ComputeState,
    ):
        """Micro-step sweep within 1e-5 around threshold epsilon = 0.15."""
        th = ThalamusAGC(d_sens=1, epsilon=0.15)
        out = th.process(x_ext=np.array([delta_in]), c_pred=np.array([0.0]))
        assert math.isclose(out.delta, delta_in, abs_tol=1e-7)
        assert out.emit_ping == expected_ping
        assert out.bypass_active == expected_bypass
        assert out.compute_state == expected_state

    def test_custom_epsilon_thresholds(self):
        """Verify FSM behaves consistently with non-default epsilon thresholds."""
        for eps in [0.05, 0.30, 0.75]:
            th = ThalamusAGC(d_sens=4, epsilon=eps)
            # Just below
            out_below = th.process(x_ext=np.full(4, eps - 1e-5), c_pred=np.zeros(4))
            assert out_below.emit_ping is False
            assert out_below.bypass_active is True
            assert out_below.compute_state == ComputeState.COMPUTE_BYPASS

            # Exactly at threshold
            out_exact = th.process(x_ext=np.full(4, eps), c_pred=np.zeros(4))
            assert out_exact.emit_ping is True
            assert out_exact.bypass_active is False
            assert out_exact.compute_state == ComputeState.COMPUTE_ACTIVE

            # Just above
            out_above = th.process(x_ext=np.full(4, eps + 1e-5), c_pred=np.zeros(4))
            assert out_above.emit_ping is True
            assert out_above.bypass_active is False
            assert out_above.compute_state == ComputeState.COMPUTE_ACTIVE

    def test_sleep_mode_nrem_hard_override(self):
        """When is_sleeping=True, even maximum dissonance (delta=1.0) MUST NOT emit Ping."""
        th = ThalamusAGC(d_sens=64, epsilon=0.15)
        out = th.process(
            x_ext=np.ones(64),
            c_pred=np.zeros(64),
            is_sleeping=True,
        )
        assert out.delta == 1.0
        assert out.emit_ping is False
        assert out.bypass_active is True
        assert out.compute_state == ComputeState.NREM_SHUTDOWN
        assert out.g_agc == 0.0
        assert np.all(out.s_mixed == 0.0)

    def test_sensory_gain_adaptation_adversarial(self):
        """
        Sensory gain g_sensor starts at 0.50, adapts:
        - min(1.0, g_sensor + 0.3) on COMPUTE_ACTIVE
        - max(0.05, 0.95 * g_sensor) on COMPUTE_BYPASS
        """
        th = ThalamusAGC(d_sens=4, epsilon=0.15)
        assert th.g_sensor == 0.50

        # Fire 3 pings: 0.5 -> 0.8 -> 1.0 -> 1.0 (clamped)
        th.process(np.full(4, 0.20), np.zeros(4))
        assert math.isclose(th.g_sensor, 0.80, abs_tol=1e-5)

        th.process(np.full(4, 0.20), np.zeros(4))
        assert math.isclose(th.g_sensor, 1.00, abs_tol=1e-5)

        th.process(np.full(4, 0.20), np.zeros(4))
        assert th.g_sensor == 1.00

        # Now 100 bypass steps: should exponentially decay towards floor 0.05
        for _ in range(100):
            th.process(np.zeros(4), np.zeros(4))

        assert math.isclose(th.g_sensor, 0.05, abs_tol=1e-3)
        assert th.g_sensor >= 0.05


class TestAdversarialDataIntegrityAndRobustness:
    """Stress-testing immutability, shape mismatch, and numerical stability."""

    def test_synesthetic_output_immutability(self):
        """s_mixed tensor in ThalamusOutput must be strictly read-only."""
        th = ThalamusAGC(d_sens=16)
        out = th.process(np.ones(16), np.zeros(16))
        assert out.s_mixed.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            out.s_mixed[0] = 999.0

    def test_mismatched_sensory_shapes_raises_value_error(self):
        """Mismatched shapes between x_ext and c_pred must raise ValueError."""
        th = ThalamusAGC(d_sens=16)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            th.compute_dissonance(np.zeros(16), np.zeros(17))

    def test_nan_and_inf_handling_robustness(self):
        """
        Investigate NaN/Inf handling:
        If sensory input contains NaN, dissonance results in NaN.
        In Python, nan >= epsilon is False, so compute bypass remains engaged (safe fail).
        """
        th = ThalamusAGC(d_sens=4, epsilon=0.15)
        x_nan = np.array([0.10, np.nan, 0.10, 0.10])
        c = np.zeros(4)
        out = th.process(x_nan, c)
        assert math.isnan(out.delta)
        # NaN does not trigger Ping! (fail-safe standby)
        assert out.emit_ping is False
        assert out.bypass_active is True
        assert out.compute_state == ComputeState.COMPUTE_BYPASS


def run_standalone_diagnostics() -> int:
    """Run diagnostics and print exact numbers for verification report."""
    print("=== STARTING ADVERSARIAL STRESS TEST FOR THALAMUS AGC ===")

    # 1. Dimensional invariance verification
    print("\n[1] DIMENSIONAL INVARIANCE:")
    dims = [1, 2, 64, 512, 4096, 65536]
    delta_val = 0.15
    for d in dims:
        th = ThalamusAGC(d_sens=d)
        x = np.full(d, delta_val, dtype=np.float64)
        c = np.zeros(d, dtype=np.float64)
        computed_delta = th.compute_dissonance(x, c)
        diff = abs(computed_delta - delta_val)
        passed = diff <= 1e-6
        print(f"  d_sens = {d:5d} | delta = {computed_delta:.10f} | target = {delta_val:.2f} | diff = {diff:.2e} | PASS={passed}")
        assert passed

    # 2. AGC Saturation verification
    print("\n[2] AGC SATURATION & BOUNDS:")
    th = ThalamusAGC(kappa_agc=0.50)
    test_l1s = [0.0, 0.90, 5.0, 10.0, 50.0, 100.0, 1000.0, 1e6]
    for l1 in test_l1s:
        v = np.full(5, l1 / 5.0, dtype=np.float64)
        g_pure = th.compute_agc_gain(v)
        g_post = th.compute_agc_gain(v, sim_time_s=10.0)
        in_bounds = (0.10 <= g_pure <= 1.00) and (0.10 <= g_post <= 1.00)
        expected = 1.0 / (1.0 + 0.50 * l1)
        expected_clamped = max(0.10, min(1.00, expected))
        diff = abs(g_pure - expected_clamped)
        print(f"  ||V||_1 = {l1:8.1f} | g_pure = {g_pure:.6f} | g_post = {g_post:.6f} | expected = {expected_clamped:.6f} | diff={diff:.2e} | in_bounds={in_bounds}")
        assert in_bounds

    # 3. Compute Bypass FSM boundary sweep around epsilon = 0.15
    print("\n[3] COMPUTE BYPASS FSM BOUNDARY SWEEP (epsilon = 0.15):")
    th = ThalamusAGC(d_sens=1, epsilon=0.15)
    boundary_tests = [
        0.149900,
        0.149990,
        0.149999,
        0.150000,
        0.150001,
        0.150010,
        0.150100,
    ]
    for val in boundary_tests:
        out = th.process(x_ext=np.array([val]), c_pred=np.array([0.0]))
        print(
            f"  delta = {out.delta:.6f} | emit_ping = {str(out.emit_ping):5s} | bypass_active = {str(out.bypass_active):5s} | state = {out.compute_state.name}"
        )
        if val < 0.15:
            assert out.emit_ping is False and out.bypass_active is True and out.compute_state == ComputeState.COMPUTE_BYPASS
        else:
            assert out.emit_ping is True and out.bypass_active is False and out.compute_state == ComputeState.COMPUTE_ACTIVE

    print("\n=== ALL ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY ===")
    return 0


if __name__ == "__main__":
    sys.exit(run_standalone_diagnostics())
