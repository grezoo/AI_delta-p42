"""
Adversarial Stress-Testing and Boundary Verification Suite for Milestone 1 (M1).
Targets:
    - core/theta_engine.py
    - core/maslow_integrator.py

Verifies:
1. Division-by-zero boundary when lambda_i < 1e-12.
2. Stability under extreme inputs: delta=1.0, V_l1=5.0, a_motor=1.0, and thousands of steps.
3. Multi-turn phase continuity and wrapping across [0, 2*pi).
4. Verification that V_i never falls below 0.0 or exceeds 1.0 without artificial clipping.
5. Absolute Zero NaN/Inf guarantees across all execution domains.
"""

from __future__ import annotations

import ast
import inspect
import math
import warnings
import numpy as np
import pytest

from core.maslow_integrator import (
    DEFAULT_INITIAL_VECTOR,
    NEED_NAMES,
    MaslowIntegrator,
    MaslowSnapshot,
)
from core.theta_engine import DEG2RAD, ThetaEngine


# ==============================================================================
# 1. DIVISION-BY-ZERO BOUNDARY (lambda_i < 1e-12)
# ==============================================================================

class TestAdversarialVanishingRates:
    """Adversarial stress-testing of the lambda_i < 1e-12 vanishing rate boundary."""

    def test_exact_zero_rates_no_warning_no_nan(self):
        """When lambda_base=0, C=0, a_motor=0 -> lambda = 0.0 across all 5 dimensions."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Any division-by-zero or numpy warning raises exception
            integrator = MaslowIntegrator(
                lambda_base=0.0,
                lambda_work=0.0,
                eta_gain=0.0,
                initial_vector=[0.11, 0.22, 0.33, 0.44, 0.55],
            )
            for _ in range(50):
                snap = integrator.step(C=0.0, a_motor=0.0, E_env=0.0)

            # State must be exactly preserved with zero NaN or Inf
            np.testing.assert_allclose(snap.vector, [0.11, 0.22, 0.33, 0.44, 0.55], rtol=1e-14)
            assert np.all(np.isfinite(snap.vector))
            assert not np.any(np.isnan(snap.vector))

    @pytest.mark.parametrize(
        "sub_lambda",
        [
            0.0,
            1e-20,
            1e-15,
            1e-13,
            1e-12 - 1e-16,
            1e-12 - 1e-20,
        ],
    )
    def test_sub_threshold_lambda_preserves_state(self, sub_lambda: float):
        """Rates strictly below 1e-12 must safely preserve state without crashing or mutating."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            integrator = MaslowIntegrator(
                lambda_base=sub_lambda,
                lambda_work=0.0,
                initial_vector=[0.20, 0.40, 0.60, 0.80, 0.50],
            )
            snap = integrator.step(C=0.0, a_motor=0.0)
            np.testing.assert_allclose(snap.vector, [0.20, 0.40, 0.60, 0.80, 0.50], rtol=1e-14)
            assert np.all(np.isfinite(snap.vector))

    @pytest.mark.parametrize(
        "crit_lambda",
        [
            1e-12,
            1e-12 + 1e-16,
            1e-11,
            1e-10,
        ],
    )
    def test_critical_threshold_boundary_continuity(self, crit_lambda: float):
        """Rates at and just above 1e-12 must compute valid exponential steps without overflow."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            integrator = MaslowIntegrator(
                lambda_base=crit_lambda,
                lambda_work=0.0,
                initial_vector=[0.20, 0.40, 0.60, 0.80, 0.50],
            )
            snap = integrator.step(C=0.0, a_motor=0.0)
            assert np.all(np.isfinite(snap.vector))
            assert not np.any(np.isnan(snap.vector))
            # Values should be within [0, 1]
            assert np.all(snap.vector >= 0.0)
            assert np.all(snap.vector <= 1.0)

    def test_heterogeneous_per_channel_boundary_rates(self):
        """Channels with rates spanning 0.0, 1e-15, 1e-12, 1e-9, 1.0 simultaneously."""
        integrator = MaslowIntegrator(
            lambda_base=0.0,
            lambda_work=0.0,
            eta_gain=1.0,
            initial_vector=[0.50, 0.50, 0.50, 0.50, 0.50],
        )
        # Construct heterogeneous environmental quenching rates
        # E_env scales quenching rate: R_i = eta_gain * a_motor * E_env_i
        e_env = np.array([0.0, 1e-15, 1e-12, 1e-6, 1.0], dtype=np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            snap = integrator.step(C=0.0, a_motor=1.0, E_env=e_env)

        v = snap.vector
        assert np.all(np.isfinite(v))
        # Channel 0 (rate 0.0): unchanged
        assert v[0] == pytest.approx(0.50, abs=1e-14)
        # Channel 1 (rate 1e-15 < 1e-12): unchanged
        assert v[1] == pytest.approx(0.50, abs=1e-14)
        # Channel 2 (rate 1e-12): active, v_infty = 0 / 1e-12 = 0, exp(-1e-12 * 0.1) ~ 1.0
        assert v[2] <= 0.50
        # Channel 3 (rate 1e-6): decayed slightly
        assert v[3] < 0.50
        # Channel 4 (rate 1.0): actively quenched
        assert v[4] < 0.46


# ==============================================================================
# 2. STABILITY UNDER EXTREME INPUTS (delta=1.0, V_l1=5.0, a_motor=1.0, thousands of steps)
# ==============================================================================

class TestAdversarialExtremeInputsStability:
    """Stress tests on ThetaEngine and MaslowIntegrator under extreme drive/damping combinations."""

    def test_sustained_maximal_drive_50000_steps(self):
        """Run ThetaEngine for 50,000 steps (1000 s) at maximum possible forward excitation."""
        engine = ThetaEngine()
        # V_l1 = 5.0 (all 5 needs saturated at 1.0), delta = 1.0 (max dissonance), a_motor = 0.0
        # Expected dot_theta in [kv*5.0 + alpha*1.0 - K, kv*5.0 + alpha*1.0 + K]
        # = [2.50 + 0.50 - 0.80, 2.50 + 0.50 + 0.80] = [2.20, 3.80] rad/s

        for step_idx in range(50000):
            th_rad, th_deg, dot_th = engine.step(V_l1=5.0, delta=1.0, a_motor=0.0)
            assert math.isfinite(th_rad)
            assert math.isfinite(th_deg)
            assert math.isfinite(dot_th)
            assert 2.19 <= dot_th <= 3.81

        assert engine.time_s == pytest.approx(1000.0, rel=1e-5)
        # Total phase advancement should be ~3,000 rad
        assert engine.theta_rad > 2000.0

    def test_sustained_maximal_damping_attractor_convergence(self):
        """Run ThetaEngine with V_l1=0, delta=0, a_motor=1.0: verify convergence to balance angle."""
        engine = ThetaEngine()
        engine.reset(theta_deg=30.0)

        # Equilibrium condition: dot_theta = 0
        # -K * sin(theta_eq - theta_0) - gamma * 1.0 = 0
        # sin(theta_eq - theta_0) = -gamma / K = -0.40 / 0.80 = -0.50
        # theta_eq - theta_0 = -pi/6 (-30 deg)
        # theta_eq = 15.0 deg - 30.0 deg = -15.0 deg (-0.261799 rad)
        expected_theta_eq_deg = 15.0 - 30.0  # -15.0 degrees

        for _ in range(1000):
            th_rad, th_deg, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=1.0)
            assert math.isfinite(th_rad)
            assert math.isfinite(dot_th)

        assert engine.theta_deg == pytest.approx(expected_theta_eq_deg, abs=1e-3)
        assert engine.dot_theta == pytest.approx(0.0, abs=1e-4)

    def test_extreme_all_ones_sustained_30000_steps(self):
        """Run ThetaEngine with delta=1.0, V_l1=5.0, a_motor=1.0 simultaneously for 30,000 steps."""
        engine = ThetaEngine()
        # dot_theta = 0.50*5.0 - 0.80*sin(...) + 0.50*1.0 - 0.40*1.0
        # = 2.50 + 0.50 - 0.40 - 0.80*sin(...) = 2.60 - 0.80*sin(...) in [1.80, 3.40] rad/s
        for _ in range(30000):
            th_rad, th_deg, dot_th = engine.step(V_l1=5.0, delta=1.0, a_motor=1.0)
            assert math.isfinite(th_rad)
            assert math.isfinite(dot_th)
            assert 1.79 <= dot_th <= 3.41

    def test_high_frequency_bang_bang_switching_stability(self):
        """Alternate every 10 steps between full positive drive and full braking for 10,000 steps."""
        engine = ThetaEngine()
        engine.reset()

        for step in range(10000):
            if (step // 10) % 2 == 0:
                # Full drive
                th_rad, th_deg, dot_th = engine.step(V_l1=5.0, delta=1.0, a_motor=0.0)
            else:
                # Full braking
                th_rad, th_deg, dot_th = engine.step(V_l1=0.0, delta=0.0, a_motor=1.0)

            assert math.isfinite(th_rad)
            assert math.isfinite(th_deg)
            assert math.isfinite(dot_th)

    def test_multi_rate_coupled_simulation_long_run(self):
        """Coupled 50 Hz ThetaEngine + 10 Hz MaslowIntegrator for 20,000 steps (400 s)."""
        maslow = MaslowIntegrator()
        theta = ThetaEngine()

        for step_50hz in range(20000):
            if step_50hz % 5 == 0:
                # Every 100 ms: advance Maslow
                # Oscillating cognitive load C(t)
                c_val = 0.5 * (1.0 + math.sin(step_50hz * 0.001))
                a_gate = 1.0 if theta.theta_deg >= 18.0 else 0.0
                snap = maslow.step(C=c_val, a_motor=a_gate)
                theta.latch_maslow(snap)

            d_sens = 0.2 * abs(math.cos(step_50hz * 0.002))
            a_cur = 1.0 if theta.theta_deg >= 18.0 else 0.0
            th_rad, th_deg, dot_th = theta.step(delta=d_sens, a_motor=a_cur)

            assert math.isfinite(th_rad)
            assert math.isfinite(dot_th)
            assert 0.0 <= maslow.vector.min()
            assert maslow.vector.max() <= 1.0


# ==============================================================================
# 3. MULTI-TURN PHASE CONTINUITY AND WRAPPING ACROSS [0, 2*pi)
# ==============================================================================

class TestAdversarialPhaseContinuityAndWrapping:
    """Rigorous verification of phase angle continuity, periodicity, and wrapping behavior."""

    def test_kuramoto_invariance_across_multi_turn_k_2pi(self):
        """Verify Kuramoto restoring torque is strictly invariant under theta + k * 2*pi."""
        engine = ThetaEngine()
        base_thetas = [0.0, 0.1, 0.261799, 0.314159, 1.0, 2.5, 3.14159, 5.0, 6.28]
        k_values = [-1000, -50, -2, -1, 0, 1, 2, 50, 1000]

        for base_th in base_thetas:
            engine.reset()
            engine._theta = base_th
            _, _, dot_ref = engine.step(V_l1=1.0, delta=0.2, a_motor=0.1)

            for k in k_values:
                engine.reset()
                engine._theta = base_th + k * 2.0 * math.pi
                _, _, dot_k = engine.step(V_l1=1.0, delta=0.2, a_motor=0.1)
                # Must match reference velocity with high numerical precision
                assert math.isclose(dot_ref, dot_k, rel_tol=1e-10, abs_tol=1e-12)

    def test_phase_continuous_smoothness_across_2pi_transition(self):
        """Ensure no discontinuous jumps or numerical spikes when phase crosses 2*pi."""
        engine = ThetaEngine()
        # Position right before 2*pi boundary
        engine.reset()
        engine._theta = 2.0 * math.pi - 0.05

        thetas = []
        dots = []
        for _ in range(100):
            th_rad, _, dot_th = engine.step(V_l1=1.0, delta=0.0, a_motor=0.0)
            thetas.append(th_rad)
            dots.append(dot_th)

        # Check phase difference smoothness: all positive, bounded increments
        diffs = np.diff(thetas)
        assert np.all(diffs > 0.0)
        assert np.all(diffs < 0.10)
        # Check acceleration (derivative of dot_theta): strictly continuous
        dot_diffs = np.diff(dots)
        assert np.max(np.abs(dot_diffs)) < 0.05

    def test_modulo_wrapped_phase_equivalence(self):
        """Verify that wrapping phase modulo 2*pi externally produces identical future trajectory."""
        engine_unwrapped = ThetaEngine()
        engine_wrapped = ThetaEngine()

        # Step both for 500 steps
        engine_unwrapped.reset(theta_deg=15.0)
        engine_wrapped.reset(theta_deg=15.0)

        for _ in range(500):
            th_u, _, dot_u = engine_unwrapped.step(V_l1=2.0, delta=0.5, a_motor=0.0)

            # Manually wrap engine_wrapped phase to [0, 2*pi)
            engine_wrapped._theta = engine_wrapped._theta % (2.0 * math.pi)
            th_w, _, dot_w = engine_wrapped.step(V_l1=2.0, delta=0.5, a_motor=0.0)

            # The angular velocities must be identical
            assert math.isclose(dot_u, dot_w, rel_tol=1e-9, abs_tol=1e-12)
            # The wrapped phases must be identical modulo 2*pi
            assert math.isclose(
                th_u % (2.0 * math.pi),
                th_w % (2.0 * math.pi),
                rel_tol=1e-7,
                abs_tol=1e-9,
            )

    def test_extreme_huge_angles_stability(self):
        """Verify stability when theta is set to +/- 1,000,000 radians."""
        engine = ThetaEngine()

        for huge_theta in [1e6, -1e6, 1e9, -1e9]:
            engine.reset()
            engine._theta = huge_theta
            th_rad, th_deg, dot_th = engine.step(V_l1=1.0, delta=0.5, a_motor=0.2)
            assert math.isfinite(th_rad)
            assert math.isfinite(th_deg)
            assert math.isfinite(dot_th)
            # Torque must remain bounded in [-K, K]
            # Restoring torque: -0.80 * sin(...)
            # Total velocity: 0.50*1.0 - 0.80*sin(...) + 0.50*0.50 - 0.40*0.20
            # = 0.50 + 0.25 - 0.08 - 0.80*sin(...) = 0.67 - 0.80*sin(...) in [-0.13, 1.47]
            assert -0.14 <= dot_th <= 1.48


# ==============================================================================
# 4. UNCONDITIONAL BOUNDS: V_i in [0.0, 1.0] WITHOUT ARTIFICIAL CLIPPING
# ==============================================================================

class TestAdversarialMaslowBoundsAndZeroClipping:
    """Stress tests on MaslowIntegrator bounds without artificial clipping."""

    def test_codebase_ast_strictly_forbids_clipping(self):
        """Inspect AST of core/maslow_integrator.py to guarantee zero np.clip or .clip()."""
        import core.maslow_integrator as mi_module
        source = inspect.getsource(mi_module)

        assert "np.clip" not in source
        assert ".clip(" not in source

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id not in ("clip", "clamp")
                elif isinstance(func, ast.Attribute):
                    assert func.attr not in ("clip", "clamp")

    def test_exhaustive_grid_search_bounds(self):
        """
        Evaluate thousands of parameter combinations across 6 orders of magnitude of dt,
        growth rates G, and quenching rates R.
        """
        dt_values = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        c_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        a_motor_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        e_env_values = [0.0, 0.1, 1.0, 10.0, 100.0]

        for dt in dt_values:
            integrator = MaslowIntegrator(dt=dt)
            for c in c_values:
                for a in a_motor_values:
                    for e in e_env_values:
                        snap = integrator.step(C=c, a_motor=a, E_env=e)
                        v = snap.vector
                        assert np.all(v >= 0.0), f"Underflow: {v} < 0.0 at dt={dt}, C={c}, a={a}, E={e}"
                        assert np.all(v <= 1.0), f"Overflow: {v} > 1.0 at dt={dt}, C={c}, a={a}, E={e}"
                        assert 0.0 <= snap.l1_norm <= 5.000000000001
                        assert 0.0 <= snap.l2_norm <= math.sqrt(5.0) + 1e-12

    def test_boundary_starting_states_zero_and_one(self):
        """Starting exactly at boundary 0.000000 and 1.000000."""
        # Start at all zeros
        int_zero = MaslowIntegrator(initial_vector=[0.0, 0.0, 0.0, 0.0, 0.0])
        snap0 = int_zero.get_snapshot()
        assert snap0.l1_norm == 0.0
        # Step with full quenching: must stay at 0.0 or slightly grow due to basal tension
        snap0_q = int_zero.step(C=0.0, a_motor=1.0, E_env=10.0)
        assert np.all(snap0_q.vector >= 0.0)
        assert np.all(snap0_q.vector <= 1.0)

        # Start at all ones
        int_one = MaslowIntegrator(initial_vector=[1.0, 1.0, 1.0, 1.0, 1.0])
        snap1 = int_one.get_snapshot()
        assert snap1.l1_norm == pytest.approx(5.0, rel=1e-12)
        # Step with maximum growth: must remain <= 1.0
        snap1_g = int_one.step(C=1.0, a_motor=0.0)
        assert np.all(snap1_g.vector <= 1.0)
        assert np.all(snap1_g.vector >= 0.0)

    def test_random_monte_carlo_fuzzing_50000_steps(self):
        """50,000 Monte Carlo steps with randomized continuous inputs."""
        rng = np.random.default_rng(1337)
        integrator = MaslowIntegrator()

        for _ in range(50000):
            c = float(rng.uniform(0.0, 1.0))
            a = float(rng.uniform(0.0, 1.0))
            e = rng.uniform(0.0, 5.0, size=5)
            snap = integrator.step(C=c, a_motor=a, E_env=e)

            v = snap.vector
            assert np.all(v >= 0.0)
            assert np.all(v <= 1.0)
            assert not np.any(np.isnan(v))
            assert not np.any(np.isinf(v))

    def test_monotonicity_guarantees(self):
        """Under pure growth (C > 0, a_motor = 0), V strictly increases; under pure quench, V decreases."""
        # Pure growth
        int_growth = MaslowIntegrator(initial_vector=[0.1, 0.2, 0.3, 0.4, 0.5])
        prev_v = int_growth.vector
        for _ in range(20):
            snap = int_growth.step(C=0.5, a_motor=0.0)
            assert np.all(snap.vector > prev_v)
            prev_v = snap.vector.copy()

        # Pure quenching (when starting high above V_infty)
        int_quench = MaslowIntegrator(initial_vector=[0.9, 0.9, 0.9, 0.9, 0.9])
        prev_vq = int_quench.vector
        for _ in range(20):
            snap_q = int_quench.step(C=0.0, a_motor=1.0, E_env=1.0)
            assert np.all(snap_q.vector < prev_vq)
            prev_vq = snap_q.vector.copy()


# ==============================================================================
# 5. ABSOLUTE ZERO NaN / Inf GUARANTEES
# ==============================================================================

class TestAdversarialZeroNaNInfGuarantees:
    """Comprehensive adversarial testing verifying absolute immunity to NaN/Inf corruptions."""

    def test_invalid_input_rejections_preserve_clean_state(self):
        """Invalid inputs must raise ValueError/TypeError immediately without corrupting state."""
        theta = ThetaEngine()
        init_th = theta.theta_rad

        # Pass NaN V_l1
        with pytest.raises(ValueError):
            theta.step(V_l1=float("nan"))
        assert theta.theta_rad == init_th
        assert math.isfinite(theta.theta_rad)

        # Pass Inf delta
        with pytest.raises(ValueError):
            theta.step(V_l1=1.0, delta=float("inf"))
        assert theta.theta_rad == init_th

        # Pass negative a_motor
        with pytest.raises(ValueError):
            theta.step(V_l1=1.0, a_motor=-0.5)
        assert theta.theta_rad == init_th

        # MaslowIntegrator invalid input rejections
        maslow = MaslowIntegrator()
        init_vec = maslow.vector.copy()

        with pytest.raises(ValueError):
            maslow.step(C=float("nan"))
        np.testing.assert_array_equal(maslow.vector, init_vec)

        with pytest.raises(ValueError):
            maslow.step(a_motor=float("inf"))
        np.testing.assert_array_equal(maslow.vector, init_vec)

        with pytest.raises(ValueError):
            maslow.step(E_env=[0.1, 0.2, float("nan"), 0.4, 0.5])
        np.testing.assert_array_equal(maslow.vector, init_vec)

    def test_massive_co_simulation_zero_nan_inf(self):
        """100,000 co-simulation steps with zero NaN, zero Inf, zero warnings."""
        theta = ThetaEngine()
        maslow = MaslowIntegrator()

        rng = np.random.default_rng(2026)

        for step in range(100000):
            if step % 5 == 0:
                c = float(rng.uniform(0.0, 1.0))
                a_m = float(rng.uniform(0.0, 1.0))
                e_e = rng.uniform(0.0, 2.0, size=5)
                snap = maslow.step(C=c, a_motor=a_m, E_env=e_e)
                assert not np.any(np.isnan(snap.vector))
                assert not np.any(np.isinf(snap.vector))
                assert math.isfinite(snap.l1_norm)
                assert math.isfinite(snap.l2_norm)
                theta.latch_maslow(snap)

            delta = float(rng.uniform(0.0, 1.0))
            a_motor = float(rng.uniform(0.0, 1.0))
            th_rad, th_deg, dot_th = theta.step(delta=delta, a_motor=a_motor)

            assert math.isfinite(th_rad)
            assert math.isfinite(th_deg)
            assert math.isfinite(dot_th)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
