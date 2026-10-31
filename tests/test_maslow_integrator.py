"""
Unit tests for core.maslow_integrator (MaslowIntegrator & MaslowSnapshot).
Validates 10 Hz exact analytical exponential integrator, basal entropy accumulation,
quenching dynamics, snapshot immutability, mathematical zero-clipping guarantee,
and division-by-zero guards.
"""

import ast
from dataclasses import FrozenInstanceError
import inspect
import math
import numpy as np
import pytest

from core.maslow_integrator import (
    DEFAULT_INITIAL_VECTOR,
    NEED_NAMES,
    MaslowIntegrator,
    MaslowSnapshot,
)


class TestMaslowIntegrator:
    """Test suite for MaslowIntegrator and MaslowSnapshot."""

    def test_initialization_and_precharge_vector(self):
        """Verify Day 0 pre-charge baseline and canonical rate constants."""
        integrator = MaslowIntegrator()

        assert integrator.dt == pytest.approx(0.10, rel=1e-6)
        assert integrator.lambda_base == pytest.approx(0.1386, rel=1e-6)
        assert integrator.lambda_work == pytest.approx(0.20, rel=1e-6)
        assert integrator.eta_gain == pytest.approx(2.0, rel=1e-6)
        assert integrator.time_s == pytest.approx(0.0, rel=1e-6)

        # Pre-charge vector: [0.15, 0.10, 0.25, 0.10, 0.30]
        np.testing.assert_allclose(integrator.vector, DEFAULT_INITIAL_VECTOR)
        assert integrator.vector.shape == (5,)

        snap = integrator.get_snapshot()
        assert snap.l1_norm == pytest.approx(0.90, rel=1e-6)
        expected_l2 = math.sqrt(0.15**2 + 0.10**2 + 0.25**2 + 0.10**2 + 0.30**2)
        assert snap.l2_norm == pytest.approx(expected_l2, rel=1e-6)

        # Dominant need is index 4 ('cog' with 0.30)
        assert snap.dominant_idx == 4
        assert snap.dominant_name == "cog"
        assert snap.dominant_name == NEED_NAMES[4]

    def test_snapshot_immutability_and_aliases(self):
        """Verify thread-safe read-only memory buffer and immutable dataclass guarantees."""
        integrator = MaslowIntegrator()
        snap1 = integrator.get_snapshot()

        # Buffer must be read-only
        assert snap1.vector.flags.writeable is False
        assert snap1.values.flags.writeable is False

        # Attempting in-place mutation of array buffer must raise ValueError
        with pytest.raises(ValueError):
            snap1.vector[0] = 0.99
        with pytest.raises(ValueError):
            snap1.values[0] = 0.99

        # Attempting modification of dataclass attributes must raise FrozenInstanceError
        with pytest.raises((FrozenInstanceError, AttributeError)):
            snap1.l1_norm = 99.0  # type: ignore

        # Alias properties
        assert snap1.dominant_index == snap1.dominant_idx
        np.testing.assert_array_equal(snap1.values, snap1.vector)

        # Integrator step must not mutate previous snapshot
        vec_before = snap1.vector.copy()
        integrator.step(C=1.0)
        np.testing.assert_array_equal(snap1.vector, vec_before)

    def test_basal_tension_accumulation_at_zero_work(self):
        """
        Verify autonomous tension accumulation under zero cognitive load (C=0, a_motor=0).
        Demonstrates that half-life is exactly 5.0 s and tension reaches 0.80 from 0.20 in 10.0 s.
        """
        # Initialize all 5 needs to 0.20
        integrator = MaslowIntegrator(initial_vector=[0.20, 0.20, 0.20, 0.20, 0.20])

        # Step 50 times (5.0 seconds = half-life t_1/2 = ln(2)/lambda_base)
        for _ in range(50):
            integrator.step(C=0.0, a_motor=0.0)

        # Expected value at t = 5.0 s:
        # V(5) = 1 - (1 - 0.20) * exp(-0.1386 * 5) = 1 - 0.80 * 0.5000735 = 0.59994
        np.testing.assert_allclose(integrator.vector, np.full(5, 0.60), atol=1e-3)

        # Step another 50 times (10.0 seconds total)
        for _ in range(50):
            integrator.step(C=0.0, a_motor=0.0)

        # Expected value at t = 10.0 s:
        # V(10) = 1 - (1 - 0.20) * exp(-0.1386 * 10) = 1 - 0.80 * 0.250074 = 0.79994 ~ 0.80
        np.testing.assert_allclose(integrator.vector, np.full(5, 0.80), atol=1e-3)
        assert integrator.time_s == pytest.approx(10.0, rel=1e-6)

    def test_cognitive_work_load_acceleration(self):
        """Verify that cognitive load C(t) accelerates need accumulation via lambda_work."""
        int_idle = MaslowIntegrator(initial_vector=[0.10, 0.10, 0.10, 0.10, 0.10])
        int_work = MaslowIntegrator(initial_vector=[0.10, 0.10, 0.10, 0.10, 0.10])

        # Run 20 steps (2.0 seconds): idle with C=0.0 vs high load with C=1.0
        for _ in range(20):
            int_idle.step(C=0.0, a_motor=0.0)
            int_work.step(C=1.0, a_motor=0.0)

        # At C=1.0, growth rate G = 0.1386 + 0.20 = 0.3386 s^-1 (2.44x faster)
        # Expected work vector > idle vector
        assert np.all(int_work.vector > int_idle.vector)

        # Check exact analytical values at t = 2.0 s:
        # Idle: V(2) = 1 - (1 - 0.10) * exp(-0.1386 * 2) = 1 - 0.90 * 0.7579 = 0.31789
        expected_idle = 1.0 - 0.90 * math.exp(-0.1386 * 2.0)
        np.testing.assert_allclose(int_idle.vector, np.full(5, expected_idle), rtol=1e-5)

        # Work: V(2) = 1 - (1 - 0.10) * exp(-0.3386 * 2) = 1 - 0.90 * 0.5080 = 0.5428
        expected_work = 1.0 - 0.90 * math.exp(-0.3386 * 2.0)
        np.testing.assert_allclose(int_work.vector, np.full(5, expected_work), rtol=1e-5)

    def test_quenching_dynamics_rapid_collapse(self):
        """
        Verify that successful motor action (a_motor=1.0, E_env=1.0) quenches
        high deficiency (0.80) to < 0.30 within 500 ms (5 steps of 100 ms).
        """
        integrator = MaslowIntegrator(initial_vector=[0.80, 0.80, 0.80, 0.80, 0.80])

        # Step 5 times (500 ms) with a_motor = 1.0, E_env = 1.0
        # G = 0.1386, R = 2.0 * 1.0 * 1.0 = 2.0 -> lambda = 2.1386 s^-1
        # V_inf = 0.1386 / 2.1386 = 0.0648
        for _ in range(5):
            integrator.step(C=0.0, a_motor=1.0, E_env=1.0)

        # Expected value at t = 0.5 s:
        # V(0.5) = 0.80 * exp(-2.1386 * 0.5) + 0.0648 * (1 - exp(-2.1386 * 0.5))
        #        = 0.80 * 0.3432 + 0.0648 * 0.6568 = 0.2746 + 0.0426 = 0.317 ~ < 0.32
        assert np.all(integrator.vector < 0.32)
        assert np.all(integrator.vector > 0.0)

        # In 1.0 second (10 steps), decays to < 0.16
        for _ in range(5):
            integrator.step(C=0.0, a_motor=1.0, E_env=1.0)
        assert np.all(integrator.vector < 0.16)

    def test_unconditional_bounds_zero_clipping_guarantee(self):
        """
        Verify mathematical proof that V_i in [0.0, 1.0] unconditionally
        without artificial clipping (np.clip is strictly prohibited).
        """
        import core.maslow_integrator as mi_module

        # Inspect source code and AST of core/maslow_integrator.py to ensure zero np.clip
        source = inspect.getsource(mi_module)
        assert "np.clip" not in source, "Violation: np.clip found in core/maslow_integrator.py!"
        assert ".clip(" not in source, "Violation: .clip() found in core/maslow_integrator.py!"

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    assert func.id != "clip", "Violation: direct clip() call in AST"
                elif isinstance(func, ast.Attribute):
                    assert func.attr != "clip", f"Violation: .{func.attr}() call in AST"

        integrator = MaslowIntegrator()

        # Test extreme conditions: large time steps, huge rates
        for dt_extreme in [0.001, 0.1, 1.0, 10.0, 100.0]:
            test_int = MaslowIntegrator(dt=dt_extreme)
            # High growth
            snap_growth = test_int.step(C=1.0, a_motor=0.0)
            assert np.all(snap_growth.vector >= 0.0)
            assert np.all(snap_growth.vector <= 1.0)

            # Extreme quenching
            snap_quench = test_int.step(C=0.0, a_motor=1.0, E_env=10.0)
            assert np.all(snap_quench.vector >= 0.0)
            assert np.all(snap_quench.vector <= 1.0)

        # Stochastic stress test: 2,000 steps of random inputs
        rng = np.random.default_rng(42)
        for _ in range(2000):
            c_rand = float(rng.uniform(0.0, 1.0))
            a_rand = float(rng.uniform(0.0, 1.0))
            e_rand = rng.uniform(0.0, 1.0, size=5)
            snap = integrator.step(C=c_rand, a_motor=a_rand, E_env=e_rand)

            assert np.all(snap.vector >= 0.0)
            assert np.all(snap.vector <= 1.0)
            assert 0.0 <= snap.l1_norm <= 5.0
            assert 0.0 <= snap.l2_norm <= math.sqrt(5.0)

    def test_vanishing_rates_division_by_zero_guard(self):
        """Verify that when lambda_i < 1e-12, the integrator preserves V_i without crashing or NaN."""
        # Initialize with lambda_base = 0.0
        integrator = MaslowIntegrator(
            lambda_base=0.0, initial_vector=[0.42, 0.55, 0.10, 0.88, 0.33]
        )

        # Step with C=0.0 and a_motor=0.0 -> G=0, R=0 -> lambda = 0.0 < 1e-12
        snap = integrator.step(C=0.0, a_motor=0.0)

        # Must preserve previous values exactly
        np.testing.assert_allclose(snap.vector, [0.42, 0.55, 0.10, 0.88, 0.33])
        assert not np.any(np.isnan(snap.vector))
        assert not np.any(np.isinf(snap.vector))

    def test_per_dimension_differential_quenching(self):
        """Verify that per-need E_env array selectively quenches specific needs."""
        integrator = MaslowIntegrator(initial_vector=[0.70, 0.70, 0.70, 0.70, 0.70])

        # Only quench need 0 ('metab') by setting E_env = [1.0, 0.0, 0.0, 0.0, 0.0]
        e_env_selective = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        for _ in range(10):
            integrator.step(C=0.0, a_motor=1.0, E_env=e_env_selective)

        v = integrator.vector
        # Need 0 quenched significantly
        assert v[0] < 0.20
        # Needs 1..4 received zero quenching (R=0) and thus accumulated basal tension
        assert np.all(v[1:] > 0.70)

    def test_invalid_inputs_and_edge_cases(self):
        """Verify error handling on invalid configurations and step inputs."""
        # Invalid dt
        with pytest.raises(ValueError):
            MaslowIntegrator(dt=0.0)
        with pytest.raises(ValueError):
            MaslowIntegrator(dt=-0.1)

        # Invalid rates
        with pytest.raises(ValueError):
            MaslowIntegrator(lambda_base=-0.1)
        with pytest.raises(ValueError):
            MaslowIntegrator(lambda_work=-0.2)
        with pytest.raises(ValueError):
            MaslowIntegrator(eta_gain=-1.0)

        # Invalid initial vector
        with pytest.raises(ValueError):
            MaslowIntegrator(initial_vector=[0.1, 0.2])  # wrong length
        with pytest.raises(ValueError):
            MaslowIntegrator(initial_vector=[0.1, 0.2, 0.3, 0.4, 1.5])  # out of bounds

        integrator = MaslowIntegrator()

        # Invalid C
        with pytest.raises(ValueError):
            integrator.step(C=-0.1)
        with pytest.raises(ValueError):
            integrator.step(C=2.0)
        with pytest.raises(ValueError):
            integrator.step(C=float("nan"))

        # Invalid a_motor
        with pytest.raises(ValueError):
            integrator.step(a_motor=-0.1)
        with pytest.raises(ValueError):
            integrator.step(a_motor=2.0)

        # Invalid E_env
        with pytest.raises(ValueError):
            integrator.step(E_env=np.array([1.0, -0.5, 0.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            integrator.step(E_env=np.array([1.0, 0.0]))  # wrong shape

    def test_reset_functionality(self):
        """Verify reset restores baseline or custom pre-charge state."""
        integrator = MaslowIntegrator()
        integrator.step(C=1.0)
        assert integrator.time_s > 0.0

        # Reset to default
        snap_def = integrator.reset()
        np.testing.assert_allclose(snap_def.vector, DEFAULT_INITIAL_VECTOR)
        assert integrator.time_s == 0.0

        # Reset to custom
        custom_vec = [0.2, 0.3, 0.4, 0.5, 0.6]
        snap_custom = integrator.reset(custom_vec)
        np.testing.assert_allclose(snap_custom.vector, custom_vec)
        assert integrator.time_s == 0.0
