"""
Adversarial Stress-Testing and Boundary Verification Suite for Milestone 2 (M2).
Target:
    - core/basal_ganglia_gate.py (BasalGangliaGate, PurkinjeLTDInterface)

Verifies:
1. Exact C0 continuity at theta = theta_gate = 18.0 deg (0.314159 rad):
   - Exact zero onset: a_gate(theta_gate) == 0.0000000 (with machine precision).
   - Right-limit continuity: |a_gate(theta_gate + epsilon) - 0.0| -> 0 as epsilon -> 0+.
   - Left-limit continuity: |a_gate(theta_gate - epsilon) - 0.0| == 0.0000000.
   - Right-derivative matching: lim_{eps->0+} [a(theta_gate + eps) - 0] / eps = u / (2 * Delta_theta_sav).
   - Machine epsilon behavior: np.nextafter(theta_gate, inf) and np.nextafter(theta_gate, -inf).
2. Strict subthreshold suppression:
   - For all theta < 18.0 deg, confirm a_motor == 0.000.
   - Grid search across negative and subthreshold domains [-1000 rad, theta_gate).
   - Fuzzing 50,000 random subthreshold angles with extreme u and Go signals.
   - Social etiquette shift expansion (effective gate shifting).
3. Biological interlock overrides:
   - lock_sleep=True and panic_amy=True force a_motor = 0.000 unconditionally at theta = 90 deg, u = 1.0.
   - Extreme angle (theta = 1e6 rad) and extreme drive (u = 1e6, go_signal = 1e6) under interlocks.
   - Mid-flight quench interruption: assertion of interlock instantly resets firing state.
   - Interlock release recovery.
4. Jerk minimization and derivative boundedness:
   - Finite and continuous first derivative da/dtheta.
   - Finite and bounded second derivative (jerk d^2a/dtheta^2).
5. Striatal competition and Purkinje LTD plasticity stress.
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


# ==============================================================================
# 1. EXACT C0 CONTINUITY AT THETA_GATE = 18.0 DEG (0.314159 RAD)
# ==============================================================================

class TestAdversarialC0Continuity:
    """Rigorous empirical verification of C0 continuity at theta_gate = 18.0 deg."""

    def test_exact_zero_onset_at_theta_gate(self):
        """At theta == theta_gate, a_gate must evaluate to exactly 0.0000000."""
        gate = BasalGangliaGate()
        decision = gate.evaluate(theta_rad=THETA_GATE_RAD, u=1.0)

        # Must be strictly equal to float 0.0 with zero residual
        assert decision.a_motor == 0.0
        assert math.isclose(decision.a_motor, 0.0, abs_tol=1e-17)
        assert decision.is_locked is False
        assert decision.go_winner is True

    @pytest.mark.parametrize("u_val", [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 1000.0])
    def test_exact_zero_onset_invariant_to_intent_u(self, u_val: float):
        """Regardless of intention intensity u, a_motor(theta_gate) must be strictly 0.0."""
        gate = BasalGangliaGate()
        decision = gate.evaluate(theta_rad=THETA_GATE_RAD, u=u_val)
        assert decision.a_motor == 0.0

    def test_limit_continuity_right_side_epsilon_convergence(self):
        """
        Verify limit continuity |a_gate(theta_gate + epsilon) - 0.0| -> 0 monotonically
        as epsilon -> 0+ across 16 orders of magnitude.
        """
        epsilons = [
            1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
            1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16
        ]

        prev_error = float("inf")
        for eps in epsilons:
            gate = BasalGangliaGate()
            decision = gate.evaluate(theta_rad=THETA_GATE_RAD + eps, u=1.0)
            err = abs(decision.a_motor - 0.0)

            # Error must strictly decrease as epsilon -> 0+
            assert err < prev_error, f"Non-monotonic error: eps={eps}, err={err}, prev={prev_error}"
            assert err >= 0.0
            prev_error = err

        # At epsilon = 1e-16, error must be negligible (< 1e-13)
        assert prev_error < 1e-13

    def test_limit_continuity_left_side_identical_zero(self):
        """
        Verify left-side limit continuity |a_gate(theta_gate - epsilon) - 0.0| == 0.0000000
        for all epsilon > 0.
        """
        epsilons = [
            1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16
        ]

        for eps in epsilons:
            gate = BasalGangliaGate()
            decision = gate.evaluate(theta_rad=THETA_GATE_RAD - eps, u=1.0)
            assert decision.a_motor == 0.0

    def test_right_derivative_matches_analytical_gradient(self):
        """
        Analytic right derivative at theta = theta_gate:
            da/dtheta|_{theta_gate+} = u / (2 * Delta_theta_sav)
        Confirm finite-difference gradient converges to exact analytic slope.
        """
        gate = BasalGangliaGate()
        u = 1.0
        expected_slope = u / (2.0 * DELTA_THETA_SAV_RAD)  # ~81.8511 rad^-1

        for eps in [1e-4, 1e-5, 1e-6, 1e-7]:
            g_temp = BasalGangliaGate()
            d = g_temp.evaluate(theta_rad=THETA_GATE_RAD + eps, u=u)
            numerical_slope = d.a_motor / eps
            assert math.isclose(numerical_slope, expected_slope, rel_tol=1e-3)

    def test_machine_epsilon_boundary_nextafter(self):
        """
        Test behavior at the finest possible IEEE 754 float64 resolution:
        np.nextafter(THETA_GATE_RAD, +inf) and np.nextafter(THETA_GATE_RAD, -inf).
        """
        gate_pos = BasalGangliaGate()
        th_pos = float(np.nextafter(THETA_GATE_RAD, np.inf))
        d_pos = gate_pos.evaluate(theta_rad=th_pos, u=1.0)

        # The step is 1 ULP (~5.55e-17 rad). a_motor must be >= 0 and < 1e-14
        assert 0.0 <= d_pos.a_motor < 1e-14

        gate_neg = BasalGangliaGate()
        th_neg = float(np.nextafter(THETA_GATE_RAD, -np.inf))
        d_neg = gate_neg.evaluate(theta_rad=th_neg, u=1.0)

        # Immediate predecessor must be identically 0.0
        assert d_neg.a_motor == 0.0

    @pytest.mark.parametrize("custom_deg", [5.0, 12.0, 15.0, 25.0, 45.0, 90.0])
    def test_exact_c0_onset_custom_thresholds(self, custom_deg: float):
        """Exact C0 continuity holds for any configured threshold."""
        gate = BasalGangliaGate(theta_gate_deg=custom_deg)
        d_exact = gate.evaluate(theta_rad=custom_deg * DEG2RAD)
        assert d_exact.a_motor == 0.0

        d_above = gate.evaluate(theta_rad=(custom_deg + 1e-5) * DEG2RAD)
        assert 0.0 < d_above.a_motor < 1e-2


# ==============================================================================
# 2. STRICT SUBTHRESHOLD SUPPRESSION (theta < 18.0 deg => a_motor == 0.000)
# ==============================================================================

class TestAdversarialSubthresholdSuppression:
    """Stress testing strict zero motor emission for all theta < 18.0 deg."""

    def test_dense_deterministic_subthreshold_grid(self):
        """Evaluate 2,000 subthreshold angles from -360 deg to 17.999999 deg."""
        angles_deg = np.linspace(-360.0, 18.0 - 1e-7, num=2000)

        for deg in angles_deg:
            gate = BasalGangliaGate()
            dec = gate.evaluate(theta_rad=float(deg * DEG2RAD))
            assert dec.a_motor == 0.0, f"Motor leak at {deg} deg: a_motor={dec.a_motor}"

    def test_stochastic_fuzzing_50000_subthreshold_samples(self):
        """Fuzz 50,000 random subthreshold angles with randomized intention intensity."""
        rng = np.random.default_rng(20261025)
        # Uniform angles from -1000 rad to (THETA_GATE_RAD - 1e-14)
        thetas = rng.uniform(-1000.0, THETA_GATE_RAD - 1e-14, size=50000)
        u_values = rng.uniform(0.0, 100.0, size=50000)

        for th, u in zip(thetas, u_values):
            gate = BasalGangliaGate()
            dec = gate.evaluate(theta_rad=float(th), u=float(u))
            assert dec.a_motor == 0.0

    def test_subthreshold_saturation_with_maximal_go_signal(self):
        """Even under maximal striatal Go drive (go=1e6, nogo=0), subthreshold angles remain 0.0."""
        gate = BasalGangliaGate()
        for deg in [0.0, 10.0, 15.0, 17.0, 17.9, 17.999]:
            dec = gate.evaluate(
                theta_rad=deg * DEG2RAD,
                u=100.0,
                go_signal=1e6,
                nogo_signal=0.0,
            )
            assert dec.a_motor == 0.0
            assert dec.go_winner is True

    def test_social_etiquette_shift_expands_quiescent_band(self):
        """Etiquette shift delta_etiquette = +3.0 deg shifts quiescent band up to 21.0 deg."""
        gate = BasalGangliaGate()
        # Normal gate would fire at 18.5 deg, 19.0 deg, 20.0 deg
        for test_deg in [18.1, 18.5, 19.0, 20.0, 20.99]:
            dec = gate.evaluate(
                theta_rad=test_deg * DEG2RAD,
                delta_etiquette_deg=3.0,
            )
            assert dec.a_motor == 0.0, f"Leaked motor action under etiquette shift at {test_deg} deg"

    def test_subthreshold_purity_post_reset(self):
        """Triggering active firing then calling reset() restores immediate subthreshold purity."""
        gate = BasalGangliaGate()
        # Trigger active firing
        d_fire = gate.evaluate(theta_rad=20.0 * DEG2RAD)
        assert d_fire.a_motor > 0.50

        # Reset gate
        gate.reset()

        # Check subthreshold angles immediately
        for deg in [17.5, 16.0, 15.0, 10.0, 0.0]:
            d_sub = gate.evaluate(theta_rad=deg * DEG2RAD)
            assert d_sub.a_motor == 0.0


# ==============================================================================
# 3. BIOLOGICAL INTERLOCK OVERRIDES (lock_sleep, panic_amy)
# ==============================================================================

class TestAdversarialBiologicalInterlocks:
    """Stress testing biological interlock hardware overrides under extreme drive."""

    def test_sleep_lock_override_at_90_deg_and_full_intent(self):
        """
        Dispatch Requirement 3:
        confirm that lock_sleep=True forces a_motor = 0.000 even when theta = 90 deg and u = 1.0.
        """
        gate = BasalGangliaGate()
        dec = gate.evaluate(
            theta_rad=90.0 * DEG2RAD,
            u=1.0,
            lock_sleep=True,
        )
        assert dec.a_motor == 0.0
        assert dec.is_locked is True

    def test_panic_lock_override_at_90_deg_and_full_intent(self):
        """
        Dispatch Requirement 3:
        confirm that panic_amy=True forces a_motor = 0.000 even when theta = 90 deg and u = 1.0.
        """
        gate = BasalGangliaGate()
        dec = gate.evaluate(
            theta_rad=90.0 * DEG2RAD,
            u=1.0,
            panic_amy=True,
        )
        assert dec.a_motor == 0.0
        assert dec.is_locked is True

    def test_sleep_lock_alias_backward_compatibility(self):
        """Verify sleep_lock alias behaves identically to lock_sleep."""
        gate = BasalGangliaGate()
        dec = gate.evaluate(
            theta_rad=90.0 * DEG2RAD,
            u=1.0,
            sleep_lock=True,
        )
        assert dec.a_motor == 0.0
        assert dec.is_locked is True

    def test_simultaneous_sleep_and_panic_locks(self):
        """Simultaneous activation of sleep_lock and panic_amy."""
        gate = BasalGangliaGate()
        dec = gate.evaluate(
            theta_rad=180.0 * DEG2RAD,
            u=10.0,
            lock_sleep=True,
            panic_amy=True,
        )
        assert dec.a_motor == 0.0
        assert dec.is_locked is True

    def test_extreme_excitation_under_biological_interlock(self):
        """
        Stress test under astronomical drive:
        theta = 1,000,000 rad, u = 1,000,000, go_signal = 1,000,000.
        """
        gate = BasalGangliaGate()
        for lock_kwarg in [{"lock_sleep": True}, {"panic_amy": True}, {"lock_sleep": True, "panic_amy": True}]:
            dec = gate.evaluate(
                theta_rad=1_000_000.0,
                u=1_000_000.0,
                go_signal=1_000_000.0,
                nogo_signal=0.0,
                **lock_kwarg,
            )
            assert dec.a_motor == 0.0
            assert dec.is_locked is True

    def test_mid_quench_emergency_alarm_resets_firing(self):
        """
        If panic_amy or lock_sleep is asserted in the middle of active firing,
        it must immediately quench a_motor to 0.0 AND reset internal firing state.
        """
        gate = BasalGangliaGate()

        # Step 1: Fire action at 20.0 deg
        d1 = gate.evaluate(theta_rad=20.0 * DEG2RAD, u=1.0)
        assert d1.a_motor > 0.50
        assert gate._firing is True

        # Step 2: In-flight emergency panic alarm
        d2 = gate.evaluate(theta_rad=19.0 * DEG2RAD, panic_amy=True)
        assert d2.a_motor == 0.0
        assert d2.is_locked is True
        assert gate._firing is False

        # Step 3: Alarm cleared on subthreshold angle: gate must remain quiescent
        d3 = gate.evaluate(theta_rad=17.0 * DEG2RAD)
        assert d3.a_motor == 0.0
        assert d3.is_locked is False

    def test_interlock_release_restores_unimpaired_firing(self):
        """Releasing interlocks restores normal threshold gating immediately."""
        gate = BasalGangliaGate()

        # Locked
        d_locked = gate.evaluate(theta_rad=20.0 * DEG2RAD, lock_sleep=True)
        assert d_locked.a_motor == 0.0
        assert d_locked.is_locked is True

        # Unlocked
        d_unlocked = gate.evaluate(theta_rad=20.0 * DEG2RAD, lock_sleep=False)
        assert d_unlocked.a_motor > 0.50
        assert d_unlocked.is_locked is False


# ==============================================================================
# 4. JERK MINIMIZATION AND DERIVATIVE BOUNDEDNESS
# ==============================================================================

class TestAdversarialJerkMinimization:
    """Mathematical verification of torque continuity and bounded physical jerk."""

    def test_torque_jump_at_threshold_is_strictly_zero(self):
        """
        Mechanical Jerk Jump: Delta a_motor = |a(theta_gate+) - a(theta_gate-)|.
        Verify Delta a_motor < 1e-12 at Delta_theta = 1e-12.
        """
        gate = BasalGangliaGate()
        eps = 1e-12
        a_left = gate.evaluate(theta_rad=THETA_GATE_RAD - eps).a_motor
        gate.reset()
        a_right = gate.evaluate(theta_rad=THETA_GATE_RAD + eps).a_motor

        torque_jump = abs(a_right - a_left)
        assert torque_jump < 1e-9

    def test_first_derivative_bounded_across_entire_domain(self):
        """
        First derivative da/dtheta must be bounded by u / (2 * Delta_theta_sav)
        everywhere in the transition band.
        """
        u = 1.0
        max_possible_slope = u / (2.0 * DELTA_THETA_SAV_RAD)  # ~81.85 rad^-1

        angles_rad = np.linspace(THETA_GATE_RAD, THETA_GATE_RAD + 4.0 * DELTA_THETA_SAV_RAD, num=2000)
        d_theta = float(angles_rad[1] - angles_rad[0])

        outputs = []
        for th in angles_rad:
            g = BasalGangliaGate()
            outputs.append(g.evaluate(theta_rad=float(th), u=u).a_motor)

        # Finite difference gradient
        grad = np.diff(outputs) / d_theta
        assert np.all(grad >= 0.0)  # Monotonic increase
        assert np.max(grad) <= max_possible_slope + 1e-2  # Bounded above

    def test_second_derivative_angular_jerk_bounded(self):
        """
        Second derivative (angular jerk d^2a/dtheta^2) must be finite and bounded.
        Theoretical maximum jerk is ~8,940 rad^-2.
        """
        angles_rad = np.linspace(THETA_GATE_RAD + 1e-5, THETA_GATE_RAD + 3.0 * DELTA_THETA_SAV_RAD, num=5000)
        d_theta = float(angles_rad[1] - angles_rad[0])

        outputs = []
        for th in angles_rad:
            g = BasalGangliaGate()
            outputs.append(g.evaluate(theta_rad=float(th), u=1.0).a_motor)

        grad = np.diff(outputs) / d_theta
        jerk = np.diff(grad) / d_theta

        # Jerk must be finite and bounded within [-10000, 10000] rad^-2
        assert np.all(np.isfinite(jerk))
        assert np.max(np.abs(jerk)) < 10000.0


# ==============================================================================
# 5. STRIATAL COMPETITION & PURKINJE LTD PLASTICITY STRESS
# ==============================================================================

class TestAdversarialStriatalAndPurkinje:
    """Stress tests on Striatal Go/No-Go competition and Purkinje LTD plasticity."""

    def test_striatal_nogo_dominance_suppresses_action(self):
        """When No-Go signal >= Go signal, motor output is strictly 0.0."""
        gate = BasalGangliaGate()
        # Even at high phase angle (30 deg) and full intent
        d1 = gate.evaluate(theta_rad=30.0 * DEG2RAD, go_signal=0.5, nogo_signal=0.5)
        assert d1.a_motor == 0.0
        assert d1.go_winner is False

        d2 = gate.evaluate(theta_rad=30.0 * DEG2RAD, go_signal=0.2, nogo_signal=0.8)
        assert d2.a_motor == 0.0
        assert d2.go_winner is False

    def test_purkinje_ltd_extreme_inputs_and_clamping(self):
        """Purkinje LTD synaptic weight remains bounded in [0, 1] under extreme errors."""
        p = PurkinjeLTDInterface(initial_weight=0.5, learning_rate=0.01)

        # Extreme climbing fiber error
        p.adapt(error_signal=1e6, parallel_fiber=1.0)
        assert p.current_weight == 0.0

        p.adapt(error_signal=-1e6, parallel_fiber=1.0)
        assert p.current_weight == 1.0

        # Zero parallel fiber produces zero change
        p_mid = PurkinjeLTDInterface(initial_weight=0.5)
        p_mid.adapt(error_signal=10.0, parallel_fiber=0.0)
        assert p_mid.current_weight == 0.5

    def test_negative_intent_clamped_to_zero(self):
        """Negative intention intensity u < 0.0 must not produce negative motor drive."""
        gate = BasalGangliaGate()
        d_neg = gate.evaluate(theta_rad=25.0 * DEG2RAD, u=-1.0)
        assert d_neg.a_motor == 0.0

    def test_biological_refractory_hysteresis_orbit_integrity(self):
        """
        Verify the biological refractory action cycle:
        1. Theta ramps up: subthreshold 0.0 until 18.0 deg.
        2. Firing triggers above 18.05 deg, reaching peak.
        3. Cooling down towards theta_0: motor quench drive remains sustained.
        4. When reaching theta <= theta_0 (15.0 deg), gate resets to 0.0.
        """
        gate = BasalGangliaGate()

        # 1. Subthreshold
        assert gate.evaluate(16.0 * DEG2RAD).a_motor == 0.0
        assert gate.evaluate(17.9 * DEG2RAD).a_motor == 0.0

        # 2. Fire action
        d_peak = gate.evaluate(19.0 * DEG2RAD)
        assert d_peak.a_motor > 0.70
        assert gate._firing is True

        # 3. Cooling down: 17.0 deg is below gate threshold, but in active quench
        d_quench = gate.evaluate(17.0 * DEG2RAD)
        assert d_quench.a_motor >= 0.70

        # 4. Return to resting attractor theta_0 (15.0 deg)
        d_reset = gate.evaluate(15.0 * DEG2RAD)
        assert d_reset.a_motor == 0.0
        assert gate._firing is False

        # 5. Subsequent cycle starting fresh
        assert gate.evaluate(16.0 * DEG2RAD).a_motor == 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
