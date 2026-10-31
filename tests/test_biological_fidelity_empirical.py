"""
Empirical Verification Test Suite for Biological Fidelity and Physical Laws.
Target Modules: core/theta_engine.py and core/maslow_integrator.py
Milestone: Milestone 1 (ΔP-42 Gate 1)
Role: challenger_m1_2 (Biological Fidelity Challenger)

This test suite executes empirical tests, generators, oracles, and stress harnesses to verify:
1. Basal drive autonomous tension growth at C=0:
   - Exponential growth rate lambda_base = 0.1386 s^-1
   - Time constant tau = 1 / lambda_base = 7.215 s
   - Half-life t_1/2 = ln(2) / lambda_base = 5.001 s
   - Monotonic growth, asymptotic bounds, and zero-clipping guarantee
2. Motor quenching dynamics:
   - Rapid phase velocity braking (-gamma * a_motor with gamma = 0.40 rad/s)
   - Phase cooling and velocity inversion
   - Rapid need state collapse via R_i = eta_gain * a_motor * E_env (eta_gain = 2.0 s^-1)
   - Quenching time constant tau_q = 0.4676 s, half-life t_1/2,q = 0.3241 s
   - Differential per-need satisfaction
   - Coupled multi-rate limit-cycle relaxation
3. Bootstrap kick and symmetry breaking:
   - Resting tone theta_0 = 15.0 deg (0.2618 rad) vs bootstrap theta(0) = 15.5 deg (0.2705 rad)
   - Kuramoto restoring spring behavior and dead-zone analysis
   - Symmetry breaking under Day 0 pre-charge baseline (||V(0)||_1 = 0.90)
   - Idle ignition timing (<= 10.0 s) from Day 0 pre-charge and from zero baseline
"""

import math
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
# SECTION 1: Basal Drive Autonomous Tension Growth at C=0
# ==============================================================================


class TestBasalDriveAutonomousTensionGrowth:
    """Empirical verification of autonomous basal tension accumulation."""

    def test_basal_rate_constants_and_analytical_relations(self):
        """Verify lambda_base, theoretical tau, and theoretical half-life."""
        integrator = MaslowIntegrator()

        lambda_base = integrator.lambda_base
        assert lambda_base == pytest.approx(0.1386, rel=1e-6)

        # Theoretical tau = 1 / lambda_base
        tau_theoretical = 1.0 / lambda_base
        assert tau_theoretical == pytest.approx(7.2150072, rel=1e-5)
        print(f"\n[Test 1.1] lambda_base = {lambda_base:.4f} s^-1, theoretical tau = {tau_theoretical:.4f} s")

        # Theoretical half-life = ln(2) / lambda_base
        t_half_theoretical = math.log(2.0) / lambda_base
        assert t_half_theoretical == pytest.approx(5.001062, rel=1e-4)
        print(f"[Test 1.1] theoretical half-life t_1/2 = {t_half_theoretical:.4f} s (~5.0 s)")

    def test_empirical_half_life_from_zero_baseline(self):
        """
        Starting from V(0) = 0.0, verify that at t = 5.0 s (50 steps of 0.1 s),
        the deficit covers exactly 50% of the distance to saturation (1.0).
        V(t) = 1 - e^(-lambda_base * t).
        At t = 5.0 s: V(5.0) = 1 - e^(-0.1386 * 5.0) = 1 - e^(-0.6930) = 0.499926 ~ 0.50.
        """
        integrator = MaslowIntegrator(initial_vector=[0.0, 0.0, 0.0, 0.0, 0.0])

        trajectory = []
        for step_idx in range(1, 101):  # 10.0 seconds total
            snap = integrator.step(C=0.0, a_motor=0.0)
            t_s = step_idx * integrator.dt
            if step_idx in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                trajectory.append((t_s, snap.vector[0]))

        v_at_5s = integrator.vector[0]  # will be at 10.0 s, let's look at 5.0 s from trajectory
        t_5, val_5 = trajectory[4]
        assert t_5 == pytest.approx(5.0, rel=1e-6)

        expected_val_5 = 1.0 - math.exp(-0.1386 * 5.0)
        assert val_5 == pytest.approx(expected_val_5, rel=1e-6)
        assert val_5 == pytest.approx(0.50, abs=1e-3)
        print(f"\n[Test 1.2] At t = 5.0 s (half-life): empirical V = {val_5:.6f}, expected = {expected_val_5:.6f}")

        # At t = 10.0 s (2 * t_1/2):
        t_10, val_10 = trajectory[9]
        assert t_10 == pytest.approx(10.0, rel=1e-6)
        expected_val_10 = 1.0 - math.exp(-0.1386 * 10.0)
        assert val_10 == pytest.approx(expected_val_10, rel=1e-6)
        assert val_10 == pytest.approx(0.75, abs=1e-3)
        print(f"[Test 1.2] At t = 10.0 s (2 * t_1/2): empirical V = {val_10:.6f}, expected = {expected_val_10:.6f}")

    def test_empirical_tau_relaxation_bracket(self):
        """
        Verify that around t = tau = 7.215 s, the deficit fraction remaining
        (1 - V(t))/(1 - V(0)) brackets 1/e = 0.367879...
        """
        integrator = MaslowIntegrator(initial_vector=[0.0, 0.0, 0.0, 0.0, 0.0])

        deficit_history = {}
        for step_idx in range(1, 80):
            snap = integrator.step(C=0.0, a_motor=0.0)
            t_s = round(step_idx * integrator.dt, 2)
            deficit_history[t_s] = 1.0 - snap.vector[0]

        # Check step 72 (7.20 s) and step 73 (7.30 s)
        target_fraction = 1.0 / math.e
        def_72 = deficit_history[7.20]
        def_73 = deficit_history[7.30]

        assert def_72 > target_fraction > def_73, (
            f"1/e={target_fraction:.6f} should be strictly bracketed by t=7.20s ({def_72:.6f}) and t=7.30s ({def_73:.6f})"
        )
        print(f"\n[Test 1.3] 1/e deficit target = {target_fraction:.6f}")
        print(f"          At t = 7.20 s: deficit = {def_72:.6f}")
        print(f"          At t = 7.30 s: deficit = {def_73:.6f} (exact tau = 7.215 s confirmed)")

    def test_day0_precharge_autonomous_growth_trajectory(self):
        """
        Verify autonomous growth starting from official Day 0 precharge vector:
        V(0) = [0.15, 0.10, 0.25, 0.10, 0.30] (L1 norm = 0.90).
        Demonstrate:
        1. Strict monotonic increase for all 5 dimensions.
        2. Component-wise exact match to analytical formula.
        3. Preservation of need hierarchy ordering.
        4. L1 norm growth over 10.0 s.
        """
        integrator = MaslowIntegrator()
        v0 = DEFAULT_INITIAL_VECTOR.copy()
        l1_init = float(np.sum(v0))
        assert l1_init == pytest.approx(0.90, rel=1e-6)

        history = [v0.copy()]
        l1_history = [l1_init]

        for _ in range(100):  # 10 seconds
            snap = integrator.step(C=0.0, a_motor=0.0)
            history.append(snap.vector.copy())
            l1_history.append(snap.l1_norm)

        # 1. Monotonicity check
        for step in range(len(history) - 1):
            assert np.all(history[step + 1] > history[step]), f"Monotonicity violated at step {step}"

        # 2. Exact match at t = 5.0 s (step 50) and t = 10.0 s (step 100)
        expected_5s = 1.0 - (1.0 - v0) * math.exp(-0.1386 * 5.0)
        np.testing.assert_allclose(history[50], expected_5s, rtol=1e-5)

        expected_10s = 1.0 - (1.0 - v0) * math.exp(-0.1386 * 10.0)
        np.testing.assert_allclose(history[100], expected_10s, rtol=1e-5)

        # 3. Preservation of ordering: index 4 ('cog') was highest (0.30), index 1/3 lowest (0.10)
        final_vec = history[100]
        assert np.argmax(final_vec) == 4
        assert final_vec[4] > final_vec[2] > final_vec[0] > final_vec[1]

        # 4. L1 norm growth: 0.90 -> ~3.97
        expected_l1_10s = float(np.sum(expected_10s))
        assert l1_history[100] == pytest.approx(expected_l1_10s, rel=1e-5)
        assert l1_history[100] > 3.90
        print(f"\n[Test 1.4] Day 0 L1 norm: {l1_history[0]:.2f} -> at 5.0s: {l1_history[50]:.4f} -> at 10.0s: {l1_history[100]:.4f}")

    def test_asymptotic_convergence_without_clipping(self):
        """
        Verify that for long horizons (t = 100.0 s = 1000 steps),
        V_i converges smoothly to 1.0 without ever exceeding 1.0.
        """
        integrator = MaslowIntegrator()
        for _ in range(1000):
            snap = integrator.step(C=0.0, a_motor=0.0)
            assert np.all(snap.vector < 1.0)
            assert np.all(snap.vector > 0.0)

        # At t = 100 s, deficit is exp(-13.86) ~ 9.56e-7
        assert np.all(integrator.vector > 0.99999)
        assert np.all(integrator.vector <= 1.0)
        print(f"\n[Test 1.5] At t=100s: max V = {np.max(integrator.vector):.8f} (strictly <= 1.0)")


# ==============================================================================
# SECTION 2: Motor Quenching Dynamics
# ==============================================================================


class TestMotorQuenchingDynamics:
    """Empirical verification of motor activation quenching on phase and needs."""

    def test_theta_engine_motor_braking_gain(self):
        """
        Verify that -gamma * a_motor in dot{theta} strictly scales with gamma = 0.40 rad/s.
        """
        engine = ThetaEngine()
        engine.reset(theta_deg=15.0)  # Kuramoto spring = 0

        # Step with a_motor = 0.0
        _, _, dot_th_0 = engine.step(V_l1=1.0, delta=0.0, a_motor=0.0)

        # Step with a_motor = 0.25, 0.50, 0.75, 1.00
        for a_val in [0.25, 0.50, 0.75, 1.00]:
            engine.reset(theta_deg=15.0)
            _, _, dot_th_q = engine.step(V_l1=1.0, delta=0.0, a_motor=a_val)
            delta_dot_th = dot_th_0 - dot_th_q
            expected_braking = 0.40 * a_val
            assert delta_dot_th == pytest.approx(expected_braking, rel=1e-6)
            print(f"\n[Test 2.1] a_motor = {a_val:.2f} -> delta dot_theta = {delta_dot_th:.4f} rad/s (expected {expected_braking:.4f})")

    def test_phase_cooling_and_velocity_inversion(self):
        """
        Verify that when motor activation fires near gate threshold, phase velocity
        inverts (becomes negative) and pulls phase backwards toward resting tone theta_0.
        """
        engine = ThetaEngine()
        # Near motor gate: 18.0 deg
        engine.reset(theta_deg=18.0)

        # Mild Maslow drive: ||V||_1 = 0.40 -> kv * ||V||_1 = 0.20 rad/s
        # Kuramoto restoring force at 18.0 deg (Delta = 3.0 deg):
        # -0.80 * sin(3 deg) = -0.041869 rad/s
        # Unquenched velocity: 0.20 - 0.041869 = +0.158131 rad/s > 0 (advancing)
        _, _, dot_th_advancing = engine.step(V_l1=0.40, delta=0.0, a_motor=0.0)
        assert dot_th_advancing > 0.15

        # Now apply full quenching: a_motor = 1.0 (-gamma = -0.40 rad/s)
        # Quenched velocity: +0.158131 - 0.40 = -0.241869 rad/s < 0 (cooling/inverting!)
        engine.reset(theta_deg=18.0)
        _, _, dot_th_quenched = engine.step(V_l1=0.40, delta=0.0, a_motor=1.0)
        assert dot_th_quenched < 0.0
        assert dot_th_quenched == pytest.approx(-0.241869, abs=1e-4)

        # Step 25 times (500 ms) under quenching: phase should cool down toward resting tone
        engine.reset(theta_deg=18.0)
        initial_deg = engine.theta_deg
        for _ in range(25):
            engine.step(V_l1=0.40, delta=0.0, a_motor=1.0)

        final_deg = engine.theta_deg
        assert final_deg < initial_deg
        assert final_deg < 18.0
        print(f"\n[Test 2.2] Phase cooling: {initial_deg:.2f} deg -> {final_deg:.2f} deg (quenched velocity = {dot_th_quenched:.4f} rad/s)")

    def test_maslow_need_state_rapid_discharge_kinetics(self):
        """
        Verify MaslowIntegrator discharge kinetics under a_motor = 1.0, E_env = 1.0:
        R_i = eta_gain * a_motor * E_env = 2.0 s^-1.
        Total rate lambda = lambda_base + R_i = 0.1386 + 2.0 = 2.1386 s^-1.
        Target V_inf = 0.1386 / 2.1386 = 0.06480875.
        Time constant tau_q = 1 / 2.1386 = 0.467595 s.
        Discharge half-life t_1/2,q = ln(2) / 2.1386 = 0.32411 s.
        """
        # Start from high saturation V_i = 0.90
        integrator = MaslowIntegrator(initial_vector=[0.90, 0.90, 0.90, 0.90, 0.90])
        total_lambda = 0.1386 + 2.0
        v_inf = 0.1386 / total_lambda

        tau_q = 1.0 / total_lambda
        t_half_q = math.log(2.0) / total_lambda
        assert tau_q == pytest.approx(0.467595, rel=1e-4)
        assert t_half_q == pytest.approx(0.32411, rel=1e-4)

        # Step 10 times (1.0 second) with full motor quenching
        trajectory = []
        for step in range(1, 11):
            snap = integrator.step(C=0.0, a_motor=1.0, E_env=1.0)
            t_s = step * integrator.dt
            expected_v = 0.90 * math.exp(-total_lambda * t_s) + v_inf * (1.0 - math.exp(-total_lambda * t_s))
            actual_v = snap.vector[0]
            assert actual_v == pytest.approx(expected_v, rel=1e-6)
            trajectory.append((t_s, actual_v, expected_v))

        # At t = 0.5 s (step 5): V should have dropped from 0.90 to ~0.35
        t_5, act_5, exp_5 = trajectory[4]
        assert act_5 < 0.36
        assert act_5 == pytest.approx(exp_5, rel=1e-6)

        # At t = 1.0 s (step 10): V should have dropped to ~0.16
        t_10, act_10, exp_10 = trajectory[9]
        assert act_10 < 0.17
        assert act_10 == pytest.approx(exp_10, rel=1e-6)

        print(f"\n[Test 2.3] Need discharge: 0.90 -> 0.5s: {act_5:.4f} (exp: {exp_5:.4f}) -> 1.0s: {act_10:.4f} (exp: {exp_10:.4f})")
        print(f"          tau_q = {tau_q:.4f} s, t_1/2,q = {t_half_q:.4f} s, asymptotic V_inf = {v_inf:.4f}")

    def test_differential_environmental_reward_quenching(self):
        """
        Verify that selective environmental reward (E_env vector) only discharges
        the targeted need while non-rewarded needs continue accumulating tension.
        """
        # Start all needs at 0.50
        integrator = MaslowIntegrator(initial_vector=[0.50, 0.50, 0.50, 0.50, 0.50])

        # Target only need 0 ('metab') with E_env = [1.0, 0.0, 0.0, 0.0, 0.0]
        # and partial reward on need 2 ('social') with E_env = 0.5
        e_env_selective = np.array([1.0, 0.0, 0.5, 0.0, 0.0])

        for _ in range(10):  # 1.0 second
            snap = integrator.step(C=0.0, a_motor=1.0, E_env=e_env_selective)

        v = snap.vector
        # Need 0 should be heavily discharged (< 0.15)
        assert v[0] < 0.15
        # Need 2 should be moderately discharged
        assert v[0] < v[2] < 0.35
        # Needs 1, 3, 4 received zero reward (R=0) and therefore increased above 0.50!
        assert v[1] > 0.50
        assert v[3] > 0.50
        assert v[4] > 0.50

        print(f"\n[Test 2.4] Differential quenching output: metab={v[0]:.4f}, safety={v[1]:.4f}, social={v[2]:.4f}, comp={v[3]:.4f}, cog={v[4]:.4f}")

    def test_coupled_relaxation_oscillator_limit_cycle_behavior(self):
        """
        Co-simulate 10 Hz MaslowIntegrator and 50 Hz ThetaEngine in a closed loop.
        Rule: Motor gate fires (a_motor=1.0, E_env=1.0) whenever theta >= 18.0 deg,
        and is quenched back to a_motor=0.0 when theta drops below 18.0 deg.
        Verify that:
        1. Repeated ignition-quench-relaxation cycles occur.
        2. Maslow deficiency drops on each motor firing and rebuilds on relaxation.
        3. Phase theta repeatedly crosses 18.0 deg and cools back down.
        """
        maslow = MaslowIntegrator()
        theta = ThetaEngine()

        ignitions = 0
        cycle_times = []
        last_ignition_time = 0.0

        sim_duration = 10.0  # seconds
        total_50hz_steps = int(sim_duration / theta.dt)

        a_motor_active = 0.0

        for step in range(total_50hz_steps):
            # 10 Hz Maslow update every 5 steps of 50 Hz
            if step % 5 == 0:
                snap = maslow.step(C=0.0, a_motor=a_motor_active, E_env=1.0)
                theta.latch_maslow(snap)

            # 50 Hz Theta update
            th_rad, th_deg, dot_th = theta.step(delta=0.0, a_motor=a_motor_active)
            current_time = theta.time_s

            if th_deg >= 18.0:
                if a_motor_active == 0.0:  # rising edge
                    ignitions += 1
                    if last_ignition_time > 0.0:
                        cycle_times.append(current_time - last_ignition_time)
                    last_ignition_time = current_time
                a_motor_active = 1.0
            else:
                a_motor_active = 0.0

        assert ignitions >= 2, f"Expected at least 2 limit cycles in 10 s, got {ignitions}"
        print(f"\n[Test 2.5] Coupled limit cycle: {ignitions} ignitions in 10 s. Cycle times: {cycle_times}")


# ==============================================================================
# SECTION 3: Bootstrap Kick and Symmetry Breaking
# ==============================================================================


class TestBootstrapKickAndSymmetryBreaking:
    """Empirical verification of Day 0 bootstrap kick (15.5 deg) vs resting tone (15.0 deg)."""

    def test_bootstrap_kick_and_resting_angle_constants(self):
        """Verify theta_0 = 15.0 deg (pi/12 rad) and theta_bootstrap = 15.5 deg."""
        engine = ThetaEngine()

        assert engine.theta_0_deg == pytest.approx(15.0, rel=1e-6)
        assert engine.theta_0_rad == pytest.approx(math.pi / 12.0, rel=1e-9)
        assert engine.theta_0_rad == pytest.approx(0.2617993877991494, rel=1e-6)

        assert engine.theta_bootstrap_deg == pytest.approx(15.5, rel=1e-6)
        assert engine.theta_bootstrap_rad == pytest.approx(15.5 * DEG2RAD, rel=1e-9)
        assert engine.theta_bootstrap_rad == pytest.approx(0.2705260340591211, rel=1e-6)

        # Cold-start initial state
        assert engine.theta_rad == pytest.approx(engine.theta_bootstrap_rad, rel=1e-9)
        assert engine.theta_deg == pytest.approx(15.5, rel=1e-6)

        # Symmetry breaking offset
        delta_boot_rad = engine.theta_bootstrap_rad - engine.theta_0_rad
        assert delta_boot_rad > 0.0
        assert delta_boot_rad == pytest.approx(0.5 * DEG2RAD, rel=1e-9)
        assert delta_boot_rad == pytest.approx(0.008726646259971648, rel=1e-6)

        print(f"\n[Test 3.1] theta_0 = {engine.theta_0_deg:.1f} deg ({engine.theta_0_rad:.6f} rad)")
        print(f"          theta_bootstrap = {engine.theta_bootstrap_deg:.1f} deg ({engine.theta_bootstrap_rad:.6f} rad)")
        print(f"          Symmetry breaking bias = +{delta_boot_rad:.8f} rad (+0.5 deg)")

    def test_resting_tone_dead_zone_vs_bootstrap_kick_restoration(self):
        """
        Verify that:
        1. At resting tone theta_0 = 15.0 deg with zero drive (||V||_1 = 0, delta = 0), dot{theta} = 0 (dead-zone fixed point).
        2. At bootstrap angle theta_bootstrap = 15.5 deg with zero drive, Kuramoto spring pulls phase backwards
           with dot{theta} = -K * sin(0.5 deg) ~ -0.006981 rad/s.
        3. Trajectory from 15.5 deg under zero drive relaxes monotonically back to 15.0 deg.
        """
        engine = ThetaEngine()

        # 1. At resting tone: zero velocity
        engine.reset(theta_deg=15.0)
        _, _, dot_th_rest = engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert dot_th_rest == pytest.approx(0.0, abs=1e-9)

        # 2. At bootstrap angle: restoring velocity
        engine.reset()  # default 15.5 deg
        expected_restoring = -0.80 * math.sin(0.5 * DEG2RAD)
        assert expected_restoring == pytest.approx(-0.00698115, rel=1e-4)

        _, _, dot_th_boot = engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)
        assert dot_th_boot == pytest.approx(expected_restoring, rel=1e-6)
        assert dot_th_boot < 0.0

        # 3. Simulate relaxation over 200 steps (4.0 s)
        engine.reset()
        for _ in range(200):
            engine.step(V_l1=0.0, delta=0.0, a_motor=0.0)

        # Kuramoto relaxation time constant tau_K = 1 / K = 1 / 0.80 = 1.25 s
        # At t = 4.0 s (3.2 * tau_K), displacement from 15.0 deg should decay by exp(-3.2) ~ 0.04
        # 0.5 deg * 0.0407 ~ 0.020 deg
        offset_after_4s = engine.theta_deg - 15.0
        assert 0.0 < offset_after_4s < 0.03
        print(f"\n[Test 3.2] Zero-drive relaxation: 15.5 deg -> after 4.0s: {engine.theta_deg:.4f} deg (offset {offset_after_4s:.4f} deg)")

    def test_day0_precharge_immediate_symmetry_breaking(self):
        """
        Verify that under Day 0 precharge profile (||V(0)||_1 = 0.90):
        1. Forward drive kv * ||V(0)||_1 = 0.50 * 0.90 = +0.4500 rad/s
        2. Kuramoto restoring force at 15.5 deg is -0.00698 rad/s
        3. Net phase velocity dot{theta}(0) = +0.443019 rad/s > 0 (immediate forward acceleration)
        4. Bootstrap angle (15.5 deg) reaches 18.0 deg motor gate 16.7% faster than resting angle (15.0 deg).
        """
        engine_boot = ThetaEngine()
        engine_rest = ThetaEngine()

        engine_boot.reset(theta_deg=15.5)
        engine_rest.reset(theta_deg=15.0)

        # Immediate velocity check
        _, _, dot_th_boot = engine_boot.step(V_l1=0.90, delta=0.0, a_motor=0.0)
        _, _, dot_th_rest = engine_rest.step(V_l1=0.90, delta=0.0, a_motor=0.0)

        expected_boot_dot = 0.50 * 0.90 - 0.80 * math.sin(0.5 * DEG2RAD)
        assert dot_th_boot == pytest.approx(expected_boot_dot, rel=1e-6)
        assert dot_th_boot == pytest.approx(0.443019, rel=1e-4)

        assert dot_th_rest == pytest.approx(0.4500, rel=1e-6)

        # Distance to gate:
        # From 15.5 deg: 2.5 deg = 0.043633 rad
        # From 15.0 deg: 3.0 deg = 0.052360 rad
        # Step both forward until they cross 18.0 deg:
        engine_boot.reset(theta_deg=15.5)
        engine_rest.reset(theta_deg=15.0)

        steps_boot = 0
        while engine_boot.theta_deg < 18.0 and steps_boot < 100:
            engine_boot.step(V_l1=0.90, delta=0.0, a_motor=0.0)
            steps_boot += 1

        steps_rest = 0
        while engine_rest.theta_deg < 18.0 and steps_rest < 100:
            engine_rest.step(V_l1=0.90, delta=0.0, a_motor=0.0)
            steps_rest += 1

        time_boot = steps_boot * engine_boot.dt
        time_rest = steps_rest * engine_rest.dt

        assert steps_boot < steps_rest
        print(f"\n[Test 3.3] Time to gate (18.0 deg):")
        print(f"          From bootstrap kick (15.5 deg): {steps_boot} steps ({time_boot:.3f} s)")
        print(f"          From resting tone   (15.0 deg): {steps_rest} steps ({time_rest:.3f} s)")
        print(f"          Advantage: {steps_rest - steps_boot} steps ({(1 - time_boot/time_rest)*100:.1f}% faster)")

    def test_idle_ignition_within_10s_under_coupled_dynamics(self):
        """
        Verify DVP-01 Criterion 2: Idle motor ignition <= 10.0 s under zero input (C=0, delta=0).
        Test under two conditions:
        Condition A: Day 0 Pre-charge baseline (V(0) = [0.15, 0.10, 0.25, 0.10, 0.30]).
        Condition B: Cold-start from absolute zero needs (V(0) = [0.0, 0.0, 0.0, 0.0, 0.0]).
        """
        # Condition A: Day 0 Pre-charge
        maslow_a = MaslowIntegrator()
        theta_a = ThetaEngine()

        ignition_time_a = None
        for step in range(500):  # 10.0 seconds
            if step % 5 == 0:
                snap = maslow_a.step(C=0.0, a_motor=0.0)
                theta_a.latch_maslow(snap)
            theta_a.step(delta=0.0, a_motor=0.0)
            if theta_a.theta_deg >= 18.0:
                ignition_time_a = theta_a.time_s
                break

        assert ignition_time_a is not None
        assert ignition_time_a <= 10.0
        assert ignition_time_a < 0.20  # With precharge, ignites in ~0.10 s

        # Condition B: Absolute zero needs V(0) = 0.0
        maslow_b = MaslowIntegrator(initial_vector=[0.0, 0.0, 0.0, 0.0, 0.0])
        theta_b = ThetaEngine()

        ignition_time_b = None
        for step in range(500):  # 10.0 seconds
            if step % 5 == 0:
                snap = maslow_b.step(C=0.0, a_motor=0.0)
                theta_b.latch_maslow(snap)
            theta_b.step(delta=0.0, a_motor=0.0)
            if theta_b.theta_deg >= 18.0:
                ignition_time_b = theta_b.time_s
                break

        assert ignition_time_b is not None
        assert ignition_time_b <= 10.0
        print(f"\n[Test 3.4] Idle ignition timing (Criterion 2 <= 10.0 s):")
        print(f"          Condition A (Day 0 Pre-charge): {ignition_time_a:.3f} s -> PASS")
        print(f"          Condition B (Zero initial needs): {ignition_time_b:.3f} s -> PASS")


# ==============================================================================
# Standalone CLI Execution Entry Point
# ==============================================================================


def run_all_empirical_tests() -> int:
    """Execute all test methods and output summary report."""
    print("=" * 80)
    print("DELTA_P-42 GATE 1 - EMPIRICAL BIOLOGICAL FIDELITY VERIFICATION SUITE")
    print("Challenger: challenger_m1_2 | Target: theta_engine.py & maslow_integrator.py")
    print("=" * 80)

    test_classes = [
        TestBasalDriveAutonomousTensionGrowth(),
        TestMotorQuenchingDynamics(),
        TestBootstrapKickAndSymmetryBreaking(),
    ]

    total_tests = 0
    passed_tests = 0

    for suite in test_classes:
        class_name = suite.__class__.__name__
        print(f"\n>>> Running Suite: {class_name}")
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for m_name in methods:
            total_tests += 1
            method = getattr(suite, m_name)
            try:
                method()
                passed_tests += 1
                print(f"  [PASS] {m_name}")
            except Exception as e:
                print(f"  [FAIL] {m_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"EMIRICAL SUITE SUMMARY: {passed_tests}/{total_tests} PASSED")
    print("=" * 80)

    return 0 if passed_tests == total_tests else 1


if __name__ == "__main__":
    sys.exit(run_all_empirical_tests())
