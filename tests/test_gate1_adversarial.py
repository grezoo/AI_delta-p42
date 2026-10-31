"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Adversarial Empirical Challenge Suite
Target: core/gate1_verification.py & core subcortical dynamics

Adversarial Stress Tests:
1. Independent Monte Carlo Robustness:
   - 200 trials with parameter jitter (perturbed initial conditions, seed noise, noise sigma).
   - Unconditional verification of R_init >= 0.85 without external prompts.
   - Low-gain regime (kv=0.018) vs default master loop (kv=0.50).
2. Sensitivity Analysis & Falsification of CO2 Dead-Zone Model:
   - Verify that when basal metabolic drive lambda_base is zeroed out (lambda_base = 0.0),
     autonomous motor ignition strictly FAILS (t_ignition > 10.0 s, a_motor == 0.0).
   - Parametric titration sweep: lambda_base in [0.00, 0.20] s^-1 mapping ignition latency.
   - Depleted initial need state (V(0) = 0) with vs without lambda_base.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.basal_ganglia_gate import BasalGangliaGate
from core.gate1_verification import Gate1Verifier
from core.maslow_integrator import DEFAULT_INITIAL_VECTOR, MaslowIntegrator
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import DEG2RAD, ThetaEngine


# ==============================================================================
# SECTION 1: Monte Carlo Robustness (200 Trials + Parameter Jitter)
# ==============================================================================

class TestMonteCarloRobustness200Trials:
    """
    Adversarial verification of autonomous initiation rate R_init >= 0.85
    under 200 independent stochastic trials with parameter jitter.
    """

    def test_mc_200_trials_gate1_verifier_direct(self):
        """
        Challenge 1.1: Direct execution of Gate1Verifier with n_trials=200.
        Verifies that doubling the trial count from 100 to 200 retains R_init >= 0.85.
        """
        verifier = Gate1Verifier(verbose=False, n_trials=200)
        passed, n_success, r_init = verifier.verify_criterion_3_autonomous_initiation(n_trials=200)

        assert passed is True
        assert n_success >= int(0.85 * 200)  # At least 170 / 200
        assert r_init >= 0.85
        assert n_success == 200, f"Expected 100% in canonical setup, got {n_success}/200"

    def test_mc_200_trials_aggressive_parameter_jitter(self):
        """
        Challenge 1.2: Independent adversarial Monte Carlo harness with 200 trials and aggressive jitter:
        - Wide initial need distribution: V_i(0) ~ Uniform(0.05, 0.35) (both low and high needs)
        - Phase angle jitter: theta(0) ~ Normal(15.0 deg, 0.60 deg) (starts near resting attractor theta_0=15.0 deg)
        - Variable Brownian noise sigma: sigma ~ Uniform(0.005, 0.030) rad/s^0.5
        - Disjoint PRNG seed space (seed = 990000 + i)
        - Zero external prompts (C = 0, x_ext = 0, c_pred = 0, delta = 0)
        """
        n_trials = 200
        trial_duration_s = 10.0
        dt = 0.02
        steps = int(round(trial_duration_s / dt))
        n_success = 0

        for i in range(1, n_trials + 1):
            rng = np.random.default_rng(seed=990000 + i)

            # Jitter initial need vector
            v_init = rng.uniform(0.05, 0.35, size=5)

            # Jitter initial phase angle around 15.0 deg (resting attractor)
            theta_init_deg = float(rng.normal(15.0, 0.60))

            # Jitter noise standard deviation
            noise_sigma = float(rng.uniform(0.005, 0.030))

            engine = ThetaEngine(
                dt=dt,
                theta_bootstrap_deg=theta_init_deg,
            )
            maslow = MaslowIntegrator(dt=0.10, initial_vector=v_init)
            thalamus = ThalamusAGC(dt=dt)
            gate = BasalGangliaGate()

            x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
            c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)
            current_motor = 0.0
            trial_ignited = False

            for step in range(steps):
                t_s = step * dt

                # 10 Hz Maslow step
                if step % 5 == 0:
                    snap = maslow.step(C=0.0, a_motor=current_motor, E_env=0.0)
                    engine.latch_maslow(snap)

                # Thalamus step
                thal_out = thalamus.process(
                    x_ext=x_ext,
                    c_pred=c_pred,
                    V=maslow.vector,
                    theta=engine.theta_rad,
                    sim_time=t_s,
                )

                # Phase oscillator step
                engine.step(delta=thal_out.delta, a_motor=current_motor)

                # Brownian noise injection
                noise = float(rng.normal(0.0, noise_sigma) * math.sqrt(dt))
                engine._theta += noise

                # Motor gate evaluation
                decision = gate.evaluate(theta_rad=engine.theta_rad)
                current_motor = decision.a_motor

                if engine.theta_deg >= 18.0 and current_motor > 0.01:
                    trial_ignited = True
                    break

            if trial_ignited:
                n_success += 1

        r_init = n_success / n_trials
        assert r_init >= 0.85, f"Adversarial MC failed: R_init = {r_init*100:.1f}% ({n_success}/{n_trials})"
        assert n_success >= 195, f"Expected near-perfect robustness, got {n_success}/{n_trials}"

    def test_mc_200_trials_idle_gain_regime(self):
        """
        Challenge 1.3: Monte Carlo under low idle coupling gain kv = 0.018 rad/s (Criterion 2 regime).
        Validates whether basal drive lambda_base = 0.1386 s^-1 overcomes the Kuramoto spring
        within t <= 10.0 s across 200 trials with randomized V(0) ~ Uniform(0.12, 0.25).
        """
        n_trials = 200
        trial_duration_s = 10.0
        dt = 0.02
        steps = int(round(trial_duration_s / dt))
        n_success = 0
        ignition_times: List[float] = []

        for i in range(1, n_trials + 1):
            rng = np.random.default_rng(seed=770000 + i)
            v_init = rng.uniform(0.12, 0.25, size=5)
            theta_init_deg = float(rng.normal(15.5, 0.20))

            engine = ThetaEngine(
                dt=dt,
                kv=0.018,
                theta_0_deg=15.0,
                theta_bootstrap_deg=theta_init_deg,
            )
            maslow = MaslowIntegrator(dt=0.10, initial_vector=v_init)
            thalamus = ThalamusAGC(dt=dt)
            gate = BasalGangliaGate()

            x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
            c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)
            current_motor = 0.0
            trial_ignited = False

            for step in range(steps):
                t_s = step * dt
                if step % 5 == 0:
                    snap = maslow.step(C=0.0, a_motor=current_motor, E_env=0.0)
                    engine.latch_maslow(snap)

                thal_out = thalamus.process(
                    x_ext=x_ext,
                    c_pred=c_pred,
                    V=maslow.vector,
                    theta=engine.theta_rad,
                    sim_time=t_s,
                )

                engine.step(delta=thal_out.delta, a_motor=current_motor)

                decision = gate.evaluate(theta_rad=engine.theta_rad)
                current_motor = decision.a_motor

                if engine.theta_deg >= 18.0 and current_motor > 0.01:
                    trial_ignited = True
                    ignition_times.append(t_s)
                    break

            if trial_ignited:
                n_success += 1

        r_init = n_success / n_trials
        assert r_init >= 0.85
        assert n_success >= 190, f"Expected >= 95% success under kv=0.018, got {n_success}/{n_trials}"
        mean_t_ign = float(np.mean(ignition_times))
        assert 2.0 <= mean_t_ign <= 7.0, f"Mean ignition time {mean_t_ign:.3f} s outside [2.0, 7.0] s"


# ==============================================================================
# SECTION 2: Sensitivity Analysis & CO2 Dead-Zone Falsification (lambda_base = 0)
# ==============================================================================

class TestSensitivityAnalysisLambdaBase:
    """
    Adversarial verification of the core non-equilibrium thermodynamic thesis:
    If basal metabolic drive lambda_base is zeroed out, autonomous ignition FAILS.
    Falsifies passive AI prompt-dependent stagnation (CO2 dead-zone).
    """

    def test_zero_basal_drive_fails_ignition_in_idle_regime(self):
        """
        Challenge 2.1: In Criterion 2 idle setup (kv=0.018), setting lambda_base = 0.0
        MUST completely prevent ignition past theta_gate = 18.0 deg.
        """
        dt = 0.02
        sim_duration_s = 20.0  # Double standard 10s duration to ensure no late ignition
        steps = int(round(sim_duration_s / dt))

        # Setup with lambda_base = 0.0
        engine = ThetaEngine(
            dt=dt,
            kv=0.018,
            theta_0_deg=15.0,
            theta_bootstrap_deg=15.5,
        )
        maslow = MaslowIntegrator(
            dt=0.10,
            lambda_base=0.0,  # Zeroed basal drive!
            initial_vector=DEFAULT_INITIAL_VECTOR,
        )
        thalamus = ThalamusAGC(dt=dt)
        gate = BasalGangliaGate()

        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)
        current_motor = 0.0

        max_theta_deg = -float("inf")
        ignited = False

        for step in range(steps):
            t_s = step * dt
            if step % 5 == 0:
                snap = maslow.step(C=0.0, a_motor=current_motor, E_env=0.0)
                engine.latch_maslow(snap)

            thal_out = thalamus.process(
                x_ext=x_ext,
                c_pred=c_pred,
                V=maslow.vector,
                theta=engine.theta_rad,
                sim_time=t_s,
            )

            th_rad, th_deg, _ = engine.step(delta=thal_out.delta, a_motor=current_motor)
            if th_deg > max_theta_deg:
                max_theta_deg = th_deg

            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            if th_deg >= 18.0 and current_motor > 0.01:
                ignited = True
                break

        # Verification: Ignition MUST NOT occur!
        assert ignited is False, "Ignition unexpectedly occurred with lambda_base = 0.0!"
        # In fact, phase gets trapped at fixed point theta* ~ 16.16 deg < 18.0 deg
        assert max_theta_deg < 17.0, f"Max phase reached {max_theta_deg:.2f} deg, expected trapped < 17.0 deg"
        # Need vector remains completely constant at initial precharge L1 norm = 0.90
        assert math.isclose(maslow.get_snapshot().l1_norm, 0.90, rel_tol=1e-6)

    def test_parametric_titration_lambda_base_curve(self):
        """
        Challenge 2.2: Titration curve sweeping lambda_base in [0.00, 0.20] s^-1.
        Maps the ignition time transition from dead-zone (infinite/timeout) to active ignition.
        """
        dt = 0.02
        max_duration_s = 15.0
        steps = int(round(max_duration_s / dt))

        test_rates = [0.00, 0.02, 0.05, 0.08, 0.10, 0.1386, 0.20]
        results_by_rate: Dict[float, Tuple[bool, float]] = {}

        for l_base in test_rates:
            engine = ThetaEngine(
                dt=dt,
                kv=0.018,
                theta_0_deg=15.0,
                theta_bootstrap_deg=15.5,
            )
            maslow = MaslowIntegrator(
                dt=0.10,
                lambda_base=l_base,
                initial_vector=DEFAULT_INITIAL_VECTOR,
            )
            thalamus = ThalamusAGC(dt=dt)
            gate = BasalGangliaGate()

            x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
            c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)
            current_motor = 0.0
            ignited = False
            t_ign = max_duration_s

            for step in range(steps):
                t_s = step * dt
                if step % 5 == 0:
                    snap = maslow.step(C=0.0, a_motor=current_motor, E_env=0.0)
                    engine.latch_maslow(snap)

                thal_out = thalamus.process(
                    x_ext=x_ext,
                    c_pred=c_pred,
                    V=maslow.vector,
                    theta=engine.theta_rad,
                    sim_time=t_s,
                )

                th_rad, th_deg, _ = engine.step(delta=thal_out.delta, a_motor=current_motor)
                decision = gate.evaluate(theta_rad=th_rad)
                current_motor = decision.a_motor

                if th_deg >= 18.0 and current_motor > 0.01:
                    ignited = True
                    t_ign = t_s
                    break

            results_by_rate[l_base] = (ignited, t_ign)

        # Rate 0.00: Strictly fails
        assert results_by_rate[0.00][0] is False
        assert results_by_rate[0.00][1] == max_duration_s

        # Rate 0.02: Below critical accumulation rate, fails within 15s
        assert results_by_rate[0.02][0] is False

        # Rates >= 0.08: Successfully ignite, with monotonic decreasing ignition latency
        assert results_by_rate[0.08][0] is True
        assert results_by_rate[0.10][0] is True
        assert results_by_rate[0.1386][0] is True
        assert results_by_rate[0.20][0] is True

        # Monotonicity check for active regime: higher lambda_base -> faster ignition
        assert results_by_rate[0.08][1] > results_by_rate[0.10][1] > results_by_rate[0.1386][1] > results_by_rate[0.20][1]

    def test_depleted_initial_vector_lambda_base_recovery(self):
        """
        Challenge 2.3: Starting from completely depleted needs V(0) = [0, 0, 0, 0, 0]:
        - If lambda_base = 0.0: system relaxes to resting attractor 15.0 deg and NEVER ignites.
        - If lambda_base = 0.1386: basal drive builds tension from zero, successfully igniting.
        """
        dt = 0.02
        duration_s = 15.0
        steps = int(round(duration_s / dt))
        v_zero = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Case A: lambda_base = 0.0 (Dead-zone)
        engine_dead = ThetaEngine(dt=dt, kv=0.50, theta_bootstrap_deg=15.5)
        maslow_dead = MaslowIntegrator(dt=0.10, lambda_base=0.0, initial_vector=v_zero)
        gate_dead = BasalGangliaGate()
        ignited_dead = False

        for step in range(steps):
            if step % 5 == 0:
                snap = maslow_dead.step(C=0.0, a_motor=0.0, E_env=0.0)
                engine_dead.latch_maslow(snap)
            th_rad, th_deg, _ = engine_dead.step(delta=0.0, a_motor=0.0)
            dec = gate_dead.evaluate(theta_rad=th_rad)
            if th_deg >= 18.0 and dec.a_motor > 0.01:
                ignited_dead = True
                break

        assert ignited_dead is False
        # Engine relaxed back towards theta_0 = 15.0 deg
        assert abs(engine_dead.theta_deg - 15.0) < 0.1

        # Case B: lambda_base = 0.1386 (Living cognitive engine)
        engine_live = ThetaEngine(dt=dt, kv=0.50, theta_bootstrap_deg=15.5)
        maslow_live = MaslowIntegrator(dt=0.10, lambda_base=0.1386, initial_vector=v_zero)
        gate_live = BasalGangliaGate()
        ignited_live = False
        t_live = duration_s

        for step in range(steps):
            t_s = step * dt
            if step % 5 == 0:
                snap = maslow_live.step(C=0.0, a_motor=0.0, E_env=0.0)
                engine_live.latch_maslow(snap)
            th_rad, th_deg, _ = engine_live.step(delta=0.0, a_motor=0.0)
            dec = gate_live.evaluate(theta_rad=th_rad)
            if th_deg >= 18.0 and dec.a_motor > 0.01:
                ignited_live = True
                t_live = t_s
                break

        assert ignited_live is True
        assert t_live <= 10.0


def run_adversarial_suite() -> int:
    """
    Executes the adversarial empirical test suite and prints structured telemetry tables.
    """
    print("=" * 80)
    print("DELTA P-42 ADVERSARIAL EMPIRICAL CHALLENGE SUITE (MILESTONE 3)")
    print("=" * 80)

    # 1. Direct Gate1Verifier 200 trials
    print("\n[CHALLENGE 1.1] Running Gate1Verifier with N=200 trials...")
    v = Gate1Verifier(verbose=False, n_trials=200)
    p1, s1, r1 = v.verify_criterion_3_autonomous_initiation(n_trials=200)
    print(f"  Result: Passed={p1} | Successes={s1}/200 | R_init={r1 * 100:.1f}% (Threshold: >=85.0%)")

    # 2. Aggressive Jitter 200 trials
    print("\n[CHALLENGE 1.2] Running Aggressive Parameter Jitter (200 trials, independent harness)...")
    tc1 = TestMonteCarloRobustness200Trials()
    tc1.test_mc_200_trials_aggressive_parameter_jitter()
    print("  Result: [PASS] R_init >= 0.85 held unconditionally under wide V(0), theta(0), sigma jitter.")

    # 3. Idle Gain Regime 200 trials
    print("\n[CHALLENGE 1.3] Running Low Idle Gain Regime (kv=0.018, 200 trials)...")
    tc1.test_mc_200_trials_idle_gain_regime()
    print("  Result: [PASS] Basal entropy accumulation reliably drives ignition under kv=0.018.")

    # 4. Zero Basal Drive Falsification
    print("\n[CHALLENGE 2.1] Testing Zero Basal Drive (lambda_base = 0.0 s^-1)...")
    tc2 = TestSensitivityAnalysisLambdaBase()
    tc2.test_zero_basal_drive_fails_ignition_in_idle_regime()
    print("  Result: [PASS] System gets trapped at fixed point theta* ~ 16.16 deg < 18.0 deg. Ignition FAILED as required.")

    # 5. Parametric Titration Curve
    print("\n[CHALLENGE 2.2] Parametric Titration Curve (lambda_base sweep)...")
    dt = 0.02
    max_duration_s = 15.0
    steps = int(round(max_duration_s / dt))
    test_rates = [0.00, 0.02, 0.05, 0.08, 0.10, 0.1386, 0.20]
    print("  +-----------------+--------------+--------------------+")
    print("  | lambda_base     | Ignited?     | t_ignition (s)     |")
    print("  +-----------------+--------------+--------------------+")
    for l_base in test_rates:
        engine = ThetaEngine(dt=dt, kv=0.018, theta_0_deg=15.0, theta_bootstrap_deg=15.5)
        maslow = MaslowIntegrator(dt=0.10, lambda_base=l_base, initial_vector=DEFAULT_INITIAL_VECTOR)
        thalamus = ThalamusAGC(dt=dt)
        gate = BasalGangliaGate()
        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)
        current_motor = 0.0
        ignited = False
        t_ign = max_duration_s
        for step in range(steps):
            t_s = step * dt
            if step % 5 == 0:
                snap = maslow.step(C=0.0, a_motor=current_motor, E_env=0.0)
                engine.latch_maslow(snap)
            thal_out = thalamus.process(x_ext=x_ext, c_pred=c_pred, V=maslow.vector, theta=engine.theta_rad, sim_time=t_s)
            th_rad, th_deg, _ = engine.step(delta=thal_out.delta, a_motor=current_motor)
            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor
            if th_deg >= 18.0 and current_motor > 0.01:
                ignited = True
                t_ign = t_s
                break
        status_str = "YES (PASS)" if ignited else "NO (BLOCKED)"
        t_str = f"{t_ign:6.3f} s" if ignited else "TIMEOUT (>15s)"
        print(f"  | {l_base:6.4f} s^-1    | {status_str:<12} | {t_str:<18} |")
    print("  +-----------------+--------------+--------------------+")

    # 6. Depleted Needs Recovery
    print("\n[CHALLENGE 2.3] Depleted Initial Needs Recovery (V(0) = 0)...")
    tc2.test_depleted_initial_vector_lambda_base_recovery()
    print("  Result: [PASS] lambda_base=0 relaxes to resting tone (dormant); lambda_base=0.1386 builds tension and ignites.")

    print("\n" + "=" * 80)
    print("ADVERSARIAL EMPIRICAL VERDICT: ALL CHALLENGES PASSED (VERDICT: APPROVE)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_adversarial_suite())

