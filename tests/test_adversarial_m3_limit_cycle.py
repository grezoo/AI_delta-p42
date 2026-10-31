"""
Adversarial Stress-Testing and Limit Cycle Orbit Challenger Suite for Milestone 3 (M3).
Target:
    - core/gate1_verification.py (Gate1Verifier, DVP01Results)
    - Closed-loop coupled subcortical ODE dynamics (ThetaEngine, MaslowIntegrator, ThalamusAGC, BasalGangliaGate)

Verification Scope:
1. Long-horizon limit cycle stability:
   - 20,000 steps (400.0 s @ 50 Hz dt=0.02s) continuous simulation.
   - Orbit boundedness: theta(t) in [14.0 deg, 20.0 deg], V_l1 in [0, 5], a_motor in [0, 1].
   - Period stability: mean T_cycle in [1.0, 5.0] s across 240+ cycles.
   - Peak amplitude CV: sigma(A_peak) / mu(A_peak) <= 0.05.
   - Zero secular frequency drift between early and late orbits.

2. Perturbation recovery:
   - Injection of sudden phase noise pulses in theta(t) during orbit.
   - Measurement of relaxation time t_relax to return to limit cycle attractor within <= 5.0 s.
   - Empirical discovery and characterization of a parasitic stable fixed point (coma node)
     at theta* ~ 18.447 deg, a_motor* ~ 0.5638, V_l1* ~ 0.5469 where positive phase kicks
     lead to fatal limit cycle collapse.
   - Verification of negative pulse resilience.
   - Proof-of-concept verification of the quench drive latch mitigation.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.1, III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.1, X.2)
- Plan Review.md (DVP-01 Quantitative Protocol)
- ORIGINAL_REQUEST.md (Requirements R1, R4)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pytest

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.basal_ganglia_gate import BasalGangliaGate, GateDecision
from core.gate1_verification import Gate1Verifier
from core.maslow_integrator import MaslowIntegrator
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import DEG2RAD, ThetaEngine


# ==============================================================================
# SECTION 1: LONG-HORIZON LIMIT CYCLE STABILITY (20,000 STEPS = 400 S)
# ==============================================================================

class TestLongHorizonLimitCycleStability:
    """
    Adversarial verification of non-equilibrium limit cycle stability over long horizons.
    Runs 20,000 steps (400.0 s at 50 Hz) in idle mode (C=0, delta=0).
    """

    @pytest.fixture(scope="class")
    @classmethod
    def long_horizon_telemetry(cls):
        """Runs the 20,000-step (400.0 s) coupled subcortical ODE simulation."""
        total_steps = 20000
        dt = 0.02
        engine = ThetaEngine(dt=dt)
        maslow = MaslowIntegrator(dt=0.10)
        thalamus = ThalamusAGC(dt=dt)
        gate = BasalGangliaGate()

        timestamps: List[float] = []
        thetas_deg: List[float] = []
        v_l1_norms: List[float] = []
        motor_activations: List[float] = []
        dot_thetas: List[float] = []

        current_motor = 0.0
        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)

        for step in range(total_steps):
            t_s = step * dt

            # 10 Hz Maslow update
            if step % 5 == 0:
                e_env = 1.0 if current_motor > 0.0 else 0.0
                snap = maslow.step(C=0.0, a_motor=current_motor, E_env=e_env)
                engine.latch_maslow(snap)

            thal_out = thalamus.process(
                x_ext=x_ext,
                c_pred=c_pred,
                V=maslow.vector,
                theta=engine.theta_rad,
                sim_time=t_s,
            )

            th_rad, th_deg, dot_th = engine.step(
                delta=thal_out.delta,
                a_motor=current_motor,
            )

            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            timestamps.append(t_s)
            thetas_deg.append(th_deg)
            v_l1_norms.append(snap.l1_norm)
            motor_activations.append(current_motor)
            dot_thetas.append(dot_th)

        return {
            "timestamps": np.asarray(timestamps, dtype=np.float64),
            "thetas_deg": np.asarray(thetas_deg, dtype=np.float64),
            "v_l1": np.asarray(v_l1_norms, dtype=np.float64),
            "motor": np.asarray(motor_activations, dtype=np.float64),
            "dot_thetas": np.asarray(dot_thetas, dtype=np.float64),
            "total_steps": total_steps,
            "dt": dt,
        }

    def test_20000_steps_orbit_boundedness(self, long_horizon_telemetry):
        """Verifies that all dynamical state variables remain strictly bounded across 20,000 steps."""
        thetas = long_horizon_telemetry["thetas_deg"]
        v_l1 = long_horizon_telemetry["v_l1"]
        motor = long_horizon_telemetry["motor"]
        dot_th = long_horizon_telemetry["dot_thetas"]

        # Absolute mathematical finiteness (zero NaN / zero Inf)
        assert np.all(np.isfinite(thetas)), "NaN or Inf detected in theta trajectory"
        assert np.all(np.isfinite(v_l1)), "NaN or Inf detected in Maslow deficiency"
        assert np.all(np.isfinite(motor)), "NaN or Inf detected in motor activation"
        assert np.all(np.isfinite(dot_th)), "NaN or Inf detected in phase velocity"

        # Phase angle strictly bounded within physiological relaxation range [14.0 deg, 20.0 deg]
        assert 14.5 <= thetas.min() <= 15.2, f"Unexpected theta minimum: {thetas.min()}"
        assert 18.4 <= thetas.max() <= 19.5, f"Unexpected theta maximum: {thetas.max()}"

        # Maslow L1 deficiency strictly bounded within [0.0, 5.0]
        assert 0.0 <= v_l1.min()
        assert v_l1.max() <= 5.0
        # Post-ignition cyclic tension bounds [0.45, 1.05]
        assert 0.45 <= v_l1.min() <= 0.50
        assert 0.90 <= v_l1.max() <= 1.05

        # Motor activation strictly bounded in [0.0, 1.0]
        assert motor.min() == 0.0
        assert 0.80 <= motor.max() <= 1.0

        # Phase angular velocity finite and bounded
        assert -0.50 <= dot_th.min() <= 0.0
        assert 0.0 <= dot_th.max() <= 1.0

    def test_20000_steps_period_stability(self, long_horizon_telemetry):
        """Verifies that inter-peak cycle period remains strictly in [1.0, 5.0] s across all cycles."""
        timestamps = long_horizon_telemetry["timestamps"]
        thetas = long_horizon_telemetry["thetas_deg"]

        # Discard transient initial 10 seconds (500 steps)
        t_steady = timestamps[500:]
        th_steady = thetas[500:]

        peaks = Gate1Verifier.detect_peaks(th_steady * DEG2RAD, min_threshold=18.0 * DEG2RAD)
        assert len(peaks) >= 240, f"Expected >= 240 cycles over 390 s, found {len(peaks)}"

        peak_times = t_steady[peaks]
        periods = np.diff(peak_times)

        # All periods must be in [1.0, 5.0] s
        assert np.all((periods >= 1.0) & (periods <= 5.0))

        # Mean period must be near canonical 1.60 s
        assert math.isclose(float(np.mean(periods)), 1.600, abs_tol=0.05)

        # Standard deviation of cycle periods across 240+ cycles must be near zero (< 0.01 s)
        assert float(np.std(periods)) < 0.01

    def test_20000_steps_peak_cv(self, long_horizon_telemetry):
        """Verifies that peak amplitude CV strictly satisfies CV <= 0.05 across 20,000 steps."""
        timestamps = long_horizon_telemetry["timestamps"]
        thetas = long_horizon_telemetry["thetas_deg"]

        t_steady = timestamps[500:]
        th_steady = thetas[500:]

        peaks = Gate1Verifier.detect_peaks(th_steady * DEG2RAD, min_threshold=18.0 * DEG2RAD)
        peak_amps = th_steady[peaks]

        mean_amp = float(np.mean(peak_amps))
        std_amp = float(np.std(peak_amps))
        cv = std_amp / mean_amp

        # Peak CV must satisfy thesis requirement <= 0.05 (5.0%)
        assert cv <= 0.05, f"Peak CV {cv*100:.3f}% exceeds 5.0%"
        assert cv < 0.001, f"Peak CV {cv*100:.3f}% is larger than expected for deterministic simulation"

    def test_secular_drift_early_vs_late(self, long_horizon_telemetry):
        """Compares the first 20 steady cycles vs the last 20 cycles to verify zero secular drift."""
        timestamps = long_horizon_telemetry["timestamps"]
        thetas = long_horizon_telemetry["thetas_deg"]

        t_steady = timestamps[500:]
        th_steady = thetas[500:]

        peaks = Gate1Verifier.detect_peaks(th_steady * DEG2RAD, min_threshold=18.0 * DEG2RAD)
        peak_times = t_steady[peaks]
        periods = np.diff(peak_times)

        early_periods = periods[:20]
        late_periods = periods[-20:]

        early_mean = float(np.mean(early_periods))
        late_mean = float(np.mean(late_periods))

        # Secular drift must be negligible (< 1 ms difference)
        assert abs(early_mean - late_mean) < 0.001


# ==============================================================================
# SECTION 2: PERTURBATION RECOVERY & ATTRACTOR COLLAPSE REPRODUCTION
# ==============================================================================

class TestPerturbationRecovery:
    """
    Adversarial verification of limit cycle perturbation recovery.
    Injects sudden angular noise pulses into theta(t) during steady orbit.
    """

    @staticmethod
    def _run_pulse_simulation(
        pulse_deg: float,
        pulse_time_s: float = 50.0,
        total_duration_s: float = 80.0,
        dt: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Runs a simulation injecting a single delta-theta pulse at pulse_time_s."""
        total_steps = int(round(total_duration_s / dt))
        pert_step = int(round(pulse_time_s / dt))

        engine = ThetaEngine(dt=dt)
        maslow = MaslowIntegrator(dt=0.10)
        thalamus = ThalamusAGC(dt=dt)
        gate = BasalGangliaGate()

        current_motor = 0.0
        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)

        times: List[float] = []
        thetas: List[float] = []
        motors: List[float] = []
        v_l1s: List[float] = []

        for step in range(total_steps):
            t_s = step * dt

            # Inject sudden pulse into phase
            if step == pert_step:
                engine._theta += float(pulse_deg * DEG2RAD)

            if step % 5 == 0:
                e_env = 1.0 if current_motor > 0.0 else 0.0
                snap = maslow.step(C=0.0, a_motor=current_motor, E_env=e_env)
                engine.latch_maslow(snap)

            thal_out = thalamus.process(
                x_ext=x_ext,
                c_pred=c_pred,
                V=maslow.vector,
                theta=engine.theta_rad,
                sim_time=t_s,
            )

            th_rad, th_deg, _ = engine.step(
                delta=thal_out.delta,
                a_motor=current_motor,
            )

            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            times.append(t_s)
            thetas.append(th_deg)
            motors.append(current_motor)
            v_l1s.append(snap.l1_norm)

        return (
            np.asarray(times, dtype=np.float64),
            np.asarray(thetas, dtype=np.float64),
            np.asarray(motors, dtype=np.float64),
            np.asarray(v_l1s, dtype=np.float64),
        )

    @pytest.mark.parametrize("pulse_deg", [-2.0, -5.0, -10.0, -20.0, -90.0, -180.0])
    def test_negative_perturbation_recovery(self, pulse_deg: float):
        """Verifies that negative phase perturbations reliably relax back within <= 5.0 s."""
        pulse_time_s = 50.0
        times, thetas, motors, _ = self._run_pulse_simulation(
            pulse_deg=pulse_deg,
            pulse_time_s=pulse_time_s,
            total_duration_s=70.0,
        )

        # Evaluate window post pulse + 5.0 s: [55.0 s, 70.0 s]
        mask_post5 = times >= (pulse_time_s + 5.0)
        t_eval = times[mask_post5]
        th_eval = thetas[mask_post5]

        # Orbit amplitude range must be active (~3.5 deg peak-to-trough)
        amp_range = float(th_eval.max() - th_eval.min())
        assert amp_range >= 3.0, f"Oscillations damped out for pulse {pulse_deg} deg"

        peaks = Gate1Verifier.detect_peaks(th_eval * DEG2RAD, min_threshold=18.0 * DEG2RAD)
        assert len(peaks) >= 8, f"Too few cycles detected post-5s: {len(peaks)}"

        periods = np.diff(t_eval[peaks])
        # Periods must match canonical limit cycle 1.60 s within 5%
        assert np.all(np.abs(periods - 1.60) / 1.60 <= 0.05)

    def test_reproduce_positive_perturbation_collapse(self):
        """
        EMPIRICAL BUG REPRODUCTION:
        Demonstrates that a +2.0 deg pulse at t=50.80 s causes permanent limit cycle collapse!
        The system enters a parasitic stable fixed point at theta ~ 18.447 deg, a_motor ~ 0.5638.
        The relaxation time t_relax is infinite (> 5.0 s), violating the requirement.
        """
        pulse_time_s = 50.80
        pulse_deg = 2.0
        times, thetas, motors, vl1s = self._run_pulse_simulation(
            pulse_deg=pulse_deg,
            pulse_time_s=pulse_time_s,
            total_duration_s=80.0,
        )

        # Evaluate trajectory after pulse + 5.0 s (from t=56.0 s to 80.0 s)
        mask_late = times >= (pulse_time_s + 5.0)
        late_thetas = thetas[mask_late]
        late_motors = motors[mask_late]
        late_vl1s = vl1s[mask_late]

        # Calculate oscillation amplitude in late window
        late_amp_range = float(late_thetas.max() - late_thetas.min())

        # Assert empirical finding: oscillations have COMPLETELY CEASED (amp_range < 0.001 deg)
        assert late_amp_range < 0.001, (
            f"Expected limit cycle collapse, but found oscillation range: {late_amp_range} deg"
        )

        # Characterize the trapped fixed point values
        assert math.isclose(float(late_thetas[-1]), 18.4469, abs_tol=0.01)
        assert math.isclose(float(late_motors[-1]), 0.5638, abs_tol=0.01)
        assert math.isclose(float(late_vl1s[-1]), 0.5469, abs_tol=0.01)

    @pytest.mark.parametrize(
        "pulse_deg, pulse_time_s",
        [
            (6.0, 50.00),
            (5.0, 50.20),
            (10.0, 50.40),
            (6.0, 50.50),
            (5.0, 50.70),
            (2.0, 50.80),
            (5.0, 50.80),
            (20.0, 50.00),
        ],
    )
    def test_positive_perturbation_failure_modes(self, pulse_deg: float, pulse_time_s: float):
        """
        Systematic empirical reproduction of the limit cycle collapse across various
        amplitudes and orbit phases.
        """
        times, thetas, motors, _ = self._run_pulse_simulation(
            pulse_deg=pulse_deg,
            pulse_time_s=pulse_time_s,
            total_duration_s=75.0,
        )

        # Check amplitude in [pulse_time + 5.0, pulse_time + 20.0]
        mask_post5 = (times >= pulse_time_s + 5.0) & (times <= pulse_time_s + 20.0)
        eval_thetas = thetas[mask_post5]
        amp_range = float(eval_thetas.max() - eval_thetas.min())

        # These specific conditions must reproduce the collapse (amp_range == 0)
        assert amp_range < 0.01, (
            f"Pulse +{pulse_deg} deg at {pulse_time_s}s did not collapse (amp_range={amp_range})"
        )


# ==============================================================================
# SECTION 3: MITIGATION PROOF-OF-CONCEPT (ROBUST GATE)
# ==============================================================================

class RobustGate(BasalGangliaGate):
    """
    Candidate architectural defense:
    Once firing is triggered (self._firing = True), maintain quench drive
    a_motor = max(instant_a, self._quench_drive) throughout the active burst,
    preventing the fatal deceleration to dot_theta = 0 inside the threshold band.
    """

    def evaluate(self, theta_rad: float, **kwargs) -> GateDecision:
        dec = super().evaluate(theta_rad, **kwargs)
        if self._firing:
            a_motor_latched = max(dec.a_motor, self._quench_drive)
            return GateDecision(
                timestamp_ns=dec.timestamp_ns,
                a_motor=a_motor_latched,
                go_winner=dec.go_winner,
                is_locked=dec.is_locked,
                theta_rad=dec.theta_rad,
                theta_deg=dec.theta_deg,
            )
        return dec


class TestMitigationProofOfConcept:
    """Verifies that the proposed quench latch mitigation prevents collapse and recovers in <= 5s."""

    @pytest.mark.parametrize("pulse_deg, pulse_time_s", [(2.0, 50.80), (6.0, 50.00), (20.0, 50.00)])
    def test_robust_gate_prevents_collapse(self, pulse_deg: float, pulse_time_s: float):
        """With RobustGate, all previously failing pulses relax back to limit cycle within <= 5.0 s."""
        dt = 0.02
        total_steps = int(round(75.0 / dt))
        pert_step = int(round(pulse_time_s / dt))

        engine = ThetaEngine(dt=dt)
        maslow = MaslowIntegrator(dt=0.10)
        thalamus = ThalamusAGC(dt=dt)
        gate = RobustGate()

        current_motor = 0.0
        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)

        times: List[float] = []
        thetas: List[float] = []

        for step in range(total_steps):
            t_s = step * dt
            if step == pert_step:
                engine._theta += float(pulse_deg * DEG2RAD)

            if step % 5 == 0:
                e_env = 1.0 if current_motor > 0.0 else 0.0
                snap = maslow.step(C=0.0, a_motor=current_motor, E_env=e_env)
                engine.latch_maslow(snap)

            thal_out = thalamus.process(
                x_ext=x_ext,
                c_pred=c_pred,
                V=maslow.vector,
                theta=engine.theta_rad,
                sim_time=t_s,
            )

            th_rad, th_deg, _ = engine.step(
                delta=thal_out.delta,
                a_motor=current_motor,
            )

            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            times.append(t_s)
            thetas.append(th_deg)

        t_arr = np.asarray(times, dtype=np.float64)
        th_arr = np.asarray(thetas, dtype=np.float64)

        # Check trajectory post pulse + 5.0 s
        mask_post5 = t_arr >= (pulse_time_s + 5.0)
        t_eval = t_arr[mask_post5]
        th_eval = th_arr[mask_post5]

        amp_range = float(th_eval.max() - th_eval.min())
        assert amp_range >= 3.0, f"RobustGate failed to sustain orbit: amp_range={amp_range}"

        peaks = Gate1Verifier.detect_peaks(th_eval * DEG2RAD, min_threshold=18.0 * DEG2RAD)
        assert len(peaks) >= 8

        # Verify orbit periods remain in [1.0, 5.0] s per DVP-01
        periods = np.diff(t_eval[peaks])
        assert np.all((periods >= 1.0) & (periods <= 5.0)), f"Periods outside [1.0, 5.0] s: {periods}"

        # Verify peak amplitude CV <= 0.05 per DVP-01
        peak_amps = th_eval[peaks]
        cv = float(np.std(peak_amps) / np.mean(peak_amps))
        assert cv <= 0.05, f"RobustGate peak CV {cv*100:.3f}% exceeds 5.0%"


# ==============================================================================
# STANDALONE CLI RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING ADVERSARIAL LIMIT CYCLE TEST SUITE")
    print("=" * 80)
    pytest.main([__file__, "-v", "-s"])
