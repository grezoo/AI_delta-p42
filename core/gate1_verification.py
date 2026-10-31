"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - DVP-01 Quantitative Verification Engine
Automated, reproducible verification runner for Stage-Gate 1 exit criteria.

Tests the coupled non-equilibrium thermodynamic engine operating as a Hybrid
Relaxation Oscillator (FitzHugh-Nagumo dynamical class). Verifies that basal
metabolic entropy accumulation (lambda_base = 0.1386 s^-1) drives autonomous
phase advancement, smooth motor gating, and closed limit cycle dynamics without
external prompts (CO2 dead-zone falsification).

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section I.3, III.1, III.2, III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.1, I.5, II.1, X.2)
- Plan Review.md (DVP-01 Quantitative Protocol, Acceptance Criteria)
- megvalósitási terv.md (Section 3.3, Phase 1 Verification)
- ORIGINAL_REQUEST.md (Requirements R1, R4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repository root is on sys.path when executed directly as script
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from core.basal_ganglia_gate import BasalGangliaGate, GateDecision
from core.maslow_integrator import (
    DEFAULT_INITIAL_VECTOR,
    MaslowIntegrator,
    MaslowSnapshot,
)
from core.thalamus_agc import ThalamusAGC, ThalamusOutput
from core.theta_engine import DEG2RAD, ThetaEngine


@dataclass(frozen=True)
class DVP01Results:
    """
    Comprehensive quantitative results across all 3 DVP-01 verification criteria.
    """
    # Criterion 1: Limit cycle stability
    criterion_1_passed: bool
    t_cycle_mean: float
    t_cycle_valid_range: Tuple[float, float]    # (1.0, 5.0) seconds
    peak_amp_mean: float                        # SI radians
    peak_amp_std: float                         # SI radians
    peak_amp_cv: float                          # sigma(A_peak) / mu(A_peak)
    peak_amp_cv_threshold: float                # 0.05 (5.0%)
    cycles_detected: int

    # Criterion 2: Idle motor ignition
    criterion_2_passed: bool
    t_ignition: float
    t_ignition_threshold: float                 # 10.0 seconds

    # Criterion 3: Autonomous initiation rate
    criterion_3_passed: bool
    n_trials: int                               # 100
    n_success: int
    r_init: float                               # n_success / 100
    r_init_threshold: float                     # 0.85 (85%)

    # Overall Gate 1 status
    all_passed: bool
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)


class Gate1Verifier:
    """
    DVP-01 Quantitative Verification Engine for ΔP-42 Gate 1.

    Instantiates the real coupled subcortical cognitive loop:
        ThetaEngine (50 Hz SI radian phase oscillator)
        MaslowIntegrator (10 Hz exact analytical exponential integrator)
        ThalamusAGC (50 Hz normalized RMS dissonance & AGC gain)
        BasalGangliaGate (50 Hz continuous C0 jerk-free motor gate)

    Executes reproducible verification protocols for:
        Criterion 1: Closed Limit Cycle Formation & Stability (10,000 steps @ 50 Hz)
        Criterion 2: Idle State Phase Tipping & Motor Ignition (<= 10.0 s)
        Criterion 3: Autonomous Initiation Rate R_init (>= 0.85 over 100 stochastic trials)
    """

    def __init__(
        self,
        verbose: bool = True,
        transient_steps: int = 500,
        noise_sigma: float = 0.01,
        n_trials: int = 100,
        dt: float = 0.02,
        t_ignition_timeout: float = 10.0,
        steps_criterion1: int = 10000,
    ) -> None:
        """
        Initialize the Gate 1 verification engine.

        Args:
            verbose: If True, status messages and formatted reports are printed to stdout.
            transient_steps: Number of initial 50 Hz steps to discard for limit cycle settling (default 500 = 10.0 s).
            noise_sigma: Standard deviation of Brownian noise for Criterion 3 [rad/s^0.5] (canonical: 0.01).
            n_trials: Number of Monte Carlo trials for Criterion 3 (canonical: 100).
            dt: Master integration time step in seconds (default 0.02 s = 50 Hz).
            t_ignition_timeout: Maximum duration allowed for idle motor ignition in seconds (default 10.0 s).
            steps_criterion1: Number of simulation steps for Criterion 1 limit cycle (default 10000 = 200.0 s).
        """
        if n_trials <= 0:
            raise ValueError(f"n_trials must be positive, got {n_trials}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if t_ignition_timeout <= 0:
            raise ValueError(f"t_ignition_timeout must be positive, got {t_ignition_timeout}")
        if steps_criterion1 <= 0:
            raise ValueError(f"steps_criterion1 must be positive, got {steps_criterion1}")
        if transient_steps < 0:
            raise ValueError(f"transient_steps must be non-negative, got {transient_steps}")
        if noise_sigma < 0.0 or not math.isfinite(noise_sigma):
            raise ValueError(f"noise_sigma must be non-negative and finite, got {noise_sigma}")

        self.verbose: bool = bool(verbose)
        self.transient_steps: int = int(transient_steps)
        self.noise_sigma: float = float(noise_sigma)
        self.n_trials: int = int(n_trials)
        self.dt: float = float(dt)
        self.t_ignition_timeout: float = float(t_ignition_timeout)
        self.steps_criterion1: int = int(steps_criterion1)

    # --------------------------------------------------------------------------
    # Helper: Peak Detection
    # --------------------------------------------------------------------------

    @staticmethod
    def detect_peaks(
        signal: np.ndarray,
        min_threshold: float = 0.0,
    ) -> List[int]:
        """
        Detect local maxima (peaks) of a 1D discrete time-series using derivative zero-crossing.
        Identifies indices i where dot{signal}[i-1] > 0 and dot{signal}[i] <= 0 with signal[i] >= min_threshold.

        Args:
            signal: 1D numpy array of values.
            min_threshold: Minimum value for peak qualification.

        Returns:
            List of integer peak indices in the signal.
        """
        sig = np.asarray(signal, dtype=np.float64)
        if len(sig) < 3:
            return []

        diff = np.diff(sig)
        peaks: List[int] = []
        for i in range(len(diff) - 1):
            if diff[i] > 0.0 and diff[i + 1] <= 0.0:
                peak_idx = i + 1
                if sig[peak_idx] >= min_threshold:
                    peaks.append(peak_idx)
        return peaks

    # --------------------------------------------------------------------------
    # CRITERION 1: Closed Limit Cycle Formation & Stability
    # --------------------------------------------------------------------------

    def verify_criterion_1_limit_cycle(
        self,
        sim_duration_s: float = 200.0,
        dt: float = 0.02,
        c_work: float = 0.0,
        delta: float = 0.0,
    ) -> Tuple[bool, float, float, float, int]:
        """
        Verify DVP-01 Criterion 1: Closed Limit Cycle Formation & Stability.

        Runs a 10,000-step (200.0 s @ 50 Hz) simulation of the coupled subcortical loop in idle.
        Closed-loop environmental reward E_env is coupled to motor gate: E_env = 1.0 when a_motor > 0.
        Discards the initial transient (transient_steps = 500 = 10.0 s).
        Identifies peaks of theta(t) via derivative zero-crossing.
        Calculates inter-peak cycle periods T_cycle and peak amplitudes A_peak.

        Pass Thresholds:
            mean(T_cycle) in [1.0, 5.0] s
            CV(A_peak) = std(A_peak) / mean(A_peak) <= 0.05 (5.0%)
            completed cycles >= 35

        Returns:
            Tuple[bool, float, float, float, int]:
                (passed, t_cycle_mean, peak_amp_cv, peak_amp_mean_rad, cycles_detected)
        """
        total_steps = int(round(sim_duration_s / dt))
        transient = min(self.transient_steps, total_steps // 2)

        # Instantiate genuine subcortical modules
        theta_engine = ThetaEngine(dt=dt)
        maslow = MaslowIntegrator(dt=0.10)
        thalamus = ThalamusAGC(dt=dt)
        gate = BasalGangliaGate()

        timestamps: List[float] = []
        theta_rad_history: List[float] = []
        current_motor: float = 0.0

        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)

        # 50 Hz master integration loop
        for step in range(total_steps):
            t_s = step * dt

            # 10 Hz Maslow update every 5 steps (dt_maslow = 0.10 s, dt_theta = 0.02 s)
            if step % 5 == 0:
                e_env = 1.0 if current_motor > 0.0 else 0.0
                snap = maslow.step(C=c_work, a_motor=current_motor, E_env=e_env)
                theta_engine.latch_maslow(snap)

            # Thalamic DSP (dissonance = 0 in idle)
            thal_out = thalamus.process(
                x_ext=x_ext,
                c_pred=c_pred,
                V=maslow.vector,
                theta=theta_engine.theta_rad,
                sim_time=t_s,
            )
            dissonance = delta if delta > 0.0 else thal_out.delta

            # 50 Hz Theta phase step
            th_rad, _, _ = theta_engine.step(delta=dissonance, a_motor=current_motor)

            # Basal Ganglia continuous motor gating
            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            # Record steady-state trajectory post-transient
            if step >= transient:
                timestamps.append(t_s)
                theta_rad_history.append(th_rad)

        t_arr = np.asarray(timestamps, dtype=np.float64)
        th_arr = np.asarray(theta_rad_history, dtype=np.float64)

        # Detect peaks using derivative zero-crossing (threshold: theta >= 18.0 deg = 0.3142 rad)
        peak_indices = self.detect_peaks(th_arr, min_threshold=18.0 * DEG2RAD)
        cycles_detected = len(peak_indices) - 1 if len(peak_indices) > 0 else 0

        if len(peak_indices) < 2:
            return (False, 0.0, 1.0, 0.0, cycles_detected)

        peak_times = t_arr[peak_indices]
        peak_amps = th_arr[peak_indices]

        # Calculate cycle periods and peak amplitude statistics
        periods = np.diff(peak_times)
        t_cycle_mean = float(np.mean(periods))
        peak_amp_mean = float(np.mean(peak_amps))
        peak_amp_std = float(np.std(peak_amps))
        peak_amp_cv = float(peak_amp_std / peak_amp_mean) if peak_amp_mean > 0.0 else 1.0

        # Acceptance thresholds
        passed = (
            (1.0 <= t_cycle_mean <= 5.0)
            and (peak_amp_cv <= 0.05)
            and (cycles_detected >= 35)
        )

        return (passed, t_cycle_mean, peak_amp_cv, peak_amp_mean, cycles_detected)

    # --------------------------------------------------------------------------
    # CRITERION 2: Idle State Phase Tipping & Motor Ignition <= 10.0 s
    # --------------------------------------------------------------------------

    def verify_criterion_2_idle_ignition(
        self,
        max_duration_s: float = 10.0,
        dt: float = 0.02,
        kv: float = 0.018,
        initial_vector: Optional[Sequence[float]] = None,
        theta_bootstrap_deg: float = 15.5,
    ) -> Tuple[bool, float]:
        """
        Verify DVP-01 Criterion 2: Idle State Phase Tipping & Motor Ignition <= 10.0 s.

        Physical Coupling Regime Note:
            - Canonical Master Loop Gain: kv = 0.50 rad/s (DELTA_P42_MUKODESI_ELVEK.md line 75).
              Under kv = 0.50 rad/s, ignition occurs rapidly (t_ignition = 0.08 s <= 10.0 s).
            - Low-Gain Adversarial Stress Regime (Default for Criterion 2): kv = 0.018 rad/s.
              Under kv = 0.018 rad/s, the initial coupling force kv * ||V(0)||_1 = 0.0162 rad/s is
              initially SMALLER than the restoring Kuramoto spring barrier K * sin(3 deg) = 0.0418 rad/s.
              This regime rigorously proves that basal metabolic hunger accumulation (lambda_base = 0.1386 s^-1)
              alone forces autonomous limit-cycle phase tipping (t_ignition = 4.38 s <= 10.0 s)
              even when the immediate drive coupling is severely attenuated.

        Starts the coupled cognitive system from a cold Day 0 precharge:
            Initial need vector: V(0) = [0.15, 0.10, 0.25, 0.10, 0.30]
            Initial phase: theta(0) = 15.5 deg
            Workload: C = 0.0 (idle)
            Sensory input: x_ext = 0, c_pred = 0, delta = 0.0
            Environmental feedback: E_env = 0.0

        Basal entropy production (lambda_base = 0.1386 s^-1) drives autonomous need growth.
        The phase advances until crossing theta_gate = 18.0 deg, firing a_motor > 0.01.
        Evaluates whether ignition occurs in t_ignition <= 10.0 s.

        Returns:
            Tuple[bool, float]: (passed, t_ignition)
        """
        steps = int(round(max_duration_s / dt))

        v_init = DEFAULT_INITIAL_VECTOR if initial_vector is None else initial_vector

        # Instantiate genuine core loop with canonical Day 0 precharge
        theta_engine = ThetaEngine(
            dt=dt,
            kv=kv,
            theta_0_deg=15.0,
            theta_bootstrap_deg=theta_bootstrap_deg,
        )
        maslow = MaslowIntegrator(dt=0.10, initial_vector=v_init)
        thalamus = ThalamusAGC(dt=dt)
        gate = BasalGangliaGate()

        x_ext = np.zeros(thalamus.d_sens, dtype=np.float64)
        c_pred = np.zeros(thalamus.d_sens, dtype=np.float64)
        current_motor = 0.0

        t_ignition: Optional[float] = None

        for step in range(steps):
            t_s = step * dt

            # 10 Hz Maslow update
            if step % 5 == 0:
                snap = maslow.step(C=0.0, a_motor=current_motor, E_env=0.0)
                theta_engine.latch_maslow(snap)

            # Thalamic DSP
            thal_out = thalamus.process(
                x_ext=x_ext,
                c_pred=c_pred,
                V=maslow.vector,
                theta=theta_engine.theta_rad,
                sim_time=t_s,
            )

            # Phase advance
            th_rad, th_deg, _ = theta_engine.step(
                delta=thal_out.delta,
                a_motor=current_motor,
            )

            # Gate evaluation
            decision = gate.evaluate(theta_rad=th_rad)
            current_motor = decision.a_motor

            # Check ignition condition: theta >= 18.0 deg and a_motor > 0.01
            if th_deg >= 18.0 and current_motor > 0.01:
                t_ignition = t_s
                break

        if t_ignition is not None and t_ignition <= max_duration_s:
            return (True, float(t_ignition))
        return (False, float(max_duration_s if t_ignition is None else t_ignition))

    # --------------------------------------------------------------------------
    # CRITERION 3: Autonomous Initiation Rate R_init over N=100 Trials
    # --------------------------------------------------------------------------

    def verify_criterion_3_autonomous_initiation(
        self,
        n_trials: Optional[int] = None,
        trial_duration_s: float = 10.0,
        dt: float = 0.02,
    ) -> Tuple[bool, int, float]:
        """
        Verify DVP-01 Criterion 3: Autonomous Initiation Rate R_init >= 0.85 (85%).

        Executes N = 100 independent Monte Carlo trials without any external prompt (E_ext = 0, C = 0).
        Each trial samples perturbed initial physiological states:
            V_i(0) ~ Uniform(0.10, 0.25)
            theta(0) ~ Normal(15.5 deg, 0.30 deg)
            Brownian phase noise ~ Normal(0, noise_sigma * sqrt(dt))
        Each trial is considered a SUCCESS if motor ignition (theta >= 18.0 deg and a_motor > 0.01)
        occurs spontaneously within t <= 10.0 s.

        Pass Threshold:
            R_init = N_success / N >= 0.85

        Returns:
            Tuple[bool, int, float]: (passed, n_success, r_init)
        """
        trials = self.n_trials if n_trials is None else int(n_trials)
        if trials <= 0:
            raise ValueError(f"n_trials must be positive, got {trials}")
        steps_per_trial = int(round(trial_duration_s / dt))
        n_success = 0

        for j in range(1, trials + 1):
            # Deterministic, reproducible RNG stream per trial
            rng = np.random.default_rng(seed=42000 + j)

            # Sample perturbed initial physiological conditions
            v_init = rng.uniform(0.10, 0.25, size=5)
            theta_init_deg = float(rng.normal(15.5, 0.30))

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

            trial_success = False

            for step in range(steps_per_trial):
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

                # Step phase oscillator
                th_rad, th_deg, _ = engine.step(
                    delta=thal_out.delta,
                    a_motor=current_motor,
                )

                # Inject Brownian stochastic noise into phase angle
                noise = float(rng.normal(0.0, self.noise_sigma) * math.sqrt(dt))
                engine._theta += noise

                decision = gate.evaluate(theta_rad=engine.theta_rad)
                current_motor = decision.a_motor

                if engine.theta_deg >= 18.0 and current_motor > 0.01:
                    trial_success = True
                    break

            if trial_success:
                n_success += 1

        r_init = float(n_success / trials)
        passed = bool(r_init >= 0.85)
        return (passed, n_success, r_init)

    # --------------------------------------------------------------------------
    # Master Execution: run_all_checks
    # --------------------------------------------------------------------------

    def run_all_checks(self) -> DVP01Results:
        """
        Execute all 3 DVP-01 verification criteria, compile telemetry, and return structured results.

        Returns:
            DVP01Results dataclass carrying quantitative metrics and overall pass/fail status.
        """
        if self.verbose:
            print("=" * 80)
            print("DELTA P-42 SOVEREIGN COGNITIVE CORE (GATE 1) - DVP-01 QUANTITATIVE VERIFICATION")
            print("=" * 80)
            print("[1/3] Running Criterion 1: Closed Limit Cycle Formation & Stability (200 s)...")

        # Criterion 1
        c1_pass, t_mean, cv, peak_mean, cycles = self.verify_criterion_1_limit_cycle(
            sim_duration_s=self.steps_criterion1 * self.dt,
            dt=self.dt,
        )
        peak_std = peak_mean * cv

        if self.verbose:
            status_1 = "PASS" if c1_pass else "FAIL"
            print(f"      Status: [{status_1}] | T_cycle = {t_mean:.4f} s | CV = {cv*100:.3f}% | Cycles = {cycles}")
            print(f"[2/3] Running Criterion 2: Idle State Phase Tipping & Motor Ignition (<= {self.t_ignition_timeout:.1f} s)...")

        # Criterion 2
        c2_pass, t_ign = self.verify_criterion_2_idle_ignition(
            max_duration_s=self.t_ignition_timeout,
            dt=self.dt,
        )

        if self.verbose:
            status_2 = "PASS" if c2_pass else "FAIL"
            print(f"      Status: [{status_2}] | t_ignition = {t_ign:.3f} s (Threshold: <= {self.t_ignition_timeout:.1f} s)")
            print(f"[3/3] Running Criterion 3: Autonomous Initiation Rate ({self.n_trials} trials)...")

        # Criterion 3
        c3_pass, n_succ, r_init = self.verify_criterion_3_autonomous_initiation(
            n_trials=self.n_trials,
            dt=self.dt,
        )

        if self.verbose:
            status_3 = "PASS" if c3_pass else "FAIL"
            print(f"      Status: [{status_3}] | R_init = {r_init*100:.1f}% ({n_succ}/{self.n_trials}) (Threshold: >= 85.0%)")

        all_passed = bool(c1_pass and c2_pass and c3_pass)

        telemetry: Dict[str, Any] = {
            "criterion_1_passed": c1_pass,
            "t_cycle_mean": t_mean,
            "t_cycle_valid_range": (1.0, 5.0),
            "peak_amp_mean_rad": peak_mean,
            "peak_amp_mean_deg": peak_mean / DEG2RAD,
            "peak_amp_std_rad": peak_std,
            "peak_amp_cv": cv,
            "peak_amp_cv_threshold": 0.05,
            "cycles_detected": cycles,
            "criterion_2_passed": c2_pass,
            "t_ignition": t_ign,
            "t_ignition_threshold": self.t_ignition_timeout,
            "criterion_3_passed": c3_pass,
            "n_trials": self.n_trials,
            "n_success": n_succ,
            "r_init": r_init,
            "r_init_threshold": 0.85,
            "all_passed": all_passed,
        }

        results = DVP01Results(
            criterion_1_passed=c1_pass,
            t_cycle_mean=t_mean,
            t_cycle_valid_range=(1.0, 5.0),
            peak_amp_mean=peak_mean,
            peak_amp_std=peak_std,
            peak_amp_cv=cv,
            peak_amp_cv_threshold=0.05,
            cycles_detected=cycles,
            criterion_2_passed=c2_pass,
            t_ignition=t_ign,
            t_ignition_threshold=self.t_ignition_timeout,
            criterion_3_passed=c3_pass,
            n_trials=self.n_trials,
            n_success=n_succ,
            r_init=r_init,
            r_init_threshold=0.85,
            all_passed=all_passed,
            telemetry_summary=telemetry,
        )

        if self.verbose:
            print("\n" + self.format_report(results))

        return results

    # --------------------------------------------------------------------------
    # Report Generator: format_report
    # --------------------------------------------------------------------------

    def format_report(self, results: Optional[DVP01Results] = None) -> str:
        """
        Generate a structured, professional ASCII report of DVP-01 verification results.

        Args:
            results: DVP01Results instance. If None, runs all checks to produce fresh results.

        Returns:
            Formatted multi-line report string.
        """
        res = self.run_all_checks() if results is None else results

        c1_mark = "[PASS]" if res.criterion_1_passed else "[FAIL]"
        c2_mark = "[PASS]" if res.criterion_2_passed else "[FAIL]"
        c3_mark = "[PASS]" if res.criterion_3_passed else "[FAIL]"
        overall_mark = "GATE 1 APPROVED - PASS" if res.all_passed else "GATE 1 REJECTED - FAIL"

        lines = [
            "+" + "-" * 78 + "+",
            f"| DELTA P-42 STAGE-GATE 1: DVP-01 QUANTITATIVE VERIFICATION REPORT{' ' * 14}|",
            "+" + "-" * 78 + "+",
            f"| Criterion 1: Closed Limit Cycle Formation & Stability{' ' * 23}|",
            f"|   - Mean Period T_cycle:      {res.t_cycle_mean:6.4f} s  (Target: [{res.t_cycle_valid_range[0]:.1f}, {res.t_cycle_valid_range[1]:.1f}] s)     {c1_mark} |",
            f"|   - Peak Amp Relative Std CV: {res.peak_amp_cv * 100:6.3f} %  (Target: <= {res.peak_amp_cv_threshold * 100:.1f} %)          {c1_mark} |",
            f"|   - Detected Cycles:          {res.cycles_detected:6d}    (Target: >= 35)               {c1_mark} |",
            "+" + "-" * 78 + "+",
            f"| Criterion 2: Idle State Phase Tipping & Motor Ignition{' ' * 23}|",
            f"|   - Time to Ignition t_ign:   {res.t_ignition:6.3f} s  (Target: <= {res.t_ignition_threshold:.1f} s)             {c2_mark} |",
            "+" + "-" * 78 + "+",
            f"| Criterion 3: Autonomous Initiation Rate (Monte Carlo N={res.n_trials}){' ' * 18}|",
            f"|   - Initiation Rate R_init:   {res.r_init * 100:6.1f} %  (Target: >= {res.r_init_threshold * 100:.1f} %)          {c3_mark} |",
            f"|   - Successful Initiations:   {res.n_success:6d} / {res.n_trials:<4d}                             {c3_mark} |",
            "+" + "-" * 78 + "+",
            f"| OVERALL STATUS: {overall_mark:<60}|",
            "+" + "-" * 78 + "+",
        ]
        return "\n".join(lines)


# ------------------------------------------------------------------------------
# Standalone CLI Entry Point
# ------------------------------------------------------------------------------

def main() -> int:
    """
    CLI entry point for core/gate1_verification.py.
    Executes DVP-01 verification engine, emits report, and exits with code 0 on PASS, 1 on FAIL.
    """
    verifier = Gate1Verifier(verbose=True)
    results = verifier.run_all_checks()
    return 0 if results.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
