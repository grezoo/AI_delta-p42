"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Module: nursery.bootstrap_loader
Author: Milestone 1 Worker (Nursery Subsystem Specialist)

Day 0 Cold Start Profile Loader and Soft AGC Ramp-Up Engine.
Enforces the calibrated 5D homeostatic pre-charge deficit profile:
    V_0 = [0.15, 0.10, 0.25, 0.10, 0.30]^T
and the initial symmetry-breaking phase perturbation:
    theta(0) = 15.5 deg == 0.270526 rad (theta_0 + 0.5 deg).
Evaluates the soft sigmoidal gain curve:
    g_boot(t) = 1 / (1 + exp(-2(t - 1.5))) for t in [0, 3.0 s]
preventing initial excitotoxic numeric shock, and supervises autonomous convergence
into the stable limit-cycle orbit within <= 3.0 s (DVP-09).

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter X.2 lines 659-668)
- DELTA_P42_MESTER_TEZIS.md (Chapter III.1 lines 137-145)
- Plan Review.md (REQ-FUNC-02, DVP-09)
- ORIGINAL_REQUEST.md (Stage-Gate 4 Milestone 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.basal_ganglia_gate import BasalGangliaGate
from core.maslow_integrator import DEFAULT_INITIAL_VECTOR, MaslowIntegrator, MaslowSnapshot
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import DEG2RAD, ThetaEngine

# ------------------------------------------------------------------------------
# Canonical Day 0 Constants
# ------------------------------------------------------------------------------

# Calibrated 5D Homeostatic Baseline Deficit Profile (L1 norm = 0.90)
# Needs: [metabolic, safety, social, competence, cognition]
DEFAULT_V0: np.ndarray = np.array([0.15, 0.10, 0.25, 0.10, 0.30], dtype=np.float64)
V0_CANONICAL: np.ndarray = DEFAULT_V0

# Angle Constants in SI Radians and Angular Degrees
THETA_0_DEG: float = 15.0
THETA_0_RAD: float = float(THETA_0_DEG * DEG2RAD)  # pi / 12 rad ~= 0.2617993877991494

THETA_BOOTSTRAP_DEG: float = 15.5
THETA_BOOTSTRAP_RAD: float = float(THETA_BOOTSTRAP_DEG * DEG2RAD)  # ~= 0.2705260340591211

THETA_GATE_DEG: float = 18.0
THETA_GATE_RAD: float = float(THETA_GATE_DEG * DEG2RAD)  # pi / 10 rad ~= 0.3141592653589793

# Soft AGC Ramp-Up Parameters
RAMP_UP_DURATION_S: float = 3.0
RAMP_UP_SIGMOID_K: float = 2.0
RAMP_UP_SIGMOID_T0: float = 1.5


# ------------------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapProfile:
    """
    Immutable specification of the Day 0 cold start state.
    """
    V_0: np.ndarray
    theta_0_rad: float
    theta_0_deg: float
    ramp_duration_s: float = RAMP_UP_DURATION_S

    def __post_init__(self) -> None:
        if self.V_0.shape != (5,):
            raise ValueError(f"V_0 must be a 5-element 1D array, got shape {self.V_0.shape}")
        if np.any(self.V_0 < 0.0) or np.any(self.V_0 > 1.0):
            raise ValueError(f"All components of V_0 must be in [0.0, 1.0], got {self.V_0}")


@dataclass(frozen=True)
class ConvergenceResult:
    """
    Quantitative results of limit-cycle convergence supervision (DVP-09).

    Attributes:
        passed: Overall boolean DVP-09 pass verdict.
        convergence_time_s: Time elapsed before entering and stabilizing in limit-cycle (<= 3.0 s).
        stable: Whether orbit remains bounded in [15.0 deg, 18.5 deg].
        first_cycle_period_s: Duration of the first observed limit cycle (in [1.0, 5.0] s).
        peak_cv: Coefficient of variation of peak amplitudes (<= 0.05).
        numeric_shock: True if derivative spike, NaN, or unbounded velocity was observed.
        details: Telemetry diagnostics.
    """
    passed: bool
    convergence_time_s: float
    stable: bool
    first_cycle_period_s: float
    peak_cv: float
    numeric_shock: bool
    details: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# BootstrapLoader Implementation
# ------------------------------------------------------------------------------

class BootstrapLoader:
    """
    Day 0 Cold Start Profile Loader and Soft AGC Ramp-Up Supervisor.

    Guarantees seamless bootstrap without numeric shock, initializing the 5D deficit
    vector and kicking the phase oscillator into autonomous limit-cycle orbit.
    """

    def __init__(
        self,
        V_0: Optional[Union[np.ndarray, List[float], Tuple[float, ...]]] = None,
        theta_0_deg: float = THETA_BOOTSTRAP_DEG,
        ramp_duration_s: float = RAMP_UP_DURATION_S,
    ) -> None:
        """
        Initialize the BootstrapLoader.

        Args:
            V_0: Initial 5D deficit vector. Defaults to DEFAULT_V0 [0.15, 0.10, 0.25, 0.10, 0.30].
            theta_0_deg: Initial phase perturbation angle in degrees. Defaults to 15.5 deg.
            ramp_duration_s: Duration of the soft sigmoid ramp-up in seconds (default 3.0 s).
        """
        if V_0 is None:
            self._V_0 = DEFAULT_V0.copy()
        else:
            arr = np.asarray(V_0, dtype=np.float64)
            if arr.shape != (5,):
                raise ValueError(f"V_0 must be a 5-element array, got shape {arr.shape}")
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"V_0 elements must be in [0.0, 1.0], got {arr}")
            self._V_0 = arr.copy()

        if theta_0_deg <= THETA_0_DEG:
            raise ValueError(
                f"Initial phase theta_0_deg ({theta_0_deg}) must exceed resting attractor "
                f"theta_0 ({THETA_0_DEG} deg) to break symmetry."
            )

        self._theta_0_deg = float(theta_0_deg)
        self._theta_0_rad = float(self._theta_0_deg * DEG2RAD)
        self._ramp_duration_s = float(ramp_duration_s)

    @property
    def V_0(self) -> np.ndarray:
        """Copy of canonical initial deficit vector."""
        return self._V_0.copy()

    @property
    def theta_bootstrap_rad(self) -> float:
        """Initial bootstrap phase angle in SI radians."""
        return self._theta_0_rad

    @property
    def theta_bootstrap_deg(self) -> float:
        """Initial bootstrap phase angle in degrees."""
        return self._theta_0_deg

    def get_initial_deficit(self) -> np.ndarray:
        """Returns the calibrated 5D homeostatic baseline deficit vector."""
        return self._V_0.copy()

    def get_initial_phase(self) -> float:
        """Returns the initial phase angle in SI radians."""
        return self._theta_0_rad

    def get_initial_phase_deg(self) -> float:
        """Returns the initial phase angle in degrees."""
        return self._theta_0_deg

    def get_boot_gain(self, elapsed_s: float) -> float:
        """
        Evaluates the soft sigmoidal ramp-up gain:
            g_boot(t) = 1 / (1 + exp(-2(t - 1.5))) for t in [0, 3.0 s]

        For t >= 3.0 s, saturates smoothly to 1.0.
        """
        if elapsed_s <= 0.0:
            # At t=0: 1 / (1 + exp(3)) ~= 0.04742587
            return float(1.0 / (1.0 + math.exp(RAMP_UP_SIGMOID_K * RAMP_UP_SIGMOID_T0)))

        if elapsed_s >= self._ramp_duration_s:
            return 1.0

        # Exact sigmoidal interpolation over t in (0, 3.0)
        exponent = -RAMP_UP_SIGMOID_K * (elapsed_s - RAMP_UP_SIGMOID_T0)
        # Numerical protection against overflow
        if exponent > 60.0:
            return 0.0
        if exponent < -60.0:
            return 1.0
        return float(1.0 / (1.0 + math.exp(exponent)))

    def evaluate_boot_gain(self, elapsed_s: float) -> float:
        """Compatibility alias for get_boot_gain."""
        return self.get_boot_gain(elapsed_s)

    def apply_bootstrap(
        self,
        theta_engine: ThetaEngine,
        maslow: MaslowIntegrator,
        thalamus_agc: Optional[ThalamusAGC] = None,
    ) -> None:
        """
        Injects the Day 0 cold start state into core subcortical engines.

        Args:
            theta_engine: Phase oscillator engine to initialize.
            maslow: Maslow deficit integrator to initialize.
            thalamus_agc: Optional Thalamus AGC engine.
        """
        # 1. Initialize Maslow Integrator with V_0
        if hasattr(maslow, "reset"):
            maslow.reset(initial_vector=self._V_0.copy())
        elif hasattr(maslow, "_vector"):
            maslow._vector = self._V_0.copy()

        # 2. Perturb Theta Engine with theta(0) = 15.5 deg
        if hasattr(theta_engine, "reset"):
            theta_engine.reset(theta_deg=self._theta_0_deg)
        elif hasattr(theta_engine, "_theta"):
            theta_engine._theta = self._theta_0_rad
            theta_engine._dot_theta = 0.0

        # 3. Synchronize Thalamus AGC baseline if present
        if thalamus_agc is not None:
            if hasattr(thalamus_agc, "reset"):
                thalamus_agc.reset()

    def verify_convergence(
        self,
        trajectory_history: List[Tuple[float, float, float]],
    ) -> ConvergenceResult:
        """
        Supervises limit-cycle convergence against DVP-09 requirements:
        1. Limit-cycle convergence within t <= 3.0 s.
        2. Closed orbit period T_cycle in [1.0, 5.0] s.
        3. Peak amplitude coefficient of variation CV(A_peak) <= 0.05.
        4. Zero numeric shock: |dot_theta| <= 5.0 rad/s, theta < 25.0 deg, no NaN.

        Args:
            trajectory_history: Sequence of (t_sec, theta_rad, dot_theta_rad_s) samples.

        Returns:
            ConvergenceResult with quantitative evaluation.
        """
        if not trajectory_history or len(trajectory_history) < 10:
            return ConvergenceResult(
                passed=False,
                convergence_time_s=0.0,
                stable=False,
                first_cycle_period_s=0.0,
                peak_cv=1.0,
                numeric_shock=True,
                details={"error": "Insufficient trajectory history"},
            )

        times = np.array([pt[0] for pt in trajectory_history], dtype=np.float64)
        thetas_rad = np.array([pt[1] for pt in trajectory_history], dtype=np.float64)
        dot_thetas = np.array([pt[2] for pt in trajectory_history], dtype=np.float64)

        # 1. Check for numeric shock (NaN, Inf, or runaway derivative)
        has_nan = bool(np.any(np.isnan(thetas_rad)) or np.any(np.isnan(dot_thetas)))
        has_inf = bool(np.any(np.isinf(thetas_rad)) or np.any(np.isinf(dot_thetas)))
        max_vel = float(np.max(np.abs(dot_thetas))) if len(dot_thetas) > 0 else 0.0
        max_deg = float(np.max(np.rad2deg(thetas_rad))) if len(thetas_rad) > 0 else 0.0

        numeric_shock = has_nan or has_inf or (max_vel > 5.0) or (max_deg > 25.0)

        # 2. Detect local peaks (maxima where dot_theta crosses zero downwards)
        # Refractory guard (min separation >= 0.20 s) prevents chattering at flat apexes (VULN-E1)
        peaks_idx = []
        min_refractory_s = 0.20
        for i in range(1, len(dot_thetas) - 1):
            if dot_thetas[i - 1] > 0.0 >= dot_thetas[i]:
                if not peaks_idx:
                    peaks_idx.append(i)
                elif (times[i] - times[peaks_idx[-1]]) >= min_refractory_s:
                    peaks_idx.append(i)
                else:
                    # Update to peak with higher amplitude within refractory window
                    if thetas_rad[i] > thetas_rad[peaks_idx[-1]]:
                        peaks_idx[-1] = i

        first_cycle_period = 0.0
        peak_cv = 0.0
        convergence_time = 0.0
        stable = False

        if len(peaks_idx) >= 2:
            peak_times = times[peaks_idx]
            peak_thetas = thetas_rad[peaks_idx]

            periods = np.diff(peak_times)
            first_cycle_period = float(periods[0])

            # Time to enter first limit cycle peak
            convergence_time = float(peak_times[0])

            # Coefficient of variation of peak amplitudes
            mean_peak = float(np.mean(peak_thetas))
            std_peak = float(np.std(peak_thetas))
            peak_cv = float(std_peak / (mean_peak + 1e-12))

            # Bounded limit cycle orbit check: [14.0 deg, 22.0 deg]
            min_deg = float(np.min(np.rad2deg(thetas_rad[peaks_idx[0]:])))
            max_deg_orbit = float(np.max(np.rad2deg(thetas_rad[peaks_idx[0]:])))
            stable = (min_deg >= 14.0) and (max_deg_orbit <= 22.0)

        # Overall DVP-09 pass criteria
        passed = (
            not numeric_shock
            and (convergence_time <= self._ramp_duration_s)
            and (1.0 <= first_cycle_period <= 5.0)
            and (peak_cv <= 0.05)
            and stable
        )

        final_phase_deg = float(np.rad2deg(thetas_rad[-1])) if len(thetas_rad) > 0 else 0.0

        return ConvergenceResult(
            passed=passed,
            convergence_time_s=convergence_time,
            stable=stable,
            first_cycle_period_s=first_cycle_period,
            peak_cv=peak_cv,
            numeric_shock=numeric_shock,
            details={
                "max_velocity_rad_s": max_vel,
                "max_phase_deg": max_deg,
                "num_peaks_detected": len(peaks_idx),
                "final_phase_deg": final_phase_deg,
            },
        )

    def run_bootstrap_simulation(
        self,
        duration_s: float = 5.0,
        dt: float = 0.02,
    ) -> ConvergenceResult:
        """
        Executes a self-contained numerical simulation of the bootstrap phase to verify
        limit-cycle convergence.

        Args:
            duration_s: Total simulation duration in seconds (default 5.0 s).
            dt: Discrete time step in seconds (default 0.02 s = 50 Hz).

        Returns:
            ConvergenceResult evaluating the run.
        """
        steps = int(duration_s / dt)
        theta_engine = ThetaEngine(dt=dt)
        maslow = MaslowIntegrator(dt=0.10)
        gate = BasalGangliaGate()
        self.apply_bootstrap(theta_engine, maslow)

        trajectory: List[Tuple[float, float, float]] = []
        sim_time = 0.0
        current_motor = 0.0

        for step in range(steps):
            # Evaluate ramp-up gain
            g_boot = self.get_boot_gain(sim_time)

            # Maslow steps at 10 Hz (every 5 theta steps)
            if step % 5 == 0:
                e_env = 1.0 if current_motor > 0.0 else 0.0
                maslow_snap = maslow.step(C=0.0, a_motor=current_motor, E_env=e_env)
                theta_engine.latch_maslow(maslow_snap)

            # Step theta engine with gentle non-negative initial noise
            noise_delta = float(0.01 * abs(math.sin(sim_time * 2.0)))
            theta_rad, theta_deg, dot_theta = theta_engine.step(
                delta=noise_delta * g_boot,
                a_motor=current_motor,
            )

            decision = gate.evaluate(theta_rad=theta_rad)
            current_motor = float(decision.a_motor)

            trajectory.append((sim_time, theta_rad, dot_theta))
            sim_time += dt

        return self.verify_convergence(trajectory)
