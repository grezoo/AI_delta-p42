"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Theta Engine
50 Hz Coupled Phase Oscillator running strictly in SI radians.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.1, III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.1, X.2)
- Plan Review.md (REQ-MATH-01, REQ-FUNC-02, DVP-01)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from core.maslow_integrator import MaslowSnapshot

# Precision conversion constant from angular degrees to SI radians
DEG2RAD: float = np.pi / 180.0


class ThetaEngine:
    """
    50 Hz Hybrid Relaxation Phase Oscillator for the ΔP-42 Cognitive Architecture.

    Evaluates the coupled phase velocity ODE:
        dot{theta}(t) = kv * ||V(t)||_1 - K * sin(theta(t) - theta_0) + alpha * delta(t) - gamma * a_motor(t)

    All internal angle states, constants, and velocities operate strictly in SI radians.
    """

    def __init__(
        self,
        dt: float = 0.02,
        kv: float = 0.50,
        K: float = 0.80,
        alpha: float = 0.50,
        gamma: float = 0.40,
        theta_0_deg: float = 15.0,
        theta_gate_deg: float = 18.0,
        theta_bootstrap_deg: float = 15.5,
        delta_theta_sav_deg: float = 0.35,
    ) -> None:
        """
        Initialize the Theta Engine with calibrated biological and physical parameters.

        Args:
            dt: Discrete time step in seconds (default 0.02 s = 50 Hz loop frequency).
            kv: Maslow coupling gain in rad/s (converts deficiency into angular drive).
            K: Kuramoto homeostatic spring restoration gain in rad/s.
            alpha: Thalamic excitatory forward coupling gain in rad/s.
            gamma: Motor quenching phase braking gain in rad/s.
            theta_0_deg: Resting homeostatic attractor in degrees (canonical: 15.0 deg = pi/12 rad).
            theta_gate_deg: Basal Ganglia motor gate threshold in degrees (canonical: 18.0 deg = pi/10 rad).
            theta_bootstrap_deg: Day 0 cold-start bootstrap kick in degrees (canonical: 15.5 deg).
            delta_theta_sav_deg: Sigmoid smoothing bandwidth in degrees (canonical: 0.35 deg).
        """
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"Time step dt must be positive and finite, got {dt}")
        if kv < 0.0 or not math.isfinite(kv):
            raise ValueError(f"Coupling gain kv must be non-negative and finite, got {kv}")
        if K < 0.0 or not math.isfinite(K):
            raise ValueError(f"Restoring gain K must be non-negative and finite, got {K}")
        if alpha < 0.0 or not math.isfinite(alpha):
            raise ValueError(f"Thalamic gain alpha must be non-negative and finite, got {alpha}")
        if gamma < 0.0 or not math.isfinite(gamma):
            raise ValueError(f"Quenching gain gamma must be non-negative and finite, got {gamma}")

        self._dt: float = float(dt)
        self._kv: float = float(kv)
        self._K: float = float(K)
        self._alpha: float = float(alpha)
        self._gamma: float = float(gamma)

        # SI Radian conversion of angular parameters
        self._theta_0: float = float(theta_0_deg * DEG2RAD)
        self._theta_gate: float = float(theta_gate_deg * DEG2RAD)
        self._theta_bootstrap: float = float(theta_bootstrap_deg * DEG2RAD)
        self._delta_theta_sav: float = float(delta_theta_sav_deg * DEG2RAD)

        # State variables
        self._theta: float = self._theta_bootstrap
        self._dot_theta: float = 0.0
        self._time_s: float = 0.0

        # Multi-rate synchronization bridge (latched 10 Hz snapshot/L1 norm)
        self._latched_V_l1: Optional[float] = None

    # --- Property Accessors ---

    @property
    def dt(self) -> float:
        """Integration time step in seconds."""
        return self._dt

    @property
    def kv(self) -> float:
        """Maslow deficiency coupling gain in rad/s."""
        return self._kv

    @property
    def K(self) -> float:
        """Kuramoto restoring spring gain in rad/s."""
        return self._K

    @property
    def alpha(self) -> float:
        """Thalamic dissonance forward acceleration gain in rad/s."""
        return self._alpha

    @property
    def gamma(self) -> float:
        """Motor quenching braking gain in rad/s."""
        return self._gamma

    @property
    def theta_0_rad(self) -> float:
        """Resting attractor angle in SI radians (15.0 deg)."""
        return self._theta_0

    @property
    def theta_0_deg(self) -> float:
        """Resting attractor angle in degrees (15.0 deg)."""
        return self._theta_0 / DEG2RAD

    @property
    def theta_gate_rad(self) -> float:
        """Basal Ganglia motor gate threshold in SI radians (18.0 deg)."""
        return self._theta_gate

    @property
    def theta_gate_deg(self) -> float:
        """Basal Ganglia motor gate threshold in degrees (18.0 deg)."""
        return self._theta_gate / DEG2RAD

    @property
    def theta_bootstrap_rad(self) -> float:
        """Day 0 bootstrap angle in SI radians (15.5 deg)."""
        return self._theta_bootstrap

    @property
    def theta_bootstrap_deg(self) -> float:
        """Day 0 bootstrap angle in degrees (15.5 deg)."""
        return self._theta_bootstrap / DEG2RAD

    @property
    def delta_theta_sav_rad(self) -> float:
        """Smoothing bandwidth in SI radians (0.35 deg)."""
        return self._delta_theta_sav

    @property
    def delta_theta_sav_deg(self) -> float:
        """Smoothing bandwidth in degrees (0.35 deg)."""
        return self._delta_theta_sav / DEG2RAD

    @property
    def theta_rad(self) -> float:
        """Current internal phase angle in SI radians."""
        return self._theta

    @property
    def theta_deg(self) -> float:
        """Current internal phase angle converted to degrees."""
        return self._theta / DEG2RAD

    @property
    def dot_theta(self) -> float:
        """Current phase angular velocity in rad/s."""
        return self._dot_theta

    @property
    def time_s(self) -> float:
        """Elapsed simulation time in seconds."""
        return self._time_s

    # --- Multi-Rate Synchronization ---

    def latch_maslow(self, snapshot: Union[MaslowSnapshot, np.ndarray, float]) -> None:
        """
        Latch a 10 Hz Maslow state into the 50 Hz Theta Engine loop.

        Args:
            snapshot: MaslowSnapshot instance, 5D numpy array, or scalar L1 norm.
        """
        self._latched_V_l1 = self._extract_l1_norm(snapshot)

    def _extract_l1_norm(self, V_input: Union[float, int, np.ndarray, MaslowSnapshot]) -> float:
        """Extract and validate the L1 norm from various input types."""
        if hasattr(V_input, "l1_norm"):
            val = float(getattr(V_input, "l1_norm"))
        elif isinstance(V_input, np.ndarray):
            if not np.all(np.isfinite(V_input)):
                raise ValueError("Maslow vector contains NaN or Inf")
            val = float(np.sum(np.abs(V_input)))
        elif isinstance(V_input, (int, float)):
            val = float(V_input)
        else:
            raise TypeError(f"Unsupported type for Maslow deficiency: {type(V_input)}")

        if not math.isfinite(val):
            raise ValueError(f"Maslow L1 norm must be finite, got {val}")
        if val < 0.0:
            raise ValueError(f"Maslow L1 norm cannot be negative, got {val}")
        return val

    # --- Step Execution ---

    def step(
        self,
        V_l1: Optional[Union[float, int, np.ndarray, MaslowSnapshot]] = None,
        delta: float = 0.0,
        a_motor: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Advance the coupled phase oscillator by one discrete time step (dt = 0.02 s).

        ODE formulation:
            dot{theta}(t) = kv * ||V||_1 - K * sin(theta - theta_0) + alpha * delta - gamma * a_motor
            theta(t + dt) = theta(t) + dot{theta}(t) * dt

        Args:
            V_l1: Maslow total L1 deficiency (float, MaslowSnapshot, or np.ndarray).
                  If None, the previously latched multi-rate value is utilized.
            delta: Normalized thalamic predictive dissonance in [0.0, 1.0].
            a_motor: Continuous Basal Ganglia motor gate activation in [0.0, 1.0].

        Returns:
            Tuple[float, float, float]: (theta_rad, theta_deg, dot_theta)
        """
        # Multi-rate latching / resolution of V_l1
        if V_l1 is not None:
            resolved_l1 = self._extract_l1_norm(V_l1)
            self._latched_V_l1 = resolved_l1
        elif self._latched_V_l1 is not None:
            resolved_l1 = self._latched_V_l1
        else:
            raise ValueError("No Maslow deficiency provided and no latched snapshot available")

        # Validate delta
        if not math.isfinite(delta):
            raise ValueError(f"Dissonance delta must be finite, got {delta}")
        if delta < 0.0:
            raise ValueError(f"Dissonance delta cannot be negative, got {delta}")
        # Numerical guard: clamp minor upper float inaccuracies to 1.0 if reasonable
        if delta > 1.0:
            if delta > 1.05:
                raise ValueError(f"Dissonance delta exceeds [0, 1] range: {delta}")
            delta = 1.0

        # Validate a_motor
        if not math.isfinite(a_motor):
            raise ValueError(f"Motor activation a_motor must be finite, got {a_motor}")
        if a_motor < 0.0:
            raise ValueError(f"Motor activation a_motor cannot be negative, got {a_motor}")
        if a_motor > 1.0:
            if a_motor > 1.05:
                raise ValueError(f"Motor activation a_motor exceeds [0, 1] range: {a_motor}")
            a_motor = 1.0

        # 1. Forward drive from Maslow deficiency vacuum
        drive_maslow = self._kv * resolved_l1

        # 2. Kuramoto non-linear sinusoidal homeostatic spring restoration
        # Using phase difference directly preserves continuity and periodicity
        restoring_torque = -self._K * math.sin(self._theta - self._theta_0)

        # 3. Thalamic predictive dissonance forward coupling
        drive_thalamus = self._alpha * float(delta)

        # 4. Motor action quenching negative feedback braking
        damping_motor = -self._gamma * float(a_motor)

        # Net phase velocity ODE with Non-Equilibrium Invariant (Delta P > 0)
        dot_theta = drive_maslow + restoring_torque + drive_thalamus + damping_motor
        if dot_theta < 0.05 and resolved_l1 > 0.10:
            # Thermodynamic Non-Equilibrium Invariant: biological pacemaker never reaches thermal death
            dot_theta = 0.08 + 0.04 * math.sin(self._time_s * 2.0 * math.pi * 0.5)

        # Forward Euler state integration
        self._theta += dot_theta * self._dt
        self._dot_theta = dot_theta
        self._time_s += self._dt

        return (self._theta, self._theta / DEG2RAD, dot_theta)

    def reset(self, theta_deg: Optional[float] = None) -> None:
        """
        Reset the engine state to the bootstrap baseline or a specified angle.

        Args:
            theta_deg: Optional starting angle in degrees. Defaults to theta_bootstrap (15.5 deg).
        """
        if theta_deg is not None:
            self._theta = float(theta_deg * DEG2RAD)
        else:
            self._theta = self._theta_bootstrap
        self._dot_theta = 0.0
        self._time_s = 0.0
        self._latched_V_l1 = None
