"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Maslow Integrator
10 Hz Exact Analytical Exponential Integrator for 5D Need-Deficiency Dynamics.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section I.3, III.2, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.1, X.2)
- Plan Review.md (REQ-MATH-03, REQ-FUNC-02, REQ-COMM-02)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional, Sequence, Tuple, Union

import numpy as np

# Canonical 5D Maslow Needs
NEED_NAMES: Tuple[str, ...] = ("metab", "safety", "social", "comp", "cog")

# Day 0 Pre-charge Baseline Deficiency Vector
DEFAULT_INITIAL_VECTOR: np.ndarray = np.array(
    [0.15, 0.10, 0.25, 0.10, 0.30], dtype=np.float64
)


@dataclass(frozen=True)
class MaslowSnapshot:
    """
    Immutable, atomic, read-only snapshot emitted by the 10 Hz Maslow Integrator.

    Thread-safe data structure carrying nanosecond timestamps, read-only 5D deficiency vector,
    L1/L2 norms, and dominant need classification.
    """

    timestamp_ns: int
    vector: np.ndarray
    l1_norm: float
    l2_norm: float
    dominant_idx: int
    dominant_name: str

    def __init__(
        self,
        timestamp_ns: int,
        vector: Optional[np.ndarray] = None,
        l1_norm: Optional[float] = None,
        l2_norm: Optional[float] = None,
        dominant_idx: Optional[int] = None,
        dominant_name: Optional[str] = None,
        *,
        values: Optional[np.ndarray] = None,
        dominant_index: Optional[int] = None,
    ) -> None:
        # Support alternative alias keywords (values, dominant_index)
        resolved_vec = values if vector is None else vector
        if resolved_vec is None:
            raise ValueError("Either 'vector' or 'values' must be provided")

        vec_arr = np.array(resolved_vec, dtype=np.float64, copy=True)
        if vec_arr.shape != (5,):
            raise ValueError(f"Maslow vector must have shape (5,), got {vec_arr.shape}")

        resolved_l1 = float(np.sum(vec_arr)) if l1_norm is None else float(l1_norm)
        resolved_l2 = float(np.linalg.norm(vec_arr)) if l2_norm is None else float(l2_norm)

        idx_cand = dominant_index if dominant_idx is None else dominant_idx
        resolved_idx = int(np.argmax(vec_arr)) if idx_cand is None else int(idx_cand)

        if dominant_name is None:
            resolved_name = NEED_NAMES[resolved_idx] if 0 <= resolved_idx < 5 else "unknown"
        else:
            resolved_name = str(dominant_name)

        # Enforce read-only memory buffer to guarantee immutability
        vec_arr.flags.writeable = False

        object.__setattr__(self, "timestamp_ns", int(timestamp_ns))
        object.__setattr__(self, "vector", vec_arr)
        object.__setattr__(self, "l1_norm", resolved_l1)
        object.__setattr__(self, "l2_norm", resolved_l2)
        object.__setattr__(self, "dominant_idx", resolved_idx)
        object.__setattr__(self, "dominant_name", resolved_name)

    @property
    def values(self) -> np.ndarray:
        """Alias for vector buffer (read-only)."""
        return self.vector

    @property
    def dominant_index(self) -> int:
        """Alias for dominant_idx."""
        return self.dominant_idx


class MaslowIntegrator:
    """
    10 Hz Exact Analytical Exponential Integrator for the 5-dimensional Maslow need vector V(t).

    Continuous governing ODE:
        dot{V}_i(t) = G_i(C(t)) * (1 - V_i(t)) - R_i(a_motor(t), E_env(t)) * V_i(t)
                    = G_i - (G_i + R_i) * V_i(t)
                    = -lambda_i * (V_i(t) - V_i^infty)

    where:
        G_i(C) = lambda_base + lambda_work * C(t)
        R_i(a, E) = eta_gain * a_motor(t) * E_env_i(t)
        lambda_i = G_i + R_i
        V_i^infty = G_i / lambda_i

    Exact discrete analytical solution over interval [t, t + dt]:
        V_i(t + dt) = V_i(t) * e^(-lambda_i * dt) + V_i^infty * (1 - e^(-lambda_i * dt))

    Unconditional Stability Guarantee:
        Since V_i(t+dt) is a strict convex combination of V_i(t) in [0, 1] and V_i^infty in [0, 1]
        with weights w = e^(-lambda_i * dt) in (0, 1] and (1 - w) in [0, 1), V_i is mathematically
        guaranteed to remain strictly within [0.0, 1.0] without artificial clipping.
    """

    def __init__(
        self,
        dt: float = 0.10,
        lambda_base: float = 0.1386,
        lambda_work: float = 0.20,
        eta_gain: float = 2.0,
        initial_vector: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initialize the Maslow Integrator with canonical physical rates and initial vector.

        Args:
            dt: Discrete integration time step in seconds (default 0.10 s = 10 Hz).
            lambda_base: Basal metabolic entropy production rate in s^-1 (tau = 7.215 s, t_1/2 = 5.0 s).
            lambda_work: Working memory load coupling rate in s^-1.
            eta_gain: Environmental satisfaction rate gain in s^-1.
            initial_vector: Initial 5D need vector. Defaults to Day 0 pre-charge profile.
        """
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"Time step dt must be positive and finite, got {dt}")
        if lambda_base < 0.0 or not math.isfinite(lambda_base):
            raise ValueError(f"lambda_base must be non-negative and finite, got {lambda_base}")
        if lambda_work < 0.0 or not math.isfinite(lambda_work):
            raise ValueError(f"lambda_work must be non-negative and finite, got {lambda_work}")
        if eta_gain < 0.0 or not math.isfinite(eta_gain):
            raise ValueError(f"eta_gain must be non-negative and finite, got {eta_gain}")

        self._dt: float = float(dt)
        self._lambda_base: float = float(lambda_base)
        self._lambda_work: float = float(lambda_work)
        self._eta_gain: float = float(eta_gain)

        # Initialize internal 5D deficiency state vector
        self._V: np.ndarray = np.zeros(5, dtype=np.float64)
        self.reset(initial_vector)

    # --- Property Accessors ---

    @property
    def dt(self) -> float:
        """Integration time step in seconds."""
        return self._dt

    @property
    def lambda_base(self) -> float:
        """Basal metabolic tension rate in s^-1."""
        return self._lambda_base

    @property
    def lambda_work(self) -> float:
        """Working memory tension coupling rate in s^-1."""
        return self._lambda_work

    @property
    def eta_gain(self) -> float:
        """Environmental satisfaction quenching gain in s^-1."""
        return self._eta_gain

    @property
    def vector(self) -> np.ndarray:
        """Copy of the current internal 5D need vector."""
        return self._V.copy()

    @property
    def time_s(self) -> float:
        """Elapsed simulation time in seconds."""
        return self._time_s

    # --- State Management ---

    def reset(self, initial_vector: Optional[Sequence[float]] = None) -> MaslowSnapshot:
        """
        Reset the deficiency state to the initial pre-charge profile.

        Args:
            initial_vector: Optional custom 5D starting vector. Defaults to DEFAULT_INITIAL_VECTOR.

        Returns:
            MaslowSnapshot: Initial state snapshot.
        """
        if initial_vector is None:
            vec_init = DEFAULT_INITIAL_VECTOR.copy()
        else:
            vec_init = np.array(initial_vector, dtype=np.float64)
            if vec_init.shape != (5,):
                raise ValueError(f"Initial vector must have shape (5,), got {vec_init.shape}")
            if np.any(vec_init < 0.0) or np.any(vec_init > 1.0):
                raise ValueError("Initial vector values must be within [0.0, 1.0]")

        self._V[:] = vec_init
        self._time_s: float = 0.0
        self._sim_time_ns: int = 0
        return self.get_snapshot()

    def get_snapshot(self) -> MaslowSnapshot:
        """
        Emit an immutable, read-only atomic snapshot of the current internal state.

        Returns:
            MaslowSnapshot: Thread-safe snapshot with flags.writeable = False.
        """
        snap_vec = self._V.copy()
        snap_vec.flags.writeable = False

        l1 = float(np.sum(snap_vec))
        l2 = float(np.linalg.norm(snap_vec))
        dom_idx = int(np.argmax(snap_vec))
        dom_name = NEED_NAMES[dom_idx]

        return MaslowSnapshot(
            timestamp_ns=self._sim_time_ns,
            vector=snap_vec,
            l1_norm=l1,
            l2_norm=l2,
            dominant_idx=dom_idx,
            dominant_name=dom_name,
        )

    # --- Step Execution ---

    def step(
        self,
        C: float = 0.0,
        a_motor: float = 0.0,
        E_env: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ) -> MaslowSnapshot:
        """
        Advance the Maslow deficiency dynamics by one discrete time step (dt = 0.10 s)
        using the exact analytical exponential integrator.

        Args:
            C: Working memory cognitive load in [0.0, 1.0].
            a_motor: Motor action activation level in [0.0, 1.0].
            E_env: Environmental satisfaction vector in [0.0, 1.0]^5 or scalar broadcast.
                   If None, defaults to 1.0 (enabling full quenching upon motor action).

        Returns:
            MaslowSnapshot: New immutable state snapshot at t + dt.
        """
        # Validate cognitive load C
        if not math.isfinite(C):
            raise ValueError(f"Cognitive load C must be finite, got {C}")
        if C < 0.0:
            raise ValueError(f"Cognitive load C cannot be negative, got {C}")
        if C > 1.0:
            if C > 1.05:
                raise ValueError(f"Cognitive load C exceeds [0, 1] range: {C}")
            C = 1.0

        # Validate motor activation a_motor
        if not math.isfinite(a_motor):
            raise ValueError(f"Motor activation a_motor must be finite, got {a_motor}")
        if a_motor < 0.0:
            raise ValueError(f"Motor activation a_motor cannot be negative, got {a_motor}")
        if a_motor > 1.0:
            if a_motor > 1.05:
                raise ValueError(f"Motor activation a_motor exceeds [0, 1] range: {a_motor}")
            a_motor = 1.0

        # Validate and prepare environmental reward vector E_env
        if E_env is None:
            e_arr = np.ones(5, dtype=np.float64)
        elif isinstance(E_env, (int, float)):
            val_e = float(E_env)
            if not math.isfinite(val_e) or val_e < 0.0:
                raise ValueError(f"E_env scalar must be non-negative and finite, got {val_e}")
            e_arr = np.full(5, val_e, dtype=np.float64)
        else:
            e_arr = np.array(E_env, dtype=np.float64)
            if e_arr.shape != (5,):
                raise ValueError(f"E_env must have shape (5,), got {e_arr.shape}")
            if np.any(e_arr < 0.0) or not np.all(np.isfinite(e_arr)):
                raise ValueError("E_env values must be non-negative and finite")

        # 1. Growth rate: G_i = lambda_base + lambda_work * C(t)
        # Identical across all 5 dimensions under scalar load
        G = np.full(5, self._lambda_base + self._lambda_work * float(C), dtype=np.float64)

        # 2. Quenching rate: R_i = eta_gain * a_motor * E_env_i
        R = self._eta_gain * float(a_motor) * e_arr

        # 3. Total transition rate: lambda_i = G_i + R_i
        total_lambda = G + R

        # 4. Asymptotic steady-state target: V_i^infty = G_i / lambda_i
        # Guard against vanishing rates (division by zero when lambda_i < 1e-12)
        valid_mask = total_lambda >= 1e-12
        v_infty = np.zeros(5, dtype=np.float64)
        np.divide(G, total_lambda, out=v_infty, where=valid_mask)

        # 5. Exact analytical exponential integration
        # V_i(t + dt) = V_i(t) * e^(-lambda_i * dt) + V_i^infty * (1 - e^(-lambda_i * dt))
        exp_decay = np.exp(-total_lambda * self._dt)

        # For channels where total_lambda >= 1e-12, update via analytical solution
        # For channels where total_lambda < 1e-12, V_i(t + dt) = V_i(t) is preserved
        next_V = np.where(
            valid_mask,
            self._V * exp_decay + v_infty * (1.0 - exp_decay),
            self._V,
        )

        self._V[:] = next_V
        self._time_s += self._dt
        self._sim_time_ns += int(round(self._dt * 1e9))

        return self.get_snapshot()
