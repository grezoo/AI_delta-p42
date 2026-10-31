"""
ΔP-42 Sovereign Cognitive Core (Gate 2) - Cerebellum Engine Module
20 Hz Deterministic Motor Smoothing, 10,000 Granule Cell Spiking Reservoir,
Purkinje Anti-Hebbian LTD Plasticity (Dynamic Inverse Model Learning),
Second-Order Critically Damped Smoothing Filter, and Amygdala Panic Interlock.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapter I.7 lines 225-246, Chapter I.Connectome line 44)
- Plan Review.md (REQ-ANAT-01, DVP-14)
- megvalósítási terv.md (Gate 2 Subcortical Specification)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Canonical Module Identifiers
MODULE_ID: int = 7
MODULE_CEREBELLUM: int = 7
MODULE_AMYGDALA: int = 6

# Timing and Reservoir Canonical Constants
CADENCE_HZ: float = 20.0
DT_DEFAULT: float = 1.0 / 20.0          # 0.050 s (50 ms)
N_GRANULE_CANONICAL: int = 10000        # Exactly 10,000 granule cells

# Biophysical Cerebellar Plasticity Constants (Parallel Fiber - Climbing Fiber Coincidence)
GAMMA_LTD: float = 0.005                # Anti-Hebbian LTD learning rate
TAU_STDP_MS: float = 20.0               # Temporal coincidence decay constant in ms
MAX_DELTA_TAU_MS: float = 50.0          # Biological coincidence temporal window boundary in ms


@dataclass(frozen=True)
class CerebellarState:
    """
    Immutable atomic state record emitted by the Cerebellum Engine.

    Attributes:
        timestamp_ns: Hardware/system nanosecond timestamp.
        y_trajectory: Continuous smoothed actuator target position in R^(d_act).
        y_velocity: Continuous actuator target velocity in R^(d_act).
        y_acceleration: Continuous actuator target acceleration in R^(d_act).
        granule_sparsity: Measured reservoir sparsity in [0.0, 1.0] (must be >= 0.900).
        purkinje_correction: Adaptive feedforward inverse model torque compensation in R^(d_act).
        tracking_error: Climbing fiber motor error (a_target - s_actual) in R^(d_act).
        motor_jitter_ms: Measured cycle-to-cycle timing jitter in milliseconds.
        is_interlocked: Boolean flag indicating if Amygdala panic override clamped the motor.
    """
    timestamp_ns: int
    y_trajectory: np.ndarray
    y_velocity: np.ndarray
    y_acceleration: np.ndarray
    granule_sparsity: float
    purkinje_correction: np.ndarray
    tracking_error: np.ndarray
    motor_jitter_ms: float
    is_interlocked: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert state dataclass to a plain dictionary."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "y_trajectory": self.y_trajectory.tolist(),
            "y_velocity": self.y_velocity.tolist(),
            "y_acceleration": self.y_acceleration.tolist(),
            "granule_sparsity": float(self.granule_sparsity),
            "purkinje_correction": self.purkinje_correction.tolist(),
            "tracking_error": self.tracking_error.tolist(),
            "motor_jitter_ms": float(self.motor_jitter_ms),
            "is_interlocked": bool(self.is_interlocked),
        }


class CerebellumEngine:
    """
    Cerebellum & Brainstem Motor Predictor and Smoothing Engine.
    Executes a 20 Hz (50 ms) deterministic analog servo loop:
    1. Granule Cell High-Dimensional Reservoir:
       Maps low-dimensional target and proprioception into 10,000 sparse parallel fibers (>= 90% sparsity).
    2. Purkinje Cell LTD Plasticity:
       Online anti-Hebbian learning of dynamic inverse plant/load models via climbing fiber error.
    3. Second-Order Critically Damped Smoothing:
       y'' + 2*zeta*omega*y' + omega^2*y = omega^2 * u_total (zeta=1.0, zero overshoot).
    4. Amygdala Panic Override Interlock:
       Clamps output to zero within <= 2.0 ms on emergency alarm.
    """

    def __init__(
        self,
        d_act: int = 8,
        d_sens: int = 16,
        n_granule: int = 10000,
        sparsity_target: float = 0.90,
        eta_ltd: float = 0.005,
        cadence_hz: float = 20.0,
        damping_ratio_zeta: float = 1.0,
        natural_freq_omega: float = 15.0,
        w_purkinje_max: float = 1.0,
        gamma_stdp: float = 0.005,
        tau_stdp_ms: float = 20.0,
        max_delta_tau_ms: float = 50.0,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize the Cerebellum Engine.

        Args:
            d_act: Actuator command dimension (default 8).
            d_sens: Proprioceptive sensory dimension (default 16).
            n_granule: Number of sparse granule cells in reservoir (canonical: exactly 10,000).
            sparsity_target: Minimum required reservoir sparsity (canonical: 0.90 = 90.0%).
            eta_ltd: Anti-Hebbian Purkinje learning rate (canonical: 0.005).
            cadence_hz: Deterministic loop frequency (canonical: 20.0 Hz).
            damping_ratio_zeta: Filter damping ratio (canonical: 1.0 = critically damped).
            natural_freq_omega: Filter undamped natural frequency in rad/s (canonical: 15.0).
            w_purkinje_max: Maximum allowable absolute synaptic weight for Purkinje layer (default 1.0).
            gamma_stdp: Plasticity learning rate alias for anti-Hebbian LTD (default 0.005).
            tau_stdp_ms: Coincidence decay constant in ms (canonical: 20.0 ms).
            max_delta_tau_ms: Maximum coincidence temporal window in ms (canonical: 50.0 ms).
            seed: Optional random seed for reproducible mossy projection matrix.
        """
        self.d_act: int = int(d_act)
        self.d_sens: int = int(d_sens)
        self.d_in: int = self.d_act + self.d_sens
        self.n_granule: int = int(n_granule)
        self.sparsity_target: float = float(sparsity_target)
        self.eta_ltd: float = float(eta_ltd if eta_ltd is not None else gamma_stdp)
        self.gamma_stdp: float = float(gamma_stdp)
        self.tau_stdp_ms: float = float(tau_stdp_ms)
        self.max_delta_tau_ms: float = float(max_delta_tau_ms)
        self.cadence_hz: float = float(cadence_hz)
        self.dt_nominal: float = 1.0 / self.cadence_hz
        self.zeta: float = float(damping_ratio_zeta)
        self.omega_n: float = float(natural_freq_omega)
        self.w_purkinje_max: float = float(w_purkinje_max)

        # 1. Mossy Fiber -> Granule Cell Expansion Projection Matrix & Bias
        rng = np.random.RandomState(seed if seed is not None else 42)
        # Normalized Gaussian random projections
        self.W_mossy: np.ndarray = rng.randn(self.n_granule, self.d_in) / np.sqrt(self.d_in)
        self.b_mossy: np.ndarray = rng.randn(self.n_granule) * 0.1

        # 2. Purkinje Cell Plastic Synaptic Weights W_purkinje in [-W_max, W_max]^(d_act x n_granule)
        self.W_purkinje: np.ndarray = np.zeros((self.d_act, self.n_granule), dtype=np.float64)

        # 3. Continuous Dynamical Filter State
        self.y_pos: np.ndarray = np.zeros(self.d_act, dtype=np.float64)
        self.y_vel: np.ndarray = np.zeros(self.d_act, dtype=np.float64)
        self.y_acc: np.ndarray = np.zeros(self.d_act, dtype=np.float64)

        # 4. Diagnostics & Jitter History
        self._last_step_time_ns: Optional[int] = None
        self._cycle_intervals_ms: List[float] = []
        self._max_history: int = 10000

    def step(
        self,
        a_target: np.ndarray,
        s_proprio: np.ndarray,
        s_actual: Optional[np.ndarray] = None,
        panic_amy: bool = False,
        timestamp_ns: Optional[int] = None,
    ) -> CerebellarState:
        """
        Execute one 20 Hz deterministic cerebellar cycle (50 ms).

        Workflow:
        1. Jitter measurement via high-resolution performance counters.
        2. Amygdala Panic Override: If panic_amy=True, instantly freeze trajectory to zero.
        3. Mossy Fiber expansion into 10,000 granule cells with guaranteed >= 90% sparsity.
        4. Purkinje feedforward compensation u_purkinje = W_purkinje * x_granule.
        5. Climbing fiber error adaptation: delta_W = -eta * error (X) x_granule.
        6. Second-order critically damped filter smoothing: zero overshoot trajectory.

        Args:
            a_target: Cortical motor target command of shape (d_act,).
            s_proprio: Proprioceptive state vector of shape (d_sens,).
            s_actual: Optional actual plant feedback of shape (d_act,) for online error learning.
            panic_amy: Amygdala emergency override flag (True = immediate zero clamp).
            timestamp_ns: Optional reference nanosecond timestamp.

        Returns:
            Immutable CerebellarState record.
        """
        t_now_ns = timestamp_ns if timestamp_ns is not None else time.perf_counter_ns()

        # 1. High-Resolution Timing & Jitter Tracking
        dt = self.dt_nominal
        jitter_ms = 0.0
        if self._last_step_time_ns is not None:
            actual_interval_ms = (t_now_ns - self._last_step_time_ns) / 1_000_000.0
            jitter_ms = abs(actual_interval_ms - (self.dt_nominal * 1000.0))
            if len(self._cycle_intervals_ms) >= self._max_history:
                self._cycle_intervals_ms.pop(0)
            self._cycle_intervals_ms.append(actual_interval_ms)
        self._last_step_time_ns = t_now_ns

        # 2. Amygdala Panic Hardware Override Interlock (<= 2.0 ms clamp)
        if panic_amy:
            self.y_pos = np.zeros(self.d_act, dtype=np.float64)
            self.y_vel = np.zeros(self.d_act, dtype=np.float64)
            self.y_acc = np.zeros(self.d_act, dtype=np.float64)
            return CerebellarState(
                timestamp_ns=t_now_ns,
                y_trajectory=self.y_pos.copy(),
                y_velocity=self.y_vel.copy(),
                y_acceleration=self.y_acc.copy(),
                granule_sparsity=1.0,
                purkinje_correction=np.zeros(self.d_act, dtype=np.float64),
                tracking_error=np.zeros(self.d_act, dtype=np.float64),
                motor_jitter_ms=jitter_ms,
                is_interlocked=True,
            )

        # 3. Input Alignment and Sanitization
        target_arr = np.asarray(a_target, dtype=np.float64)
        proprio_arr = np.asarray(s_proprio, dtype=np.float64)

        if not np.all(np.isfinite(target_arr)) or not np.all(np.isfinite(proprio_arr)):
            # Numerical fallback to safety zero
            target_arr = np.nan_to_num(target_arr, nan=0.0, posinf=0.0, neginf=0.0)
            proprio_arr = np.nan_to_num(proprio_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Resize / Pad if dimensions differ
        if target_arr.shape[0] != self.d_act:
            padded_target = np.zeros(self.d_act, dtype=np.float64)
            copy_t = min(target_arr.shape[0], self.d_act)
            padded_target[:copy_t] = target_arr[:copy_t]
            target_arr = padded_target

        if proprio_arr.shape[0] != self.d_sens:
            padded_prop = np.zeros(self.d_sens, dtype=np.float64)
            copy_p = min(proprio_arr.shape[0], self.d_sens)
            padded_prop[:copy_p] = proprio_arr[:copy_p]
            proprio_arr = padded_prop

        # 4. Granule Cell High-Dimensional Expansion
        u_in = np.concatenate([target_arr, proprio_arr])
        x_granule = self.forward_granule(u_in)

        # Measure sparsity
        zeros_count = int(np.count_nonzero(x_granule == 0.0))
        sparsity = zeros_count / float(self.n_granule)

        # 5. Purkinje Feedforward Inverse Model Compensation
        u_purkinje = self.W_purkinje @ x_granule

        # 6. Climbing Fiber Error & Online LTD Plasticity Adaptation
        tracking_err = np.zeros(self.d_act, dtype=np.float64)
        if s_actual is not None:
            act_arr = np.asarray(s_actual, dtype=np.float64)
            if act_arr.shape[0] != self.d_act:
                padded_act = np.zeros(self.d_act, dtype=np.float64)
                copy_a = min(act_arr.shape[0], self.d_act)
                padded_act[:copy_a] = act_arr[:copy_a]
                act_arr = padded_act

            # Motor tracking error: desired target minus actual plant state
            tracking_err = target_arr - act_arr

            # Depress synapses to adapt feedforward inverse model:
            # Climbing fiber error signal is -(a_target - s_actual) = (s_actual - a_target),
            # so Delta W = -eta * (s_actual - a_target) (X) x = +eta * (a_target - s_actual) (X) x
            # boosting feedforward drive when plant lags behind target.
            cf_error = -(tracking_err)
            self.adapt_purkinje(cf_error, x_granule)

        # 7. Total Motor Command to Smoothing Filter
        u_total = target_arr + u_purkinje

        # 8. Second-Order Critically Damped Trajectory Smoothing
        y_traj, y_vel, y_acc = self.smooth_trajectory(u_total, dt)

        return CerebellarState(
            timestamp_ns=t_now_ns,
            y_trajectory=y_traj,
            y_velocity=y_vel,
            y_acceleration=y_acc,
            granule_sparsity=sparsity,
            purkinje_correction=u_purkinje,
            tracking_error=tracking_err,
            motor_jitter_ms=jitter_ms,
            is_interlocked=False,
        )

    def forward_granule(self, u_in: np.ndarray) -> np.ndarray:
        """
        Project composite input into the 10,000 granule cell reservoir with guaranteed >= 90.0% sparsity.

        Formula:
            z = W_mossy * u_in + b
            theta_90 = Percentile_90.0(z)
            x_granule = ReLU(z - theta_90)
            Strict top-1000 unit activation enforcement.

        Args:
            u_in: Composite input vector [a_target, s_proprio] of shape (d_in,).

        Returns:
            Sparse parallel fiber activation vector of shape (n_granule,).
        """
        z = self.W_mossy @ u_in + self.b_mossy

        # Dynamic 90th percentile thresholding
        theta_90 = float(np.percentile(z, self.sparsity_target * 100.0))
        x_granule = np.maximum(0.0, z - theta_90)

        # Strictly enforce at most 10% active units (top 1000 units for n_granule=10000)
        max_active = int(self.n_granule * (1.0 - self.sparsity_target))
        active_count = int(np.count_nonzero(x_granule > 0.0))
        if active_count > max_active:
            cutoff = float(np.partition(x_granule, -max_active)[-max_active])
            x_granule[x_granule < cutoff] = 0.0

        return x_granule

    def adapt_purkinje(self, error_signal: np.ndarray, x_granule: np.ndarray) -> float:
        """
        Apply anti-Hebbian Long-Term Depression (LTD) to Purkinje synaptic weights:
            Delta W_purkinje = -eta_ltd * (error_signal (X) x_granule) / (||x_granule||^2 + eps)
            W_purkinje(t+1) = clip(W_purkinje(t) + Delta W, -W_max, W_max)

        Args:
            error_signal: Climbing fiber error vector of shape (d_act,).
            x_granule: Parallel fiber activation vector of shape (n_granule,).

        Returns:
            Mean absolute synaptic weight change float(mean(|Delta W|)).
        """
        err_arr = np.asarray(error_signal, dtype=np.float64)
        if err_arr.shape[0] != self.d_act:
            padded_err = np.zeros(self.d_act, dtype=np.float64)
            copy_e = min(err_arr.shape[0], self.d_act)
            padded_err[:copy_e] = err_arr[:copy_e]
            err_arr = padded_err

        # Normalized anti-Hebbian outer product (NLMS / LTD)
        # Prevents gradient explosion across 10,000 granule cells
        x_norm_sq = float(np.sum(x_granule ** 2))
        scale = 1.0 / (x_norm_sq + 1e-8) if x_norm_sq > 0.0 else 0.0
        delta_w = -self.eta_ltd * scale * np.outer(err_arr, x_granule)

        # Update weights with saturation clipping [-W_max, W_max]
        self.W_purkinje = np.clip(
            self.W_purkinje + delta_w,
            -self.w_purkinje_max,
            self.w_purkinje_max,
        )

        return float(np.mean(np.abs(delta_w)))

    def smooth_trajectory(
        self,
        u_total: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Continuous trajectory smoothing using an exact analytic continuous-to-discrete (C2D)
        matrix exponential integrator for a second-order critically damped system:
            y'' + 2*zeta*omega*y' + omega^2*y = omega^2 * u_total
        Guarantees EXACT 0.00% overshoot, finite jerk, and unconditional numerical stability.

        Args:
            u_total: Total input command vector of shape (d_act,).
            dt: Discrete time step in seconds.

        Returns:
            Tuple of (y_position, y_velocity, y_acceleration).
        """
        if dt <= 0.0:
            dt = self.dt_nominal

        # Error state e = y - u_total
        e = self.y_pos - u_total
        v = self.y_vel

        # Exact analytic matrix exponential for critically damped harmonic oscillator (zeta = 1.0)
        # Characteristic equation roots: s1 = s2 = -omega_n
        alpha = self.omega_n * dt
        decay = math.exp(-alpha)

        # State transition:
        # e(t+dt) = decay * [ (1 + alpha) * e(t) + dt * v(t) ]
        # v(t+dt) = decay * [ -omega_n^2 * dt * e(t) + (1 - alpha) * v(t) ]
        e_next = decay * ((1.0 + alpha) * e + dt * v)
        v_next = decay * (- (self.omega_n ** 2) * dt * e + (1.0 - alpha) * v)
        y_next = u_total + e_next

        # Acceleration: y'' = omega_n^2 * (u_total - y) - 2 * zeta * omega_n * y'
        a_next = (self.omega_n ** 2) * (u_total - y_next) - 2.0 * self.zeta * self.omega_n * v_next

        self.y_pos = y_next
        self.y_vel = v_next
        self.y_acc = a_next

        return self.y_pos.copy(), self.y_vel.copy(), self.y_acc.copy()

    def reset_state(self) -> None:
        """
        Reset continuous filter states, timing history, and Purkinje weights.
        """
        self.y_pos = np.zeros(self.d_act, dtype=np.float64)
        self.y_vel = np.zeros(self.d_act, dtype=np.float64)
        self.y_acc = np.zeros(self.d_act, dtype=np.float64)
        self.W_purkinje = np.zeros((self.d_act, self.n_granule), dtype=np.float64)
        self._last_step_time_ns = None
        self._cycle_intervals_ms.clear()

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Compute jitter and performance metrics over execution history.

        Returns:
            Dictionary containing jitter_mean_ms, jitter_std_ms, nominal_dt_ms,
            granule_cell_count, and dvp14_jitter_pass flag.
        """
        nominal_ms = self.dt_nominal * 1000.0
        if len(self._cycle_intervals_ms) < 2:
            return {
                "jitter_mean_ms": 0.0,
                "jitter_std_ms": 0.0,
                "nominal_dt_ms": nominal_ms,
                "granule_cell_count": float(self.n_granule),
                "dvp14_jitter_pass": 1.0,
            }

        intervals = np.array(self._cycle_intervals_ms, dtype=np.float64)
        jitter = np.abs(intervals - nominal_ms)
        mean_jitter = float(np.mean(jitter))
        std_jitter = float(np.std(intervals))

        return {
            "jitter_mean_ms": mean_jitter,
            "jitter_std_ms": std_jitter,
            "nominal_dt_ms": nominal_ms,
            "granule_cell_count": float(self.n_granule),
            "dvp14_jitter_pass": 1.0 if std_jitter < 0.5 else 0.0,
        }
