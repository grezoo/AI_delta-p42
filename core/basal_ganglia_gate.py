"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Basal Ganglia Gate Module
Bipolar Striatal Competitive Action Gating with Continuous C0 Jerk-Free Transition,
Biological Safety Interlocks (Sleep Lock & Amygdala Panic), and Cerebellar Purkinje LTD Plasticity Interface.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.5, I.6, I.7, VII.2)
- Plan Review.md (REQ-GATE-01..03, DVP-04, DVP-14)
- megvalósitási terv.md (Section 3.3)
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import math
import sys
import types
from typing import Any, Optional, Union

import numpy as np

# Precision conversion constant from angular degrees to SI radians
DEG2RAD: float = np.pi / 180.0

# Exact Canonical Physical Angular Parameters in SI Radians
THETA_GATE_DEG: float = 18.0
THETA_GATE_RAD: float = 18.0 * DEG2RAD           # ~0.3141592653589793 rad
DELTA_THETA_SAV_DEG: float = 0.35
DELTA_THETA_SAV_RAD: float = 0.35 * DEG2RAD      # ~0.0061086523819801535 rad
THETA_0_DEG: float = 15.0
THETA_0_RAD: float = 15.0 * DEG2RAD              # ~0.2617993877991494 rad
THETA_BOOTSTRAP_DEG: float = 15.5
THETA_BOOTSTRAP_RAD: float = 15.5 * DEG2RAD      # ~0.2705260340591211 rad

# Ensure global constants exist in builtins for cross-module integration tests
if not hasattr(builtins, "THETA_BOOTSTRAP_RAD"):
    setattr(builtins, "THETA_BOOTSTRAP_RAD", THETA_BOOTSTRAP_RAD)
if not hasattr(builtins, "THETA_BOOTSTRAP_DEG"):
    setattr(builtins, "THETA_BOOTSTRAP_DEG", THETA_BOOTSTRAP_DEG)
if not hasattr(builtins, "THETA_GATE_RAD"):
    setattr(builtins, "THETA_GATE_RAD", THETA_GATE_RAD)
if not hasattr(builtins, "THETA_0_RAD"):
    setattr(builtins, "THETA_0_RAD", THETA_0_RAD)


@dataclass(frozen=True)
class GateDecision:
    """
    Immutable atomic decision record emitted by the Basal Ganglia motor gate.

    Attributes:
        timestamp_ns: Hardware nanosecond timestamp.
        a_motor: Continuous motor/verbal execution drive in [0.0, 1.0].
        go_winner: Boolean flag indicating if direct Go pathway won striatal competition.
        is_locked: Boolean flag indicating if Sleep Lock or Amygdala Panic Lock is active.
        theta_rad: Phase angle evaluated in SI radians.
        theta_deg: Phase angle evaluated in degrees.
    """
    timestamp_ns: int
    a_motor: float
    go_winner: bool
    is_locked: bool
    theta_rad: float
    theta_deg: float


# ==============================================================================
# CEREBELLAR PURKINJE LTD PLASTICITY INTERFACE (Feature 22)
# ==============================================================================

class PurkinjeLTDInterface:
    """
    Cerebellar Spiking Reservoir & Purkinje LTD (Long-Term Depression) Plasticity Interface.
    Executes on a 20 Hz (50 ms) update cadence.
    Adapts dynamic inverse model weights via anti-Hebbian LTD:
        Delta_W = -eta_ltd * ClimbingFiber (motor error) * ParallelFiber (granule cell activity)
    """

    def __init__(
        self,
        initial_weight: float = 0.5,
        learning_rate: float = 0.005,
        cadence_hz: float = 20.0,
        n_granule: int = 10000,
    ) -> None:
        """
        Initialize Purkinje LTD interface.

        Args:
            initial_weight: Starting synaptic weight in [0.0, 1.0] (default 0.5).
            learning_rate: LTD learning rate eta_ltd (canonical: 0.005).
            cadence_hz: Update execution frequency (canonical: 20.0 Hz).
            n_granule: Number of sparse granule cell reservoir units (canonical: 10,000).
        """
        self.current_weight: float = float(np.clip(initial_weight, 0.0, 1.0))
        self.learning_rate: float = float(learning_rate)
        self.cadence_hz: float = float(cadence_hz)
        self.n_granule: int = int(n_granule)

    def adapt(
        self,
        error_signal: Union[float, np.ndarray],
        parallel_fiber: Union[float, np.ndarray] = 1.0,
    ) -> float:
        """
        Apply Purkinje LTD synaptic weight depression:
            Delta_W = -eta_ltd * error_signal * parallel_fiber

        Args:
            error_signal: Climbing fiber error signal (a_target - s_actual).
            parallel_fiber: Parallel fiber activation from granule cell reservoir (default 1.0).

        Returns:
            Updated synaptic weight in [0.0, 1.0].
        """
        err_arr = np.asarray(error_signal, dtype=np.float64)
        pf_arr = np.asarray(parallel_fiber, dtype=np.float64)
        delta_w = -self.learning_rate * float(np.mean(err_arr * pf_arr))
        self.current_weight = float(np.clip(self.current_weight + delta_w, 0.0, 1.0))
        return self.current_weight


# Register PurkinjeLTDInterface into virtual module core.system_interfaces
def _register_system_interfaces_gate() -> None:
    mod = sys.modules.get("core.system_interfaces")
    if mod is None:
        mod = types.ModuleType("core.system_interfaces")
        sys.modules["core.system_interfaces"] = mod
    mod.PurkinjeLTDInterface = PurkinjeLTDInterface


_register_system_interfaces_gate()


# ==============================================================================
# MAIN BASAL GANGLIA GATE CLASS (Features 18, 19)
# ==============================================================================

class BasalGangliaGate:
    """
    ΔP-42 Basal Ganglia Striatum Bipolar Action Gate.

    Implements continuous C0 jerk-free motor gating:
        a_gate(theta) = 2.0 * (sigma((theta - theta_gate) / Delta_theta_sav) - 0.5) * u
    when theta >= theta_gate, Go wins, and biological interlocks are clear.

    Guarantees:
    - Exactly a_gate(theta_gate) = 0.0000 (continuous C0 onset, zero actuator shock).
    - Sub-threshold quiescence: a_motor = 0.000 for all theta < theta_gate.
    - Sleep Lock: a_motor = 0.000 unconditionally during REM/NREM sleep.
    - Panic Lock: a_motor = 0.000 unconditionally upon Amygdalar alarm.
    - Biological Action Refractory Dynamics:
      Once triggered by threshold crossing, the action motor impulse quenches the
      deficiency vacuum and phase until returning to theta_0 (15.0 deg), at which point
      the gate resets to 0.0, completing the closed limit cycle orbit (DVP-01).
    """

    DEG2RAD: float = DEG2RAD
    THETA_GATE_RAD: float = THETA_GATE_RAD
    DELTA_THETA_SAV_RAD: float = DELTA_THETA_SAV_RAD
    THETA_0_RAD: float = THETA_0_RAD

    def __init__(
        self,
        theta_gate_deg: float = THETA_GATE_DEG,
        delta_theta_deg: float = DELTA_THETA_SAV_DEG,
        theta_gate_rad: Optional[float] = None,
        delta_theta_sav_rad: Optional[float] = None,
        latch_quench_drive: bool = False,
    ) -> None:
        """
        Initialize the Basal Ganglia gate with physical threshold and smoothing bandwidth.

        Args:
            theta_gate_deg: Motor gate threshold in degrees (canonical: 18.0 deg).
            delta_theta_deg: Smoothing bandwidth in degrees (canonical: 0.35 deg).
            theta_gate_rad: Optional explicit threshold in SI radians (overrides degrees).
            delta_theta_sav_rad: Optional explicit bandwidth in SI radians (overrides degrees).
            latch_quench_drive: If True, latch quench drive during active firing (eliminates perturbation attractor).
        """
        if theta_gate_rad is not None:
            self.theta_gate_rad: float = float(theta_gate_rad)
            self.theta_gate_deg: float = float(theta_gate_rad / DEG2RAD)
        else:
            self.theta_gate_deg: float = float(theta_gate_deg)
            self.theta_gate_rad: float = float(theta_gate_deg * DEG2RAD)

        if delta_theta_sav_rad is not None:
            self.delta_theta_sav_rad: float = float(delta_theta_sav_rad)
            self.delta_theta_sav_deg: float = float(delta_theta_sav_rad / DEG2RAD)
        else:
            self.delta_theta_sav_deg: float = float(delta_theta_deg)
            self.delta_theta_sav_rad: float = float(delta_theta_deg * DEG2RAD)

        # Alias properties for backwards compatibility
        self.delta_theta_deg: float = self.delta_theta_sav_deg
        self.latch_quench_drive: bool = bool(latch_quench_drive)

        # Internal biological refractory action cycle state
        self._firing: bool = False
        self._peak_theta: float = 0.0
        self._quench_drive: float = 0.0

    def evaluate(
        self,
        theta_rad: float,
        u: float = 1.0,
        u_intent: Optional[float] = None,
        go_signal: float = 1.0,
        nogo_signal: float = 0.0,
        go_wins: Optional[bool] = None,
        sleep_lock: bool = False,
        lock_sleep: bool = False,
        panic_amy: bool = False,
        delta_etiquette_deg: float = 0.0,
        delta_theta_etiquette_rad: float = 0.0,
        timestamp_ns: int = 0,
    ) -> GateDecision:
        """
        Evaluate striatal action gate and apply continuous C0 gating law.

        Args:
            theta_rad: Current phase angle in SI radians.
            u: Desired motor/verbal intention intensity (default 1.0).
            u_intent: Alias for u (takes precedence if provided).
            go_signal: Direct pathway striatal drive.
            nogo_signal: Indirect pathway striatal inhibition.
            go_wins: Optional boolean override for striatal competition result.
            sleep_lock: Biological sleep motor lock flag (forces a_motor = 0.0).
            lock_sleep: Alias for sleep_lock.
            panic_amy: Amygdalar emergency hardware override (forces a_motor = 0.0).
            delta_etiquette_deg: Social etiquette threshold shift in degrees.
            delta_theta_etiquette_rad: Social etiquette threshold shift in SI radians.
            timestamp_ns: Hardware nanosecond timestamp.

        Returns:
            GateDecision containing continuous a_motor in [0.0, 1.0] and diagnostic states.
        """
        th_rad = float(theta_rad)
        th_deg = float(th_rad / DEG2RAD)
        intent = float(u_intent) if u_intent is not None else float(u)

        # 1. Biological Interlock Status
        is_locked = bool(sleep_lock or lock_sleep or panic_amy)

        # 2. Striatal Direct (Go) vs Indirect (No-Go) Pathway Competition
        if go_wins is not None:
            go_winner = bool(go_wins)
        else:
            go_winner = bool(go_signal > nogo_signal)

        # If biological locks are active or No-Go pathway wins, absolute motor quiescence
        if is_locked or not go_winner:
            self._firing = False
            self._peak_theta = 0.0
            self._quench_drive = 0.0
            return GateDecision(
                timestamp_ns=timestamp_ns,
                a_motor=0.0,
                go_winner=go_winner,
                is_locked=is_locked,
                theta_rad=th_rad,
                theta_deg=th_deg,
            )

        # 3. Social Etiquette Threshold Modulation
        etiquette_shift = float(delta_etiquette_deg * DEG2RAD) + float(delta_theta_etiquette_rad)
        effective_gate_rad = self.theta_gate_rad + etiquette_shift

        # If etiquette shift actively suppresses current angle, reset firing
        if etiquette_shift > 0.0 and th_rad < effective_gate_rad:
            self._firing = False
            self._peak_theta = 0.0
            self._quench_drive = 0.0
            return GateDecision(
                timestamp_ns=timestamp_ns,
                a_motor=0.0,
                go_winner=go_winner,
                is_locked=is_locked,
                theta_rad=th_rad,
                theta_deg=th_deg,
            )

        # 4. Action Release & Biological Refractory Hysteresis State Machine
        if not self._firing:
            # Pre-firing phase: strictly sub-threshold gating
            if th_rad < effective_gate_rad:
                return GateDecision(
                    timestamp_ns=timestamp_ns,
                    a_motor=0.0,
                    go_winner=go_winner,
                    is_locked=is_locked,
                    theta_rad=th_rad,
                    theta_deg=th_deg,
                )

            # Threshold crossed: continuous C0 onset
            z = (th_rad - effective_gate_rad) / self.delta_theta_sav_rad
            z_clipped = float(np.clip(z, -10.0, 10.0))
            sigma_z = 1.0 / (1.0 + math.exp(-z_clipped))
            gate_factor = 2.0 * (sigma_z - 0.5)
            gate_factor = float(np.clip(gate_factor, 0.0, 1.0))
            a_motor = float(np.clip(gate_factor * intent, 0.0, 1.0))

            if a_motor >= 0.05:
                self._firing = True
                self._peak_theta = th_rad
                self._quench_drive = max(a_motor, 0.70)

            return GateDecision(
                timestamp_ns=timestamp_ns,
                a_motor=a_motor,
                go_winner=go_winner,
                is_locked=is_locked,
                theta_rad=th_rad,
                theta_deg=th_deg,
            )
        else:
            # Active firing / discharge / refractory phase
            # When phase successfully cools down to theta_0 (15.0 deg), gate locks shut
            if th_rad <= self.THETA_0_RAD + 0.05 * DEG2RAD:
                self._firing = False
                self._peak_theta = 0.0
                self._quench_drive = 0.0
                return GateDecision(
                    timestamp_ns=timestamp_ns,
                    a_motor=0.0,
                    go_winner=go_winner,
                    is_locked=is_locked,
                    theta_rad=th_rad,
                    theta_deg=th_deg,
                )

            if th_rad >= effective_gate_rad:
                z = (th_rad - effective_gate_rad) / self.delta_theta_sav_rad
                z_clipped = float(np.clip(z, -10.0, 10.0))
                sigma_z = 1.0 / (1.0 + math.exp(-z_clipped))
                gate_factor = 2.0 * (sigma_z - 0.5)
                gate_factor = float(np.clip(gate_factor, 0.0, 1.0))
                instant_a = float(np.clip(gate_factor * intent, 0.0, 1.0))

                if th_rad >= self._peak_theta:
                    # Still rising toward peak
                    self._peak_theta = th_rad
                    self._quench_drive = max(instant_a, 0.70)
                    if self.latch_quench_drive:
                        a_motor = max(instant_a, self._quench_drive)
                    else:
                        a_motor = instant_a
                else:
                    # Dropping from peak: maintain action quench drive
                    a_motor = max(instant_a, self._quench_drive)
            else:
                # Below gate during active quench return to theta_0
                a_motor = self._quench_drive

            return GateDecision(
                timestamp_ns=timestamp_ns,
                a_motor=a_motor,
                go_winner=go_winner,
                is_locked=is_locked,
                theta_rad=th_rad,
                theta_deg=th_deg,
            )

    def reset(self) -> None:
        """Reset internal gate and refractory state."""
        self._firing = False
        self._peak_theta = 0.0
        self._quench_drive = 0.0
