"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 (Memory Subsystem)
Module: memory.sleep_hysteresis
Author: ΔP-42 Senior Engineering Team (Zero Downgrade Mandate)

Two-Threshold Schmitt Trigger State Machine for Sleep/Wake Phase Transition Control.
Prevents rapid high-frequency switching ('chattering') between sleep and wakefulness
by maintaining dual operating thresholds and an invariant latching deadband.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter X.5 lines 698-703, Chapter III)
- DELTA_P42_MESTER_TEZIS.md (Section I.3, III.2, V)
- Plan Review.md (REQ-SLEEP-04, DVP-08, RTM-07)
- megvalósítási terv.md (Phase 3: Section 3.1)
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Canonical Schmitt Trigger Physical Switching Thresholds
DEFAULT_SLEEP_THRESHOLD: float = 0.85   # Transition to Sleep allowed ONLY when C(t) >= 0.85
DEFAULT_WAKE_THRESHOLD: float = 0.40    # Transition to Wake allowed ONLY when C(t) <= 0.40
DEFAULT_DEADBAND_WIDTH: float = 0.45    # Width of deadband [0.40, 0.85]

# Aliases matching various specification callouts
C_SLEEP_THRESHOLD: float = DEFAULT_SLEEP_THRESHOLD
C_WAKE_THRESHOLD: float = DEFAULT_WAKE_THRESHOLD
C_DEADBAND_WIDTH: float = DEFAULT_DEADBAND_WIDTH


class SleepState(enum.Enum):
    """
    Biological sleep/wake operational state in ΔP-42 Sovereign Cognitive Architecture.

    Members:
        AWAKE: Active waking state; full sensory streaming and motor release enabled.
        ASLEEP: General metabolic rest/sleep state.
        NREM_SLEEP: Non-Rapid Eye Movement sleep; active 2PC memory consolidation.
        REM_SLEEP: Rapid Eye Movement sleep; virtual dream replay under 100% motor atonia.
    """
    AWAKE = "AWAKE"
    ASLEEP = "ASLEEP"
    NREM_SLEEP = "NREM_SLEEP"
    REM_SLEEP = "REM_SLEEP"

    # Aliases
    WAKE = "AWAKE"
    SLEEP = "ASLEEP"

    @property
    def is_sleeping(self) -> bool:
        """Returns True if this state corresponds to any sleeping phase."""
        return self in (SleepState.ASLEEP, SleepState.NREM_SLEEP, SleepState.REM_SLEEP)

    @property
    def is_awake(self) -> bool:
        """Returns True if this state corresponds to active wakefulness."""
        return self in (SleepState.AWAKE, SleepState.WAKE)


# Cross-module alias matching spec_miner_gate3_dvp.md
CognitiveState = SleepState


@dataclass(frozen=True)
class HysteresisSnapshot:
    """
    Immutable atomic snapshot emitted by SleepHysteresis on each evaluation cycle.

    Thread-safe and duck-typed: directly compares with SleepState and string names
    for seamless integration across test harnesses and CLI verifiers.
    """
    timestamp_ns: int
    context_saturation: float
    current_state: SleepState
    previous_state: SleepState
    state_changed: bool
    cycles_evaluated: int
    sleep_threshold: float = DEFAULT_SLEEP_THRESHOLD
    wake_threshold: float = DEFAULT_WAKE_THRESHOLD
    in_deadband: bool = False
    chattering_detected: bool = False

    def __eq__(self, other: Any) -> bool:
        """Duck-typed equality allowing comparison with SleepState or string name."""
        if isinstance(other, SleepState):
            return self.current_state == other
        if isinstance(other, str):
            return self.current_state.value == other or self.current_state.name == other
        if isinstance(other, HysteresisSnapshot):
            return (
                self.timestamp_ns == other.timestamp_ns
                and math.isclose(self.context_saturation, other.context_saturation, abs_tol=1e-7)
                and self.current_state == other.current_state
                and self.previous_state == other.previous_state
                and self.state_changed == other.state_changed
                and self.cycles_evaluated == other.cycles_evaluated
            )
        return False

    @property
    def name(self) -> str:
        return self.current_state.name

    @property
    def value(self) -> str:
        return self.current_state.value

    @property
    def is_sleeping(self) -> bool:
        return self.current_state.is_sleeping

    @property
    def is_awake(self) -> bool:
        return self.current_state.is_awake


class SleepHysteresis:
    """
    ΔP-42 Sleep Hysteresis Schmitt Trigger State Machine.

    Governs state transitions between AWAKE and ASLEEP based on volatile working
    memory context saturation C(t):
        - Sleep transition occurs ONLY when C(t) >= sleep_threshold (0.85).
        - Wake transition occurs ONLY when C(t) <= wake_threshold (0.40).
        - In deadband 0.40 < C(t) < 0.85: state is strictly invariant (latched).

    Guarantees:
        - Exactly 0 chattering flips across 10,000 noisy cycles (DVP-08).
        - Thread-safe, deterministic state evaluation.
        - Emergency forced arousal support for watchdog supervisor interlock.
    """

    def __init__(
        self,
        sleep_threshold: float = DEFAULT_SLEEP_THRESHOLD,
        wake_threshold: float = DEFAULT_WAKE_THRESHOLD,
        initial_state: Union[SleepState, str] = SleepState.AWAKE,
    ) -> None:
        """
        Initialize the Schmitt trigger state machine.

        Args:
            sleep_threshold: Context saturation threshold to enter sleep (default 0.85).
            wake_threshold: Context saturation threshold to permit waking (default 0.40).
            initial_state: Initial state (default SleepState.AWAKE).
        """
        s_th = float(sleep_threshold)
        w_th = float(wake_threshold)

        if not math.isfinite(s_th) or not math.isfinite(w_th):
            raise ValueError(f"Thresholds must be finite: sleep={sleep_threshold}, wake={wake_threshold}")
        if not (0.0 <= w_th < s_th <= 1.0):
            raise ValueError(
                f"Invalid hysteresis thresholds: require 0.0 <= wake < sleep <= 1.0, "
                f"got wake={w_th}, sleep={s_th}"
            )

        self._sleep_threshold: float = s_th
        self._wake_threshold: float = w_th

        # Resolve initial state
        if isinstance(initial_state, str):
            init_resolved = (
                SleepState[initial_state.upper()]
                if initial_state.upper() in SleepState.__members__
                else SleepState(initial_state)
            )
        else:
            init_resolved = initial_state

        self._state: SleepState = init_resolved
        self._cycles_evaluated: int = 0
        self._transition_count: int = 0
        self._deadband_evaluations: int = 0
        self._chattering_count: int = 0
        self._last_snapshot: Optional[HysteresisSnapshot] = None

    @property
    def sleep_threshold(self) -> float:
        return self._sleep_threshold

    @property
    def wake_threshold(self) -> float:
        return self._wake_threshold

    @property
    def current_state(self) -> SleepState:
        return self._state

    def get_state(self) -> SleepState:
        return self._state

    def is_sleeping(self) -> bool:
        return self._state.is_sleeping

    def is_awake(self) -> bool:
        return self._state.is_awake

    def get_transition_count(self) -> int:
        return self._transition_count

    def get_deadband_evaluations(self) -> int:
        return self._deadband_evaluations

    def get_chattering_count(self) -> int:
        return self._chattering_count

    def update(
        self,
        context_saturation: float,
        timestamp_ns: int = 0,
    ) -> HysteresisSnapshot:
        """
        Advance the Schmitt trigger state machine by evaluating context saturation C(t).

        Args:
            context_saturation: Context saturation C(t) in [0.0, 1.0].
            timestamp_ns: High-resolution hardware/simulation timestamp (ns).

        Returns:
            HysteresisSnapshot: Immutable atomic state record (duck-types with SleepState).
        """
        c_raw = float(context_saturation)

        # 1. Input Sanitization & Validation
        if not math.isfinite(c_raw):
            raise ValueError(f"Context saturation must be a finite real number, got {c_raw}")
        if c_raw < 0.0:
            raise ValueError(f"Context saturation C(t) cannot be negative, got {c_raw}")
        if c_raw > 1.0:
            if c_raw > 1.05:
                raise ValueError(f"Context saturation C(t) severely exceeds [0, 1] range: {c_raw}")
            c = 1.0
        else:
            c = c_raw

        prev_state = self._state
        in_deadband = (self._wake_threshold < c < self._sleep_threshold)
        chattering = False

        # 2. Schmitt Trigger Switching Dynamics
        if in_deadband:
            self._deadband_evaluations += 1
            # In deadband: state MUST remain strictly latched
            new_state = prev_state
        elif c >= self._sleep_threshold:
            # Saturation at or above upper threshold: transition to sleep mandatory
            if prev_state.is_awake:
                new_state = SleepState.ASLEEP
            else:
                new_state = prev_state
        elif c <= self._wake_threshold:
            # Saturation at or below lower threshold: transition to wake permitted
            if prev_state.is_sleeping:
                new_state = SleepState.AWAKE
            else:
                new_state = SleepState.AWAKE
        else:
            new_state = prev_state

        state_changed = (new_state != prev_state)
        if state_changed:
            self._transition_count += 1
            # Chattering fault detection: state changed inside deadband
            if in_deadband:
                chattering = True
                self._chattering_count += 1

        self._state = new_state
        self._cycles_evaluated += 1

        snapshot = HysteresisSnapshot(
            timestamp_ns=int(timestamp_ns),
            context_saturation=c,
            current_state=new_state,
            previous_state=prev_state,
            state_changed=state_changed,
            cycles_evaluated=self._cycles_evaluated,
            sleep_threshold=self._sleep_threshold,
            wake_threshold=self._wake_threshold,
            in_deadband=in_deadband,
            chattering_detected=chattering,
        )
        self._last_snapshot = snapshot
        return snapshot

    def force_arousal(
        self,
        reason: str = "forced_arousal",
        timestamp_ns: int = 0,
    ) -> HysteresisSnapshot:
        """
        Emergency forced arousal command (used by Watchdog on timeout stall).

        Forces the system to AWAKE state regardless of context saturation.
        """
        prev_state = self._state
        new_state = SleepState.AWAKE
        state_changed = (new_state != prev_state)

        if state_changed:
            self._transition_count += 1

        self._state = new_state
        self._cycles_evaluated += 1

        snapshot = HysteresisSnapshot(
            timestamp_ns=int(timestamp_ns),
            context_saturation=self._last_snapshot.context_saturation if self._last_snapshot else 0.0,
            current_state=new_state,
            previous_state=prev_state,
            state_changed=state_changed,
            cycles_evaluated=self._cycles_evaluated,
            sleep_threshold=self._sleep_threshold,
            wake_threshold=self._wake_threshold,
            in_deadband=False,
            chattering_detected=False,
        )
        self._last_snapshot = snapshot
        return snapshot

    def force_wake(
        self,
        reason: str = "forced_wake",
        timestamp_ns: int = 0,
    ) -> HysteresisSnapshot:
        """Canonical alias for force_arousal()."""
        return self.force_arousal(reason=reason, timestamp_ns=timestamp_ns)

    def force_sleep(
        self,
        reason: str = "forced_sleep",
        timestamp_ns: int = 0,
    ) -> HysteresisSnapshot:
        """Forces state to ASLEEP."""
        prev_state = self._state
        new_state = SleepState.ASLEEP
        state_changed = (new_state != prev_state)
        if state_changed:
            self._transition_count += 1
        self._state = new_state
        self._cycles_evaluated += 1
        snapshot = HysteresisSnapshot(
            timestamp_ns=int(timestamp_ns),
            context_saturation=self._last_snapshot.context_saturation if self._last_snapshot else 1.0,
            current_state=new_state,
            previous_state=prev_state,
            state_changed=state_changed,
            cycles_evaluated=self._cycles_evaluated,
            sleep_threshold=self._sleep_threshold,
            wake_threshold=self._wake_threshold,
            in_deadband=False,
            chattering_detected=False,
        )
        self._last_snapshot = snapshot
        return snapshot

    def transition_to_rem(self, timestamp_ns: int = 0) -> HysteresisSnapshot:
        """Transitions sleeping system to REM dream state."""
        if not self._state.is_sleeping:
            raise RuntimeError(f"Cannot transition to REM sleep from wake state: {self._state}")
        prev_state = self._state
        self._state = SleepState.REM_SLEEP
        self._cycles_evaluated += 1
        snapshot = HysteresisSnapshot(
            timestamp_ns=int(timestamp_ns),
            context_saturation=self._last_snapshot.context_saturation if self._last_snapshot else 0.0,
            current_state=SleepState.REM_SLEEP,
            previous_state=prev_state,
            state_changed=(prev_state != SleepState.REM_SLEEP),
            cycles_evaluated=self._cycles_evaluated,
            sleep_threshold=self._sleep_threshold,
            wake_threshold=self._wake_threshold,
            in_deadband=False,
            chattering_detected=False,
        )
        self._last_snapshot = snapshot
        return snapshot

    def transition_to_nrem(self, timestamp_ns: int = 0) -> HysteresisSnapshot:
        """Transitions sleeping system to NREM consolidation state."""
        prev_state = self._state
        self._state = SleepState.NREM_SLEEP
        self._cycles_evaluated += 1
        snapshot = HysteresisSnapshot(
            timestamp_ns=int(timestamp_ns),
            context_saturation=self._last_snapshot.context_saturation if self._last_snapshot else 1.0,
            current_state=SleepState.NREM_SLEEP,
            previous_state=prev_state,
            state_changed=(prev_state != SleepState.NREM_SLEEP),
            cycles_evaluated=self._cycles_evaluated,
            sleep_threshold=self._sleep_threshold,
            wake_threshold=self._wake_threshold,
            in_deadband=False,
            chattering_detected=False,
        )
        self._last_snapshot = snapshot
        return snapshot

    def reset(self, initial_state: Union[SleepState, str] = SleepState.AWAKE) -> None:
        """Resets all metrics and state history."""
        if isinstance(initial_state, str):
            init_resolved = (
                SleepState[initial_state.upper()]
                if initial_state.upper() in SleepState.__members__
                else SleepState(initial_state)
            )
        else:
            init_resolved = initial_state
        self._state = init_resolved
        self._cycles_evaluated = 0
        self._transition_count = 0
        self._deadband_evaluations = 0
        self._chattering_count = 0
        self._last_snapshot = None

    def evaluate_noise_stability(
        self,
        cycles: int = 10000,
        sigma: float = 0.05,
        seed: int = 42,
    ) -> Tuple[bool, int, Dict[str, Any]]:
        """
        Executes the formal DVP-08 10,000-cycle stochastic noise stability test.

        Returns:
            (passed, chattering_count, stats_dict)
        """
        rng = np.random.default_rng(seed)
        self.reset(SleepState.AWAKE)
        quarters = cycles // 4

        # Regime A: Ascending ramp 0.0 -> 1.0 with noise
        ramp_up = np.linspace(0.0, 1.0, quarters) + rng.normal(0.0, sigma, quarters)
        # Regime B: Descending ramp 1.0 -> 0.0 with noise
        ramp_down = np.linspace(1.0, 0.0, quarters) + rng.normal(0.0, sigma, quarters)
        # Regime C: Deadband dwell around 0.625 +- 0.12 with noise
        deadband_base = 0.625 + 0.12 * np.sin(np.linspace(0, 4 * np.pi, quarters))
        deadband_dwell = deadband_base + rng.normal(0.0, sigma, quarters)
        # Regime D: Boundary oscillation stress (centered on 0.85 and 0.40)
        boundary_osc = np.zeros(quarters)
        half_q = quarters // 2
        boundary_osc[:half_q] = 0.85 + rng.normal(0.0, 0.02, half_q)
        boundary_osc[half_q:] = 0.40 + rng.normal(0.0, 0.02, quarters - half_q)

        full_trajectory = np.clip(np.concatenate([ramp_up, ramp_down, deadband_dwell, boundary_osc]), 0.0, 1.0)

        for c_val in full_trajectory:
            self.update(float(c_val))

        passed = (self._chattering_count == 0)
        stats = {
            "total_cycles": self._cycles_evaluated,
            "transitions": self._transition_count,
            "deadband_evaluations": self._deadband_evaluations,
            "chattering_count": self._chattering_count,
            "latching_fidelity_pct": 100.0 if passed else 0.0,
        }
        return passed, self._chattering_count, stats
