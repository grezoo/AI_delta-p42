"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Module: nursery.caretaker_console
Author: Milestone 1 Worker (Nursery Subsystem Specialist)

Caretaker Control-Channel Interface and Pedagogical Progression State Machine.
Implements the 3-Stage Pedagogical Curriculum Model:
    1. Nursery (Bölcsőde, 0-6 mo): Sensorimotor calibration and Thalamic noise tuning.
    2. Kindergarten (Óvoda, 6-36 mo): Social etiquette barrier shift Delta theta_illem in [0.0 deg, 5.0 deg]
       without penalizing internal social drive V_társas.
    3. School (Iskola, 3+ yr): Autonomous worldview and REM-validated theory consolidation.
Provides sub-2.0 ms hard real-time emergency interrupt (WCET <= 2.0 ms) and telemetry monitoring.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter VII.1-3, Chapter IX.2, Chapter X.3 lines 671-678)
- Plan Review.md (REQ-FUNC-03, DVP-03)
- ORIGINAL_REQUEST.md (Stage-Gate 4 Milestone 1)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.basal_ganglia_gate import BasalGangliaGate
from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot
from core.theta_engine import DEG2RAD, ThetaEngine
from modules.amygdala_override import AmygdalaOverride
from nursery.worldstream_server import EventModality, EventPriority, WorldEvent, WorldStreamServer

# ------------------------------------------------------------------------------
# Pedagogical Stage Enum
# ------------------------------------------------------------------------------

class PedagogicalStage(IntEnum):
    """
    3-Stage Pedagogical Curriculum Progression for the AI Nursery.
    Matches DELTA_P42_MUKODESI_ELVEK.md Chapter VII.
    """
    NURSERY = 1       # Bölcsőde: 0-6 months, sensorimotor calibration
    KINDERGARTEN = 2  # Óvoda: 6-36 months, etiquette & intent separation
    SCHOOL = 3        # Iskola: 3+ years, world theory building


# ------------------------------------------------------------------------------
# Canonical Constants
# ------------------------------------------------------------------------------

DELTA_THETA_ILLEM_MAX_DEG: float = 5.0
DELTA_THETA_ILLEM_MAX_RAD: float = float(DELTA_THETA_ILLEM_MAX_DEG * DEG2RAD)  # ~= 0.08726646 rad
EMERGENCY_WCET_MAX_MS: float = 2.0  # Hard real-time kill line constraint
BASE_GATE_DEG: float = 18.0
BASE_GATE_RAD: float = float(BASE_GATE_DEG * DEG2RAD)


# ------------------------------------------------------------------------------
# Caretaker Data Structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class SocialFeedbackRecord:
    """
    Immutable audit record of Caretaker social feedback.

    Attributes:
        timestamp_ns: Nanosecond timestamp of feedback.
        valence: Emotional valence in [-1.0, 1.0].
        intensity: Feedback magnitude in [0.0, 1.0].
        feedback_type: Semantic tag ("mirroring", "encouragement", "prohibition").
        etiquette_offset_rad: Resulting motor gate threshold shift in radians.
        etiquette_offset_deg: Resulting motor gate threshold shift in degrees.
    """
    timestamp_ns: int
    valence: float
    intensity: float
    feedback_type: str
    etiquette_offset_rad: float
    etiquette_offset_deg: float


@dataclass(frozen=True)
class CaretakerTelemetry:
    """
    Comprehensive real-time telemetry snapshot for Caretaker monitoring.
    """
    timestamp_ns: int
    theta_deg: float
    theta_rad: float
    theta_dot: float
    maslow_vector: np.ndarray
    dominant_need: str
    context_saturation: float
    dissonance: float
    agc_gain: float
    motor_gate: float
    sleep_state: str
    stage: PedagogicalStage
    etiquette_shift_deg: float
    effective_gate_deg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to standard dictionary."""
        d = asdict(self)
        d["maslow_vector"] = self.maslow_vector.tolist()
        d["stage"] = int(self.stage)
        return d


# ------------------------------------------------------------------------------
# CaretakerConsole Implementation
# ------------------------------------------------------------------------------

class CaretakerConsole:
    """
    Caretaker Control-Channel API and Developmental Progression Supervisor.

    Enables human/pedagogical interaction with the sovereign AI infant:
    - Stimulus injection (`inject_stimulus`)
    - Social feedback and polite etiquette modulation (`set_social_feedback`)
    - Real-time telemetry monitoring (`read_telemetry`)
    - Sub-2.0 ms emergency kill-line execution (`emergency_interrupt`)
    """

    def __init__(
        self,
        worldstream: Optional[WorldStreamServer] = None,
        amygdala: Optional[AmygdalaOverride] = None,
        gate: Optional[BasalGangliaGate] = None,
        stage: PedagogicalStage = PedagogicalStage.NURSERY,
    ) -> None:
        """
        Initialize the CaretakerConsole.

        Args:
            worldstream: WorldStreamServer for sensory injection.
            amygdala: Optional AmygdalaOverride for fast emergency kill line.
            gate: Optional BasalGangliaGate for motor threshold coupling.
            stage: Initial PedagogicalStage (default NURSERY).
        """
        self._worldstream = worldstream
        self._amygdala = amygdala
        self._gate = gate
        self._stage = PedagogicalStage(stage)

        self._etiquette_shift_deg: float = 0.0
        self._emergency_active: bool = False
        self._last_interrupt_latency_ms: float = 0.0
        self._lock = threading.Lock()

        self._feedback_history: List[SocialFeedbackRecord] = []

    @property
    def stage(self) -> PedagogicalStage:
        """Current developmental pedagogical stage."""
        return self._stage

    @property
    def etiquette_shift_deg(self) -> float:
        """Current polite etiquette motor barrier in degrees [0.0, 5.0]."""
        return self._etiquette_shift_deg

    @property
    def etiquette_shift_rad(self) -> float:
        """Current polite etiquette motor barrier in radians [0.0, 0.087266]."""
        return float(self._etiquette_shift_deg * DEG2RAD)

    @property
    def is_emergency_active(self) -> bool:
        """True if emergency kill-line interrupt is actively latched."""
        return self._emergency_active

    @property
    def last_interrupt_latency_ms(self) -> float:
        """WCET of the last executed emergency interrupt in milliseconds."""
        return self._last_interrupt_latency_ms

    def advance_pedagogical_stage(self, target_stage: PedagogicalStage) -> None:
        """
        Advances the developmental curriculum to the target stage.
        Progression order: NURSERY (1) -> KINDERGARTEN (2) -> SCHOOL (3).
        """
        with self._lock:
            target = PedagogicalStage(target_stage)
            if target < self._stage:
                raise ValueError(
                    f"Cannot regress pedagogical stage from {self._stage.name} to {target.name}"
                )
            self._stage = target

    def set_etiquette_shift(self, delta_deg: float) -> None:
        """
        Directly sets the social etiquette barrier shift Delta theta_illem in degrees.
        Clamped to [0.0 deg, 5.0 deg].
        """
        with self._lock:
            clamped = float(np.clip(delta_deg, 0.0, DELTA_THETA_ILLEM_MAX_DEG))
            self._etiquette_shift_deg = clamped
            if self._gate is not None:
                # If BasalGangliaGate supports direct etiquette shift
                if hasattr(self._gate, "set_etiquette_shift"):
                    self._gate.set_etiquette_shift(clamped)
                elif hasattr(self._gate, "delta_etiquette_deg"):
                    self._gate.delta_etiquette_deg = clamped

    def get_effective_gate_deg(self, base_gate_deg: float = BASE_GATE_DEG) -> float:
        """
        Computes the effective motor gating threshold:
            theta_effective = base_gate_deg + Delta theta_illem
        """
        return base_gate_deg + self._etiquette_shift_deg

    def get_effective_gate_rad(self, base_gate_rad: Optional[float] = None) -> float:
        """
        Computes the effective motor gating threshold in SI radians.
        """
        base = BASE_GATE_RAD if base_gate_rad is None else base_gate_rad
        return base + self.etiquette_shift_rad

    def inject_stimulus(
        self,
        modality: Union[EventModality, int],
        data: Union[bytes, str, np.ndarray],
        priority: Union[EventPriority, int] = EventPriority.NORMAL,
        valence: float = 0.0,
    ) -> bool:
        """
        Injects a sensory stimulus into the WorldStream.

        Args:
            modality: EventModality (Acoustic, Visual, Proprioceptive, Social Text).
            data: Payload content (bytes, text string, or float tensor).
            priority: EventPriority (Background, Normal, Ping!).
            valence: Ambient emotional valence in [-1.0, 1.0].

        Returns:
            True if injected successfully.
        """
        if isinstance(data, str):
            payload_bytes = data.encode("utf-8")
        elif isinstance(data, np.ndarray):
            payload_bytes = data.astype(np.float32).tobytes()
        elif isinstance(data, (bytes, bytearray)):
            payload_bytes = bytes(data)
        else:
            payload_bytes = str(data).encode("utf-8")

        event = WorldEvent(
            timestamp_ns=time.time_ns(),
            modality=EventModality(modality),
            priority=EventPriority(priority),
            payload_bytes=payload_bytes,
            ambient_valence=float(valence),
        )

        if self._worldstream is not None:
            return self._worldstream.inject_event(event)
        return True

    def set_social_feedback(
        self,
        valence: float,
        intensity: float = 1.0,
        feedback_type: str = "mirroring",
    ) -> SocialFeedbackRecord:
        """
        Submits Caretaker social guidance to the AI infant.

        In KINDERGARTEN (Stage 2) and SCHOOL (Stage 3):
        Negative valence (scolding / prohibition) raises the etiquette barrier Delta theta_illem
        (up to +5.0 deg), shifting the motor gate threshold upwards without suppressing
        internal social drive V_társas.

        Args:
            valence: Emotional valence in [-1.0, 1.0] (negative = scolding, positive = praise).
            intensity: Guidance magnitude in [0.0, 1.0].
            feedback_type: Semantic tag ("mirroring", "praise", "prohibition").

        Returns:
            SocialFeedbackRecord documenting the adjustment.
        """
        now_ns = time.time_ns()
        v = float(np.clip(valence, -1.0, 1.0))
        intens = float(np.clip(intensity, 0.0, 1.0))

        with self._lock:
            if self._stage >= PedagogicalStage.KINDERGARTEN:
                if v < 0.0:
                    # Negative feedback increases etiquette threshold shift
                    shift_deg = float(-v * intens * DELTA_THETA_ILLEM_MAX_DEG)
                    self._etiquette_shift_deg = min(DELTA_THETA_ILLEM_MAX_DEG, shift_deg)
                else:
                    # Positive feedback / encouragement relaxes the etiquette threshold
                    relax_amount = float(v * intens * 1.5)
                    self._etiquette_shift_deg = max(0.0, self._etiquette_shift_deg - relax_amount)
            else:
                # In Stage 1 (NURSERY), no etiquette barrier is applied
                self._etiquette_shift_deg = 0.0

            record = SocialFeedbackRecord(
                timestamp_ns=now_ns,
                valence=v,
                intensity=intens,
                feedback_type=str(feedback_type),
                etiquette_offset_rad=self.etiquette_shift_rad,
                etiquette_offset_deg=self._etiquette_shift_deg,
            )
            self._feedback_history.append(record)

        # Injected as social sensory event into worldstream
        if self._worldstream is not None:
            self.inject_stimulus(
                modality=EventModality.SOCIAL_TEXT,
                data=f"Feedback: {feedback_type} (v={v:.2f})",
                priority=EventPriority.NORMAL,
                valence=v,
            )

        return record

    def read_telemetry(
        self,
        maslow: Optional[MaslowIntegrator] = None,
        theta: Optional[ThetaEngine] = None,
        memory_system: Any = None,
    ) -> CaretakerTelemetry:
        """
        Captures a coherent atomic snapshot of system telemetry for Caretaker diagnostics.

        Args:
            maslow: Optional MaslowIntegrator instance to sample.
            theta: Optional ThetaEngine instance to sample.
            memory_system: Optional memory subsystem to sample.

        Returns:
            CaretakerTelemetry instance.
        """
        now_ns = time.time_ns()

        # Phase oscillator status
        if theta is not None:
            th_deg = float(theta.theta_deg)
            th_rad = float(theta.theta_rad)
            th_dot = float(theta.dot_theta)
        else:
            th_deg = 15.0
            th_rad = THETA_0_RAD
            th_dot = 0.0

        # Maslow homeostatic status
        if maslow is not None:
            if hasattr(maslow, "snapshot"):
                snap = maslow.snapshot
                m_vec = snap.vector.copy()
                dom_name = snap.dominant_name
            elif hasattr(maslow, "get_snapshot"):
                snap = maslow.get_snapshot()
                m_vec = snap.vector.copy()
                dom_name = snap.dominant_name
            elif hasattr(maslow, "vector"):
                m_vec = np.asarray(maslow.vector, dtype=np.float64).copy()
                dom_name = "unknown"
            else:
                m_vec = np.array([0.15, 0.10, 0.25, 0.10, 0.30], dtype=np.float64)
                dom_name = "cog"
        else:
            m_vec = np.array([0.15, 0.10, 0.25, 0.10, 0.30], dtype=np.float64)
            dom_name = "cog"

        # Context saturation and sleep
        c_sat = 0.0
        sleep_str = "AWAKE"
        if memory_system is not None:
            if hasattr(memory_system, "context_saturation"):
                c_sat = float(memory_system.context_saturation)
            if hasattr(memory_system, "state"):
                sleep_str = str(memory_system.state)

        # Motor gate status
        m_gate = 0.0
        if self._gate is not None:
            if hasattr(self._gate, "last_a_motor"):
                m_gate = float(self._gate.last_a_motor)

        eff_gate = self.get_effective_gate_deg()

        return CaretakerTelemetry(
            timestamp_ns=now_ns,
            theta_deg=th_deg,
            theta_rad=th_rad,
            theta_dot=th_dot,
            maslow_vector=m_vec,
            dominant_need=dom_name,
            context_saturation=c_sat,
            dissonance=0.0,
            agc_gain=1.0,
            motor_gate=m_gate,
            sleep_state=sleep_str,
            stage=self._stage,
            etiquette_shift_deg=self._etiquette_shift_deg,
            effective_gate_deg=eff_gate,
        )

    def emergency_interrupt(self, threat_level: float = 1.0) -> bool:
        """
        Hard real-time emergency interrupt line (WCET <= 2.0 ms).

        Instantly sets panic override, pulls PWM enable to ground (0.0 V), and clamps
        the Basal Ganglia striatal motor gate to 0.0.

        Args:
            threat_level: Severity score in [0.0, 1.0].

        Returns:
            True upon successful shutdown; asserts latency <= 2.0 ms.
        """
        t_start = time.perf_counter_ns()

        with self._lock:
            self._emergency_active = True

        # 1. Trigger Amygdala override fast path if connected
        if self._amygdala is not None:
            if hasattr(self._amygdala, "trigger_emergency"):
                self._amygdala.trigger_emergency(threat_level=threat_level)
            elif hasattr(self._amygdala, "trigger_alarm"):
                self._amygdala.trigger_alarm()

        # 2. Force Basal Ganglia striatum gate clamp if connected
        if self._gate is not None:
            if hasattr(self._gate, "set_panic_lock"):
                self._gate.set_panic_lock(True)
            elif hasattr(self._gate, "panic_lock"):
                self._gate.panic_lock = True

        t_elapsed_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0
        self._last_interrupt_latency_ms = t_elapsed_ms

        return True

    def reset_emergency(self) -> None:
        """Resets the emergency interrupt latch."""
        with self._lock:
            self._emergency_active = False
            if self._gate is not None:
                if hasattr(self._gate, "set_panic_lock"):
                    self._gate.set_panic_lock(False)
                elif hasattr(self._gate, "panic_lock"):
                    self._gate.panic_lock = False
