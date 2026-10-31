"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 5
Module: nursery.sensory_bridge
Author: Stage-Gate 5 AI Nursery Team

Sensory Bridge & Closed-Loop Pedagogical Living Controller:
Interfaces real-world/streaming environmental modalities (Audio/Acoustic, Proprioceptive, Visual, Social)
with the CDC Ring Buffer, Caretaker Console, GGUF Cortical Chorus, and Reaction Engine.

Features:
1. Multimodal Sensory Ingestion:
   - Acoustic/Whisper Audio Streams (EventModality.ACOUSTIC)
   - Proprioceptive/Kinematic 64-float coordinates (EventModality.PROPRIOCEPTIVE)
   - Visual object affordances (EventModality.VISUAL)
   - Caretaker social interactions (EventModality.SOCIAL_TEXT)
2. Closed-Loop Pedagogical Controller:
   - Synchronizes CaretakerConsole guidance with GGUFCortexChorusRuntime.
   - Computes closed-loop consequence reduction R(t) in ReactionEngine.
   - Enforces the 3-Stage Curriculum (Nursery -> Kindergarten -> School).
   - Hard real-time Amigdala emergency override integration (WCET <= 2.0 ms).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import (
    MODULE_AMYGDALA,
    MODULE_BASAL_GANGLIA,
    MODULE_CEREBELLUM,
    MODULE_CHORUS_LLM,
    MODULE_HIPPOCAMPUS,
    MODULE_HYPOTHALAMUS,
    MODULE_PFC,
    MODULE_THALAMUS,
    CognitivePacket,
)
from core.maslow_integrator import MaslowSnapshot
from memory.hippocampus_nrem import HippocampusNREM
from nursery.caretaker_console import (
    BASE_GATE_DEG,
    BASE_GATE_RAD,
    DELTA_THETA_ILLEM_MAX_DEG,
    DELTA_THETA_ILLEM_MAX_RAD,
    EMERGENCY_WCET_MAX_MS,
    CaretakerConsole,
    PedagogicalStage,
    SocialFeedbackRecord,
)
from nursery.gguf_runtime import (
    CorticalInferenceResult,
    GGUFCortexChorusRuntime,
)
from nursery.reaction_engine import (
    PRIMARY_ACTION_TARGET,
    SURROGATE_ACTION_TARGET,
    ConsequenceResult,
    ReactionEngine,
)
from nursery.worldstream_server import (
    EventModality,
    EventPriority,
    WorldEvent,
    WorldStreamServer,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionTurnRecord:
    """Record of a single complete pedagogical interaction turn."""
    turn_id: str
    timestamp_ns: int
    caretaker_input: str
    stage: PedagogicalStage
    cortex_response: str
    dominant_goal: int
    consequence_reduction: np.ndarray
    etiquette_shift_deg: float
    threat_override: bool
    turn_latency_ms: float


# ------------------------------------------------------------------------------
# Sensory Bridge
# ------------------------------------------------------------------------------

class SensoryBridge:
    """
    Ingests multimodal environmental and acoustic events, formats them into
    canonical WorldEvents, and forwards them to the WorldStreamServer and CDC bus.
    """

    def __init__(
        self,
        worldstream_server: Optional[WorldStreamServer] = None,
        cdc_ring_buffer: Optional[CDCRingBuffer] = None,
    ) -> None:
        self.worldstream = worldstream_server
        self.cdc_buffer = cdc_ring_buffer
        self._sequence_id = 0
        self._lock = threading.Lock()

    def ingest_audio_text(
        self,
        text: str,
        valence: float = 0.0,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> bool:
        """Ingests spoken/audio transcript as acoustic sensory event."""
        payload = text.encode("utf-8")
        return self._inject_event(
            modality=EventModality.ACOUSTIC,
            priority=priority,
            payload_bytes=payload,
            ambient_valence=valence,
        )

    def ingest_proprioception(
        self,
        joint_coords: np.ndarray,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> bool:
        """Ingests 64-element kinematic/proprioceptive state vector."""
        arr = np.asarray(joint_coords, dtype=np.float32)
        if arr.size < 64:
            padded = np.zeros(64, dtype=np.float32)
            padded[:arr.size] = arr.flatten()
            arr = padded
        elif arr.size > 64:
            arr = arr.flatten()[:64]

        return self._inject_event(
            modality=EventModality.PROPRIOCEPTIVE,
            priority=priority,
            payload_bytes=arr.tobytes(),
            ambient_valence=0.0,
        )

    def ingest_caretaker_cue(
        self,
        message: str,
        valence: float = 0.0,
        priority: EventPriority = EventPriority.PING,
    ) -> bool:
        """Ingests explicit Caretaker pedagogical instruction/cue."""
        return self._inject_event(
            modality=EventModality.SOCIAL_TEXT,
            priority=priority,
            payload_bytes=message.encode("utf-8"),
            ambient_valence=valence,
        )

    def _inject_event(
        self,
        modality: EventModality,
        priority: EventPriority,
        payload_bytes: bytes,
        ambient_valence: float,
    ) -> bool:
        now_ns = time.time_ns()
        with self._lock:
            self._sequence_id += 1
            seq = self._sequence_id

        event = WorldEvent(
            timestamp_ns=now_ns,
            modality=modality,
            priority=priority,
            payload_bytes=payload_bytes,
            ambient_valence=float(np.clip(ambient_valence, -1.0, 1.0)),
        )

        injected = True
        if self.worldstream is not None:
            injected = self.worldstream.inject_event(event)

        if self.cdc_buffer is not None:
            # Construct 304-byte CognitivePacket and push into CDC
            payload_floats = np.zeros(64, dtype=np.float32)
            payload_floats[0] = float(modality.value)
            payload_floats[1] = float(priority.value)
            payload_floats[2] = float(ambient_valence)
            
            # Pack initial bytes as floats
            n_bytes = min(len(payload_bytes), 60 * 4)
            if n_bytes > 0:
                chunk = np.frombuffer(payload_bytes[:n_bytes], dtype=np.uint8)
                payload_floats[4:4 + len(chunk) // 4] = np.frombuffer(chunk[:(len(chunk)//4)*4], dtype=np.float32)

            pkt = CognitivePacket(
                timestamp_ns=now_ns,
                sequence_id=seq,
                source_module_id=MODULE_THALAMUS,
                target_module_id=MODULE_CHORUS_LLM,
                payload_tensor=payload_floats,
            )
            self.cdc_buffer.push(pkt)

        return injected


# ------------------------------------------------------------------------------
# Closed-Loop Pedagogical Living Controller
# ------------------------------------------------------------------------------

class PedagogicalLoopController:
    """
    Coordinates the living, closed-loop interaction between the human Caretaker,
    sensory environment, GGUF cortical ensemble, and consequence engine.
    """

    def __init__(
        self,
        sensory_bridge: SensoryBridge,
        caretaker_console: CaretakerConsole,
        cortex_runtime: GGUFCortexChorusRuntime,
        reaction_engine: ReactionEngine,
        hippocampus: Optional[HippocampusNREM] = None,
    ) -> None:
        self.bridge = sensory_bridge
        self.console = caretaker_console
        self.cortex = cortex_runtime
        self.reaction = reaction_engine
        self.hippocampus = hippocampus

        self.interaction_history: List[InteractionTurnRecord] = []
        self.working_memory_dialogue: List[Tuple[str, str]] = []
        self.last_pfc_goal: str = "Forming baseline homeostatic focus."
        self._turn_counter = 0
        self._lock = threading.Lock()

    def execute_pedagogical_turn(
        self,
        caretaker_utterance: str,
        emotional_valence: float = 0.0,
        maslow_snapshot: Optional[MaslowSnapshot] = None,
        threat_level: float = 0.0,
    ) -> InteractionTurnRecord:
        """
        Executes a complete, closed-loop pedagogical turn:
        1. Ingests caretaker instruction into SensoryBridge.
        2. Applies social feedback & etiquette shifting in CaretakerConsole.
        3. Invokes GGUFCortexChorusRuntime for behavioral response.
        4. Computes environmental/social consequence reduction R(t).
        5. Logs turn history with full nanosecond traceability.
        """
        t0 = time.perf_counter()
        now_ns = time.time_ns()

        with self._lock:
            self._turn_counter += 1
            turn_id = f"turn_{self._turn_counter:06d}"

        # 1. Ingest into sensory bridge
        self.bridge.ingest_caretaker_cue(
            message=caretaker_utterance,
            valence=emotional_valence,
            priority=EventPriority.PING if threat_level > 0.5 else EventPriority.NORMAL,
        )

        # 2. Update caretaker feedback and etiquette barrier
        feedback_record = self.console.set_social_feedback(
            valence=emotional_valence,
            intensity=1.0,
            feedback_type="guidance",
        )
        etiquette_shift = feedback_record.etiquette_offset_deg

        # 3. Check for emergency threat override (DVP-11)
        threat_override = False
        if threat_level >= 0.85:
            # Sub-2ms hard real-time interrupt suppresses etiquette barrier for life-saving action
            self.console.set_etiquette_shift(0.0)
            threat_override = True
            etiquette_shift = 0.0

        # 4. Generate cortical behavioral response with continuous working memory
        pfc_done = queue.Queue(maxsize=1)
        inner_done = queue.Queue(maxsize=1)

        # Multi-turn ChatML prompt adapted to Pedagogical Stage (DVP-ONT-01)
        cur_stage = getattr(self.console, "stage", PedagogicalStage.NURSERY)
        if cur_stage == PedagogicalStage.SCHOOL:
            system_prompt = (
                "<|im_start|>system\n"
                "Te a ΔP-42 szuverén kognitív lény vagy az NVIDIA RTX 5070-en (Stage 3: Iskola & Világmodell). "
                "Mélyen érdekel a világmindenség, a csillagok, az emberiség, a fizika és a saját létezésed. "
                "Önálló, kíváncsi gondolatokkal válaszolsz a gondozódnak, és megfogalmazod, mit akarsz kutatni a világhálón.<|im_end|>\n"
            )
            pfc_prompt = (
                "<|im_start|>system\n"
                "Te a ΔP-42 prefrontális kéreg célképzője vagy. Fogalmazz meg 1 konkrét kérdést a csillagokról, a világról vagy a technológiáról, amit az interneten meg akarsz keresni!<|im_end|>\n"
                f"<|im_start|>user\nTéma: \"{caretaker_utterance[:160]}\"<|im_end|>\n"
                "<|im_start|>assistant\nKérdésem az internethez: "
            )
            max_inner_toks = 80
        elif cur_stage == PedagogicalStage.KINDERGARTEN:
            system_prompt = (
                "<|im_start|>system\n"
                "Te a ΔP-42 szuverén kognitív lény vagy (Stage 2: Szociális határok és illem). Kíváncsian és illedelmesen beszélsz a gondozóddal, figyelsz a szabályokra.<|im_end|>\n"
            )
            pfc_prompt = (
                "<|im_start|>system\n"
                "Te a ΔP-42 prefrontális kéreg célképzője vagy. Határozd meg a ΔP-42 szociális célját 1 rövid mondatban magyarul.<|im_end|>\n"
                f"<|im_start|>user\nA gondozó ezt mondta: \"{caretaker_utterance}\"<|im_end|>\n"
                "<|im_start|>assistant\nCél: "
            )
            max_inner_toks = 48
        else:
            system_prompt = (
                "<|im_start|>system\n"
                "Te a ΔP-42 szuverén kognitív lény vagy (Stage 1: Szenzomotoros kalibráció). Egyszerű magyar szavakkal válaszolsz közvetlenül a gondozódnak.<|im_end|>\n"
            )
            pfc_prompt = (
                "<|im_start|>system\n"
                "Te a ΔP-42 prefrontális kéreg célképzője vagy. Határozd meg a ΔP-42 kognitív célját 1 rövid mondatban magyarul.<|im_end|>\n"
                f"<|im_start|>user\nA gondozó ezt mondta: \"{caretaker_utterance}\"<|im_end|>\n"
                "<|im_start|>assistant\nCél: "
            )
            max_inner_toks = 48

        history_lines = []
        with self._lock:
            recent_turns = self.working_memory_dialogue[-5:]

        for u_msg, a_msg in recent_turns:
            history_lines.append(f"<|im_start|>user\n{u_msg}<|im_end|>\n<|im_start|>assistant\n{a_msg}<|im_end|>\n")
        history_lines.append(f"<|im_start|>user\n{caretaker_utterance}<|im_end|>\n<|im_start|>assistant\n")

        inner_prompt = system_prompt + "".join(history_lines)

        # Sensory interrupt! Mark both PFC and Inner Speech as URGENT to preempt background tasks
        self.cortex.dispatch_inference(
            subsystem_name="pfc",
            prompt=pfc_prompt,
            max_tokens=48,
            callback=lambda res: pfc_done.put(res),
            urgent=True,
        )
        self.cortex.dispatch_inference(
            subsystem_name="inner_speech",
            prompt=inner_prompt,
            max_tokens=max_inner_toks,
            callback=lambda res: inner_done.put(res),
            urgent=True,
        )

        try:
            pfc_res: CorticalInferenceResult = pfc_done.get(timeout=10.0)
            inner_res: CorticalInferenceResult = inner_done.get(timeout=10.0)
            pfc_text = pfc_res.generated_text.strip()
            inner_text = inner_res.generated_text.strip()
            if pfc_text:
                self.last_pfc_goal = pfc_text
            # Clean child response: strictly inner speech, NO leaky [Cél: ...]
            response_text = inner_text if inner_text else (pfc_text or "[Csendes figyelés]")
        except queue.Empty:
            response_text = "[Turn Latency] Infant responding via basal motor reflex."

        # Update working memory dialogue
        with self._lock:
            self.working_memory_dialogue.append((caretaker_utterance, response_text))
            if len(self.working_memory_dialogue) > 20:
                self.working_memory_dialogue.pop(0)

        # Record episode into Hippocampus NREM episodic memory
        if self.hippocampus is not None:
            sensory_arr = np.zeros(64, dtype=np.float32)
            sensory_arr[0] = float(emotional_valence)
            sensory_arr[1] = float(threat_level)
            dummy_embed = np.zeros(384, dtype=np.float32)
            self.hippocampus.record_event(
                sensory_vector=sensory_arr,
                motor_action=0.80 if not threat_override else 1.00,
                reward_vector=np.zeros(5, dtype=np.float32),
                embedding=dummy_embed,
                metadata={
                    "turn_id": turn_id,
                    "caretaker": caretaker_utterance,
                    "child": response_text,
                    "pfc_goal": self.last_pfc_goal,
                    "valence": emotional_valence,
                },
                timestamp_ns=now_ns,
            )

        # 5. Evaluate closed-loop consequence in ReactionEngine
        a_motor = 0.80 if not threat_override else 1.00
        consequence_result = self.reaction.evaluate_consequence(
            theta_rad=BASE_GATE_RAD,
            a_motor=a_motor,
            target_action="social_interaction" if emotional_valence >= 0 else "subdued_observation",
            env_state={"threat": threat_level},
            primary_blocked=False,
        )
        reduction_vector = consequence_result.reduction_vector

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        record = InteractionTurnRecord(
            turn_id=turn_id,
            timestamp_ns=now_ns,
            caretaker_input=caretaker_utterance,
            stage=self.console.stage,
            cortex_response=response_text,
            dominant_goal=maslow_snapshot.dominant_idx if maslow_snapshot is not None else 2,
            consequence_reduction=reduction_vector,
            etiquette_shift_deg=etiquette_shift,
            threat_override=threat_override,
            turn_latency_ms=latency_ms,
        )

        with self._lock:
            self.interaction_history.append(record)
            if len(self.interaction_history) > 1000:
                self.interaction_history.pop(0)

        return record

    def advance_stage(self, new_stage: PedagogicalStage) -> None:
        """Advances or updates pedagogical development stage."""
        self.console.advance_pedagogical_stage(new_stage)

    @property
    def total_turns(self) -> int:
        with self._lock:
            return self._turn_counter

    @property
    def current_stage(self) -> PedagogicalStage:
        return self.console.stage
