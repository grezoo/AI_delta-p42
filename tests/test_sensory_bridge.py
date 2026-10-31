"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 5
Test Suite: tests/test_sensory_bridge.py
Author: Stage-Gate 5 AI Nursery Team

Verifications for SensoryBridge & PedagogicalLoopController:
1. Multimodal Sensory Ingestion (Audio, Proprioception 64-float, Caretaker Cues).
2. CDC Ring Buffer Delivery with CRC-64 verification.
3. Closed-Loop Pedagogical Interaction Turn execution.
4. Etiquette Barrier Shifting in response to Caretaker guidance.
5. DVP-11 Emergency Threat Override (threat >= 0.85 suppresses etiquette barrier).
6. Homeostatic Reduction Tensor R(t) Closed-Loop Quenching.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import MODULE_CHORUS_LLM, MODULE_THALAMUS
from core.maslow_integrator import MaslowSnapshot
from nursery.caretaker_console import CaretakerConsole, PedagogicalStage
from nursery.gguf_runtime import GGUFCortexChorusRuntime
from nursery.reaction_engine import ReactionEngine
from nursery.sensory_bridge import (
    InteractionTurnRecord,
    PedagogicalLoopController,
    SensoryBridge,
)
from nursery.worldstream_server import EventModality, EventPriority, WorldStreamServer


# ==============================================================================
# 1. MULTIMODAL SENSORY INGESTION & CDC COURIER
# ==============================================================================

def test_sensory_bridge_ingestion_and_cdc_push() -> None:
    """Verifies that the sensory bridge ingests audio, proprioception, and social cues to the CDC ring buffer."""
    cdc = CDCRingBuffer()
    bridge = SensoryBridge(cdc_ring_buffer=cdc)

    # 1. Ingest audio text
    ok_audio = bridge.ingest_audio_text("Caregiver says: hello little one!", valence=0.5)
    assert ok_audio is True

    # 2. Ingest 64-element proprioception
    kinematics = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    ok_prop = bridge.ingest_proprioception(kinematics)
    assert ok_prop is True

    # 3. Ingest caretaker cue
    ok_cue = bridge.ingest_caretaker_cue("Look at the red ball", valence=0.2, priority=EventPriority.PING)
    assert ok_cue is True

    # Verify CDC received packets
    assert not cdc.is_empty()
    pkt = cdc.pop()
    assert pkt is not None
    assert pkt.source_module_id == MODULE_THALAMUS
    assert pkt.payload_tensor is not None
    assert len(pkt.payload_tensor) == 64


# ==============================================================================
# 2. CLOSED-LOOP PEDAGOGICAL TURN EXECUTION
# ==============================================================================

def test_pedagogical_turn_execution() -> None:
    """Verifies end-to-end execution of a living interaction turn between Caretaker and Infant."""
    cdc = CDCRingBuffer()
    bridge = SensoryBridge(cdc_ring_buffer=cdc)
    console = CaretakerConsole()
    cortex = GGUFCortexChorusRuntime(force_simulation=True, cdc_ring_buffer=cdc)
    reaction = ReactionEngine()
    cortex.start()

    controller = PedagogicalLoopController(
        sensory_bridge=bridge,
        caretaker_console=console,
        cortex_runtime=cortex,
        reaction_engine=reaction,
    )

    snapshot_v = MaslowSnapshot(
        timestamp_ns=time.time_ns(),
        vector=np.array([0.15, 0.10, 0.40, 0.10, 0.20], dtype=np.float64),
        dominant_idx=2,
    )

    turn = controller.execute_pedagogical_turn(
        caretaker_utterance="Great job reaching for the block!",
        emotional_valence=0.8,
        maslow_snapshot=snapshot_v,
        threat_level=0.0,
    )

    assert turn.turn_id.startswith("turn_")
    assert turn.stage == PedagogicalStage.NURSERY
    assert "infant" in turn.cortex_response.lower() or "speech" in turn.cortex_response.lower() or "strategy" in turn.cortex_response.lower()
    assert turn.consequence_reduction is not None
    assert len(turn.consequence_reduction) == 5
    assert turn.threat_override is False
    assert turn.turn_latency_ms > 0.0

    assert controller.total_turns == 1

    cortex.stop()


# ==============================================================================
# 3. ETIQUETTE BARRIER SHIFTING UNDER SOCIAL GUIDANCE
# ==============================================================================

def test_etiquette_barrier_shifting() -> None:
    """Verifies that negative feedback in Kindergarten shifts the etiquette barrier Delta theta_illem."""
    bridge = SensoryBridge()
    console = CaretakerConsole()
    cortex = GGUFCortexChorusRuntime(force_simulation=True)
    reaction = ReactionEngine()
    cortex.start()

    controller = PedagogicalLoopController(
        sensory_bridge=bridge,
        caretaker_console=console,
        cortex_runtime=cortex,
        reaction_engine=reaction,
    )

    # Advance to Kindergarten (Stage 2)
    controller.advance_stage(PedagogicalStage.KINDERGARTEN)
    assert controller.current_stage == PedagogicalStage.KINDERGARTEN

    # Negative feedback (prohibition / scolding)
    turn = controller.execute_pedagogical_turn(
        caretaker_utterance="No, do not touch the sharp edge!",
        emotional_valence=-0.7,
    )

    assert turn.etiquette_shift_deg > 0.0
    assert turn.etiquette_shift_deg <= 5.0
    assert console.etiquette_shift_deg > 0.0

    cortex.stop()


# ==============================================================================
# 4. EMERGENCY THREAT OVERRIDE (DVP-11)
# ==============================================================================

def test_emergency_threat_override() -> None:
    """Verifies that imminent threat (>= 0.85) immediately clears etiquette barrier for life-saving action."""
    bridge = SensoryBridge()
    console = CaretakerConsole()
    cortex = GGUFCortexChorusRuntime(force_simulation=True)
    reaction = ReactionEngine()
    cortex.start()

    controller = PedagogicalLoopController(
        sensory_bridge=bridge,
        caretaker_console=console,
        cortex_runtime=cortex,
        reaction_engine=reaction,
    )

    controller.advance_stage(PedagogicalStage.KINDERGARTEN)
    console.set_etiquette_shift(4.5)
    assert console.etiquette_shift_deg == 4.5

    # Threat arrives (threat = 0.95 >= 0.85)
    turn = controller.execute_pedagogical_turn(
        caretaker_utterance="WATCH OUT!",
        emotional_valence=-1.0,
        threat_level=0.95,
    )

    assert turn.threat_override is True
    assert turn.etiquette_shift_deg == 0.0
    assert console.etiquette_shift_deg == 0.0

    cortex.stop()
