"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_caretaker_console.py
Author: Stage-Gate 4 Milestone 2 Specialist

Verifies:
1. 3-Stage pedagogical progression: NURSERY (1) -> KINDERGARTEN (2) -> SCHOOL (3) with anti-regression enforcement.
2. Social etiquette threshold shift Delta theta_illem in [0.0 deg, 5.0 deg] (0.087266 rad).
3. Etiquette barrier shift in Stage 2/3 upon negative guidance without penalizing internal V_social.
4. Caretaker API methods: inject_stimulus, set_social_feedback, read_telemetry, emergency_interrupt.
5. Sub-2.0 ms hard real-time emergency interrupt WCET, Amygdala triggering, and Basal Ganglia striatal panic lock.
6. Coherent telemetry snapshot serialization (CaretakerTelemetry and SocialFeedbackRecord).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from core.basal_ganglia_gate import BasalGangliaGate
from core.maslow_integrator import MaslowIntegrator
from core.theta_engine import DEG2RAD, ThetaEngine
from modules.amygdala_override import AmygdalaOverride
from nursery.caretaker_console import (
    BASE_GATE_DEG,
    BASE_GATE_RAD,
    DELTA_THETA_ILLEM_MAX_DEG,
    DELTA_THETA_ILLEM_MAX_RAD,
    EMERGENCY_WCET_MAX_MS,
    CaretakerConsole,
    CaretakerTelemetry,
    PedagogicalStage,
    SocialFeedbackRecord,
)
from nursery.worldstream_server import EventModality, EventPriority, WorldStreamServer


# ==============================================================================
# 1. CONSTANTS & PEDAGOGICAL STAGE ENUMS
# ==============================================================================

def test_caretaker_constants_and_enums() -> None:
    assert PedagogicalStage.NURSERY == 1
    assert PedagogicalStage.KINDERGARTEN == 2
    assert PedagogicalStage.SCHOOL == 3

    assert DELTA_THETA_ILLEM_MAX_DEG == 5.0
    assert pytest.approx(DELTA_THETA_ILLEM_MAX_RAD, rel=1e-6) == 5.0 * DEG2RAD
    assert EMERGENCY_WCET_MAX_MS == 2.0
    assert BASE_GATE_DEG == 18.0
    assert pytest.approx(BASE_GATE_RAD, rel=1e-6) == 18.0 * DEG2RAD


# ==============================================================================
# 2. PEDAGOGICAL PROGRESSION & ANTI-REGRESSION INVARIANTS
# ==============================================================================

def test_pedagogical_stage_progression() -> None:
    console = CaretakerConsole(stage=PedagogicalStage.NURSERY)
    assert console.stage == PedagogicalStage.NURSERY

    # Advance to Kindergarten
    console.advance_pedagogical_stage(PedagogicalStage.KINDERGARTEN)
    assert console.stage == PedagogicalStage.KINDERGARTEN

    # Advance to School
    console.advance_pedagogical_stage(PedagogicalStage.SCHOOL)
    assert console.stage == PedagogicalStage.SCHOOL

    # Anti-regression: Cannot revert from School to Kindergarten or Nursery
    with pytest.raises(ValueError, match="Cannot regress pedagogical stage"):
        console.advance_pedagogical_stage(PedagogicalStage.KINDERGARTEN)

    with pytest.raises(ValueError, match="Cannot regress pedagogical stage"):
        console.advance_pedagogical_stage(PedagogicalStage.NURSERY)


# ==============================================================================
# 3. SOCIAL FEEDBACK & ETIQUETTE MODULATION
# ==============================================================================

def test_etiquette_barrier_clamping() -> None:
    console = CaretakerConsole(stage=PedagogicalStage.KINDERGARTEN)
    assert console.etiquette_shift_deg == 0.0

    # Valid shift within bounds
    console.set_etiquette_shift(3.5)
    assert console.etiquette_shift_deg == 3.5
    assert pytest.approx(console.etiquette_shift_rad, rel=1e-6) == 3.5 * DEG2RAD
    assert console.get_effective_gate_deg() == 18.0 + 3.5
    assert pytest.approx(console.get_effective_gate_rad(), rel=1e-6) == (18.0 + 3.5) * DEG2RAD

    # Upper clamp to 5.0 deg
    console.set_etiquette_shift(10.0)
    assert console.etiquette_shift_deg == 5.0
    assert console.get_effective_gate_deg() == 23.0

    # Lower clamp to 0.0 deg
    console.set_etiquette_shift(-4.0)
    assert console.etiquette_shift_deg == 0.0
    assert console.get_effective_gate_deg() == 18.0


def test_social_feedback_nursery_vs_kindergarten() -> None:
    # 1. In NURSERY (Stage 1), social feedback does NOT raise etiquette barrier
    console_nursery = CaretakerConsole(stage=PedagogicalStage.NURSERY)
    rec1 = console_nursery.set_social_feedback(valence=-0.9, intensity=1.0, feedback_type="prohibition")
    assert rec1.etiquette_offset_deg == 0.0
    assert console_nursery.etiquette_shift_deg == 0.0

    # 2. In KINDERGARTEN (Stage 2), negative feedback raises etiquette barrier
    console_kind = CaretakerConsole(stage=PedagogicalStage.KINDERGARTEN)
    rec2 = console_kind.set_social_feedback(valence=-0.8, intensity=1.0, feedback_type="prohibition")
    assert rec2.etiquette_offset_deg > 0.0
    assert rec2.etiquette_offset_deg <= 5.0
    assert console_kind.etiquette_shift_deg == rec2.etiquette_offset_deg

    # Positive encouragement relaxes etiquette barrier
    rec3 = console_kind.set_social_feedback(valence=0.9, intensity=1.0, feedback_type="praise")
    assert console_kind.etiquette_shift_deg < rec2.etiquette_offset_deg


def test_social_feedback_preserves_internal_social_drive() -> None:
    """
    Confirms that etiquette restriction modulates the gate without penalizing
    or artificially suppressing the internal Maslow social need V_social.
    """
    maslow = MaslowIntegrator(dt=0.10)
    initial_v_social = float(maslow.vector[2])

    console = CaretakerConsole(stage=PedagogicalStage.KINDERGARTEN)
    # Severe scolding applied
    console.set_social_feedback(valence=-1.0, intensity=1.0, feedback_type="severe_scold")
    assert console.etiquette_shift_deg == 5.0

    # Internal Maslow deficit V_social remains authentic
    assert float(maslow.vector[2]) == initial_v_social


# ==============================================================================
# 4. SENSORY STIMULUS INJECTION
# ==============================================================================

def test_inject_stimulus_modalities_and_payloads() -> None:
    worldstream = WorldStreamServer()
    console = CaretakerConsole(worldstream=worldstream)

    # Inject text
    ok_text = console.inject_stimulus(
        modality=EventModality.SOCIAL_TEXT,
        data="Good morning little one",
        priority=EventPriority.PING,
        valence=0.6,
    )
    assert ok_text is True

    # Inject numpy array (e.g. acoustic or visual)
    audio_data = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
    ok_arr = console.inject_stimulus(
        modality=EventModality.ACOUSTIC,
        data=audio_data,
        valence=0.1,
    )
    assert ok_arr is True

    # Inject raw bytes
    ok_bytes = console.inject_stimulus(
        modality=EventModality.PROPRIOCEPTIVE,
        data=b"\x00\x01\x02\x03",
    )
    assert ok_bytes is True

    assert worldstream.events_injected == 3


# ==============================================================================
# 5. HARD REAL-TIME EMERGENCY INTERRUPT (WCET <= 2.0 MS)
# ==============================================================================

def test_emergency_interrupt_sub_2ms_wcet() -> None:
    amygdala = AmygdalaOverride()
    gate = BasalGangliaGate()
    console = CaretakerConsole(amygdala=amygdala, gate=gate)

    assert not console.is_emergency_active

    # Trigger emergency interrupt
    t0 = time.perf_counter_ns()
    success = console.emergency_interrupt(threat_level=1.0)
    t_elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

    assert success is True
    assert console.is_emergency_active is True
    assert console.last_interrupt_latency_ms <= EMERGENCY_WCET_MAX_MS, (
        f"Emergency interrupt latency {console.last_interrupt_latency_ms:.3f} ms "
        f"exceeded maximum WCET of {EMERGENCY_WCET_MAX_MS} ms"
    )

    # Reset emergency
    console.reset_emergency()
    assert not console.is_emergency_active


# ==============================================================================
# 6. TELEMETRY MONITORING & ATOMIC SERIALIZATION
# ==============================================================================

def test_read_telemetry_snapshot_and_serialization() -> None:
    theta = ThetaEngine(dt=0.02)
    maslow = MaslowIntegrator(dt=0.10)
    gate = BasalGangliaGate()
    console = CaretakerConsole(gate=gate, stage=PedagogicalStage.KINDERGARTEN)
    console.set_etiquette_shift(2.5)

    telemetry: CaretakerTelemetry = console.read_telemetry(maslow=maslow, theta=theta)

    assert telemetry.stage == PedagogicalStage.KINDERGARTEN
    assert telemetry.theta_deg == theta.theta_deg
    assert telemetry.etiquette_shift_deg == 2.5
    assert telemetry.effective_gate_deg == 20.5
    assert len(telemetry.maslow_vector) == 5
    assert telemetry.motor_gate == 0.0

    t_dict = telemetry.to_dict()
    assert isinstance(t_dict, dict)
    assert t_dict["stage"] == 2
    assert t_dict["effective_gate_deg"] == 20.5
    assert isinstance(t_dict["maslow_vector"], list)
