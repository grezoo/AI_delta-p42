"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_cortex_ensemble.py
Author: Stage-Gate 4 Milestone 2 Specialist

Verifies:
1. Coordination harness for the 6 cortical subsystems (PFC 1.5B, Inner Speech 0.5B,
   Social Mirror 1B, Doubt Critic 360M, Sensorimotor 0.5B, Dream Embedding All-MiniLM-L6-v2).
2. Architectural budget invariants (4.17B parameters, 2.90 GB weights, 4.20 GB runtime VRAM).
3. GGUF_CHORUS_ROSTER telemetry metadata key mapping and sub-module telemetry retrieval.
4. Dual execution mode: high-fidelity analytical simulation and live GGUF fallback.
5. PFC goal resonance selection (rho = W_PFC * V) and inner speech verbalization.
6. CDC Ring Buffer packet generation, packet routing, and thread-safe courier exchange.
"""

from __future__ import annotations

import concurrent.futures
import time

import numpy as np
import pytest

from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import (
    MODULE_BASAL_GANGLIA,
    MODULE_PFC,
    CognitivePacket,
)
from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot
from core.thalamus_agc import GGUF_CHORUS_ROSTER
from nursery.cortex_ensemble import (
    CORTEX_SUBSYSTEM_NAMES,
    PFC_GOAL_TEMPLATES,
    RUNTIME_VRAM_GB,
    SUBSYSTEM_TO_ROSTER_KEY,
    TOTAL_PARAMS,
    TOTAL_WEIGHTS_GB,
    CorticalMessage,
    CortexEnsemble,
    EnsembleState,
)


# ==============================================================================
# 1. CANONICAL CONSTANTS & ARCHITECTURAL INVARIANTS
# ==============================================================================

def test_cortex_ensemble_constants() -> None:
    assert TOTAL_PARAMS == "4.17B"
    assert TOTAL_WEIGHTS_GB == 2.90
    assert RUNTIME_VRAM_GB == 4.20

    assert len(CORTEX_SUBSYSTEM_NAMES) == 6
    assert "pfc" in CORTEX_SUBSYSTEM_NAMES
    assert "inner_speech" in CORTEX_SUBSYSTEM_NAMES
    assert "social_mirror" in CORTEX_SUBSYSTEM_NAMES
    assert "doubt_critic" in CORTEX_SUBSYSTEM_NAMES
    assert "sensorimotor" in CORTEX_SUBSYSTEM_NAMES
    assert "dream_embedding" in CORTEX_SUBSYSTEM_NAMES

    assert len(PFC_GOAL_TEMPLATES) == 5


def test_roster_key_mapping() -> None:
    """Verifies that all 6 cortical subsystems map correctly into GGUF_CHORUS_ROSTER."""
    for sub_name in CORTEX_SUBSYSTEM_NAMES:
        roster_key = SUBSYSTEM_TO_ROSTER_KEY.get(sub_name)
        assert roster_key is not None, f"Missing mapping for {sub_name}"
        assert roster_key in GGUF_CHORUS_ROSTER, f"Roster key {roster_key} not in GGUF_CHORUS_ROSTER"

        # Check metadata fields
        meta = GGUF_CHORUS_ROSTER[roster_key]
        assert "model_id" in meta or "params" in meta


# ==============================================================================
# 2. INITIALIZATION & DUAL EXECUTION MODES
# ==============================================================================

def test_cortex_ensemble_initialization() -> None:
    ensemble = CortexEnsemble(simulation_mode=True)
    assert ensemble.simulation_mode is True
    assert len(ensemble.active_subsystems) == 6
    assert all(ensemble.active_subsystems.values())

    # Mode toggle to live GGUF
    ensemble_live = CortexEnsemble(simulation_mode=False)
    assert ensemble_live.simulation_mode is False


def test_get_subsystem_telemetry() -> None:
    ensemble = CortexEnsemble(simulation_mode=True)
    telemetry = ensemble.get_subsystem_telemetry()
    assert len(telemetry) == 6
    for name in CORTEX_SUBSYSTEM_NAMES:
        assert name in telemetry
        assert telemetry[name]["active"] is True
        assert telemetry[name]["simulation_mode"] is True
        assert isinstance(telemetry[name]["model_meta"], dict)


# ==============================================================================
# 3. PFC GOAL RESONANCE & BEHAVIORAL DYNAMICS
# ==============================================================================

def test_pfc_goal_selection_driven_by_maslow() -> None:
    ensemble = CortexEnsemble(simulation_mode=True)

    # 1. Deficit where social need (idx 2) is dominant
    social_snap = MaslowSnapshot(
        timestamp_ns=time.time_ns(),
        vector=np.array([0.10, 0.10, 0.85, 0.10, 0.10]),
        l1_norm=1.25,
        l2_norm=0.87,
        dominant_idx=2,
        dominant_name="soc",
    )
    res_social = ensemble.step_chorus(maslow_snapshot=social_snap)
    assert ensemble.get_pfc_goal() == 2
    assert res_social["pfc"]["goal_id"] == 2
    assert "social" in res_social["pfc"]["goal_description"].lower()
    assert "communicative social chirp" in ensemble.get_inner_monologue()

    # 2. Deficit where metabolic need (idx 0) is dominant
    metabolic_snap = MaslowSnapshot(
        timestamp_ns=time.time_ns(),
        vector=np.array([0.90, 0.10, 0.10, 0.10, 0.10]),
        l1_norm=1.30,
        l2_norm=0.91,
        dominant_idx=0,
        dominant_name="met",
    )
    res_met = ensemble.step_chorus(maslow_snapshot=metabolic_snap)
    assert ensemble.get_pfc_goal() == 0
    assert res_met["pfc"]["goal_id"] == 0
    assert "metabolic" in res_met["pfc"]["goal_description"].lower()
    assert "sustenance" in ensemble.get_inner_monologue()


def test_social_mirror_and_doubt_critic() -> None:
    ensemble = CortexEnsemble(simulation_mode=True)

    # High safety deficit (idx 1 = 0.80) -> Risk score = 0.80 * 1.2 = 0.96 >= 0.70
    # Critic should raise doubt (approved = False)
    danger_snap = MaslowSnapshot(
        timestamp_ns=time.time_ns(),
        vector=np.array([0.10, 0.80, 0.10, 0.10, 0.10]),
        l1_norm=1.20,
        l2_norm=0.81,
        dominant_idx=1,
        dominant_name="saf",
    )
    res_danger = ensemble.step_chorus(maslow_snapshot=danger_snap)
    assert res_danger["doubt_critic"]["approved"] is False
    assert res_danger["doubt_critic"]["risk_score"] > 0.70

    # Low safety deficit (idx 1 = 0.10) -> Critic approves (approved = True)
    safe_snap = MaslowSnapshot(
        timestamp_ns=time.time_ns(),
        vector=np.array([0.10, 0.10, 0.10, 0.10, 0.10]),
        l1_norm=0.50,
        l2_norm=0.22,
        dominant_idx=0,
        dominant_name="met",
    )
    res_safe = ensemble.step_chorus(maslow_snapshot=safe_snap)
    assert res_safe["doubt_critic"]["approved"] is True


def test_dream_embedding_normalization() -> None:
    """Verifies that dream embedding produces a 384-dimensional normalized unit vector."""
    ensemble = CortexEnsemble(simulation_mode=True)
    res = ensemble.step_chorus()

    emb = ensemble.get_last_dream_embedding()
    assert emb.shape == (384,)
    assert emb.dtype == np.float32
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, abs=1e-5) == 1.0
    assert res["dream_embedding"]["dimension"] == 384
    assert pytest.approx(res["dream_embedding"]["norm"], abs=1e-5) == 1.0


# ==============================================================================
# 4. CDC BUS PACKET EXCHANGE & ROUTING
# ==============================================================================

def test_cortex_cdc_outbound_generation_and_polling() -> None:
    cdc_buf = CDCRingBuffer()
    cdc_buf.clear()
    ensemble = CortexEnsemble(cdc_buffer=cdc_buf, simulation_mode=True)

    # Step chorus without input
    res = ensemble.step_chorus()
    assert res is not None

    # Verify outbound packet was generated and pushed to CDC ring buffer
    assert cdc_buf.size() == 1
    popped = cdc_buf.pop()
    assert popped is not None
    assert popped.source_module_id == MODULE_PFC
    assert popped.target_module_id == MODULE_BASAL_GANGLIA
    assert popped.payload_tensor.shape == (64,)

    # Verify poll_cortex_outputs flushes internal buffer
    polled = ensemble.poll_cortex_outputs()
    assert len(polled) == 1
    assert len(ensemble.poll_cortex_outputs()) == 0


def test_dispatch_packet_inbox_routing() -> None:
    ensemble = CortexEnsemble(simulation_mode=True)

    pkt = CognitivePacket(
        timestamp_ns=time.time_ns(),
        sequence_id=99,
        source_module_id=2,  # Thalamus
        target_module_id=4,  # PFC
        payload_tensor=np.ones(64, dtype=np.float32) * 0.5,
    )

    accepted = ensemble.dispatch_packet(pkt)
    assert accepted is True

    # Step chorus with incoming packet
    res = ensemble.step_chorus(packet=pkt)
    assert res is not None


# ==============================================================================
# 5. MULTITHREADED CONCURRENCY
# ==============================================================================

def test_cortex_ensemble_concurrent_steps() -> None:
    cdc_buf = CDCRingBuffer()
    cdc_buf.clear()
    ensemble = CortexEnsemble(cdc_buffer=cdc_buf, simulation_mode=True)

    num_threads = 4
    steps_per_thread = 25

    def worker_step(thread_id: int) -> None:
        for s in range(steps_per_thread):
            snap = MaslowSnapshot(
                timestamp_ns=time.time_ns(),
                vector=np.array([0.15, 0.10, 0.25, 0.10, 0.30]),
                l1_norm=0.90,
                l2_norm=0.45,
                dominant_idx=4,
                dominant_name="cog",
            )
            ensemble.step_chorus(maslow_snapshot=snap)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_step, t) for t in range(num_threads)]
        concurrent.futures.wait(futures)

    expected_total = num_threads * steps_per_thread
    outbound = ensemble.poll_cortex_outputs()
    assert len(outbound) == expected_total
