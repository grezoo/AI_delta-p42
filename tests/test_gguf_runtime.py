"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 5
Test Suite: tests/test_gguf_runtime.py
Author: Stage-Gate 5 AI Nursery Team

Comprehensive Verifications for the Local GGUF Cortical Runtime:
1. Canonical Architecture Budget: 6 cortical subsystems, <= 2.90 GB VRAM (2.88 GB), 4.17B parameters.
2. Thread-safe Asynchronous Worker Lifecycle: Start, submit requests, callbacks, clean stop.
3. Concurrent 6-Model Chorus Dispatch: Validates parallel execution across all subsystems.
4. Physical Kuramoto Coherence Computation: Computes order parameter r(t) and phase shift angles from latencies.
5. CDC Ring Buffer Courier Integration: Verifies CognitivePacket creation and queue dispatch.
6. Role-Specific Syntheses: Validates PFC, inner speech, social, critic, sensorimotor, and 384-dim embedding.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import List

import numpy as np
import pytest

from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import MODULE_CHORUS_LLM, CognitivePacket
from core.maslow_integrator import MaslowSnapshot
from nursery.gguf_runtime import (
    DEFAULT_MODEL_DIR,
    MAX_VRAM_BUDGET_MB,
    TOTAL_CANONICAL_VRAM_MB,
    CorticalInferenceRequest,
    CorticalInferenceResult,
    CorticalModelWorker,
    CorticalTelemetrySnapshot,
    GGUFCortexChorusRuntime,
)


# ==============================================================================
# 1. ARCHITECTURAL BUDGET AND INVARIANTS
# ==============================================================================

def test_gguf_runtime_architectural_invariants() -> None:
    """Verifies that the GGUF runtime strictly conforms to canonical VRAM budgets."""
    assert MAX_VRAM_BUDGET_MB == 2900.0
    assert TOTAL_CANONICAL_VRAM_MB <= MAX_VRAM_BUDGET_MB
    assert TOTAL_CANONICAL_VRAM_MB == 2879.0

    runtime = GGUFCortexChorusRuntime(force_simulation=True)
    assert runtime.total_subsystems == 6
    assert runtime.vram_used_mb <= runtime.vram_budget_mb
    assert "pfc" in runtime.workers
    assert "inner_speech" in runtime.workers
    assert "social_mirror" in runtime.workers
    assert "doubt_critic" in runtime.workers
    assert "sensorimotor" in runtime.workers
    assert "dream_embedding" in runtime.workers


# ==============================================================================
# 2. WORKER LIFECYCLE AND ASYNCHRONOUS DISPATCH
# ==============================================================================

def test_cortical_worker_single_dispatch() -> None:
    """Tests single worker async execution, response callback, and latency tracking."""
    runtime = GGUFCortexChorusRuntime(force_simulation=True)
    runtime.start()

    received_results: List[CorticalInferenceResult] = []
    done_event = concurrent.futures.Future()

    def on_result(res: CorticalInferenceResult) -> None:
        received_results.append(res)
        done_event.set_result(res)

    req_id = runtime.dispatch_inference(
        subsystem_name="pfc",
        prompt="Analyze current metabolic deficit and select homeostatic goal.",
        callback=on_result,
    )
    assert req_id.startswith("req_pfc_")

    res = done_event.result(timeout=3.0)
    assert res.subsystem_name == "pfc"
    assert "homeostatic" in res.generated_text.lower() or "strategy" in res.generated_text.lower()
    assert res.latency_ms > 0.0
    assert res.ttft_ms > 0.0
    assert res.simulated is True

    runtime.stop()


# ==============================================================================
# 3. CONCURRENT 6-MODEL CHORUS DISPATCH
# ==============================================================================

def test_concurrent_chorus_step() -> None:
    """Verifies that all 6 cortical models process stimuli concurrently without deadlock."""
    runtime = GGUFCortexChorusRuntime(force_simulation=True)
    runtime.start()

    snapshot_v = MaslowSnapshot(
        timestamp_ns=time.time_ns(),
        vector=np.array([0.20, 0.15, 0.30, 0.10, 0.25], dtype=np.float64),
        dominant_idx=2,
    )

    req_ids = runtime.dispatch_chorus_step(
        stimulus_text="Sensory input: visual ball moving across grid.",
        maslow_snapshot=snapshot_v,
    )
    assert len(req_ids) == 6

    # Allow workers to complete
    time.sleep(0.3)

    telemetry = runtime.get_telemetry_snapshot()
    assert telemetry.active_workers == 6
    assert telemetry.vram_used_mb <= 2900.0
    assert len(telemetry.subsystem_latencies_ms) == 6
    assert telemetry.total_inferences >= 6

    runtime.stop()


# ==============================================================================
# 4. PHYSICAL KURAMOTO COHERENCE AND PHASE SHIFTS
# ==============================================================================

def test_kuramoto_phase_coherence_computation() -> None:
    """
    Tests the physical Kuramoto order parameter r(t) calculation derived from
    actual inference latencies across the 6 models.
    """
    runtime = GGUFCortexChorusRuntime(force_simulation=True)
    r, phase_angles = runtime.compute_kuramoto_phase_coherence()

    assert 0.0 <= r <= 1.0
    assert len(phase_angles) == 6
    for name, angle in phase_angles.items():
        assert 0.0 <= angle < 360.0


# ==============================================================================
# 5. CDC RING BUFFER INTEGRATION
# ==============================================================================

def test_cdc_ring_buffer_integration() -> None:
    """Verifies that completed cortical inference pushes valid CognitivePackets into CDC ring buffer."""
    cdc = CDCRingBuffer()
    runtime = GGUFCortexChorusRuntime(force_simulation=True, cdc_ring_buffer=cdc)
    runtime.start()

    done_future = concurrent.futures.Future()

    runtime.dispatch_inference(
        subsystem_name="dream_embedding",
        prompt="Consolidate episodic memory trace from recent spatial exploration.",
        callback=lambda res: done_future.set_result(res),
    )

    res = done_future.result(timeout=3.0)
    assert res.embedding is not None
    assert res.embedding.shape == (384,)
    # Embedding must be normalized (unit length)
    norm = np.linalg.norm(res.embedding)
    assert pytest.approx(norm, rel=1e-3) == 1.0

    # Wait for CDC push
    time.sleep(0.1)
    assert not cdc.is_empty()

    pkt = cdc.pop()
    assert pkt is not None
    assert pkt.source_module_id == MODULE_CHORUS_LLM
    assert pkt.payload_tensor is not None
    assert len(pkt.payload_tensor) == 64

    runtime.stop()


# ==============================================================================
# 6. ROLE-SPECIFIC SYNTHESIS CONFORMANCE
# ==============================================================================

def test_all_role_specific_outputs() -> None:
    """Verifies that all 6 subsystems emit domain-authentic synthese."""
    runtime = GGUFCortexChorusRuntime(force_simulation=True)

    expected_keywords = {
        "pfc": "strategy",
        "inner_speech": "speech",
        "social_mirror": "social",
        "doubt_critic": "critic",
        "sensorimotor": "sensorimotor",
        "dream_embedding": "dream",
    }

    for name, keyword in expected_keywords.items():
        worker = runtime.workers[name]
        text, tokens, embedding = worker._synthesize_role_output("Test context")
        assert keyword in text.lower()
        assert len(tokens) > 0
        if name == "dream_embedding":
            assert embedding is not None
            assert embedding.shape == (384,)
        else:
            assert embedding is None
