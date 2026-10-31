"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_worldstream_server.py
Author: Stage-Gate 4 Milestone 2 Specialist

Verifies:
1. 50 Hz cadence (dt = 0.020 s), timing constants, and modality/priority enums.
2. Canonical 280-byte binary layout ('<QHH I 256s f 4x') struct packing and unpacking.
3. Clamping of ambient valence in [-1.0, 1.0] and payload clamping to <= 256 bytes.
4. Conversion to canonical 304-byte CognitivePacket with hardware-grade CRC-64/ECMA-182.
5. WorldStreamServer life cycle, sequence numbering, and multimodal synthesis.
6. CDCRingBuffer integration, drop-oldest overfill handling, and multithreaded concurrency.
"""

from __future__ import annotations

import concurrent.futures
import math
import struct
import time
import uuid

import numpy as np
import pytest

from comm.cdc_ring_buffer import CDCRingBuffer, _PurePythonRingBuffer
from comm.cognitive_packet import (
    MODULE_PFC,
    MODULE_THALAMUS,
    PACKET_SIZE,
    compute_crc64,
    validate_crc,
)
from nursery.worldstream_server import (
    MAX_PAYLOAD_BYTES,
    VALENCE_MAX,
    VALENCE_MIN,
    WORLD_EVENT_FORMAT,
    WORLD_EVENT_SIZE,
    WORLDSTREAM_CADENCE_HZ,
    WORLDSTREAM_DT,
    EventModality,
    EventPriority,
    ModalityType,
    WorldEvent,
    WorldStreamServer,
)


# ==============================================================================
# 1. CONSTANTS & ENUMS INTEGRITY TESTS
# ==============================================================================

def test_worldstream_constants() -> None:
    assert WORLDSTREAM_CADENCE_HZ == 50.0
    assert pytest.approx(WORLDSTREAM_DT, rel=1e-6) == 0.020
    assert WORLD_EVENT_SIZE == 280
    assert MAX_PAYLOAD_BYTES == 256
    assert VALENCE_MIN == -1.0
    assert VALENCE_MAX == 1.0
    assert struct.calcsize(WORLD_EVENT_FORMAT) == 280


def test_modality_and_priority_enums() -> None:
    assert EventModality.ACOUSTIC == 1
    assert EventModality.VISUAL == 2
    assert EventModality.PROPRIOCEPTIVE == 3
    assert EventModality.SOCIAL_TEXT == 4
    assert ModalityType is EventModality

    assert EventPriority.BACKGROUND == 0
    assert EventPriority.NORMAL == 1
    assert EventPriority.PING == 2


# ==============================================================================
# 2. WORLDEVENT STRUCT PACKING, UNPACKING & VALIDATION
# ==============================================================================

def test_world_event_instantiation_defaults() -> None:
    now_ns = time.time_ns()
    payload = b"test payload"
    ev = WorldEvent(
        timestamp_ns=now_ns,
        modality=EventModality.ACOUSTIC,
        priority=EventPriority.NORMAL,
        payload_bytes=payload,
    )
    assert ev.timestamp_ns == now_ns
    assert ev.modality == EventModality.ACOUSTIC
    assert ev.priority == EventPriority.NORMAL
    assert ev.payload_bytes == payload
    assert ev.payload == payload
    assert ev.payload_size == len(payload)
    assert ev.ambient_valence == 0.0
    assert len(ev.event_uuid) == 36  # Valid UUID string format


def test_world_event_clamping_and_coercion() -> None:
    # Test ambient valence out-of-bounds clamping
    ev_high = WorldEvent(
        timestamp_ns=1000,
        modality=1,
        priority=1,
        payload_bytes=b"data",
        ambient_valence=2.5,
    )
    assert ev_high.ambient_valence == 1.0

    ev_low = WorldEvent(
        timestamp_ns=1000,
        modality=1,
        priority=1,
        payload_bytes=b"data",
        ambient_valence=-3.5,
    )
    assert ev_low.ambient_valence == -1.0

    # Test payload truncation to 256 bytes
    oversized = b"A" * 300
    ev_oversized = WorldEvent(
        timestamp_ns=1000,
        modality=EventModality.VISUAL,
        priority=EventPriority.NORMAL,
        payload_bytes=oversized,
    )
    assert len(ev_oversized.payload_bytes) == 256
    assert ev_oversized.payload_bytes == b"A" * 256

    # Test string payload coercion to bytes
    ev_str = WorldEvent(
        timestamp_ns=1000,
        modality=EventModality.SOCIAL_TEXT,
        priority=EventPriority.PING,
        payload_bytes="hello world",  # type: ignore[arg-type]
    )
    assert ev_str.payload_bytes == b"hello world"

    # Test invalid payload type
    with pytest.raises(TypeError, match="payload_bytes must be bytes-like"):
        WorldEvent(
            timestamp_ns=1000,
            modality=1,
            priority=1,
            payload_bytes=12345,  # type: ignore[arg-type]
        )


def test_world_event_binary_serialization_exact_roundtrip() -> None:
    now_ns = 1718000000123456789
    payload = b"Sonic waveform sample data 1234567890"
    original_uuid = str(uuid.uuid4())
    ev = WorldEvent(
        timestamp_ns=now_ns,
        modality=EventModality.PROPRIOCEPTIVE,
        priority=EventPriority.PING,
        payload_bytes=payload,
        ambient_valence=0.45,
        event_uuid=original_uuid,
    )

    data = ev.to_bytes()
    assert isinstance(data, bytes)
    assert len(data) == 280

    reconstructed = WorldEvent.from_bytes(data, event_uuid=original_uuid)
    assert reconstructed.timestamp_ns == now_ns
    assert reconstructed.modality == EventModality.PROPRIOCEPTIVE
    assert reconstructed.priority == EventPriority.PING
    assert reconstructed.payload_bytes == payload
    assert pytest.approx(reconstructed.ambient_valence, abs=1e-5) == 0.45
    assert reconstructed.event_uuid == original_uuid


def test_world_event_from_bytes_short_buffer() -> None:
    short_data = b"\x00" * 200
    with pytest.raises(ValueError, match="Expected at least 280 bytes"):
        WorldEvent.from_bytes(short_data)


# ==============================================================================
# 3. CONVERSION TO COGNITIVEPACKET & CRC-64 INTEGRITY
# ==============================================================================

def test_world_event_to_packet_crc_integrity() -> None:
    payload = b"\x01\x02\x03\x04" * 16  # 64 bytes
    ev = WorldEvent(
        timestamp_ns=1000,
        modality=EventModality.ACOUSTIC,
        priority=EventPriority.NORMAL,
        payload_bytes=payload,
        ambient_valence=0.2,
    )

    pkt = ev.to_packet(sequence_id=42, source_module_id=MODULE_THALAMUS, target_module_id=MODULE_PFC)
    assert pkt.sequence_id == 42
    assert pkt.source_module_id == MODULE_THALAMUS
    assert pkt.target_module_id == MODULE_PFC
    assert pkt.payload_bytes == len(payload)
    assert pkt.payload_tensor.shape == (64,)
    assert pkt.payload_tensor.dtype == np.float32

    # Pack into canonical 304-byte binary packet
    pkt_bytes = pkt.to_bytes()
    assert len(pkt_bytes) == PACKET_SIZE

    # Verify hardware CRC-64/ECMA-182 integrity
    computed_crc = compute_crc64(pkt_bytes[:296])
    unpacked_crc = struct.unpack("<Q", pkt_bytes[296:304])[0]
    assert computed_crc == unpacked_crc
    assert pkt.crc64_checksum == computed_crc
    assert validate_crc(pkt_bytes) is True


def test_world_event_to_packet_nan_inf_protection() -> None:
    # Craft payload containing NaN and Inf IEEE 754 float bytes
    nan_inf_bytes = struct.pack("<4f", float("nan"), float("inf"), float("-inf"), 42.0)
    ev = WorldEvent(
        timestamp_ns=2000,
        modality=EventModality.VISUAL,
        priority=EventPriority.NORMAL,
        payload_bytes=nan_inf_bytes,
    )
    pkt = ev.to_packet()
    tensor = pkt.payload_tensor
    assert not np.isnan(tensor[0])
    assert tensor[0] == 0.0  # NaN replaced by 0.0
    assert not np.isinf(tensor[1])
    assert tensor[1] == 1.0  # +inf replaced by 1.0
    assert not np.isinf(tensor[2])
    assert tensor[2] == -1.0 # -inf replaced by -1.0
    assert tensor[3] == 42.0


# ==============================================================================
# 4. WORLDSTREAM SERVER LIFE CYCLE & MULTIMODAL SYNTHESIS
# ==============================================================================

def test_worldstream_server_initialization() -> None:
    server = WorldStreamServer(sample_rate_hz=50.0)
    assert server.sample_rate_hz == 50.0
    assert server.dt == 0.020
    assert not server.is_running()
    assert server.events_generated == 0
    assert server.events_injected == 0
    assert server.packets_dropped == 0

    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        WorldStreamServer(sample_rate_hz=0.0)

    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        WorldStreamServer(sample_rate_hz=-10.0)


def test_worldstream_server_start_stop() -> None:
    server = WorldStreamServer()
    assert not server.is_running()
    server.start()
    assert server.is_running()
    server.stop()
    assert not server.is_running()


def test_worldstream_server_modality_rotation_and_payloads() -> None:
    server = WorldStreamServer()

    # Step 1: Modality 1 -> ACOUSTIC (sinusoidal waveform)
    ev1, pkt1 = server.generate_step(sim_time=0.0)
    assert ev1.modality == EventModality.ACOUSTIC
    assert pkt1.sequence_id == 1
    assert len(ev1.payload_bytes) == 256  # 64 floats * 4 bytes
    assert server.events_generated == 1

    # Step 2: Modality 2 -> VISUAL (optical flow grid)
    ev2, pkt2 = server.generate_step(sim_time=0.02)
    assert ev2.modality == EventModality.VISUAL
    assert pkt2.sequence_id == 2
    assert len(ev2.payload_bytes) == 256

    # Step 3: Modality 3 -> PROPRIOCEPTIVE (6-DOF joint states)
    ev3, pkt3 = server.generate_step(sim_time=0.04)
    assert ev3.modality == EventModality.PROPRIOCEPTIVE
    assert pkt3.sequence_id == 3
    assert len(ev3.payload_bytes) == 256

    # Step 4: Modality 4 -> SOCIAL_TEXT (caregiver phrase)
    ev4, pkt4 = server.generate_step(sim_time=0.06)
    assert ev4.modality == EventModality.SOCIAL_TEXT
    assert pkt4.sequence_id == 4
    assert len(ev4.payload_bytes) > 0
    assert b"baby" in ev4.payload_bytes or b"toy" in ev4.payload_bytes or b"job" in ev4.payload_bytes or b"touch" in ev4.payload_bytes

    # Step 5: Loops back to ACOUSTIC
    ev5, pkt5 = server.generate_step(sim_time=0.08)
    assert ev5.modality == EventModality.ACOUSTIC
    assert pkt5.sequence_id == 5


# ==============================================================================
# 5. CDC RING BUFFER INTEGRATION & OVERFILL STRESS
# ==============================================================================

def test_worldstream_server_with_cdc_buffer() -> None:
    cdc_buffer = CDCRingBuffer()
    cdc_buffer.clear()
    server = WorldStreamServer(cdc_buffer=cdc_buffer)

    # Generate 10 steps and verify buffer receipt
    for i in range(10):
        ev, pkt = server.generate_step(sim_time=i * 0.02)
        assert pkt.sequence_id == i + 1

    assert cdc_buffer.size() == 10
    assert server.events_generated == 10
    assert server.packets_dropped == 0

    # Pop packets and verify sequence ordering
    for i in range(10):
        popped = cdc_buffer.pop()
        assert popped is not None
        assert popped.sequence_id == i + 1


def test_worldstream_server_inject_event() -> None:
    cdc_buffer = CDCRingBuffer()
    cdc_buffer.clear()
    server = WorldStreamServer(cdc_buffer=cdc_buffer)

    custom_event = WorldEvent(
        timestamp_ns=time.time_ns(),
        modality=EventModality.SOCIAL_TEXT,
        priority=EventPriority.PING,
        payload_bytes=b"External Caregiver Call",
        ambient_valence=0.8,
    )

    accepted = server.inject_event(custom_event)
    assert accepted is True
    assert server.events_injected == 1
    assert cdc_buffer.size() == 1

    popped = cdc_buffer.pop()
    assert popped is not None
    assert popped.sequence_id == 1

    with pytest.raises(TypeError, match="Expected WorldEvent"):
        server.inject_event("not a world event")  # type: ignore[arg-type]


def test_worldstream_server_buffer_overfill_drop_oldest() -> None:
    # Use _PurePythonRingBuffer with small capacity 8 to test overfill
    small_buf = _PurePythonRingBuffer(capacity=8)
    server = WorldStreamServer(cdc_buffer=small_buf)  # type: ignore[arg-type]

    # Fill buffer to capacity (8 slots)
    for i in range(8):
        ev, _ = server.generate_step(sim_time=i * 0.02)

    assert small_buf.is_full()
    assert server.packets_dropped == 0

    # Pushing with drop_oldest=False drops the packet
    ev_extra = WorldEvent(
        timestamp_ns=1000,
        modality=EventModality.ACOUSTIC,
        priority=EventPriority.NORMAL,
        payload_bytes=b"Extra",
    )
    pushed = server.stream_event(ev_extra, drop_oldest=False)
    assert pushed is False
    assert server.packets_dropped == 1

    # Pushing with drop_oldest=True succeeds by dropping the oldest unread packet
    pushed_drop = server.stream_event(ev_extra, drop_oldest=True)
    assert pushed_drop is True
    assert small_buf.is_full()


# ==============================================================================
# 6. MULTITHREADED CONCURRENCY & THREAD SAFETY
# ==============================================================================

def test_worldstream_server_concurrent_streaming() -> None:
    cdc_buffer = CDCRingBuffer()
    cdc_buffer.clear()
    server = WorldStreamServer(cdc_buffer=cdc_buffer)

    num_threads = 4
    steps_per_thread = 50

    def worker_generate(thread_id: int) -> None:
        for s in range(steps_per_thread):
            server.generate_step(sim_time=s * 0.02)

    def worker_inject(thread_id: int) -> None:
        for s in range(steps_per_thread):
            ev = WorldEvent(
                timestamp_ns=time.time_ns(),
                modality=EventModality.SOCIAL_TEXT,
                priority=EventPriority.NORMAL,
                payload_bytes=f"Worker {thread_id} step {s}".encode("utf-8"),
            )
            server.inject_event(ev)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads * 2) as executor:
        futures = []
        for t in range(num_threads):
            futures.append(executor.submit(worker_generate, t))
            futures.append(executor.submit(worker_inject, t))
        concurrent.futures.wait(futures)

    expected_total = num_threads * steps_per_thread * 2
    assert server.events_generated == num_threads * steps_per_thread
    assert server.events_injected == num_threads * steps_per_thread
    assert cdc_buffer.size() == expected_total
