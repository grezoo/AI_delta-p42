"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Adversarial Stress Test: WorldStream Server & Bootstrap Loader (Milestone 1)
Author: Empirical Challenger 1 (Sensory & Bootstrap Specialist)

Tests:
1. WorldEvent binary struct packing/unpacking: exact 280 bytes, IEEE 754 precision, padding.
2. WorldStream & CDCRingBuffer throughput: 10^4 events, 100% CRC-64 integrity, zero loss under capacity.
3. Burst overfill & drop-oldest policy: 65,536 capacity saturation + 10^4 overflow.
4. Backpressure zero-drop rejection when buffer is full.
5. Concurrent multithreaded injection.
6. Cold start Monte Carlo perturbation sweep (500 iterations): theta(0) in [15.1 deg, 16.5 deg], V_0 +- 0.05.
7. Extreme corner cases and boundary condition enforcement.
"""

from __future__ import annotations

import concurrent.futures
import math
import struct
import threading
import time
from typing import List, Tuple
import uuid

import numpy as np
import pytest

from comm.cdc_ring_buffer import CDCRingBuffer, is_native_backend_available
from comm.cognitive_packet import (
    CognitivePacket,
    PACKET_SIZE,
    compute_crc64,
    validate_crc,
)
from core.maslow_integrator import MaslowIntegrator
from core.theta_engine import DEG2RAD, ThetaEngine
from nursery.bootstrap_loader import (
    DEFAULT_V0,
    RAMP_UP_DURATION_S,
    THETA_0_DEG,
    THETA_BOOTSTRAP_DEG,
    BootstrapLoader,
    ConvergenceResult,
)
from nursery.worldstream_server import (
    MAX_PAYLOAD_BYTES,
    WORLD_EVENT_FORMAT,
    WORLD_EVENT_SIZE,
    EventModality,
    EventPriority,
    WorldEvent,
    WorldStreamServer,
)


# ==============================================================================
# SECTION 1: WorldEvent 280-Byte Binary Layout & Unpack Stress Tests
# ==============================================================================

class TestWorldEventBinaryLayout:
    """Stress-test binary layout, padding, and edge-case serialization for WorldEvent."""

    def test_exact_struct_size(self):
        """Verify binary format string produces exactly 280 bytes."""
        assert WORLD_EVENT_SIZE == 280
        assert struct.calcsize(WORLD_EVENT_FORMAT) == 280

    def test_roundtrip_edge_case_payloads(self):
        """Verify byte-exact roundtrip with empty, full, and binary payloads."""
        test_payloads = [
            b"",                                  # 0 bytes
            b"A",                                 # 1 byte
            b"X" * 128,                           # 128 bytes
            b"\x00" * 256,                        # 256 null bytes
            b"\xff" * 256,                        # 256 max bytes
            bytes(range(256)),                    # Full range 0x00..0xFF
            np.random.bytes(256),                 # Random binary noise
        ]

        for payload in test_payloads:
            ev = WorldEvent(
                timestamp_ns=1_700_000_000_123_456_789,
                modality=EventModality.VISUAL,
                priority=EventPriority.NORMAL,
                payload_bytes=payload,
                ambient_valence=-0.75,
            )
            raw = ev.to_bytes()
            assert len(raw) == 280, f"Expected 280 bytes, got {len(raw)}"

            # Unpack and verify exact fields
            restored = WorldEvent.from_bytes(raw, event_uuid=ev.event_uuid)
            assert restored.timestamp_ns == ev.timestamp_ns
            assert restored.modality == ev.modality
            assert restored.priority == ev.priority
            assert restored.payload_bytes == ev.payload_bytes
            assert restored.payload_size == len(payload)
            assert abs(restored.ambient_valence - ev.ambient_valence) < 1e-6
            assert restored.event_uuid == ev.event_uuid

    def test_valence_clamping(self):
        """Verify ambient valence clamping to [-1.0, 1.0]."""
        ev_low = WorldEvent(
            timestamp_ns=100,
            modality=EventModality.ACOUSTIC,
            priority=EventPriority.BACKGROUND,
            payload_bytes=b"low",
            ambient_valence=-5.0,
        )
        assert ev_low.ambient_valence == -1.0

        ev_high = WorldEvent(
            timestamp_ns=100,
            modality=EventModality.ACOUSTIC,
            priority=EventPriority.BACKGROUND,
            payload_bytes=b"high",
            ambient_valence=5.0,
        )
        assert ev_high.ambient_valence == 1.0

    def test_oversized_payload_truncation(self):
        """Payloads exceeding 256 bytes must be clamped cleanly to 256 bytes."""
        oversized = b"Z" * 512
        ev = WorldEvent(
            timestamp_ns=100,
            modality=EventModality.PROPRIOCEPTIVE,
            priority=EventPriority.PING,
            payload_bytes=oversized,
        )
        assert len(ev.payload_bytes) == 256
        assert ev.payload_bytes == b"Z" * 256
        raw = ev.to_bytes()
        assert len(raw) == 280

    def test_malformed_buffer_unpack_rejection(self):
        """Unpacking a buffer shorter than 280 bytes must raise ValueError."""
        with pytest.raises(ValueError, match="Expected at least 280 bytes"):
            WorldEvent.from_bytes(b"\x00" * 279)


# ==============================================================================
# SECTION 2: WorldStream & CDC Ring Buffer Throughput (10^4 Events)
# ==============================================================================

class TestWorldStreamCDCThroughput:
    """Empirical throughput, CRC-64 verification, and drop behavior under load."""

    @pytest.mark.parametrize("backend", ["python", "c"] if is_native_backend_available() else ["python"])
    def test_throughput_10k_events_zero_drop(self, backend: str):
        """
        Push 10,000 synthetic events into CDCRingBuffer.
        Verify:
        - 0 dropped packets
        - Buffer size == 10,000
        - Strict sequential monotonic order
        - 100% CRC-64 integrity across all 10,000 packets
        - Clean pop until empty
        """
        buffer = CDCRingBuffer(backend=backend)
        server = WorldStreamServer(cdc_buffer=buffer, sample_rate_hz=50.0)

        n_events = 10_000
        start_time = time.perf_counter()

        # Push 10,000 events
        for step_idx in range(n_events):
            sim_time = step_idx * server.dt
            ev, pkt = server.generate_step(sim_time=sim_time)
            assert pkt.sequence_id == step_idx + 1

        elapsed = time.perf_counter() - start_time

        # Verify counters
        assert server.events_generated == n_events
        assert server.packets_dropped == 0
        assert buffer.dropped_count == 0
        assert buffer.size() == n_events

        # Pop all packets and verify 100% CRC-64 integrity & sequencing
        popped_count = 0
        expected_seq = 1

        while not buffer.is_empty():
            pkt = buffer.pop()
            assert pkt is not None
            assert pkt.sequence_id == expected_seq
            assert pkt.validate_crc(), f"CRC-64 verification failed on packet {expected_seq}"

            raw = pkt.to_bytes()
            assert len(raw) == PACKET_SIZE
            assert validate_crc(raw), f"Byte-level CRC-64 validation failed on packet {expected_seq}"

            popped_count += 1
            expected_seq += 1

        assert popped_count == n_events
        assert buffer.size() == 0
        assert buffer.is_empty()

    @pytest.mark.parametrize("backend", ["python", "c"] if is_native_backend_available() else ["python"])
    def test_burst_overfill_drop_oldest(self, backend: str):
        """
        Fill buffer to maximum capacity (65,536), then push 10,000 additional events
        with drop_oldest=True.
        Verify:
        - Buffer size stays at 65,536
        - dropped_count == 10,000
        - Popped packets start from sequence 10,001 (oldest 10,000 dropped)
        - 100% CRC-64 integrity on all retained packets
        """
        buffer = CDCRingBuffer(backend=backend)
        server = WorldStreamServer(cdc_buffer=buffer)

        # 1. Fill buffer to capacity (65,536)
        capacity = buffer.capacity
        for i in range(capacity):
            ev = WorldEvent(
                timestamp_ns=i,
                modality=EventModality.VISUAL,
                priority=EventPriority.NORMAL,
                payload_bytes=struct.pack("<I", i),
            )
            ok = server.stream_event(ev, drop_oldest=False)
            assert ok, f"Failed to push packet {i} below capacity"

        assert buffer.is_full()
        assert buffer.size() == capacity
        assert buffer.dropped_count == 0

        # 2. Push 10,000 overflow events with drop_oldest=True
        overflow_count = 10_000
        for i in range(overflow_count):
            ev = WorldEvent(
                timestamp_ns=capacity + i,
                modality=EventModality.ACOUSTIC,
                priority=EventPriority.NORMAL,
                payload_bytes=struct.pack("<I", capacity + i),
            )
            ok = server.stream_event(ev, drop_oldest=True)
            assert ok, f"drop_oldest push failed on packet {i}"

        assert buffer.size() == capacity
        assert buffer.dropped_count == overflow_count

        # 3. Read first retained packet - sequence must be 10,001 (1-based: 10001)
        first_pkt = buffer.pop()
        assert first_pkt is not None
        assert first_pkt.sequence_id == overflow_count + 1
        assert first_pkt.validate_crc()

        # Check a sample of subsequent packets
        retained_seq = overflow_count + 2
        for _ in range(1000):
            p = buffer.pop()
            assert p is not None
            assert p.sequence_id == retained_seq
            assert p.validate_crc()
            retained_seq += 1

    def test_zero_drop_backpressure_rejection(self):
        """
        Verify that when buffer is full, drop_oldest=False cleanly rejects new events
        without data loss or corruption.
        """
        buffer = CDCRingBuffer(backend="python")
        server = WorldStreamServer(cdc_buffer=buffer)

        # Fill to capacity
        for i in range(buffer.capacity):
            ev = WorldEvent(
                timestamp_ns=i,
                modality=EventModality.VISUAL,
                priority=EventPriority.NORMAL,
                payload_bytes=b"fill",
            )
            server.stream_event(ev, drop_oldest=False)

        assert buffer.is_full()
        initial_dropped = server.packets_dropped

        # Attempt to generate a step (defaults to drop_oldest=False)
        _, pkt = server.generate_step()
        assert server.packets_dropped == initial_dropped + 1
        assert buffer.size() == buffer.capacity

        # Attempt to inject an event
        ev_extra = WorldEvent(
            timestamp_ns=999999,
            modality=EventModality.SOCIAL_TEXT,
            priority=EventPriority.PING,
            payload_bytes=b"extra",
        )
        ok = server.inject_event(ev_extra, push_to_buffer=True)
        assert not ok
        assert server.packets_dropped == initial_dropped + 2

    def test_concurrent_multithreaded_push_native_mpmc(self):
        """
        Stress-test WorldStreamServer under concurrent multi-producer worker injection
        using the native C++ backend (which supports MPMC via atomic CAS).
        """
        if not is_native_backend_available():
            pytest.skip("Native C++ CDC DLL not available on this platform")

        buffer = CDCRingBuffer(backend="c")
        server = WorldStreamServer(cdc_buffer=buffer)
        num_threads = 4
        events_per_thread = 2_500
        total_events = num_threads * events_per_thread

        def worker_push(worker_id: int):
            for i in range(events_per_thread):
                ev = WorldEvent(
                    timestamp_ns=time.time_ns(),
                    modality=EventModality.ACOUSTIC,
                    priority=EventPriority.NORMAL,
                    payload_bytes=f"Worker {worker_id} msg {i}".encode("utf-8"),
                )
                server.inject_event(ev, push_to_buffer=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_push, tid) for tid in range(num_threads)]
            concurrent.futures.wait(futures)

        assert server.events_injected == total_events
        assert buffer.size() == total_events
        assert server.packets_dropped == 0

        # Verify all 10,000 packets are intact and valid
        popped = 0
        seqs = set()
        while not buffer.is_empty():
            p = buffer.pop()
            if p is not None:
                popped += 1
                seqs.add(p.sequence_id)
                assert p.validate_crc()

        assert popped == total_events
        assert len(seqs) == total_events

    def test_concurrent_spsc_streaming_python_backend(self):
        """
        Test pure Python backend under the canonical Single-Producer Single-Consumer (SPSC)
        concurrent streaming model: Producer thread pushes 10,000 events while Consumer thread pops.
        """
        buffer = CDCRingBuffer(backend="python")
        server = WorldStreamServer(cdc_buffer=buffer)
        total_events = 10_000

        popped_packets: List[CognitivePacket] = []
        consumer_done = threading.Event()

        def producer_worker():
            for i in range(total_events):
                ev = WorldEvent(
                    timestamp_ns=time.time_ns(),
                    modality=EventModality.VISUAL,
                    priority=EventPriority.NORMAL,
                    payload_bytes=f"SPSC msg {i}".encode("utf-8"),
                )
                server.stream_event(ev, drop_oldest=False)

        def consumer_worker():
            while len(popped_packets) < total_events:
                pkt = buffer.pop(timeout=0.01)
                if pkt is not None:
                    popped_packets.append(pkt)
            consumer_done.set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f_prod = executor.submit(producer_worker)
            f_cons = executor.submit(consumer_worker)
            f_prod.result()
            f_cons.result()

        assert len(popped_packets) == total_events
        assert buffer.is_empty()
        for idx, p in enumerate(popped_packets):
            assert p.sequence_id == idx + 1
            assert p.validate_crc()


# ==============================================================================
# SECTION 3: Cold Start & Limit Cycle Parameter Sweep (Monte Carlo)
# ==============================================================================

class TestColdStartMonteCarloSweep:
    """
    Monte Carlo perturbation sweep of initial angle theta(0) in [15.1 deg, 16.5 deg]
    and deficit noise V_0 +- 0.05.
    Verifies 100% convergence, bounded orbit, and zero numeric shock.
    """

    def test_monte_carlo_sweep_500_iterations(self):
        """
        Execute 500 Monte Carlo runs across the continuous parameter domain:
          theta(0) ~ Uniform(15.1 deg, 16.5 deg)
          V_0,i = clip(DEFAULT_V0,i + Uniform(-0.05, +0.05), 0.01, 1.0)
        """
        np.random.seed(1337)
        num_trials = 500
        v0_base = DEFAULT_V0.copy()

        thetas = np.random.uniform(15.1, 16.5, size=num_trials)
        v0_noises = np.random.uniform(-0.05, 0.05, size=(num_trials, 5))

        passed_count = 0
        convergence_times = []
        periods = []
        cvs = []

        for i in range(num_trials):
            th_init = float(thetas[i])
            v0_perturbed = np.clip(v0_base + v0_noises[i], 0.01, 1.0)

            loader = BootstrapLoader(V_0=v0_perturbed, theta_0_deg=th_init)
            res = loader.run_bootstrap_simulation(duration_s=6.0, dt=0.02)

            assert not res.numeric_shock, f"Numeric shock in trial {i}: th={th_init}, res={res}"
            assert res.stable, f"Instability in trial {i}: th={th_init}, res={res}"
            assert res.convergence_time_s <= 3.0, f"Slow convergence in trial {i}: {res.convergence_time_s} s"
            assert 1.0 <= res.first_cycle_period_s <= 5.0, f"Period out of bounds: {res.first_cycle_period_s} s"
            assert res.peak_cv <= 0.05, f"Peak CV too high: {res.peak_cv}"
            assert res.passed, f"Trial {i} failed DVP-09: th={th_init}, v0={v0_perturbed}"

            passed_count += 1
            convergence_times.append(res.convergence_time_s)
            periods.append(res.first_cycle_period_s)
            cvs.append(res.peak_cv)

        # 100% convergence rate
        assert passed_count == num_trials
        # Verify stats
        assert max(convergence_times) <= 3.0
        assert max(cvs) <= 0.05

    def test_corner_cases_extrema(self):
        """
        Test the 4 extreme corners of the parameter bounding box:
        Corner 1: theta=15.1 deg (min), V_0 = base - 0.05 (min)
        Corner 2: theta=15.1 deg (min), V_0 = base + 0.05 (max)
        Corner 3: theta=16.5 deg (max), V_0 = base - 0.05 (min)
        Corner 4: theta=16.5 deg (max), V_0 = base + 0.05 (max)
        """
        corners = [
            (15.1, DEFAULT_V0 - 0.05),
            (15.1, DEFAULT_V0 + 0.05),
            (16.5, DEFAULT_V0 - 0.05),
            (16.5, DEFAULT_V0 + 0.05),
        ]

        for idx, (th_deg, v0) in enumerate(corners):
            v0_clean = np.clip(v0, 0.01, 1.0)
            loader = BootstrapLoader(V_0=v0_clean, theta_0_deg=th_deg)
            res = loader.run_bootstrap_simulation(duration_s=6.0)

            assert res.passed, f"Corner {idx + 1} failed: th={th_deg}, v0={v0_clean}, res={res}"
            assert not res.numeric_shock
            assert res.convergence_time_s <= 3.0
            assert 1.0 <= res.first_cycle_period_s <= 5.0
            assert res.peak_cv <= 0.05
            assert res.stable

    def test_bootstrap_loader_validation_and_sigmoid(self):
        """Test input rejection and mathematical properties of g_boot(t)."""
        # theta <= 15.0 deg must raise ValueError (must exceed resting attractor to break symmetry)
        with pytest.raises(ValueError, match="must exceed resting attractor"):
            BootstrapLoader(theta_0_deg=15.0)

        with pytest.raises(ValueError, match="must exceed resting attractor"):
            BootstrapLoader(theta_0_deg=14.5)

        # V_0 dimension check
        with pytest.raises(ValueError, match="must be a 5-element array"):
            BootstrapLoader(V_0=[0.1, 0.2, 0.3])

        # V_0 range check
        with pytest.raises(ValueError, match="elements must be in"):
            BootstrapLoader(V_0=[-0.1, 0.1, 0.2, 0.3, 0.4])

        loader = BootstrapLoader()
        # g_boot(0) ~= 1 / (1 + exp(3)) ~= 0.0474
        g0 = loader.get_boot_gain(0.0)
        assert abs(g0 - 0.04742587) < 1e-4

        # g_boot(1.5) == 0.50
        g15 = loader.get_boot_gain(1.5)
        assert abs(g15 - 0.50) < 1e-6

        # g_boot(3.0) == 1.00
        g30 = loader.get_boot_gain(3.0)
        assert g30 == 1.00

        # Monotonically non-decreasing
        t_vals = np.linspace(0.0, 3.5, 36)
        g_vals = [loader.get_boot_gain(t) for t in t_vals]
        for i in range(1, len(g_vals)):
            assert g_vals[i] >= g_vals[i - 1] - 1e-12
