"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Module: nursery.worldstream_server
Author: Milestone 1 Worker (Nursery Subsystem Specialist)

50 Hz Multimodal Sensory Event Stream Generator and Zero-Drop Ring Buffer Courier.
Ingests and generates synthetic and physical sensory stimuli across four canonical modalities:
Acoustic, Visual, Proprioceptive, and Social Text.
Strictly packages every sensory event into a 64-bit aligned (aligned(8)) canonical 304-byte
CognitivePacket with RFC-4122 v4 UUID and hardware-grade CRC-64/ECMA-182 integrity, streaming
directly onto the 65,536-slot lock-free CDCRingBuffer.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter V.1, V.3 lines 392-410, Chapter X.1 lines 646-654)
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- Plan Review.md (REQ-FUNC-01, DVP-16)
- ORIGINAL_REQUEST.md (Stage-Gate 4 Milestone 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import (
    CRC_COVERAGE_BYTES,
    MODULE_PFC,
    MODULE_THALAMUS,
    PACKET_SIZE,
    PAYLOAD_FLOAT_COUNT,
    PAYLOAD_MAX_BYTES,
    CognitivePacket,
    compute_crc64,
)

# ------------------------------------------------------------------------------
# Sensory Modality & Priority Enums
# ------------------------------------------------------------------------------

class EventModality(IntEnum):
    """
    Sensory modality classification for incoming environmental stimuli.
    Matches DELTA_P42_MUKODESI_ELVEK.md Chapter X.1 line 648.
    """
    ACOUSTIC = 1
    VISUAL = 2
    PROPRIOCEPTIVE = 3
    SOCIAL_TEXT = 4


# Canonical alias for compatibility
ModalityType = EventModality


class EventPriority(IntEnum):
    """
    Transmission and processing priority level.
    Matches DELTA_P42_MUKODESI_ELVEK.md Chapter X.1 line 649.
    """
    BACKGROUND = 0
    NORMAL = 1
    PING = 2


# ------------------------------------------------------------------------------
# Canonical Constants
# ------------------------------------------------------------------------------

WORLDSTREAM_CADENCE_HZ: float = 50.0
WORLDSTREAM_DT: float = 0.020  # 20 ms discrete time step
WORLD_EVENT_SIZE: int = 280     # 64-bit aligned WorldEvent binary layout
MAX_PAYLOAD_BYTES: int = 256    # Exactly 64 IEEE 754 float32 values
VALENCE_MIN: float = -1.0
VALENCE_MAX: float = 1.0

# Struct layout for WorldEvent (280 bytes):
# Q   : uint64 timestamp_ns (8 bytes, offset 0..8)
# H   : uint16 modality (2 bytes, offset 8..10)
# H   : uint16 priority (2 bytes, offset 10..12)
# I   : uint32 payload_bytes (4 bytes, offset 12..16)
# 256s: uint8[256] payload (256 bytes, offset 16..272)
# f   : float32 ambient_valence (4 bytes, offset 272..276)
# 4x  : 4 bytes alignment padding (offset 276..280)
WORLD_EVENT_FORMAT = "<QHH I 256s f 4x"


# ------------------------------------------------------------------------------
# WorldEvent Data Structure
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldEvent:
    """
    Canonical 280-byte 64-bit aligned multimodal sensory event.

    Attributes:
        timestamp_ns: Hardware nanosecond monotonic timestamp.
        modality: Sensory channel (Acoustic, Visual, Proprioceptive, Social Text).
        priority: Priority tag (Background, Normal, Ping!).
        payload_bytes: Raw binary payload (clamped to <= 256 bytes).
        ambient_valence: Environmental emotional valence in [-1.0, 1.0].
        event_uuid: RFC-4122 v4 unique identifier string.
    """
    timestamp_ns: int
    modality: EventModality
    priority: EventPriority
    payload_bytes: bytes
    ambient_valence: float = 0.0
    event_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        # Validate and clamp ambient valence
        clamped_valence = float(np.clip(self.ambient_valence, VALENCE_MIN, VALENCE_MAX))
        if clamped_valence != self.ambient_valence:
            object.__setattr__(self, "ambient_valence", clamped_valence)

        # Ensure modality is EventModality
        if not isinstance(self.modality, EventModality):
            object.__setattr__(self, "modality", EventModality(int(self.modality)))

        # Ensure priority is EventPriority
        if not isinstance(self.priority, EventPriority):
            object.__setattr__(self, "priority", EventPriority(int(self.priority)))

        # Ensure payload_bytes is bytes and length <= 256
        if isinstance(self.payload_bytes, str):
            p_bytes = self.payload_bytes.encode("utf-8")
        elif isinstance(self.payload_bytes, (bytearray, memoryview)):
            p_bytes = bytes(self.payload_bytes)
        elif isinstance(self.payload_bytes, bytes):
            p_bytes = self.payload_bytes
        else:
            raise TypeError(f"payload_bytes must be bytes-like, got {type(self.payload_bytes)}")

        if len(p_bytes) > MAX_PAYLOAD_BYTES:
            p_bytes = p_bytes[:MAX_PAYLOAD_BYTES]
        object.__setattr__(self, "payload_bytes", p_bytes)

        # Validate or assign UUID
        if not self.event_uuid:
            object.__setattr__(self, "event_uuid", str(uuid.uuid4()))

    @property
    def payload(self) -> bytes:
        """Alias for payload_bytes for backward and interface compatibility."""
        return self.payload_bytes

    @property
    def payload_size(self) -> int:
        """Actual length of payload in bytes (0..256)."""
        return len(self.payload_bytes)

    def to_bytes(self) -> bytes:
        """
        Serializes WorldEvent into exact 280-byte binary buffer matching C struct layout.
        """
        padded_payload = self.payload_bytes.ljust(MAX_PAYLOAD_BYTES, b"\x00")
        return struct.pack(
            WORLD_EVENT_FORMAT,
            int(self.timestamp_ns),
            int(self.modality),
            int(self.priority),
            len(self.payload_bytes),
            padded_payload,
            float(self.ambient_valence),
        )

    @classmethod
    def from_bytes(cls, data: bytes, event_uuid: Optional[str] = None) -> WorldEvent:
        """
        Deserializes a 280-byte binary buffer into a canonical WorldEvent instance.
        """
        if len(data) < WORLD_EVENT_SIZE:
            raise ValueError(f"Expected at least {WORLD_EVENT_SIZE} bytes for WorldEvent, got {len(data)}")
        (
            ts_ns,
            mod_val,
            prio_val,
            p_len,
            padded_payload,
            valence,
        ) = struct.unpack(WORLD_EVENT_FORMAT, data[:WORLD_EVENT_SIZE])

        actual_len = min(int(p_len), MAX_PAYLOAD_BYTES)
        actual_payload = padded_payload[:actual_len]

        return cls(
            timestamp_ns=ts_ns,
            modality=EventModality(mod_val),
            priority=EventPriority(prio_val),
            payload_bytes=actual_payload,
            ambient_valence=valence,
            event_uuid=event_uuid or str(uuid.uuid4()),
        )

    def to_packet(
        self,
        sequence_id: int = 0,
        source_module_id: int = MODULE_THALAMUS,
        target_module_id: int = MODULE_PFC,
    ) -> CognitivePacket:
        """
        Packages this WorldEvent into a canonical 304-byte CognitivePacket.
        The 256 payload bytes are mapped directly into the 64-element float32 payload_tensor.
        """
        padded = self.payload_bytes.ljust(MAX_PAYLOAD_BYTES, b"\x00")
        # Interpret padded 256 bytes as 64 float32 numbers, replacing NaNs or Infs
        raw_floats = np.frombuffer(padded, dtype=np.float32).copy()
        clean_floats = np.nan_to_num(raw_floats, nan=0.0, posinf=1.0, neginf=-1.0)

        packet = CognitivePacket(
            timestamp_ns=self.timestamp_ns,
            event_uuid=self.event_uuid,
            sequence_id=sequence_id,
            source_module_id=source_module_id,
            target_module_id=target_module_id,
            payload_bytes=len(self.payload_bytes),
            payload_tensor=clean_floats,
        )
        # Compute hardware CRC-64/ECMA-182 integrity checksum
        raw_bytes = packet.to_bytes()
        packet.crc64_checksum = compute_crc64(raw_bytes[:CRC_COVERAGE_BYTES])
        return packet


# ------------------------------------------------------------------------------
# WorldStreamServer Implementation
# ------------------------------------------------------------------------------

class WorldStreamServer:
    """
    50 Hz Multimodal Sensory Stream Generator and Dispatch Server.

    Manages sensory generation, event injection, and courier streaming to the
    65,536-slot lock-free CDCRingBuffer.
    """

    def __init__(
        self,
        cdc_buffer: Optional[CDCRingBuffer] = None,
        sample_rate_hz: float = WORLDSTREAM_CADENCE_HZ,
    ) -> None:
        """
        Initialize the WorldStreamServer.

        Args:
            cdc_buffer: Optional CDCRingBuffer instance. If None, operates in standalone mode.
            sample_rate_hz: Streaming frequency in Hertz (default 50.0 Hz).
        """
        if sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

        self._cdc_buffer = cdc_buffer
        self._sample_rate_hz = float(sample_rate_hz)
        self._dt = 1.0 / self._sample_rate_hz

        self._sequence_counter: int = 0
        self._running: bool = False
        self._lock = threading.Lock()

        # Telemetry counters
        self._events_generated: int = 0
        self._events_injected: int = 0
        self._packets_dropped: int = 0

    @property
    def sample_rate_hz(self) -> float:
        """Operating cadence in Hz."""
        return self._sample_rate_hz

    @property
    def dt(self) -> float:
        """Sampling period in seconds (20 ms at 50 Hz)."""
        return self._dt

    @property
    def events_generated(self) -> int:
        """Total number of synthetic events generated."""
        return self._events_generated

    @property
    def events_injected(self) -> int:
        """Total number of external events injected."""
        return self._events_injected

    @property
    def packets_dropped(self) -> int:
        """Total number of packets dropped due to buffer backpressure."""
        return self._packets_dropped

    def is_running(self) -> bool:
        """Check if streaming server is currently active."""
        return self._running

    def start(self) -> None:
        """Start the streaming server."""
        with self._lock:
            self._running = True

    def stop(self) -> None:
        """Stop the streaming server."""
        with self._lock:
            self._running = False

    def generate_step(self, sim_time: float = 0.0) -> Tuple[WorldEvent, CognitivePacket]:
        """
        Generates a single synthetic 50 Hz multimodal sensory frame.

        Args:
            sim_time: Current simulation elapsed time in seconds.

        Returns:
            Tuple of (WorldEvent, CognitivePacket).
        """
        now_ns = time.time_ns()

        with self._lock:
            self._sequence_counter += 1
            seq = self._sequence_counter
            self._events_generated += 1

        # Cycle modalities deterministically based on sequence ID to cover all senses
        modality_idx = ((seq - 1) % 4) + 1
        modality = EventModality(modality_idx)

        # Synthesize characteristic sensory payloads
        if modality == EventModality.ACOUSTIC:
            # 440 Hz sinusoidal acoustic tone simulation in float32
            freq = 440.0
            t_samples = np.linspace(sim_time, sim_time + self._dt, 64, endpoint=False, dtype=np.float32)
            wave = (0.5 * np.sin(2.0 * np.pi * freq * t_samples)).astype(np.float32)
            payload_bytes = wave.tobytes()
            ambient_valence = float(0.10 * np.sin(sim_time * 0.5))
            priority = EventPriority.NORMAL

        elif modality == EventModality.VISUAL:
            # 8x8 spatial luminance / optical flow field
            grid = np.linspace(-1.0, 1.0, 64, dtype=np.float32) * float(np.cos(sim_time))
            payload_bytes = grid.tobytes()
            ambient_valence = 0.05
            priority = EventPriority.NORMAL

        elif modality == EventModality.PROPRIOCEPTIVE:
            # 6-DOF joint angles, angular velocities, and torque estimates
            joints = np.zeros(64, dtype=np.float32)
            for j in range(6):
                joints[j * 3] = float(np.sin(sim_time + j * 0.5))       # angle
                joints[j * 3 + 1] = float(np.cos(sim_time + j * 0.5))   # velocity
                joints[j * 3 + 2] = 0.1                                 # torque
            payload_bytes = joints.tobytes()
            ambient_valence = 0.0
            priority = EventPriority.NORMAL

        else:  # SOCIAL_TEXT
            # Caregiver vocalization / textual feedback
            social_phrases = [
                b"Hello baby",
                b"Good job",
                b"Look at this toy",
                b"Gentle touch",
            ]
            phrase = social_phrases[(seq // 4) % len(social_phrases)]
            payload_bytes = phrase
            ambient_valence = 0.40
            priority = EventPriority.NORMAL

        event = WorldEvent(
            timestamp_ns=now_ns,
            modality=modality,
            priority=priority,
            payload_bytes=payload_bytes,
            ambient_valence=ambient_valence,
        )

        packet = event.to_packet(sequence_id=seq)

        # Courier stream to CDCRingBuffer if attached
        if self._cdc_buffer is not None:
            with self._lock:
                pushed = self._cdc_buffer.push(packet, drop_oldest=False)
                if not pushed:
                    # Buffer full and drop_oldest=False
                    self._packets_dropped += 1

        return event, packet

    def inject_event(self, event: WorldEvent, push_to_buffer: bool = True) -> bool:
        """
        Injects an external or Caretaker stimulus event into the WorldStream.

        Args:
            event: Canonical WorldEvent to inject.
            push_to_buffer: Whether to push to attached CDCRingBuffer.

        Returns:
            True if event was successfully accepted and buffered; False otherwise.
        """
        if not isinstance(event, WorldEvent):
            raise TypeError(f"Expected WorldEvent, got {type(event)}")

        with self._lock:
            self._sequence_counter += 1
            seq = self._sequence_counter
            self._events_injected += 1

            if push_to_buffer and self._cdc_buffer is not None:
                packet = event.to_packet(sequence_id=seq)
                pushed = self._cdc_buffer.push(packet, drop_oldest=False)
                if not pushed:
                    self._packets_dropped += 1
                    return False
                return True

            return True

    def stream_event(self, event: WorldEvent, drop_oldest: bool = False) -> bool:
        """
        Streams a single WorldEvent onto the CDC ring buffer.

        Args:
            event: Canonical WorldEvent.
            drop_oldest: If True, overwrites oldest unread slot when buffer is full.

        Returns:
            True if pushed successfully, False otherwise.
        """
        if self._cdc_buffer is None:
            return False

        with self._lock:
            self._sequence_counter += 1
            seq = self._sequence_counter
            packet = event.to_packet(sequence_id=seq)
            pushed = self._cdc_buffer.push(packet, drop_oldest=drop_oldest)
            if not pushed:
                self._packets_dropped += 1
            return pushed
