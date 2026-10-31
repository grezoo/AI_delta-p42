"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Deterministic Bus Package
Package: comm
Author: Milestone 1 Worker (Comm Subsystem Specialist)

Initializes the communication subsystem, exporting:
  - CognitivePacket: 304-byte, 64-bit aligned binary/tensor packet with CRC-64/ECMA-182.
  - CDCRingBuffer: 65,536-slot lock-free CDC FIFO ring buffer (C++20 DLL / Python emulator).
  - CDCRingBufferSpec: Specification dataclass with bitmask wrap-around indexing (& 0xFFFF).
  - compute_crc64: Hardware/vectorized CRC-64 checksum calculator.
  - validate_crc: Packet integrity validator.
"""

from comm.cognitive_packet import (
    CognitivePacket,
    PACKET_SIZE,
    CRC_COVERAGE_BYTES,
    PAYLOAD_FLOAT_COUNT,
    PAYLOAD_MAX_BYTES,
    MODULE_HYPOTHALAMUS,
    MODULE_THALAMUS,
    MODULE_CHORUS_LLM,
    MODULE_PFC,
    MODULE_AMYGDALA,
    MODULE_BASAL_GANGLIA,
    MODULE_CEREBELLUM,
    MODULE_HIPPOCAMPUS,
    compute_crc64,
    validate_crc,
)
from comm.cdc_ring_buffer import (
    CDCRingBuffer,
    CDCRingBufferSpec,
    CAPACITY,
    MASK,
    is_native_backend_available,
)

__all__ = [
    "CognitivePacket",
    "CDCRingBuffer",
    "CDCRingBufferSpec",
    "compute_crc64",
    "validate_crc",
    "CAPACITY",
    "MASK",
    "PACKET_SIZE",
    "CRC_COVERAGE_BYTES",
    "PAYLOAD_FLOAT_COUNT",
    "PAYLOAD_MAX_BYTES",
    "MODULE_HYPOTHALAMUS",
    "MODULE_THALAMUS",
    "MODULE_CHORUS_LLM",
    "MODULE_PFC",
    "MODULE_AMYGDALA",
    "MODULE_BASAL_GANGLIA",
    "MODULE_CEREBELLUM",
    "MODULE_HIPPOCAMPUS",
    "is_native_backend_available",
]
