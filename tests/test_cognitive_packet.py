"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Deterministic Bus Unit Tests
Test Suite: tests/test_cognitive_packet.py
Author: Milestone 1 Worker (Comm Subsystem Specialist)

Comprehensive unit and property tests for CognitivePacket:
  - Strict 304 bytes binary structure verification
  - 64-bit memory address alignment (aligned(8))
  - RFC-4122 v4 UUID generation, parsing, formatting, and collision resistance
  - Hardware/Vector CRC-64/ECMA-182 polynomial verification (0x42F0E1EBA9EA3693)
  - Canonical test vector: b"P42_COGNITIVE_CORE_CRC_TEST" == 0xD3307F914841FA1A
  - Single-bit corruption detection and quarantine
  - Struct boundary error handling (underfill/overfill)
  - Full backward compatibility with Gate 1 aliases (packet_id, payload)
"""

import struct
import uuid
import numpy as np
import pytest

from comm.cognitive_packet import (
    CognitivePacket,
    PACKET_FORMAT,
    HEADER_PAYLOAD_FORMAT,
    PACKET_SIZE,
    CRC_COVERAGE_BYTES,
    PAYLOAD_FLOAT_COUNT,
    PAYLOAD_MAX_BYTES,
    CRC64_ECMA_POLY,
    MODULE_THALAMUS,
    MODULE_PFC,
    compute_crc64,
    validate_crc,
)


class TestCognitivePacketStructure:
    """Verifies physical memory layout, alignment, and struct geometry."""

    def test_packet_size_is_strictly_304_bytes(self):
        """Struct format '<Q16sIHHII64fQ' must evaluate to exactly 304 bytes."""
        assert struct.calcsize(PACKET_FORMAT) == 304
        assert PACKET_SIZE == 304

        pkt = CognitivePacket(timestamp_ns=1000, sequence_id=1)
        raw = pkt.to_bytes()
        assert len(raw) == 304

    def test_header_payload_format_is_strictly_296_bytes(self):
        """Header + Payload covered by CRC-64 must be exactly 296 bytes."""
        assert struct.calcsize(HEADER_PAYLOAD_FORMAT) == 296
        assert CRC_COVERAGE_BYTES == 296

    def test_memory_alignment_64bit(self):
        """Packet and internal payload buffer must satisfy 64-bit (8-byte) alignment."""
        pkt = CognitivePacket(timestamp_ns=5000, sequence_id=10)
        assert pkt.is_aligned_64bit()
        # Verify 304 bytes is divisible by 8
        assert PACKET_SIZE % 8 == 0
        # Verify underlying numpy array pointer is 8-byte aligned
        assert (pkt.payload_tensor.ctypes.data % 8) == 0

    def test_invalid_buffer_size_raises_value_error(self):
        """Buffers not equal to 304 bytes must raise ValueError on deserialization."""
        with pytest.raises(ValueError, match="Invalid CognitivePacket buffer size"):
            CognitivePacket.from_bytes(b"\x00" * 300)

        with pytest.raises(ValueError, match="Invalid CognitivePacket buffer size"):
            CognitivePacket.from_bytes(b"\x00" * 308)

        with pytest.raises(ValueError, match="Invalid CognitivePacket buffer size"):
            CognitivePacket.from_bytes(b"")


class TestCRC64ECMA182Integrity:
    """Verifies CRC-64/ECMA-182 polynomial, test vectors, and error detection."""

    def test_crc64_canonical_test_vector(self):
        """
        Validates the exact canonical reference test vector:
        b"P42_COGNITIVE_CORE_CRC_TEST" must produce 0xD3307F914841FA1A.
        """
        test_payload = b"P42_COGNITIVE_CORE_CRC_TEST"
        result = compute_crc64(test_payload)
        assert result == 0xD3307F914841FA1A
        assert f"0x{result:016X}" == "0xD3307F914841FA1A"

    def test_crc64_deterministic_repeatability(self):
        """Identical inputs must yield identical checksums every time."""
        data = b"Arbitrary deterministic telemetry test string 12345"
        assert compute_crc64(data) == compute_crc64(data)

    def test_crc64_single_bit_flip_sensitivity(self):
        """Flipping a single bit anywhere in input must alter the CRC-64 checksum."""
        original = b"A" * 296
        base_crc = compute_crc64(original)

        for byte_idx in [0, 10, 100, 200, 295]:
            corrupted = bytearray(original)
            corrupted[byte_idx] ^= 0x01
            assert compute_crc64(corrupted) != base_crc

    def test_packet_crc_validation_success(self):
        """A freshly constructed and serialized packet must validate CRC successfully."""
        pkt = CognitivePacket(
            timestamp_ns=999999,
            sequence_id=42,
            source_module_id=MODULE_THALAMUS,
            target_module_id=MODULE_PFC,
            payload_tensor=np.linspace(0.1, 1.0, 64, dtype=np.float32),
        )
        raw = pkt.to_bytes()
        assert validate_crc(raw) is True
        assert pkt.validate_crc() is True

    def test_packet_corruption_detection(self):
        """Any bit-flip in the 296-byte body must trigger CRC validation failure."""
        pkt = CognitivePacket(
            timestamp_ns=100000,
            sequence_id=1,
            payload_tensor=np.ones(64, dtype=np.float32),
        )
        raw = bytearray(pkt.to_bytes())

        # Corrupt header byte (timestamp)
        raw[2] ^= 0xFF
        assert validate_crc(raw) is False
        with pytest.raises(ValueError, match="CRC-64 integrity violation"):
            CognitivePacket.from_bytes(raw, validate_checksum=True)

        # Corrupt payload float byte
        raw = bytearray(pkt.to_bytes())
        raw[150] ^= 0x80
        assert validate_crc(raw) is False


class TestRFC4122V4UUIDTracking:
    """Verifies RFC-4122 version 4 UUID creation, validation, and collision resistance."""

    def test_auto_generates_valid_v4_uuid(self):
        """Omitting UUID auto-generates a valid RFC-4122 v4 UUID."""
        pkt = CognitivePacket()
        parsed_uuid = uuid.UUID(pkt.event_uuid)
        assert parsed_uuid.version == 4
        assert len(pkt.event_uuid_bytes) == 16

    def test_explicit_uuid_types_accepted(self):
        """Accepts uuid.UUID instance, string, and raw 16 bytes."""
        u = uuid.uuid4()
        pkt1 = CognitivePacket(event_uuid=u)
        assert pkt1.event_uuid == str(u)
        assert pkt1.event_uuid_bytes == u.bytes

        pkt2 = CognitivePacket(event_uuid=str(u))
        assert pkt2.event_uuid == str(u)
        assert pkt2.event_uuid_bytes == u.bytes

        pkt3 = CognitivePacket(event_uuid=u.bytes)
        assert pkt3.event_uuid == str(u)
        assert pkt3.event_uuid_bytes == u.bytes

    def test_invalid_uuid_length_raises_error(self):
        """Invalid byte lengths raise ValueError."""
        with pytest.raises(ValueError):
            CognitivePacket(event_uuid=b"too_short")

    def test_uuid_collision_resistance_10000_packets(self):
        """10,000 consecutively generated packets must have zero duplicate UUIDs."""
        seen = set()
        for i in range(10000):
            pkt = CognitivePacket(sequence_id=i)
            seen.add(pkt.event_uuid_bytes)
        assert len(seen) == 10000


class TestPayloadTensorOperations:
    """Verifies tensor formatting, padding, truncation, and roundtrip precision."""

    def test_underfilled_tensor_padded_to_64_floats(self):
        """A 5D vector (e.g. Maslow vector) is padded to 64 floats, payload_bytes=20."""
        maslow_5d = [0.15, 0.10, 0.25, 0.10, 0.30]
        pkt = CognitivePacket(payload_tensor=maslow_5d)
        assert pkt.payload_bytes == 20
        assert pkt.payload_tensor.shape == (64,)
        np.testing.assert_allclose(pkt.payload_tensor[:5], maslow_5d, rtol=1e-5)
        np.testing.assert_allclose(pkt.payload_tensor[5:], 0.0)

    def test_overfilled_tensor_raises_value_error(self):
        """Tensors exceeding 64 floats raise ValueError."""
        with pytest.raises(ValueError, match="exceeds"):
            CognitivePacket(payload_tensor=np.zeros(65, dtype=np.float32))

    def test_float32_roundtrip_precision(self):
        """Floating point values preserve IEEE 754 single precision across roundtrip."""
        rng = np.random.default_rng(42)
        random_floats = rng.uniform(-100.0, 100.0, size=64).astype(np.float32)

        pkt = CognitivePacket(
            timestamp_ns=1234567890,
            sequence_id=777,
            payload_tensor=random_floats,
        )
        raw = pkt.to_bytes()
        pkt_deser = CognitivePacket.from_bytes(raw, validate_checksum=True)

        np.testing.assert_allclose(pkt_deser.payload_tensor, random_floats, rtol=1e-6)
        assert pkt_deser.timestamp_ns == 1234567890
        assert pkt_deser.sequence_id == 777


class TestBackwardCompatibilityGate1:
    """Verifies full backward compatibility with Gate 1 tests and interfaces."""

    def test_packet_id_alias_works(self):
        """packet_id maps seamlessly to sequence_id."""
        pkt = CognitivePacket(packet_id=101)
        assert pkt.sequence_id == 101
        assert pkt.packet_id == 101

        pkt.packet_id = 202
        assert pkt.sequence_id == 202
        assert pkt.packet_id == 202

    def test_payload_alias_works(self):
        """payload maps seamlessly to payload_tensor."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        pkt = CognitivePacket(payload=vec)
        assert pkt.payload[0] == 1.0
        assert pkt.payload_tensor[0] == 1.0

        pkt.payload = np.array([4.0, 5.0], dtype=np.float32)
        assert pkt.payload[0] == 4.0
        assert pkt.payload_tensor[0] == 4.0

    def test_to_dict_and_repr(self):
        """to_dict produces structured representation; repr is descriptive."""
        pkt = CognitivePacket(sequence_id=42, timestamp_ns=1000)
        d = pkt.to_dict()
        assert d["sequence_id"] == 42
        assert d["timestamp_ns"] == 1000
        assert "event_uuid" in d
        assert "crc64_checksum" in d

        rep = repr(pkt)
        assert "CognitivePacket" in rep
        assert "seq=42" in rep
