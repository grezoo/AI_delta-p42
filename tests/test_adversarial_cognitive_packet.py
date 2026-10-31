"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Adversarial Verification Suite
Test Suite: tests/test_adversarial_cognitive_packet.py
Author: Challenger 1 (Empirical Challenger - Gate 2 / Milestone 1)

Empirical Adversarial Challenges:
1. Exhaustive Fuzzing & Corruption:
   - 100% of all single-bit flips across header and payload bytes (0..295) [2,368 distinct bit positions].
   - 100% of all single-bit flips across CRC checksum bytes (296..303) [64 distinct bit positions].
   - 10,000 distinct multi-bit corruptions (2..32 simultaneous bit flips without replacement).
   - Byte-level burst corruptions, span zeroing, and span inversion.
2. Boundary Conditions & Numerical Edge Cases:
   - Empty tensors (0 floats), partial vectors (1..63 floats), full tensors (64 floats).
   - IEEE 754 special floats: quiet NaN, signaling NaN, +Inf, -Inf, subnormal numbers (1.4e-45),
     negative zero (-0.0), max float32 (3.4028235e+38), min normal float32 (1.17549435e-38).
   - Extreme timestamps: 0, 2^64 - 1, and negative/overflow rejection.
   - Sequence ID wrap: 0, 2^32 - 1, modular arithmetic, and negative/overflow rejection.
   - Module ID validation and range behavior (1..8 vs invalid IDs).
   - Strict buffer size rejection (< 304 bytes, > 304 bytes, empty buffer).
3. 64-bit Memory Alignment:
   - 10,000 iterations verifying strict 64-bit address alignment of internal and unpacked tensors.
4. RFC-4122 v4 UUID Compliance & Collision Resistance:
   - 100,000 packet generation verifying 16-byte length, version=4, variant=1, and 0 collisions.
5. Micro-benchmarks & Throughput Characterization:
   - Quantification of CRC-64 throughput, serialization throughput, and deserialization latency.
"""

from __future__ import annotations

import re
import struct
import time
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
    VALID_MODULE_IDS,
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


# ==============================================================================
# 1. FUZZING & BIT CORRUPTION SUITE
# ==============================================================================

class TestFuzzingAndBitCorruption:
    """
    Adversarial fuzzing verifying that CRC-64/ECMA-182 guarantees 100% detection
    of single-bit and multi-bit corruptions across header, payload, and CRC bytes.
    """

    @pytest.fixture
    def canonical_packet(self) -> CognitivePacket:
        """Constructs a deterministic reference CognitivePacket."""
        return CognitivePacket(
            timestamp_ns=1729876543210987654,
            sequence_id=42000,
            source_module_id=MODULE_THALAMUS,
            target_module_id=MODULE_PFC,
            payload_tensor=np.linspace(-50.0, 50.0, PAYLOAD_FLOAT_COUNT, dtype=np.float32),
        )

    def test_exhaustive_all_2368_single_bit_flips_header_and_payload(self, canonical_packet):
        """
        Exhaustively inverts EVERY SINGLE BIT across the entire 296 bytes (0..295)
        covered by CRC-64 (296 * 8 = 2,368 bit positions).
        Must achieve EXACTLY 100.0% detection rate (0 undetected).
        """
        raw = canonical_packet.to_bytes()
        assert len(raw) == PACKET_SIZE
        assert validate_crc(raw) is True

        undetected_count = 0
        total_bit_positions = CRC_COVERAGE_BYTES * 8  # 296 * 8 = 2368

        for byte_idx in range(CRC_COVERAGE_BYTES):
            for bit_idx in range(8):
                corrupted = bytearray(raw)
                corrupted[byte_idx] ^= (1 << bit_idx)

                # 1. validate_crc must return False
                if validate_crc(corrupted):
                    undetected_count += 1

                # 2. from_bytes with validate_checksum=True must raise ValueError
                with pytest.raises(ValueError, match="CRC-64 integrity violation"):
                    CognitivePacket.from_bytes(corrupted, validate_checksum=True)

        assert undetected_count == 0, (
            f"CRC-64 failed to detect {undetected_count} / {total_bit_positions} single-bit corruptions!"
        )

    def test_exhaustive_all_64_single_bit_flips_in_crc_checksum(self, canonical_packet):
        """
        Exhaustively inverts EVERY SINGLE BIT in the 8-byte stored checksum itself (bytes 296..303).
        Must achieve EXACTLY 100.0% detection rate.
        """
        raw = canonical_packet.to_bytes()
        undetected_count = 0

        for byte_idx in range(CRC_COVERAGE_BYTES, PACKET_SIZE):
            for bit_idx in range(8):
                corrupted = bytearray(raw)
                corrupted[byte_idx] ^= (1 << bit_idx)

                if validate_crc(corrupted):
                    undetected_count += 1

                with pytest.raises(ValueError, match="CRC-64 integrity violation"):
                    CognitivePacket.from_bytes(corrupted, validate_checksum=True)

        assert undetected_count == 0, (
            f"CRC-64 failed to detect corruption in checksum field itself: {undetected_count} undetected"
        )

    def test_randomized_multi_bit_corruptions_10000_trials(self, canonical_packet):
        """
        Evaluates 10,000 distinct multi-bit corruption patterns (2 to 32 bits flipped
        without replacement across the 2,368 bit positions).
        Must achieve 100.0% detection rate.
        """
        raw = canonical_packet.to_bytes()
        rng = np.random.default_rng(20261025)
        num_trials = 10000
        undetected = 0
        total_bits = CRC_COVERAGE_BYTES * 8

        for _ in range(num_trials):
            n_flips = int(rng.integers(2, 33))
            bit_positions = rng.choice(total_bits, size=n_flips, replace=False)
            corrupted = bytearray(raw)
            for pos in bit_positions:
                corrupted[pos // 8] ^= (1 << (pos % 8))

            assert corrupted != raw, "Corruption harness generated identical buffer"

            if validate_crc(corrupted):
                undetected += 1

        assert undetected == 0, f"Multi-bit corruption undetected: {undetected} / {num_trials}"

    def test_burst_byte_corruptions(self, canonical_packet):
        """
        Simulates physical bus burst interference: contiguous corruptions
        of 2, 4, 8, and 16 bytes across arbitrary offsets.
        """
        raw = canonical_packet.to_bytes()
        rng = np.random.default_rng(9999)

        for burst_len in [2, 4, 8, 16]:
            for offset in range(0, CRC_COVERAGE_BYTES - burst_len, 7):
                corrupted = bytearray(raw)
                burst = rng.bytes(burst_len)
                corrupted[offset : offset + burst_len] = burst
                if corrupted != raw:
                    assert validate_crc(corrupted) is False
                    with pytest.raises(ValueError, match="CRC-64 integrity violation"):
                        CognitivePacket.from_bytes(corrupted, validate_checksum=True)

    def test_span_zeroing_and_inversion(self, canonical_packet):
        """
        Tests destructive bus failure modes: partial span zeroing (0x00)
        and partial span saturation (0xFF).
        """
        raw = canonical_packet.to_bytes()
        for span_len in [4, 16, 64, 128]:
            # Zeroing span
            corrupted_zeros = bytearray(raw)
            corrupted_zeros[40 : 40 + span_len] = b"\x00" * span_len
            if corrupted_zeros != raw:
                assert validate_crc(corrupted_zeros) is False

            # Saturating span
            corrupted_ones = bytearray(raw)
            corrupted_ones[40 : 40 + span_len] = b"\xFF" * span_len
            if corrupted_ones != raw:
                assert validate_crc(corrupted_ones) is False


# ==============================================================================
# 2. BOUNDARY CONDITIONS & NUMERICAL EDGE CASES
# ==============================================================================

class TestBoundaryConditionsAndEdgeCases:
    """
    Adversarial boundary stress tests covering tensors, IEEE 754 float types,
    sequence numbers, timestamps, module IDs, and buffer boundaries.
    """

    def test_empty_tensor_boundary(self):
        """
        Empty tensor (0 elements) must pad to 64 zeros, set payload_bytes=0,
        serialize to exactly 304 bytes with valid CRC, and roundtrip cleanly.
        """
        pkt = CognitivePacket(payload_tensor=[])
        assert pkt.payload_bytes == 0
        assert pkt.payload_tensor.shape == (64,)
        np.testing.assert_array_equal(pkt.payload_tensor, np.zeros(64, dtype=np.float32))
        assert pkt.validate_crc() is True

        raw = pkt.to_bytes()
        assert len(raw) == PACKET_SIZE
        deser = CognitivePacket.from_bytes(raw, validate_checksum=True)
        assert deser.payload_bytes == 0
        np.testing.assert_array_equal(deser.payload_tensor, np.zeros(64, dtype=np.float32))
        assert deser.validate_crc() is True

    @pytest.mark.parametrize("length", [1, 5, 16, 32, 63])
    def test_partial_tensor_lengths_exact_padding(self, length: int):
        """Partial tensor lengths must pad with zeros to 64 elements and set payload_bytes=len*4."""
        data = list(range(1, length + 1))
        pkt = CognitivePacket(payload_tensor=data)
        assert pkt.payload_bytes == length * 4
        np.testing.assert_array_equal(pkt.payload_tensor[:length], np.array(data, dtype=np.float32))
        np.testing.assert_array_equal(pkt.payload_tensor[length:], np.zeros(64 - length, dtype=np.float32))

        raw = pkt.to_bytes()
        deser = CognitivePacket.from_bytes(raw, validate_checksum=True)
        assert deser.payload_bytes == length * 4
        np.testing.assert_array_equal(deser.payload_tensor[:length], np.array(data, dtype=np.float32))

    def test_ieee754_special_floats_preservation(self):
        """
        Verifies exact bit-level transmission and roundtrip of IEEE 754 special floats:
        quiet NaN, positive infinity, negative infinity, negative zero, subnormals, and extrema.
        """
        subnormal = np.float32(1.401298e-45)  # Smallest positive subnormal in float32
        max_float = np.finfo(np.float32).max
        tiny_float = np.finfo(np.float32).tiny

        special_values = [
            np.nan,
            np.inf,
            -np.inf,
            -0.0,
            subnormal,
            max_float,
            -max_float,
            tiny_float,
        ]
        # Pad up to 64 floats
        padded_specials = special_values + [0.0] * (64 - len(special_values))
        pkt = CognitivePacket(payload_tensor=padded_specials)
        assert pkt.validate_crc() is True

        raw = pkt.to_bytes()
        assert validate_crc(raw) is True

        deser = CognitivePacket.from_bytes(raw, validate_checksum=True)
        deser_tensor = deser.payload_tensor

        # Assert special value representations
        assert np.isnan(deser_tensor[0])
        assert np.isposinf(deser_tensor[1])
        assert np.isneginf(deser_tensor[2])
        # Negative zero check (copysign check)
        assert deser_tensor[3] == 0.0 and np.signbit(deser_tensor[3])
        # Subnormal precision check
        assert deser_tensor[4] == subnormal
        # Max float check
        assert deser_tensor[5] == max_float
        assert deser_tensor[6] == -max_float
        assert deser_tensor[7] == tiny_float

    def test_extreme_timestamps_uint64_limits(self):
        """Timestamp limits: 0 (Unix epoch) and 2^64 - 1 (18446744073709551615)."""
        # Epoch zero
        pkt_zero = CognitivePacket(timestamp_ns=0)
        raw_zero = pkt_zero.to_bytes()
        deser_zero = CognitivePacket.from_bytes(raw_zero, validate_checksum=True)
        assert deser_zero.timestamp_ns == 0

        # Maximum uint64
        max_uint64 = (1 << 64) - 1
        pkt_max = CognitivePacket(timestamp_ns=max_uint64)
        raw_max = pkt_max.to_bytes()
        deser_max = CognitivePacket.from_bytes(raw_max, validate_checksum=True)
        assert deser_max.timestamp_ns == max_uint64

        # Negative timestamp rejection
        with pytest.raises(struct.error):
            CognitivePacket(timestamp_ns=-1)

        # Overflow timestamp rejection
        with pytest.raises(struct.error):
            CognitivePacket(timestamp_ns=max_uint64 + 1)

    def test_sequence_id_uint32_limits_and_wraparound(self):
        """Sequence ID limits: 0, 2^32 - 1, and overflow boundaries."""
        pkt_zero = CognitivePacket(sequence_id=0)
        assert pkt_zero.sequence_id == 0

        max_uint32 = (1 << 32) - 1
        pkt_max = CognitivePacket(sequence_id=max_uint32)
        raw_max = pkt_max.to_bytes()
        deser_max = CognitivePacket.from_bytes(raw_max, validate_checksum=True)
        assert deser_max.sequence_id == max_uint32

        # Wrap-around simulation: max_uint32 -> 0
        wrapped_seq = (max_uint32 + 1) % (1 << 32)
        assert wrapped_seq == 0
        pkt_wrap = CognitivePacket(sequence_id=wrapped_seq)
        assert pkt_wrap.sequence_id == 0

        # Negative sequence rejection
        with pytest.raises(struct.error):
            CognitivePacket(sequence_id=-1)

        # Overflow sequence rejection
        with pytest.raises(struct.error):
            CognitivePacket(sequence_id=max_uint32 + 1)

    def test_module_ids_all_valid_constants(self):
        """All 8 canonical module IDs must be valid and roundtrip correctly."""
        for mod_id in VALID_MODULE_IDS:
            pkt = CognitivePacket(
                source_module_id=mod_id,
                target_module_id=((mod_id % 8) + 1),
            )
            raw = pkt.to_bytes()
            deser = CognitivePacket.from_bytes(raw, validate_checksum=True)
            assert deser.source_module_id == mod_id
            assert deser.target_module_id == ((mod_id % 8) + 1)

    def test_module_ids_out_of_range_rejections(self):
        """Negative and uint16 overflow module IDs must be rejected by struct packing."""
        with pytest.raises(struct.error):
            CognitivePacket(source_module_id=-1)

        with pytest.raises(struct.error):
            CognitivePacket(source_module_id=65536)

    @pytest.mark.parametrize("bad_size", [0, 1, 16, 128, 295, 296, 303, 305, 512, 1024])
    def test_strict_buffer_length_enforcement(self, bad_size: int):
        """Buffers with length != 304 must fail validate_crc and raise ValueError in from_bytes."""
        buf = b"\x00" * bad_size
        assert validate_crc(buf) is False

        with pytest.raises(ValueError, match="Invalid CognitivePacket buffer size"):
            CognitivePacket.from_bytes(buf)

    def test_oversized_payload_tensor_rejection(self):
        """Tensors exceeding 64 floats must raise ValueError immediately."""
        with pytest.raises(ValueError, match="exceeds maximum 64 floats"):
            CognitivePacket(payload_tensor=list(range(65)))

        with pytest.raises(ValueError, match="exceeds maximum 64 floats"):
            CognitivePacket(payload_tensor=np.ones(100, dtype=np.float32))


# ==============================================================================
# 3. 64-BIT MEMORY ALIGNMENT SUITE
# ==============================================================================

class TestMemoryAlignment64Bit:
    """
    Empirically verifies that serialized bytes, internal arrays, and unpacked
    tensors strictly maintain 64-bit (8-byte) memory alignment across 10,000 iterations.
    """

    def test_memory_alignment_across_10000_iterations(self):
        """
        Runs 10,000 iterations with varying tensor sizes and randomized data,
        verifying that internal NumPy buffers and unpacked buffers satisfy address % 8 == 0.
        """
        rng = np.random.default_rng(777)
        alignment_failures = 0
        total_iterations = 10000

        for i in range(total_iterations):
            # Vary tensor size between 0 and 64
            k = int(rng.integers(0, 65))
            tensor_data = rng.standard_normal(size=k).astype(np.float32)

            pkt = CognitivePacket(
                timestamp_ns=int(rng.integers(0, 10**12)),
                sequence_id=i,
                source_module_id=int(rng.choice(list(VALID_MODULE_IDS))),
                target_module_id=int(rng.choice(list(VALID_MODULE_IDS))),
                payload_tensor=tensor_data,
            )

            # 1. Instance alignment predicate
            if not pkt.is_aligned_64bit():
                alignment_failures += 1

            # 2. Internal payload tensor memory address must be 8-byte aligned
            if (pkt.payload_tensor.ctypes.data % 8) != 0:
                alignment_failures += 1

            # 3. Serialized byte length must be divisible by 8
            raw = pkt.to_bytes()
            if (len(raw) % 8) != 0:
                alignment_failures += 1

            # 4. Deserialized packet alignment
            deser = CognitivePacket.from_bytes(raw, validate_checksum=True)
            if not deser.is_aligned_64bit():
                alignment_failures += 1
            if (deser.payload_tensor.ctypes.data % 8) != 0:
                alignment_failures += 1

            # 5. to_numpy export alignment
            exported = deser.to_numpy()
            if (exported.ctypes.data % 8) != 0:
                alignment_failures += 1

        assert alignment_failures == 0, (
            f"Detected {alignment_failures} alignment violations across {total_iterations} iterations!"
        )


# ==============================================================================
# 4. RFC-4122 V4 UUID COMPLIANCE & ZERO COLLISION SUITE
# ==============================================================================

class TestRFC4122V4UUIDUniqueness:
    """
    Stress-tests UUID generation across 100,000 consecutively generated packets.
    Verifies:
      - RFC-4122 version 4 conformance (octet 6 high nibble == 4).
      - RFC-4122 variant 1 conformance (octet 8 high 2 bits == 2, '10' binary).
      - Zero collisions across 100,000 samples.
      - Canonical string regex matching: ^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$
    """

    UUID_V4_REGEX = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    def test_rfc4122_v4_compliance_and_zero_collisions_100k(self):
        """
        Generates 100,000 packets and validates RFC-4122 v4 compliance
        and absolute uniqueness (zero collisions).
        """
        num_packets = 100000
        seen_uuids = set()
        version_failures = 0
        variant_failures = 0
        regex_failures = 0
        length_failures = 0

        t0 = time.perf_counter()

        for i in range(num_packets):
            pkt = CognitivePacket(sequence_id=i)
            raw_uuid = pkt.event_uuid_bytes

            if len(raw_uuid) != 16:
                length_failures += 1

            # Version: bits 4..7 of octet 6 must equal 4
            version = raw_uuid[6] >> 4
            if version != 4:
                version_failures += 1

            # Variant: bits 6..7 of octet 8 must equal 2 (binary 10)
            variant = raw_uuid[8] >> 6
            if variant != 2:
                variant_failures += 1

            # Canonical hyphenated representation regex
            uuid_str = pkt.event_uuid
            if not self.UUID_V4_REGEX.match(uuid_str):
                regex_failures += 1

            seen_uuids.add(raw_uuid)

        t1 = time.perf_counter()
        elapsed = t1 - t0

        assert length_failures == 0, f"Length failures: {length_failures}"
        assert version_failures == 0, f"Version 4 violations: {version_failures}"
        assert variant_failures == 0, f"Variant violations: {variant_failures}"
        assert regex_failures == 0, f"String formatting violations: {regex_failures}"
        assert len(seen_uuids) == num_packets, (
            f"Collision detected! {num_packets - len(seen_uuids)} duplicate UUIDs in 100,000 packets"
        )

        # Performance metric check: generation rate should exceed 50,000 pkt/s
        rate = num_packets / elapsed
        assert rate > 50000, f"UUID generation throughput too slow: {rate:.0f} pkt/s"


# ==============================================================================
# 5. THROUGHPUT & BENCHMARK SUITE
# ==============================================================================

class TestPerformanceBenchmarks:
    """
    Benchmarks throughput and WCET latency of CRC-64 computation,
    serialization (to_bytes), and deserialization (from_bytes).
    """

    def test_crc64_and_serialization_throughput(self):
        """
        Quantifies throughput over 25,000 operations.
        Ensures performance is compatible with sub-millisecond CDC bus operations.
        """
        num_iterations = 25000
        pkt = CognitivePacket(
            timestamp_ns=1000000000,
            sequence_id=1,
            source_module_id=MODULE_THALAMUS,
            target_module_id=MODULE_PFC,
            payload_tensor=np.ones(64, dtype=np.float32),
        )
        raw = pkt.to_bytes()
        header_payload = raw[:CRC_COVERAGE_BYTES]

        # 1. CRC-64 throughput
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            _ = compute_crc64(header_payload)
        t1 = time.perf_counter()
        dt_crc = t1 - t0
        crc_ops_per_sec = num_iterations / dt_crc
        crc_mb_per_sec = (num_iterations * CRC_COVERAGE_BYTES) / (dt_crc * 1024 * 1024)
        crc_us_per_op = (dt_crc / num_iterations) * 1e6

        # 2. to_bytes serialization throughput
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            _ = pkt.to_bytes(compute_checksum=True)
        t1 = time.perf_counter()
        dt_ser = t1 - t0
        ser_ops_per_sec = num_iterations / dt_ser
        ser_us_per_op = (dt_ser / num_iterations) * 1e6

        # 3. from_bytes deserialization with CRC verification
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            _ = CognitivePacket.from_bytes(raw, validate_checksum=True)
        t1 = time.perf_counter()
        dt_deser = t1 - t0
        deser_ops_per_sec = num_iterations / dt_deser
        deser_us_per_op = (dt_deser / num_iterations) * 1e6

        # Verifications
        assert crc_ops_per_sec > 100000, f"CRC throughput too low: {crc_ops_per_sec:.0f} ops/s"
        assert ser_ops_per_sec > 50000, f"Serialization throughput too low: {ser_ops_per_sec:.0f} pkts/s"
        assert deser_ops_per_sec > 25000, f"Deserialization throughput too low: {deser_ops_per_sec:.0f} pkts/s"

        # Latency check: Each operation should take <= 50 microseconds
        assert crc_us_per_op < 50.0, f"CRC latency excessive: {crc_us_per_op:.2f} us"
        assert ser_us_per_op < 50.0, f"Serialization latency excessive: {ser_us_per_op:.2f} us"
        assert deser_us_per_op < 50.0, f"Deserialization latency excessive: {deser_us_per_op:.2f} us"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
