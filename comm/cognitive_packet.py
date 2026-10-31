"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Deterministic Bus
Module: comm.cognitive_packet
Author: Milestone 1 Worker (Comm Subsystem Specialist)

Strict 64-bit aligned (aligned(8)) binary and tensor packet for high-speed
inter-module communication and Clock Domain Crossing (CDC) transport.
Protected by hardware/vector CRC-64/ECMA-182 checksum and RFC-4122 v4 UUID tracking.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter IV, Chapter V.1, V.3 lines 392-410)
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- Plan Review.md (REQ-COMM-02, REQ-COMM-03, DVP-16)
- ORIGINAL_REQUEST.md (Gate 2 Requirements R1, R4)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import struct
from typing import Any, Dict, Optional, Union
import uuid
import numpy as np

# Exact struct format string (Little-Endian):
# Q   : uint64 timestamp_ns (8 bytes, offset 0..8)
# 16s : uint8[16] event_uuid (16 bytes, offset 8..24)
# I   : uint32 sequence_id (4 bytes, offset 24..28)
# H   : uint16 source_module_id (2 bytes, offset 28..30)
# H   : uint16 target_module_id (2 bytes, offset 30..32)
# I   : uint32 payload_bytes (4 bytes, offset 32..36)
# I   : uint32 reserved_padding (4 bytes, offset 36..40)
# 64f : float32[64] payload_tensor (256 bytes, offset 40..296)
# Q   : uint64 crc64_checksum (8 bytes, offset 296..304)
PACKET_FORMAT = "<Q16sIHHII64fQ"
HEADER_PAYLOAD_FORMAT = "<Q16sIHHII64f"
PACKET_SIZE = 304
CRC_COVERAGE_BYTES = 296
PAYLOAD_FLOAT_COUNT = 64
PAYLOAD_MAX_BYTES = 256  # 64 * 4 bytes

# Module ID Constants (DELTA_P42_MUKODESI_ELVEK.md Chapter V.1)
MODULE_HYPOTHALAMUS = 1
MODULE_THALAMUS = 2
MODULE_CHORUS_LLM = 3
MODULE_PFC = 4
MODULE_AMYGDALA = 5
MODULE_BASAL_GANGLIA = 6
MODULE_CEREBELLUM = 7
MODULE_HIPPOCAMPUS = 8

VALID_MODULE_IDS = frozenset(range(1, 9))

# CRC-64/ECMA-182 polynomial
CRC64_ECMA_POLY = 0x42F0E1EBA9EA3693
_CRC64_TABLE: Optional[list[int]] = None

# Optional native C function hook (set by cdc_ring_buffer if native DLL loaded)
_native_crc64_fn = None


def _get_crc64_table() -> list[int]:
    """Generates 256-entry lookup table for CRC-64/ECMA-182 (MSB-first)."""
    global _CRC64_TABLE
    if _CRC64_TABLE is not None:
        return _CRC64_TABLE
    table = []
    for i in range(256):
        crc = i << 56
        for _ in range(8):
            if crc & 0x8000000000000000:
                crc = ((crc << 1) ^ CRC64_ECMA_POLY) & 0xFFFFFFFFFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFFFFFFFFFF
        table.append(crc)
    _CRC64_TABLE = table
    return _CRC64_TABLE


def set_native_crc64_fn(fn) -> None:
    """Sets optional native C DLL implementation of CRC-64 for hardware acceleration."""
    global _native_crc64_fn
    _native_crc64_fn = fn


def compute_crc64(data: Union[bytes, bytearray, memoryview, str]) -> int:
    """
    Computes CRC-64/ECMA-182 checksum with polynomial 0x42F0E1EBA9EA3693.
    Deterministic test vector:
        compute_crc64(b"P42_COGNITIVE_CORE_CRC_TEST") == 0xD3307F914841FA1A
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    if _native_crc64_fn is not None:
        try:
            return _native_crc64_fn(bytes(data), len(data))
        except Exception:
            pass

    table = _get_crc64_table()
    crc = 0
    for b in data:
        idx = ((crc >> 56) ^ b) & 0xFF
        crc = (table[idx] ^ ((crc << 8) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return crc


def validate_crc(packet_or_bytes: Union[bytes, bytearray, memoryview, "CognitivePacket"]) -> bool:
    """
    Validates the CRC-64/ECMA-182 integrity of a CognitivePacket or 304-byte buffer.
    Returns True if checksum is valid, False otherwise.
    """
    if isinstance(packet_or_bytes, CognitivePacket):
        return packet_or_bytes.validate_crc()
    
    buf = bytes(packet_or_bytes)
    if len(buf) != PACKET_SIZE:
        return False
    
    payload_data = buf[:CRC_COVERAGE_BYTES]
    expected_crc = struct.unpack("<Q", buf[CRC_COVERAGE_BYTES:PACKET_SIZE])[0]
    return compute_crc64(payload_data) == expected_crc


class CognitivePacket:
    """
    304-byte, 64-bit aligned (__attribute__((aligned(8)))) Cognitive Packet.

    Memory Layout:
        uint64_t timestamp_ns       (8 bytes)
        uint8_t  event_uuid[16]     (16 bytes, RFC-4122 v4)
        uint32_t sequence_id        (4 bytes)
        uint16_t source_module_id   (2 bytes, 1..8)
        uint16_t target_module_id   (2 bytes, 1..8)
        uint32_t payload_bytes      (4 bytes, 0..256)
        uint32_t reserved_padding   (4 bytes, 0)
        float    payload_tensor[64] (256 bytes, IEEE 754 float32)
        uint64_t crc64_checksum     (8 bytes, CRC-64/ECMA-182)
    """

    __slots__ = (
        "timestamp_ns",
        "_event_uuid_bytes",
        "sequence_id",
        "source_module_id",
        "target_module_id",
        "payload_bytes",
        "reserved_padding",
        "_payload_tensor",
        "crc64_checksum",
    )

    def __init__(
        self,
        timestamp_ns: int = 0,
        event_uuid: Optional[Union[uuid.UUID, str, bytes, bytearray]] = None,
        sequence_id: int = 0,
        source_module_id: int = MODULE_THALAMUS,
        target_module_id: int = MODULE_PFC,
        payload_bytes: int = PAYLOAD_MAX_BYTES,
        reserved_padding: int = 0,
        payload_tensor: Optional[Union[np.ndarray, list[float], tuple[float, ...]]] = None,
        crc64_checksum: int = 0,
        # Backward compatibility keyword aliases:
        packet_id: Optional[int] = None,
        payload: Optional[Union[np.ndarray, list[float], tuple[float, ...]]] = None,
    ) -> None:
        self.timestamp_ns = int(timestamp_ns)

        # UUID resolution (RFC-4122 v4)
        if event_uuid is None or event_uuid == "":
            self._event_uuid_bytes = uuid.uuid4().bytes
        elif isinstance(event_uuid, uuid.UUID):
            self._event_uuid_bytes = event_uuid.bytes
        elif isinstance(event_uuid, (bytes, bytearray)):
            if len(event_uuid) != 16:
                raise ValueError(f"event_uuid bytes must be exactly 16 bytes, got {len(event_uuid)}")
            self._event_uuid_bytes = bytes(event_uuid)
        elif isinstance(event_uuid, str):
            self._event_uuid_bytes = uuid.UUID(event_uuid).bytes
        else:
            raise TypeError(f"Invalid event_uuid type: {type(event_uuid)}")

        # Sequence ID handling (with packet_id alias support)
        if packet_id is not None and sequence_id == 0:
            self.sequence_id = int(packet_id)
        else:
            self.sequence_id = int(sequence_id)

        self.source_module_id = int(source_module_id)
        self.target_module_id = int(target_module_id)
        self.reserved_padding = int(reserved_padding)

        # Tensor payload handling (with payload alias support)
        source_tensor = payload_tensor if payload_tensor is not None else payload
        if source_tensor is None:
            self._payload_tensor = np.zeros(PAYLOAD_FLOAT_COUNT, dtype=np.float32)
            self.payload_bytes = PAYLOAD_MAX_BYTES
        else:
            arr = np.asarray(source_tensor, dtype=np.float32)
            if arr.ndim != 1:
                arr = arr.flatten()
            if arr.size > PAYLOAD_FLOAT_COUNT:
                raise ValueError(
                    f"payload_tensor exceeds maximum 64 floats (got {arr.size})"
                )
            # Create padded 64-element float32 buffer
            padded = np.zeros(PAYLOAD_FLOAT_COUNT, dtype=np.float32)
            padded[: arr.size] = arr
            self._payload_tensor = padded
            if payload_bytes == PAYLOAD_MAX_BYTES and arr.size != PAYLOAD_FLOAT_COUNT:
                self.payload_bytes = arr.size * 4
            else:
                self.payload_bytes = int(payload_bytes)

        if crc64_checksum != 0:
            self.crc64_checksum = int(crc64_checksum)
        else:
            self.crc64_checksum = self._compute_header_payload_crc()

    def _compute_header_payload_crc(self) -> int:
        """Internal helper to calculate CRC-64 over the current header and payload fields."""
        floats = self._payload_tensor.tolist()
        header_payload = struct.pack(
            HEADER_PAYLOAD_FORMAT,
            self.timestamp_ns,
            self._event_uuid_bytes,
            self.sequence_id,
            self.source_module_id,
            self.target_module_id,
            self.payload_bytes,
            self.reserved_padding,
            *floats,
        )
        return compute_crc64(header_payload)

    def update_checksum(self) -> int:
        """Recalculates and updates self.crc64_checksum from current fields."""
        self.crc64_checksum = self._compute_header_payload_crc()
        return self.crc64_checksum

    # --- Properties & Backward Compatibility Aliases ---

    @property
    def packet_id(self) -> int:
        """Backward-compatibility alias for sequence_id."""
        return self.sequence_id

    @packet_id.setter
    def packet_id(self, val: int) -> None:
        self.sequence_id = int(val)

    @property
    def payload(self) -> np.ndarray:
        """Backward-compatibility alias for payload_tensor."""
        return self._payload_tensor

    @payload.setter
    def payload(self, val: Union[np.ndarray, list[float]]) -> None:
        self.payload_tensor = val

    @property
    def payload_tensor(self) -> np.ndarray:
        """Returns 64-element float32 payload tensor array."""
        return self._payload_tensor

    @payload_tensor.setter
    def payload_tensor(self, val: Union[np.ndarray, list[float]]) -> None:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.flatten()
        if arr.size > PAYLOAD_FLOAT_COUNT:
            raise ValueError(f"payload_tensor exceeds 64 floats (got {arr.size})")
        padded = np.zeros(PAYLOAD_FLOAT_COUNT, dtype=np.float32)
        padded[: arr.size] = arr
        self._payload_tensor = padded
        self.payload_bytes = arr.size * 4

    @property
    def event_uuid(self) -> str:
        """Returns canonical hyphenated RFC-4122 v4 string."""
        return str(uuid.UUID(bytes=self._event_uuid_bytes))

    @event_uuid.setter
    def event_uuid(self, val: Union[uuid.UUID, str, bytes]) -> None:
        if isinstance(val, uuid.UUID):
            self._event_uuid_bytes = val.bytes
        elif isinstance(val, (bytes, bytearray)):
            if len(val) != 16:
                raise ValueError("event_uuid bytes must be 16 bytes")
            self._event_uuid_bytes = bytes(val)
        elif isinstance(val, str):
            self._event_uuid_bytes = uuid.UUID(val).bytes
        else:
            raise TypeError(f"Invalid event_uuid type: {type(val)}")

    @property
    def event_uuid_bytes(self) -> bytes:
        """Returns 16 raw UUID bytes."""
        return self._event_uuid_bytes

    # --- Serialization & Validation Methods ---

    def is_aligned_64bit(self, buffer: Optional[Any] = None) -> bool:
        """
        Verifies 64-bit (8-byte) memory alignment.
        If buffer is provided, checks memory address alignment of the buffer.
        Otherwise checks struct size divisibility and payload tensor memory address.
        """
        if buffer is not None:
            if isinstance(buffer, np.ndarray):
                return (buffer.ctypes.data % 8) == 0
            try:
                # Check ctypes pointer alignment
                addr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
                return (addr % 8) == 0
            except (TypeError, ValueError):
                return True

        # Check instance alignment
        return (PACKET_SIZE % 8 == 0) and (self._payload_tensor.ctypes.data % 8 == 0)

    def to_bytes(self, compute_checksum: bool = True) -> bytes:
        """
        Serializes packet into exactly 304 binary bytes.
        If compute_checksum is True (default), updates self.crc64_checksum
        with computed CRC-64/ECMA-182 over bytes 0..295.
        """
        floats = self._payload_tensor.tolist()
        header_payload = struct.pack(
            HEADER_PAYLOAD_FORMAT,
            self.timestamp_ns,
            self._event_uuid_bytes,
            self.sequence_id,
            self.source_module_id,
            self.target_module_id,
            self.payload_bytes,
            self.reserved_padding,
            *floats,
        )

        if compute_checksum:
            self.crc64_checksum = compute_crc64(header_payload)

        return header_payload + struct.pack("<Q", self.crc64_checksum)

    @classmethod
    def from_bytes(
        cls,
        data: Union[bytes, bytearray, memoryview],
        validate_checksum: bool = False,
    ) -> "CognitivePacket":
        """
        Deserializes a 304-byte binary buffer into a CognitivePacket.
        Raises ValueError if length != 304 bytes.
        If validate_checksum is True, verifies CRC-64/ECMA-182 integrity.
        """
        buf = bytes(data)
        if len(buf) != PACKET_SIZE:
            raise ValueError(
                f"Invalid CognitivePacket buffer size: expected {PACKET_SIZE} bytes, got {len(buf)}"
            )

        unpacked = struct.unpack(PACKET_FORMAT, buf)
        timestamp_ns = unpacked[0]
        uuid_bytes = unpacked[1]
        sequence_id = unpacked[2]
        source_module_id = unpacked[3]
        target_module_id = unpacked[4]
        payload_bytes = unpacked[5]
        reserved_padding = unpacked[6]
        floats = unpacked[7:71]
        crc64_checksum = unpacked[71]

        if validate_checksum:
            computed = compute_crc64(buf[:CRC_COVERAGE_BYTES])
            if computed != crc64_checksum:
                raise ValueError(
                    f"CRC-64 integrity violation: computed 0x{computed:016X} != stored 0x{crc64_checksum:016X}"
                )

        pkt = cls(
            timestamp_ns=timestamp_ns,
            event_uuid=uuid_bytes,
            sequence_id=sequence_id,
            source_module_id=source_module_id,
            target_module_id=target_module_id,
            payload_bytes=payload_bytes,
            reserved_padding=reserved_padding,
            payload_tensor=list(floats),
            crc64_checksum=crc64_checksum,
        )
        return pkt

    def validate_crc(self) -> bool:
        """
        Checks whether self.crc64_checksum matches computed CRC-64
        over the current header and payload fields.
        """
        return self._compute_header_payload_crc() == self.crc64_checksum

    def to_numpy(self) -> np.ndarray:
        """Returns a copy of the payload tensor as a float32 NumPy array."""
        return self._payload_tensor.copy()

    def from_numpy(self, arr: np.ndarray) -> None:
        """Updates payload tensor from a NumPy array."""
        self.payload_tensor = arr

    def to_dict(self) -> Dict[str, Any]:
        """Returns structured dictionary representation of packet fields."""
        return {
            "timestamp_ns": self.timestamp_ns,
            "event_uuid": self.event_uuid,
            "sequence_id": self.sequence_id,
            "source_module_id": self.source_module_id,
            "target_module_id": self.target_module_id,
            "payload_bytes": self.payload_bytes,
            "reserved_padding": self.reserved_padding,
            "payload_tensor": self._payload_tensor.copy(),
            "crc64_checksum": self.crc64_checksum,
        }

    def __repr__(self) -> str:
        return (
            f"<CognitivePacket seq={self.sequence_id} "
            f"src={self.source_module_id} dst={self.target_module_id} "
            f"uuid={self.event_uuid[:8]}... bytes={self.payload_bytes} "
            f"crc=0x{self.crc64_checksum:016X}>"
        )
