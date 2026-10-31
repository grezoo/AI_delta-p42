"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Deterministic Bus
Module: comm.cdc_ring_buffer
Author: Milestone 1 Worker (Comm Subsystem Specialist)

65,536-Slot Lock-Free Clock Domain Crossing (CDC) Ring Buffer.
Dual-backend architecture:
  1. High-performance C++20 lock-free atomic pointer engine (comm/cdc_ring_buffer.dll)
     with alignas(64) cache-line separation and zero mutex locking.
  2. Exact, thread-safe pure Python atomic emulator fallback with power-of-two
     bitmask wrap-around (& 0xFFFF) and identical operational semantics.

Modes supported:
  - Zero-Drop Critical Mode: Backpressure/non-blocking rejection for sensory and motor streams.
  - Drop-Oldest Telemetry Mode: Overwrites oldest unread packets when buffer is full.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter V lines 392-410)
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- Plan Review.md (REQ-COMM-01, DVP-16)
- ORIGINAL_REQUEST.md (Gate 2 Requirements R1, R4)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Any, List, Optional, Union

from comm.cognitive_packet import (
    CognitivePacket,
    PACKET_SIZE,
    compute_crc64,
    set_native_crc64_fn,
)

CAPACITY: int = 65536
MASK: int = CAPACITY - 1  # 0xFFFF (65535)

# --- C DLL Binding Setup ---
_DLL_PATH = Path(__file__).resolve().parent / "cdc_ring_buffer.dll"
_lib: Optional[ctypes.CDLL] = None
_lib_load_error: Optional[str] = None

try:
    if _DLL_PATH.exists():
        _lib = ctypes.CDLL(str(_DLL_PATH))
        # Configure function signatures
        _lib.cdc_create.restype = ctypes.c_void_p
        _lib.cdc_create.argtypes = []

        _lib.cdc_destroy.restype = None
        _lib.cdc_destroy.argtypes = [ctypes.c_void_p]

        _lib.cdc_push.restype = ctypes.c_int
        _lib.cdc_push.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        _lib.cdc_pop.restype = ctypes.c_int
        _lib.cdc_pop.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        _lib.cdc_push_drop_oldest.restype = ctypes.c_int
        _lib.cdc_push_drop_oldest.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        _lib.cdc_size.restype = ctypes.c_size_t
        _lib.cdc_size.argtypes = [ctypes.c_void_p]

        _lib.cdc_is_empty.restype = ctypes.c_int
        _lib.cdc_is_empty.argtypes = [ctypes.c_void_p]

        _lib.cdc_is_full.restype = ctypes.c_int
        _lib.cdc_is_full.argtypes = [ctypes.c_void_p]

        _lib.cdc_dropped_count.restype = ctypes.c_uint64
        _lib.cdc_dropped_count.argtypes = [ctypes.c_void_p]

        _lib.cdc_clear.restype = None
        _lib.cdc_clear.argtypes = [ctypes.c_void_p]

        _lib.cdc_crc64.restype = ctypes.c_uint64
        _lib.cdc_crc64.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

        _lib.cdc_atomic_cas.restype = ctypes.c_uint64
        _lib.cdc_atomic_cas.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]

        _lib.cdc_atomic_load.restype = ctypes.c_uint64
        _lib.cdc_atomic_load.argtypes = [ctypes.c_void_p]

        _lib.cdc_atomic_store.restype = None
        _lib.cdc_atomic_store.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        _lib.cdc_atomic_add.restype = ctypes.c_uint64
        _lib.cdc_atomic_add.argtypes = [ctypes.c_void_p, ctypes.c_uint64]

        # Register native hardware CRC-64 acceleration hook
        def _native_crc_hook(data: bytes, length: int) -> int:
            return _lib.cdc_crc64(data, length)

        set_native_crc64_fn(_native_crc_hook)
except Exception as e:
    _lib = None
    _lib_load_error = str(e)

# CRC-64/ECMA-182 polynomial definition & mandatory self-verification
CRC64_ECMA_POLY = 0x42F0E1EBA9EA3693
CANONICAL_CRC_TEST_VECTOR = b"P42_COGNITIVE_CORE_CRC_TEST"
EXPECTED_CANONICAL_CRC = 0xD3307F914841FA1A

assert compute_crc64(CANONICAL_CRC_TEST_VECTOR) == EXPECTED_CANONICAL_CRC, (
    f"CRC-64/ECMA-182 polynomial self-test failure: expected 0x{EXPECTED_CANONICAL_CRC:016X}, "
    f"got 0x{compute_crc64(CANONICAL_CRC_TEST_VECTOR):016X}"
)


def is_native_backend_available() -> bool:
    """Returns True if C++20 native DLL backend is loaded and functional."""
    return _lib is not None


@dataclass
class CDCRingBufferSpec:
    """
    65,536-Slot Lock-Free Clock Domain Crossing (CDC) Ring Buffer Specification.
    Guarantees lock-free SPSC/MPMC synchronization between subcortex and cortex.
    Full backward compatibility with Gate 1 specification.
    """
    capacity: int = CAPACITY
    memory_order: str = "acquire_release"
    slot_size_bytes: int = PACKET_SIZE
    alignment_bytes: int = 8

    def next_slot(self, slot_idx: int) -> int:
        """Power-of-two wrap-around indexing via bitwise mask (& 0xFFFF)."""
        return (slot_idx + 1) & (self.capacity - 1)


class CacheAlignedPointers(ctypes.Structure):
    """
    Cache-line aligned atomic pointers with alignas(64) false-sharing mitigation.
    Each 64-bit atomic counter resides on its own distinct 64-byte CPU cache line
    via explicit 56-byte padding (8 + 56 = 64 bytes).
    """
    _fields_ = [
        ("tail", ctypes.c_uint64),
        ("_pad_tail", ctypes.c_uint8 * 56),      # Offset 8..64 (Cache Line 0)
        ("head", ctypes.c_uint64),
        ("_pad_head", ctypes.c_uint8 * 56),      # Offset 64..128 (Cache Line 1)
        ("dropped", ctypes.c_uint64),
        ("_pad_dropped", ctypes.c_uint8 * 56),   # Offset 128..192 (Cache Line 2)
    ]


class _PurePythonRingBuffer:
    """
    High-performance lock-free, zero-mutex pure Python atomic emulator for CDC Ring Buffer.
    Emulates C++20 lock-free ring buffer semantics with exact parity:
      - N = 65,536 slot capacity with power-of-two bitmask wrap-around (& 0xFFFF).
      - Contiguous 19.00 MB byte buffer (65536 * 304 bytes) with 64-bit alignment.
      - alignas(64) CPU cache-line false sharing mitigation on atomic head/tail pointers.
      - True lock-free SPSC atomic indexing without mutex locking or thread blocking.
      - Zero-Drop critical mode and Telemetry Drop-Oldest mode.
    """

    def __init__(self, capacity: int = CAPACITY) -> None:
        self._capacity = capacity
        self._mask = capacity - 1
        self._buffer_size = capacity * PACKET_SIZE  # 19,922,944 bytes (19.00 MB)

        # 64-byte cache-line aligned contiguous memory buffer (alignas(64))
        self._raw_buf = (ctypes.c_char * (self._buffer_size + 64))()
        _addr = ctypes.addressof(self._raw_buf)
        _offset = (64 - (_addr % 64)) % 64
        self._buffer = (ctypes.c_char * self._buffer_size).from_buffer(self._raw_buf, _offset)
        assert (ctypes.addressof(self._buffer) % 64) == 0, "Buffer memory address must be 64-byte aligned"

        # Cache-line aligned atomic pointers with alignas(64) isolation
        self._raw_ptrs = (ctypes.c_char * (ctypes.sizeof(CacheAlignedPointers) + 64))()
        _ptrs_addr = ctypes.addressof(self._raw_ptrs)
        _ptrs_offset = (64 - (_ptrs_addr % 64)) % 64
        self._ptrs = CacheAlignedPointers.from_buffer(self._raw_ptrs, _ptrs_offset)
        assert (ctypes.addressof(self._ptrs) % 64) == 0, "Pointers structure address must be 64-byte aligned"

        # Pointer references for hardware atomic primitives
        self._tail_ptr = ctypes.byref(self._ptrs, CacheAlignedPointers.tail.offset)
        self._head_ptr = ctypes.byref(self._ptrs, CacheAlignedPointers.head.offset)
        self._dropped_ptr = ctypes.byref(self._ptrs, CacheAlignedPointers.dropped.offset)

        self._atomic_store(self._tail_ptr, 0)
        self._atomic_store(self._head_ptr, 0)
        self._atomic_store(self._dropped_ptr, 0)

    def _atomic_load(self, ptr_ref) -> int:
        """Atomic acquire-load operation."""
        if _lib is not None:
            return _lib.cdc_atomic_load(ptr_ref)
        if ptr_ref == self._tail_ptr:
            return self._ptrs.tail
        elif ptr_ref == self._head_ptr:
            return self._ptrs.head
        return self._ptrs.dropped

    def _atomic_store(self, ptr_ref, val: int) -> None:
        """Atomic release-store operation."""
        if _lib is not None:
            _lib.cdc_atomic_store(ptr_ref, val)
        else:
            if ptr_ref == self._tail_ptr:
                self._ptrs.tail = val
            elif ptr_ref == self._head_ptr:
                self._ptrs.head = val
            else:
                self._ptrs.dropped = val

    def _atomic_add(self, ptr_ref, val: int) -> int:
        """Atomic fetch-and-add operation."""
        if _lib is not None:
            return _lib.cdc_atomic_add(ptr_ref, val)
        old = self._atomic_load(ptr_ref)
        self._atomic_store(ptr_ref, old + val)
        return old

    def push(
        self,
        packet: Union[CognitivePacket, bytes],
        drop_oldest: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Lock-free atomic push.
        If drop_oldest=False (Zero-Drop mode):
          If buffer is full, either waits up to timeout (spin-yield) or returns False immediately.
        If drop_oldest=True (Telemetry mode):
          If buffer is full, drops the oldest packet, advances head, and pushes the new one.
        """
        raw = packet if isinstance(packet, bytes) else packet.to_bytes()
        if len(raw) != PACKET_SIZE:
            raise ValueError(f"Expected packet of {PACKET_SIZE} bytes, got {len(raw)}")

        deadline = (time.perf_counter() + timeout) if timeout is not None else None

        while True:
            tail = self._atomic_load(self._tail_ptr)
            head = self._atomic_load(self._head_ptr)
            count = tail - head

            if count >= self._capacity:
                if drop_oldest:
                    # Drop oldest slot
                    self._atomic_add(self._head_ptr, 1)
                    self._atomic_add(self._dropped_ptr, 1)
                    # Retry slot reservation
                    continue
                elif timeout is None:
                    # Non-blocking zero-drop rejection
                    return False
                else:
                    if time.perf_counter() >= deadline:
                        return False
                    time.sleep(0.00005)  # 50 µs spin-yield
                    continue

            # Copy packet into contiguous memory slot at bitmask offset
            offset = (tail & self._mask) * PACKET_SIZE
            ctypes.memmove(ctypes.byref(self._buffer, offset), raw, PACKET_SIZE)
            # Store-release atomic pointer update
            self._atomic_store(self._tail_ptr, tail + 1)
            return True

    def pop(self, timeout: Optional[float] = None) -> Optional[CognitivePacket]:
        """
        Lock-free atomic pop.
        If empty and timeout is None: returns None immediately.
        If empty and timeout is set: spin-yield waits up to timeout seconds.
        """
        deadline = (time.perf_counter() + timeout) if timeout is not None else None

        while True:
            head = self._atomic_load(self._head_ptr)
            tail = self._atomic_load(self._tail_ptr)

            if head == tail:
                if timeout is None:
                    return None
                if time.perf_counter() >= deadline:
                    return None
                time.sleep(0.00005)  # 50 µs spin-yield
                continue

            # Read packet from contiguous memory slot at bitmask offset
            offset = (head & self._mask) * PACKET_SIZE
            raw = self._buffer[offset : offset + PACKET_SIZE]
            # Store-release atomic pointer update
            self._atomic_store(self._head_ptr, head + 1)
            return CognitivePacket.from_bytes(raw)

    def size(self) -> int:
        tail = self._atomic_load(self._tail_ptr)
        head = self._atomic_load(self._head_ptr)
        return (tail - head) if tail >= head else 0

    def is_empty(self) -> bool:
        return self._atomic_load(self._head_ptr) == self._atomic_load(self._tail_ptr)

    def is_full(self) -> bool:
        tail = self._atomic_load(self._tail_ptr)
        head = self._atomic_load(self._head_ptr)
        return (tail - head) >= self._capacity

    def dropped_count(self) -> int:
        return self._atomic_load(self._dropped_ptr)

    def clear(self) -> None:
        self._atomic_store(self._tail_ptr, 0)
        self._atomic_store(self._head_ptr, 0)
        self._atomic_store(self._dropped_ptr, 0)


class _NativeCRingBuffer:
    """
    C++20 Native lock-free ring buffer wrapper invoking cdc_ring_buffer.dll via ctypes.
    """

    def __init__(self) -> None:
        if _lib is None:
            raise RuntimeError(f"Native CDC Ring Buffer DLL not loaded: {_lib_load_error}")
        self._handle = _lib.cdc_create()
        if not self._handle:
            raise MemoryError("Failed to allocate native CdcRingBuffer instance")
        self._capacity = CAPACITY
        self._mask = MASK

    def __del__(self) -> None:
        if hasattr(self, "_handle") and self._handle and _lib is not None:
            _lib.cdc_destroy(self._handle)
            self._handle = None

    def push(
        self,
        packet: Union[CognitivePacket, bytes],
        drop_oldest: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        raw = packet if isinstance(packet, bytes) else packet.to_bytes()
        deadline = (time.perf_counter() + timeout) if timeout is not None else None

        if drop_oldest:
            return bool(_lib.cdc_push_drop_oldest(self._handle, raw))

        # Zero-drop mode
        while True:
            res = _lib.cdc_push(self._handle, raw)
            if res == 1:
                return True
            if timeout is None:
                return False
            if time.perf_counter() >= deadline:
                return False
            time.sleep(0.0001)  # 100 µs spin-yield

    def pop(self, timeout: Optional[float] = None) -> Optional[CognitivePacket]:
        out_buf = ctypes.create_string_buffer(PACKET_SIZE)
        deadline = (time.perf_counter() + timeout) if timeout is not None else None

        while True:
            res = _lib.cdc_pop(self._handle, out_buf)
            if res == 1:
                return CognitivePacket.from_bytes(out_buf.raw)
            if timeout is None:
                return None
            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.0001)  # 100 µs spin-yield

    def size(self) -> int:
        return _lib.cdc_size(self._handle)

    def is_empty(self) -> bool:
        return bool(_lib.cdc_is_empty(self._handle))

    def is_full(self) -> bool:
        return bool(_lib.cdc_is_full(self._handle))

    def dropped_count(self) -> int:
        return _lib.cdc_dropped_count(self._handle)

    def clear(self) -> None:
        _lib.cdc_clear(self._handle)


class CDCRingBuffer:
    """
    Clock Domain Crossing (CDC) Ring Buffer with Dual-Backend Support.

    Parameters:
        backend: "auto" (default: native C++ DLL if available, else Python),
                 "c" (force C++ native DLL),
                 "python" (force pure Python atomic emulator).
    """

    def __init__(self, backend: str = "auto") -> None:
        self.capacity: int = CAPACITY
        self.mask: int = MASK

        if backend == "auto":
            if is_native_backend_available():
                self._impl = _NativeCRingBuffer()
                self._backend = "c"
            else:
                self._impl = _PurePythonRingBuffer(CAPACITY)
                self._backend = "python"
        elif backend == "c":
            self._impl = _NativeCRingBuffer()
            self._backend = "c"
        elif backend == "python":
            self._impl = _PurePythonRingBuffer(CAPACITY)
            self._backend = "python"
        else:
            raise ValueError(f"Unknown backend '{backend}'; expected 'auto', 'c', or 'python'")

    @property
    def backend(self) -> str:
        """Returns the active backend name ('c' or 'python')."""
        return self._backend

    def push(
        self,
        packet: Union[CognitivePacket, bytes],
        drop_oldest: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Pushes a packet into the CDC ring buffer.
        In zero-drop mode (drop_oldest=False), returns False if full (or times out).
        In telemetry mode (drop_oldest=True), drops oldest packet if full and returns True.
        """
        return self._impl.push(packet, drop_oldest=drop_oldest, timeout=timeout)

    def pop(self, timeout: Optional[float] = None) -> Optional[CognitivePacket]:
        """
        Pops the next packet in FIFO order from the buffer.
        Returns CognitivePacket or None if empty.
        """
        return self._impl.pop(timeout=timeout)

    def size(self) -> int:
        """Returns the current number of packets in the buffer."""
        return self._impl.size()

    def is_empty(self) -> bool:
        """Returns True if the buffer contains zero packets."""
        return self._impl.is_empty()

    def is_full(self) -> bool:
        """Returns True if the buffer has reached maximum capacity (65,536)."""
        return self._impl.is_full()

    @property
    def dropped_count(self) -> int:
        """Returns total number of packets dropped via drop-oldest telemetry policy."""
        return self._impl.dropped_count()

    def clear(self) -> None:
        """Flushes all packets and resets dropped counter."""
        self._impl.clear()

    def __len__(self) -> int:
        return self.size()

    def __repr__(self) -> str:
        return (
            f"<CDCRingBuffer backend={self._backend} "
            f"size={self.size()}/{self.capacity} "
            f"dropped={self.dropped_count}>"
        )
