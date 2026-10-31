#pragma once
/**
 * ΔP-42 Sovereign Cognitive Architecture - Gate 2 Deterministic Bus
 * Header: comm/cdc_ring_buffer.h
 * Author: Milestone 1 Worker (Comm Subsystem Specialist)
 *
 * C++20 Lock-Free 65,536-Slot Clock Domain Crossing (CDC) Ring Buffer.
 * Provides high-throughput, zero-mutex, zero-drop synchronization between
 * subcortical fast loops (1-5 ms) and cortical LLM chorus (50-200 ms).
 *
 * Memory alignment:
 *   - CognitivePacket: alignas(8), exactly 304 bytes.
 *   - Head / Tail pointers: alignas(64) to prevent CPU cache-line false sharing.
 *   - Capacity: N = 65,536 (2^16), bitmask wrap-around (index & 0xFFFF).
 *
 * Concurrency:
 *   - Multi-Producer Multi-Consumer (MPMC) lock-free bounded queue.
 *   - Atomic std::memory_order_acquire / release sequence synchronization.
 *   - Modes: Zero-Drop critical mode & Drop-Oldest telemetry mode.
 */

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <cstring>
#include <new>

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef CDC_EXPORTS
    #define CDC_API __declspec(dllexport)
  #else
    #define CDC_API __declspec(dllimport)
  #endif
#else
  #define CDC_API __attribute__((visibility("default")))
#endif

namespace dp42::comm {

#pragma pack(push, 1)
/**
 * Exactly 304-byte binary cognitive packet matching Python struct '<Q16sIHHII64fQ'.
 * 64-bit aligned (alignas(8)).
 */
struct alignas(8) CognitivePacket {
    uint64_t timestamp_ns;       // 8 bytes (offset 0..8)
    uint8_t  event_uuid[16];     // 16 bytes (offset 8..24, RFC-4122 v4)
    uint32_t sequence_id;        // 4 bytes (offset 24..28)
    uint16_t source_module_id;   // 2 bytes (offset 28..30, 1..8)
    uint16_t target_module_id;   // 2 bytes (offset 30..32, 1..8)
    uint32_t payload_bytes;      // 4 bytes (offset 32..36, valid tensor bytes)
    uint32_t reserved_padding;   // 4 bytes (offset 36..40, 64-bit alignment padding)
    float    payload_tensor[64]; // 256 bytes (offset 40..296, IEEE 754 float32)
    uint64_t crc64_checksum;     // 8 bytes (offset 296..304, CRC-64/ECMA-182)
};
#pragma pack(pop)

static_assert(sizeof(CognitivePacket) == 304, "CognitivePacket must be exactly 304 bytes");
static_assert(alignof(CognitivePacket) == 8, "CognitivePacket must be 8-byte aligned");

constexpr size_t CDC_RING_BUFFER_CAPACITY = 65536;
constexpr size_t CDC_RING_BUFFER_MASK = CDC_RING_BUFFER_CAPACITY - 1;

/**
 * Lock-Free Multi-Producer Multi-Consumer (MPMC) Bounded FIFO Queue.
 * Based on Dmitry Vyukov's bounded queue with cache-line false-sharing mitigation.
 */
class CdcRingBuffer {
public:
    struct Slot {
        std::atomic<uint64_t> sequence;
        CognitivePacket packet;
    };

    CdcRingBuffer();
    ~CdcRingBuffer();

    CdcRingBuffer(const CdcRingBuffer&) = delete;
    CdcRingBuffer& operator=(const CdcRingBuffer&) = delete;

    /**
     * Enqueues a packet in zero-drop mode.
     * Returns true if successfully enqueued; false if buffer is currently full.
     */
    bool push(const CognitivePacket& packet);

    /**
     * Dequeues a packet.
     * Returns true if successfully dequeued; false if buffer is currently empty.
     */
    bool pop(CognitivePacket& packet);

    /**
     * Enqueues a packet in telemetry drop-oldest mode.
     * If buffer is full, drops the oldest packet and increments dropped counter.
     * Returns true always.
     */
    bool push_drop_oldest(const CognitivePacket& packet);

    size_t size() const;
    bool is_empty() const;
    bool is_full() const;
    uint64_t dropped_count() const;
    void clear();

private:
    alignas(64) std::atomic<uint64_t> tail_{0};
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> dropped_{0};

    Slot* slots_{nullptr};
};

/**
 * Hardware/Vectorized CRC-64/ECMA-182 checksum calculation.
 * Polynomial: 0x42F0E1EBA9EA3693.
 */
uint64_t compute_crc64(const uint8_t* data, size_t len);

} // namespace dp42::comm

// C ABI Exports for ctypes / interop
extern "C" {
    typedef struct cdc_ring_buffer_t cdc_ring_buffer_t;

    CDC_API cdc_ring_buffer_t* cdc_create(void);
    CDC_API void cdc_destroy(cdc_ring_buffer_t* ring);
    CDC_API int cdc_push(cdc_ring_buffer_t* ring, const dp42::comm::CognitivePacket* packet);
    CDC_API int cdc_pop(cdc_ring_buffer_t* ring, dp42::comm::CognitivePacket* packet);
    CDC_API int cdc_push_drop_oldest(cdc_ring_buffer_t* ring, const dp42::comm::CognitivePacket* packet);
    CDC_API size_t cdc_size(const cdc_ring_buffer_t* ring);
    CDC_API int cdc_is_empty(const cdc_ring_buffer_t* ring);
    CDC_API int cdc_is_full(const cdc_ring_buffer_t* ring);
    CDC_API uint64_t cdc_dropped_count(const cdc_ring_buffer_t* ring);
    CDC_API void cdc_clear(cdc_ring_buffer_t* ring);
    CDC_API uint64_t cdc_crc64(const uint8_t* data, size_t len);

    // Hardware atomic primitives for lock-free Python interop (acquire/release semantics)
    CDC_API uint64_t cdc_atomic_cas(uint64_t* ptr, uint64_t expected, uint64_t desired);
    CDC_API uint64_t cdc_atomic_load(const uint64_t* ptr);
    CDC_API void cdc_atomic_store(uint64_t* ptr, uint64_t val);
    CDC_API uint64_t cdc_atomic_add(uint64_t* ptr, uint64_t val);
}
