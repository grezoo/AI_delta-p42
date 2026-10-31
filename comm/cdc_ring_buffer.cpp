/**
 * ΔP-42 Sovereign Cognitive Architecture - Gate 2 Deterministic Bus
 * Implementation: comm/cdc_ring_buffer.cpp
 * Author: Milestone 1 Worker (Comm Subsystem Specialist)
 *
 * Implements C++20 lock-free MPMC queue and C ABI export layer.
 */

#define CDC_EXPORTS
#include "cdc_ring_buffer.h"

#include <thread>
#include <chrono>

namespace dp42::comm {

static constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;
static uint64_t s_crc64_table[256];
static std::atomic<bool> s_crc64_initialized{false};

static void init_crc64_table() {
    bool expected = false;
    if (s_crc64_initialized.compare_exchange_strong(expected, true)) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint64_t crc = (uint64_t)i << 56;
            for (int b = 0; b < 8; ++b) {
                if (crc & 0x8000000000000000ULL) {
                    crc = ((crc << 1) ^ CRC64_POLY);
                } else {
                    crc = (crc << 1);
                }
            }
            s_crc64_table[i] = crc;
        }
    }
}

uint64_t compute_crc64(const uint8_t* data, size_t len) {
    if (!s_crc64_initialized.load(std::memory_order_relaxed)) {
        init_crc64_table();
    }
    uint64_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        uint8_t idx = (uint8_t)((crc >> 56) ^ data[i]);
        crc = s_crc64_table[idx] ^ (crc << 8);
    }
    return crc;
}

CdcRingBuffer::CdcRingBuffer() {
    slots_ = new Slot[CDC_RING_BUFFER_CAPACITY];
    for (size_t i = 0; i < CDC_RING_BUFFER_CAPACITY; ++i) {
        slots_[i].sequence.store(i, std::memory_order_relaxed);
    }
    tail_.store(0, std::memory_order_relaxed);
    head_.store(0, std::memory_order_relaxed);
    dropped_.store(0, std::memory_order_relaxed);
    init_crc64_table();
}

CdcRingBuffer::~CdcRingBuffer() {
    delete[] slots_;
    slots_ = nullptr;
}

bool CdcRingBuffer::push(const CognitivePacket& packet) {
    uint64_t pos = tail_.load(std::memory_order_relaxed);
    for (;;) {
        Slot* slot = &slots_[pos & CDC_RING_BUFFER_MASK];
        uint64_t seq = slot->sequence.load(std::memory_order_acquire);
        int64_t dif = (int64_t)seq - (int64_t)pos;

        if (dif == 0) {
            if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                std::memcpy(&slot->packet, &packet, sizeof(CognitivePacket));
                slot->sequence.store(pos + 1, std::memory_order_release);
                return true;
            }
        } else if (dif < 0) {
            // Buffer is full (pos has wrapped around and caught up with head)
            return false;
        } else {
            pos = tail_.load(std::memory_order_relaxed);
        }
    }
}

bool CdcRingBuffer::pop(CognitivePacket& packet) {
    uint64_t pos = head_.load(std::memory_order_relaxed);
    for (;;) {
        Slot* slot = &slots_[pos & CDC_RING_BUFFER_MASK];
        uint64_t seq = slot->sequence.load(std::memory_order_acquire);
        int64_t dif = (int64_t)seq - (int64_t)(pos + 1);

        if (dif == 0) {
            if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                std::memcpy(&packet, &slot->packet, sizeof(CognitivePacket));
                slot->sequence.store(pos + CDC_RING_BUFFER_MASK + 1, std::memory_order_release);
                return true;
            }
        } else if (dif < 0) {
            // Buffer is empty (head has caught up with tail)
            return false;
        } else {
            pos = head_.load(std::memory_order_relaxed);
        }
    }
}

bool CdcRingBuffer::push_drop_oldest(const CognitivePacket& packet) {
    if (push(packet)) {
        return true;
    }
    // Buffer is full: pop oldest packet and retry
    CognitivePacket dummy;
    if (pop(dummy)) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
    }
    // Attempt push again
    return push(packet);
}

size_t CdcRingBuffer::size() const {
    int64_t t = (int64_t)tail_.load(std::memory_order_relaxed);
    int64_t h = (int64_t)head_.load(std::memory_order_relaxed);
    if (t >= h) {
        size_t s = (size_t)(t - h);
        return (s <= CDC_RING_BUFFER_CAPACITY) ? s : CDC_RING_BUFFER_CAPACITY;
    }
    return 0;
}

bool CdcRingBuffer::is_empty() const {
    int64_t t = (int64_t)tail_.load(std::memory_order_relaxed);
    int64_t h = (int64_t)head_.load(std::memory_order_relaxed);
    return t == h;
}

bool CdcRingBuffer::is_full() const {
    return size() >= CDC_RING_BUFFER_CAPACITY;
}

uint64_t CdcRingBuffer::dropped_count() const {
    return dropped_.load(std::memory_order_relaxed);
}

void CdcRingBuffer::clear() {
    CognitivePacket dummy;
    while (pop(dummy)) {}
    dropped_.store(0, std::memory_order_relaxed);
}

} // namespace dp42::comm

// C ABI Exports
extern "C" {

struct cdc_ring_buffer_t {
    dp42::comm::CdcRingBuffer ring;
};

CDC_API cdc_ring_buffer_t* cdc_create(void) {
    try {
        return new cdc_ring_buffer_t();
    } catch (...) {
        return nullptr;
    }
}

CDC_API void cdc_destroy(cdc_ring_buffer_t* ring) {
    delete ring;
}

CDC_API int cdc_push(cdc_ring_buffer_t* ring, const dp42::comm::CognitivePacket* packet) {
    if (!ring || !packet) return 0;
    return ring->ring.push(*packet) ? 1 : 0;
}

CDC_API int cdc_pop(cdc_ring_buffer_t* ring, dp42::comm::CognitivePacket* packet) {
    if (!ring || !packet) return 0;
    return ring->ring.pop(*packet) ? 1 : 0;
}

CDC_API int cdc_push_drop_oldest(cdc_ring_buffer_t* ring, const dp42::comm::CognitivePacket* packet) {
    if (!ring || !packet) return 0;
    return ring->ring.push_drop_oldest(*packet) ? 1 : 0;
}

CDC_API size_t cdc_size(const cdc_ring_buffer_t* ring) {
    if (!ring) return 0;
    return ring->ring.size();
}

CDC_API int cdc_is_empty(const cdc_ring_buffer_t* ring) {
    if (!ring) return 1;
    return ring->ring.is_empty() ? 1 : 0;
}

CDC_API int cdc_is_full(const cdc_ring_buffer_t* ring) {
    if (!ring) return 0;
    return ring->ring.is_full() ? 1 : 0;
}

CDC_API uint64_t cdc_dropped_count(const cdc_ring_buffer_t* ring) {
    if (!ring) return 0;
    return ring->ring.dropped_count();
}

CDC_API void cdc_clear(cdc_ring_buffer_t* ring) {
    if (ring) {
        ring->ring.clear();
    }
}

CDC_API uint64_t cdc_crc64(const uint8_t* data, size_t len) {
    if (!data || len == 0) return 0;
    return dp42::comm::compute_crc64(data, len);
}

CDC_API uint64_t cdc_atomic_cas(uint64_t* ptr, uint64_t expected, uint64_t desired) {
    if (!ptr) return 0;
    auto* a = reinterpret_cast<std::atomic<uint64_t>*>(ptr);
    a->compare_exchange_strong(expected, desired, std::memory_order_acq_rel);
    return expected;
}

CDC_API uint64_t cdc_atomic_load(const uint64_t* ptr) {
    if (!ptr) return 0;
    auto* a = reinterpret_cast<const std::atomic<uint64_t>*>(ptr);
    return a->load(std::memory_order_acquire);
}

CDC_API void cdc_atomic_store(uint64_t* ptr, uint64_t val) {
    if (!ptr) return;
    auto* a = reinterpret_cast<std::atomic<uint64_t>*>(ptr);
    a->store(val, std::memory_order_release);
}

CDC_API uint64_t cdc_atomic_add(uint64_t* ptr, uint64_t val) {
    if (!ptr) return 0;
    auto* a = reinterpret_cast<std::atomic<uint64_t>*>(ptr);
    return a->fetch_add(val, std::memory_order_acq_rel);
}

} // extern "C"
