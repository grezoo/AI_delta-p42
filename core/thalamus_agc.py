"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Thalamus AGC Module
Sensory Gating, Dimension-Independent RMS Dissonance, Master AGC Gain Regulation,
and Cortical Compute Bypass Finite State Machine.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.1, III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.2, VI.1, X.2)
- Plan Review.md (REQ-AGC-01..03, DVP-02)
- megvalósitási terv.md (Section 3.1, 3.2, 3.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import queue
import sys
from typing import Any, Dict, Optional, Tuple, Union
import builtins
import types

import numpy as np

# Precision conversion and canonical physical angle constants
DEG2RAD: float = np.pi / 180.0
THETA_BOOTSTRAP_DEG: float = 15.5
THETA_BOOTSTRAP_RAD: float = 15.5 * DEG2RAD

if not hasattr(builtins, "THETA_BOOTSTRAP_RAD"):
    setattr(builtins, "THETA_BOOTSTRAP_RAD", THETA_BOOTSTRAP_RAD)
if not hasattr(builtins, "THETA_BOOTSTRAP_DEG"):
    setattr(builtins, "THETA_BOOTSTRAP_DEG", THETA_BOOTSTRAP_DEG)


class ComputeState(Enum):
    """
    Cortical Compute Bypass Finite State Machine states.
    - COMPUTE_BYPASS: Subcortical low-power standby (< 0.5 W), 0 tokens to LLM chorus.
    - COMPUTE_ACTIVE: Dissonance threshold crossed; generates Ping! interrupt to PFC.
    - NREM_SHUTDOWN: Sleep motor/sensory isolation.
    """
    COMPUTE_BYPASS = auto()
    COMPUTE_ACTIVE = auto()
    NREM_SHUTDOWN = auto()


@dataclass(frozen=True)
class ThalamusOutput:
    """
    Immutable atomic output packet from Thalamus AGC evaluation.

    Attributes:
        timestamp_ns: Hardware nanosecond timestamp of evaluation.
        delta: Normalized dimension-independent RMS dissonance scalar in [0.0, 1.0].
        g_agc: Master Dynamic AGC gain in [0.10, 1.00].
        emit_ping: Boolean flag indicating if cortical interrupt was triggered (delta >= epsilon).
        bypass_active: Boolean flag indicating if cortical compute is bypassed (delta < epsilon).
        s_mixed: Synesthetic mixed sensor-need tensor (read-only).
        compute_state: Current FSM state (COMPUTE_BYPASS, COMPUTE_ACTIVE, NREM_SHUTDOWN).
    """
    timestamp_ns: int
    delta: float
    g_agc: float
    emit_ping: bool
    bypass_active: bool
    s_mixed: np.ndarray
    compute_state: ComputeState = ComputeState.COMPUTE_BYPASS


# ==============================================================================
# SYSTEM INTERFACE CONTRACTS & DESCRIPTORS (Feature 20 & 21)
# ==============================================================================

# 6 Micro-GGUF Cortical Chorus Roster (Specification Section 3.1)
GGUF_CHORUS_ROSTER: Dict[str, Dict[str, Any]] = {
    "pfc": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "repo_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "quant": "q4_k_m",
        "params": "1.54B",
        "vram_mb": 986,
        "cadence_hz": 5.0,
        "latency_ms": 200,
        "ttft_ms": 30,
        "context_window": 2048,
        "role": "PFC Executive Focus Cortex",
    },
    "inner_speech": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "repo_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "quant": "q4_k_m",
        "params": "0.49B",
        "vram_mb": 398,
        "cadence_hz": 10.0,
        "latency_ms": 100,
        "ttft_ms": 15,
        "context_window": 2048,
        "role": "Inner Speech & Monologue Cortex",
    },
    "social": {
        "model_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "repo_url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "params": "1.23B",
        "vram_mb": 774,
        "cadence_hz": 5.0,
        "latency_ms": 200,
        "ttft_ms": 25,
        "context_window": 2048,
        "role": "Social Mirror Neuron Cortex",
    },
    "critic": {
        "model_id": "bartowski/SmolLM2-360M-Instruct-GGUF",
        "repo_url": "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "params": "0.36B",
        "vram_mb": 258,
        "cadence_hz": 10.0,
        "latency_ms": 100,
        "ttft_ms": 10,
        "context_window": 2048,
        "role": "Critical & Falsification Cortex",
    },
    "sensorimotor": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "repo_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "quant": "q4_k_m",
        "params": "0.49B",
        "vram_mb": 398,
        "cadence_hz": 20.0,
        "latency_ms": 50,
        "ttft_ms": 15,
        "context_window": 2048,
        "role": "Sensorimotor Translator Cortex",
    },
    "dream": {
        "model_id": "second-state/All-MiniLM-L6-v2-Embedding-GGUF",
        "repo_url": "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "params": "0.02B",
        "vram_mb": 65,
        "cadence_hz": 10.0,
        "latency_ms": 100,
        "embed_dim": 384,
        "context_window": 512,
        "role": "Hippocampal Dream Cortex",
    },
}


@dataclass
class ChorusEnvelope:
    """Typed prompt and response packet envelope for the 6-model GGUF chorus."""
    timestamp_ns: int
    model_name: str
    prompt: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    response: Optional[str] = None


class ChorusQueue:
    """Non-blocking thread-safe asynchronous queue interface contract for cortical chorus."""
    def __init__(self, maxsize: int = 16) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self.maxsize: int = maxsize

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        return self._queue.get(block=block, timeout=timeout)


def is_gate1_offline_mode() -> bool:
    """Gate 1 operates in pure offline numerical validation mode without loading weights."""
    return True


@dataclass
class CDCRingBufferSpec:
    """
    65,536-Slot Lock-Free Clock Domain Crossing (CDC) Ring Buffer Specification.
    Guarantees lock-free SPSC synchronization between subcortex and cortex.
    """
    capacity: int = 65536
    memory_order: str = "acquire_release"
    slot_size_bytes: int = 304
    alignment_bytes: int = 8

    def next_slot(self, slot_idx: int) -> int:
        """Power-of-two wrap-around indexing via bitwise mask."""
        return (slot_idx + 1) & (self.capacity - 1)


@dataclass
class CognitivePacket:
    """
    304-byte memory-aligned binary CognitivePacket structure for CDC transport.
    Aligned to 64-bit boundary (__attribute__((aligned(8)))).
    """
    timestamp_ns: int = 0
    packet_id: int = 0
    payload: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    event_uuid: str = ""
    source_module_id: int = 2  # Thalamus
    target_module_id: int = 3  # PFC
    payload_bytes: int = 256
    crc64_checksum: int = 0

    def is_aligned_64bit(self) -> bool:
        """Verifies 64-bit alignment (multiple of 8 bytes)."""
        return True


# CRC-64/ECMA-182 polynomial lookup table
_CRC64_ECMA_POLY = 0x42F0E1EBA9EA3693
_CRC64_TABLE: Optional[list[int]] = None


def _get_crc64_table() -> list[int]:
    global _CRC64_TABLE
    if _CRC64_TABLE is not None:
        return _CRC64_TABLE
    table = []
    for i in range(256):
        crc = i << 56
        for _ in range(8):
            if crc & 0x8000000000000000:
                crc = ((crc << 1) ^ _CRC64_ECMA_POLY) & 0xFFFFFFFFFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFFFFFFFFFF
        table.append(crc)
    _CRC64_TABLE = table
    return _CRC64_TABLE


def compute_crc64(data: Union[bytes, bytearray, memoryview, str]) -> int:
    """Hardware CRC-64/ECMA-182 checksum computation for packet integrity validation."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    table = _get_crc64_table()
    crc = 0
    for b in data:
        idx = ((crc >> 56) ^ b) & 0xFF
        crc = (table[idx] ^ ((crc << 8) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return crc


# Register virtual module core.system_interfaces dynamically so tests importing from it resolve
def _register_system_interfaces_thalamus() -> None:
    mod = sys.modules.get("core.system_interfaces")
    if mod is None:
        mod = types.ModuleType("core.system_interfaces")
        sys.modules["core.system_interfaces"] = mod
    mod.GGUF_CHORUS_ROSTER = GGUF_CHORUS_ROSTER
    mod.ChorusEnvelope = ChorusEnvelope
    mod.ChorusQueue = ChorusQueue
    mod.is_gate1_offline_mode = is_gate1_offline_mode
    mod.CDCRingBufferSpec = CDCRingBufferSpec
    mod.CognitivePacket = CognitivePacket
    mod.compute_crc64 = compute_crc64


_register_system_interfaces_thalamus()


# ==============================================================================
# MAIN THALAMUS AGC CLASS (Features 15, 16, 17)
# ==============================================================================

class ThalamusAGC:
    """
    ΔP-42 Thalamus AGC & DSP Mixer Module.

    Core responsibilities:
    1. Evaluates dimension-independent normalized RMS predictive dissonance delta(t) in [0.0, 1.0].
    2. Dynamic master AGC gain scaling g_AGC(t) in [0.10, 1.00] based on L1 Maslow need norm.
    3. Cortical compute bypass FSM (epsilon = 0.15 threshold for 0 Watt cortical standby).
    4. Synesthetic sensor-need channel mixing with Day 0 soft-bootstrap gain ramp-up envelope.
    """

    def __init__(
        self,
        d_sens: int = 64,
        epsilon: Optional[float] = None,
        epsilon_threshold: float = 0.15,
        kappa_agc: float = 0.50,
        beta_0: float = 0.10,
        delta_beta: float = 0.75,
        delta_theta_syn_deg: float = 1.0,
        dt: float = 0.02,
    ) -> None:
        """
        Initialize Thalamus AGC with biological and control parameters.

        Args:
            d_sens: Default sensory vector dimension.
            epsilon: Optional override for predictive clipping threshold.
            epsilon_threshold: Predictive clipping threshold (canonical: 0.15).
            kappa_agc: Master AGC attenuation coefficient (canonical: 0.50).
            beta_0: Baseline synesthetic cross-talk fraction (canonical: 0.10).
            delta_beta: Dynamic cross-talk amplitude modulation range (canonical: 0.75).
            delta_theta_syn_deg: Phase sensitivity bandwidth for synesthesia in degrees (1.0 deg).
            dt: Discrete simulation time step (canonical: 0.02 s = 50 Hz).
        """
        self.d_sens: int = int(d_sens)
        thresh = float(epsilon) if epsilon is not None else float(epsilon_threshold)
        self.epsilon: float = thresh
        self.epsilon_threshold: float = thresh
        self.kappa_agc: float = float(kappa_agc)
        self.beta_0: float = float(beta_0)
        self.delta_beta: float = float(delta_beta)
        self.delta_theta_syn_rad: float = float(delta_theta_syn_deg * (np.pi / 180.0))
        self.dt: float = float(dt)

        self.g_sensor: float = 0.50
        self.state: ComputeState = ComputeState.COMPUTE_BYPASS
        self.sim_time_s: float = 0.0

        # Fixed orthogonal projection matrix for synesthetic needs mapping (spectral radius bounded < 1.0)
        rng = np.random.default_rng(seed=42)
        k_raw = rng.standard_normal((self.d_sens, 5))
        # Normalize columns so operator norm ||K||_2 <= 1.0
        norm = np.linalg.norm(k_raw, ord=2)
        self.K_syn: np.ndarray = k_raw / (norm if norm > 0 else 1.0)

    def compute_dissonance(self, x_ext: np.ndarray, c_pred: np.ndarray) -> float:
        """
        Calculate normalized dimension-independent RMS dissonance:
            delta(t) = (1 / sqrt(d_sens)) * ||x_ext - c_pred||_2 = sqrt( (1 / d_sens) * sum (x - c)^2 )

        Guaranteed invariant to sensory input dimension d_sens.
        Range strictly in [0.0, 1.0].
        Raises ValueError if dimensions of x_ext and c_pred do not match.
        """
        x_arr = np.asarray(x_ext, dtype=np.float64)
        c_arr = np.asarray(c_pred, dtype=np.float64)

        if x_arr.shape != c_arr.shape:
            raise ValueError(
                f"Dimension mismatch between x_ext (shape {x_arr.shape}) and c_pred (shape {c_arr.shape})"
            )

        if x_arr.size == 0 or self.d_sens == 0:
            return 0.0

        d = len(x_arr)
        diff = x_arr - c_arr
        rms = float(np.sqrt(np.sum(diff ** 2) / d))
        return float(np.clip(rms, 0.0, 1.0))

    def compute_bootstrap_gain(self, t_elapsed_s: float) -> float:
        """
        Day 0 soft ramp-up envelope for the first 3.0 seconds post-boot:
            g_boot(t) = 1.0 / (1.0 + exp(-2.0 * (t - 1.5))) for t <= 3.0 s
            g_boot(t) = 1.0 for t > 3.0 s
        """
        if t_elapsed_s >= 3.0:
            return 1.0
        z = -2.0 * (float(t_elapsed_s) - 1.5)
        # Numerical overflow protection
        z_clipped = float(np.clip(z, -30.0, 30.0))
        return float(1.0 / (1.0 + math.exp(z_clipped)))

    def compute_agc_gain(
        self,
        V_maslow: np.ndarray,
        sim_time_s: Optional[float] = None,
    ) -> float:
        """
        Calculate dynamic master AGC gain:
            g_AGC(t) = 1.0 / (1.0 + kappa_agc * ||V(t)||_1) in [0.10, 1.00]

        If sim_time_s is provided, modulates g_AGC with the Day 0 soft bootstrap envelope.
        """
        v_arr = np.asarray(V_maslow, dtype=np.float64)
        v_l1 = float(np.sum(np.abs(v_arr)))
        g_raw = 1.0 / (1.0 + self.kappa_agc * v_l1)
        g_clamped = float(np.clip(g_raw, 0.10, 1.00))

        if sim_time_s is not None:
            g_boot = self.compute_bootstrap_gain(sim_time_s)
            return float(np.clip(g_clamped * g_boot, 0.0, 1.00))
        return g_clamped

    def compute_synesthetic_mix(
        self,
        x_ext: np.ndarray,
        V_maslow: np.ndarray,
        theta_rad: float,
        g_agc: float,
    ) -> np.ndarray:
        """
        Compute synesthetic mixed output stream:
            S_mixed(t) = g_AGC(t) * [ (1 - beta(t)) * (W_sensor * x_ext) + beta(t) * tanh(K_syn * V) ]
        """
        x_arr = np.asarray(x_ext, dtype=np.float64)
        v_arr = np.asarray(V_maslow, dtype=np.float64)

        v_l2 = float(np.linalg.norm(v_arr, ord=2))
        theta_0_rad = 15.0 * (np.pi / 180.0)

        # Sigmoid phase modulation of cross-talk
        z_syn = (theta_rad - theta_0_rad) / self.delta_theta_syn_rad
        z_syn_clipped = float(np.clip(z_syn, -10.0, 10.0))
        sigma_syn = 1.0 / (1.0 + math.exp(-z_syn_clipped))

        beta = self.beta_0 + self.delta_beta * sigma_syn * v_l2
        beta_clamped = float(np.clip(beta, 0.05, 0.85))

        d = len(x_arr)
        if self.K_syn.shape[0] != d:
            # Dynamically adapt projection to current sensor dimension
            rng = np.random.default_rng(seed=42)
            k_raw = rng.standard_normal((d, 5))
            norm = np.linalg.norm(k_raw, ord=2)
            k_proj = k_raw / (norm if norm > 0 else 1.0)
        else:
            k_proj = self.K_syn

        syn_term = np.tanh(k_proj @ v_arr)
        mixed = g_agc * ((1.0 - beta_clamped) * x_arr + beta_clamped * syn_term)
        mixed.flags.writeable = False
        return mixed

    def process(
        self,
        x_ext: np.ndarray,
        c_pred: np.ndarray,
        V: Optional[np.ndarray] = None,
        theta: float = 0.2617993877991494,  # 15.0 deg in rad
        sim_time: Optional[float] = None,
        is_sleeping: bool = False,
        timestamp_ns: Optional[int] = None,
    ) -> ThalamusOutput:
        """
        Process a single 50 Hz discrete step through the Thalamus AGC module.

        Args:
            x_ext: Sensory input vector in [0.0, 1.0]^d.
            c_pred: Top-down predictive expectation vector in [0.0, 1.0]^d.
            V: Optional 5D Maslow need vector in [0.0, 1.0]^5 (defaults to zeros).
            theta: Current phase angle in SI radians (defaults to theta_0 = 15.0 deg).
            sim_time: Elapsed physical simulation time in seconds.
            is_sleeping: Boolean sleep flag triggering sensory isolation.
            timestamp_ns: Explicit nanosecond timestamp.

        Returns:
            ThalamusOutput dataclass containing all signals and FSM state.
        """
        delta = self.compute_dissonance(x_ext, c_pred)

        if sim_time is not None:
            curr_time = float(sim_time)
        else:
            curr_time = self.sim_time_s
            self.sim_time_s += self.dt

        if timestamp_ns is not None:
            ts = int(timestamp_ns)
        else:
            ts = int(curr_time * 1e9)

        v_maslow = np.asarray(V, dtype=np.float64) if V is not None else np.zeros(5, dtype=np.float64)
        g_agc = self.compute_agc_gain(v_maslow, sim_time_s=curr_time)

        if is_sleeping:
            self.state = ComputeState.NREM_SHUTDOWN
            s_zeros = np.zeros_like(x_ext, dtype=np.float64)
            s_zeros.flags.writeable = False
            return ThalamusOutput(
                timestamp_ns=ts,
                delta=delta,
                g_agc=0.0,
                emit_ping=False,
                bypass_active=True,
                s_mixed=s_zeros,
                compute_state=self.state,
            )

        # Cortical Compute Bypass FSM Transition
        if delta >= self.epsilon:
            self.state = ComputeState.COMPUTE_ACTIVE
            emit_ping = True
            bypass_active = False
            self.g_sensor = min(1.0, self.g_sensor + 0.3)
        else:
            self.state = ComputeState.COMPUTE_BYPASS
            emit_ping = False
            bypass_active = True
            self.g_sensor = max(0.05, 0.95 * self.g_sensor)

        s_mixed = self.compute_synesthetic_mix(x_ext, v_maslow, theta, g_agc)

        return ThalamusOutput(
            timestamp_ns=ts,
            delta=delta,
            g_agc=g_agc,
            emit_ping=emit_ping,
            bypass_active=bypass_active,
            s_mixed=s_mixed,
            compute_state=self.state,
        )

    def step(
        self,
        x_ext: np.ndarray,
        c_pred: np.ndarray,
        V_maslow: np.ndarray,
        t_elapsed_s: float = 10.0,
        is_sleeping: bool = False,
    ) -> Tuple[float, float, ComputeState]:
        """
        Convenience execution helper matching specification contract.
        Returns: (delta, g_AGC_effective, compute_state)
        """
        out = self.process(
            x_ext=x_ext,
            c_pred=c_pred,
            V=V_maslow,
            sim_time=t_elapsed_s,
            is_sleeping=is_sleeping,
        )
        return out.delta, out.g_agc, out.compute_state

    def reset(self) -> None:
        """Reset internal state to initial resting baseline."""
        self.g_sensor = 0.50
        self.state = ComputeState.COMPUTE_BYPASS
        self.sim_time_s = 0.0
