"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Unit Test Suite: Thalamus AGC Module
Tests dimension-independent RMS dissonance, dynamic master AGC gain regulation,
predictive clipping, Cortical Compute Bypass FSM, Day 0 soft bootstrap envelope,
synesthetic channel mixing, GGUF chorus roster, and CDC lock-free ring buffer structures.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Section III.1, III.3, V)
- DELTA_P42_MUKODESI_ELVEK.md (Section I.2, VI.1, X.2)
- Plan Review.md (REQ-AGC-01..03, DVP-02, DVP-05)
- megvalósitási terv.md (Section 3.1, 3.2)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from core.thalamus_agc import (
    ComputeState,
    ThalamusOutput,
    ThalamusAGC,
    GGUF_CHORUS_ROSTER,
    ChorusEnvelope,
    ChorusQueue,
    is_gate1_offline_mode,
    CDCRingBufferSpec,
    CognitivePacket,
    compute_crc64,
)


class TestNormalizedRMSDissonance:
    """Mathematical and empirical verification of dimension-independent RMS dissonance (Feature 15)."""

    def test_rms_dissonance_canonical_formula(self):
        """Validates delta = 1/sqrt(d_sens) * ||x_ext - c_pred||_2."""
        th = ThalamusAGC(d_sens=4)
        x = np.array([1.0, 1.0, 1.0, 1.0])
        c = np.array([0.0, 0.0, 0.0, 0.0])
        delta = th.compute_dissonance(x, c)
        assert math.isclose(delta, 1.0, rel_tol=1e-7)

    def test_dimension_invariance_across_multiple_orders_of_magnitude(self):
        """Constant per-channel discrepancy yields identical delta across d = 1, 4, 16, 64, 256, 1024."""
        perturbation = 0.35
        for d in [1, 4, 16, 64, 256, 1024]:
            th = ThalamusAGC(d_sens=d)
            x = np.full(d, perturbation)
            c = np.zeros(d)
            delta = th.compute_dissonance(x, c)
            assert math.isclose(delta, perturbation, rel_tol=1e-6)

    def test_zero_dissonance_on_perfect_prediction(self):
        """Perfect expectation x_ext == c_pred yields strictly delta == 0.0."""
        th = ThalamusAGC(d_sens=32)
        rng = np.random.default_rng(101)
        x = rng.uniform(0.0, 1.0, size=32)
        delta = th.compute_dissonance(x, x)
        assert math.isclose(delta, 0.0, abs_tol=1e-12)

    def test_maximum_dissonance_saturation(self):
        """Opposite binary saturation yields strictly delta == 1.0."""
        th = ThalamusAGC(d_sens=64)
        x = np.ones(64)
        c = np.zeros(64)
        delta = th.compute_dissonance(x, c)
        assert math.isclose(delta, 1.0, rel_tol=1e-7)

    def test_empty_or_zero_dimension_safeguard(self):
        """Empty inputs or d_sens == 0 safely returns 0.0 without division by zero."""
        th = ThalamusAGC(d_sens=0)
        assert th.compute_dissonance(np.array([]), np.array([])) == 0.0
        th64 = ThalamusAGC(d_sens=64)
        assert th64.compute_dissonance(np.array([]), np.array([])) == 0.0

    def test_dimension_mismatch_raises_value_error(self):
        """Shape mismatch between sensory vector and prediction vector raises ValueError."""
        th = ThalamusAGC(d_sens=8)
        x = np.ones(8)
        c = np.ones(12)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            th.compute_dissonance(x, c)


class TestDynamicAGCGainScaling:
    """Verification of negative feedback AGC gain regulation based on Maslow need tension (Feature 16)."""

    def test_agc_formula_unloaded(self):
        """Zero Maslow tension ||V||_1 == 0 yields maximum gain g_AGC == 1.000."""
        th = ThalamusAGC(kappa_agc=0.5)
        g = th.compute_agc_gain(np.zeros(5), sim_time_s=10.0)
        assert math.isclose(g, 1.000, rel_tol=1e-6)

    def test_agc_formula_day0_baseline(self):
        """Day 0 baseline pre-charge ||V_0||_1 == 0.90 yields g_AGC = 1 / (1 + 0.5 * 0.9) = 1 / 1.45."""
        th = ThalamusAGC(kappa_agc=0.5)
        v0 = np.array([0.15, 0.10, 0.25, 0.10, 0.30])
        g = th.compute_agc_gain(v0, sim_time_s=10.0)
        assert math.isclose(g, 1.0 / 1.45, rel_tol=1e-5)

    def test_agc_formula_full_saturation(self):
        """Complete need vacuum ||V||_1 == 5.0 yields g_AGC = 1 / 3.5 = 0.2857 (attenuation to 28.6%)."""
        th = ThalamusAGC(kappa_agc=0.5)
        g = th.compute_agc_gain(np.ones(5), sim_time_s=10.0)
        assert math.isclose(g, 1.0 / 3.5, rel_tol=1e-5)

    def test_monotonic_gain_attenuation(self):
        """Rising need vector norm strictly and monotonically decreases AGC transmission gain."""
        th = ThalamusAGC(kappa_agc=0.5)
        g_prev = 2.0
        for tension in np.linspace(0.0, 1.0, 11):
            g = th.compute_agc_gain(np.full(5, tension), sim_time_s=10.0)
            assert g <= g_prev
            g_prev = g

    def test_gain_clamping_lower_bound(self):
        """AGC gain never drops below configured floor of 0.10 even under extreme L1 inputs."""
        th = ThalamusAGC(kappa_agc=0.5)
        g_extreme = th.compute_agc_gain(np.full(5, 100.0), sim_time_s=10.0)
        assert g_extreme >= 0.10

    def test_day0_soft_bootstrap_envelope(self):
        """Day 0 soft bootstrap envelope smoothly ramps gain during initial 3.0 seconds."""
        th = ThalamusAGC()
        # At t = 0.0 s: g_boot ~ 0.0474
        g_t0 = th.compute_bootstrap_gain(0.0)
        assert 0.04 < g_t0 < 0.06
        # At t = 1.5 s: g_boot == 0.5000
        g_t15 = th.compute_bootstrap_gain(1.5)
        assert math.isclose(g_t15, 0.5000, rel_tol=1e-5)
        # At t = 3.0 s: g_boot == 1.0
        g_t30 = th.compute_bootstrap_gain(3.0)
        assert math.isclose(g_t30, 1.0000, rel_tol=1e-5)
        # At t > 3.0 s: strictly 1.0
        assert th.compute_bootstrap_gain(10.0) == 1.0


class TestComputeBypassFSM:
    """Verification of Cortical Compute Bypass Finite State Machine & Event-driven Ping (Feature 17)."""

    def test_subthreshold_compute_bypass(self):
        """When delta < epsilon (0.15), system operates in 0 Watt compute bypass."""
        th = ThalamusAGC(d_sens=64, epsilon=0.15)
        out = th.process(x_ext=np.zeros(64), c_pred=np.zeros(64))
        assert out.bypass_active is True
        assert out.emit_ping is False
        assert out.compute_state == ComputeState.COMPUTE_BYPASS

    def test_suprathreshold_ping_interrupt(self):
        """When delta >= epsilon (0.15), cortical interrupt Ping! is emitted."""
        th = ThalamusAGC(d_sens=64, epsilon=0.15)
        x = np.full(64, 0.25)
        c = np.zeros(64)
        out = th.process(x_ext=x, c_pred=c)
        assert out.emit_ping is True
        assert out.bypass_active is False
        assert out.compute_state == ComputeState.COMPUTE_ACTIVE

    def test_exact_threshold_boundary_ping(self):
        """At exactly delta == 0.15, Ping interrupt triggers reliably."""
        th = ThalamusAGC(d_sens=1, epsilon=0.15)
        out = th.process(x_ext=np.array([0.15]), c_pred=np.array([0.0]))
        assert math.isclose(out.delta, 0.15, abs_tol=1e-6)
        assert out.emit_ping is True
        assert out.bypass_active is False

    def test_sensor_gain_adaptation_dynamics(self):
        """Sensor gain increases upon surprising events and decays exponentially during bypass."""
        th = ThalamusAGC(d_sens=16, epsilon=0.15)
        initial_gain = th.g_sensor

        # Bypass step: sensory habituation
        th.process(x_ext=np.zeros(16), c_pred=np.zeros(16))
        assert th.g_sensor < initial_gain

        # Ping step: sensory arousal boost
        th.process(x_ext=np.ones(16), c_pred=np.zeros(16))
        assert th.g_sensor > initial_gain

    def test_sleep_mode_nrem_shutdown(self):
        """Sleep flag is_sleeping == True forces complete sensory disconnect and muting."""
        th = ThalamusAGC(d_sens=16)
        out = th.process(
            x_ext=np.ones(16),
            c_pred=np.zeros(16),
            is_sleeping=True,
        )
        assert out.compute_state == ComputeState.NREM_SHUTDOWN
        assert out.emit_ping is False
        assert out.bypass_active is True
        assert out.g_agc == 0.0
        assert np.all(out.s_mixed == 0.0)

    def test_synesthetic_mixing_output(self):
        """Synesthetic mixed stream is immutable, non-empty, and bounded."""
        th = ThalamusAGC(d_sens=16)
        out = th.process(
            x_ext=np.full(16, 0.5),
            c_pred=np.zeros(16),
            V=np.full(5, 0.4),
        )
        assert out.s_mixed.shape == (16,)
        assert out.s_mixed.flags.writeable is False
        assert np.all(np.isfinite(out.s_mixed))


class TestSystemInterfacesGGUFAndCDC:
    """Verification of 6 Micro-GGUF Chorus descriptors and 65536 CDC Ring Buffer contracts (Features 20, 21)."""

    def test_gguf_roster_composition(self):
        """Verifies exact 6 model roster IDs, quantizations, and context windows."""
        assert len(GGUF_CHORUS_ROSTER) == 6
        expected_models = {"pfc", "inner_speech", "social", "critic", "sensorimotor", "dream"}
        assert set(GGUF_CHORUS_ROSTER.keys()) == expected_models

        # Specific architecture constraints
        assert GGUF_CHORUS_ROSTER["dream"]["embed_dim"] == 384
        assert GGUF_CHORUS_ROSTER["pfc"]["context_window"] == 2048
        assert GGUF_CHORUS_ROSTER["inner_speech"]["cadence_hz"] == 10.0

    def test_chorus_envelope_dataclass(self):
        """ChorusEnvelope carries structured dispatch contracts."""
        env = ChorusEnvelope(
            timestamp_ns=1_000_000,
            model_name="pfc",
            prompt="Evaluate priority need",
            token_count=12,
            metadata={"priority": 1},
        )
        assert env.model_name == "pfc"
        assert env.token_count == 12
        assert env.metadata["priority"] == 1

    def test_chorus_queue_interface(self):
        """ChorusQueue supports non-blocking thread-safe transfer."""
        cq = ChorusQueue(maxsize=4)
        assert cq.empty()
        cq.put("packet_1")
        assert cq.qsize() == 1
        assert cq.get() == "packet_1"
        assert cq.empty()

    def test_gate1_offline_execution_mode(self):
        """Gate 1 operates in pure offline validation mode."""
        assert is_gate1_offline_mode() is True

    def test_cdc_ring_buffer_specification(self):
        """CDC Ring Buffer specifies exactly 65,536 slots with bitwise power-of-two wrap-around."""
        spec = CDCRingBufferSpec()
        assert spec.capacity == 65536
        assert spec.memory_order == "acquire_release"
        assert spec.next_slot(0) == 1
        assert spec.next_slot(65535) == 0

    def test_cognitive_packet_alignment(self):
        """CognitivePacket satisfies 64-bit atomic alignment."""
        pkt = CognitivePacket(
            timestamp_ns=500,
            packet_id=1,
            payload=np.zeros(64, dtype=np.float32),
        )
        assert pkt.is_aligned_64bit()
        assert pkt.source_module_id == 2  # Thalamus

    def test_crc64_ecma_checksum_validation(self):
        """CRC-64/ECMA-182 calculates non-zero, deterministic checksum."""
        data1 = b"P42_COGNITIVE_CORE_CRC_TEST"
        crc1 = compute_crc64(data1)
        assert isinstance(crc1, int)
        assert crc1 != 0

        # Deterministic repeatability
        crc2 = compute_crc64(data1)
        assert crc1 == crc2

        # Sensitivity to bit change
        crc_modified = compute_crc64(b"P42_COGNITIVE_CORE_CRC_TESU")
        assert crc1 != crc_modified
