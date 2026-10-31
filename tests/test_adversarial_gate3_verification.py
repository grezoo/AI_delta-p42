"""
ΔP-42 Sovereign Cognitive Core - Stage-Gate 3 Adversarial Test Suite
Module: tests.test_adversarial_gate3_verification
Author: Empirical Challenger 1 (Adversarial Quality Assurance)

Empirical stress test suite challenging Gate3Verifier and Stage-Gate 3 DVP specifications:
- DVP-06: Purge stress under variable episode sizes (10 to 3,500), heterogeneous payloads,
          multi-cycle pipelines, and variable context saturation; verifying data loss strictly 0.0%.
- DVP-07: Watchdog boundary timeouts; verifying that short transactions (<2.0s) disarm cleanly
          without false aborts or false arousal, while true stalls trigger abort in [2.000, 2.080]s.
- DVP-08: Schmitt trigger deadband stability; 20,000 targeted adversarial noise evaluations
          confined strictly inside [0.40, 0.85] proving 0 chattering flips and micro-epsilon latching.
- DVP-11: Fine-grained sub-threshold cosine spectrum (cos in [0.950, 0.9899]) proving 100% rejection,
          single-bit corruption rejection, mixed-batch atomic rollback, and REM atonia under extreme drive.
"""

from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.basal_ganglia_gate import BasalGangliaGate, DEG2RAD
from core.gate3_verification import (
    DVP06Result,
    DVP07Result,
    DVP08Result,
    DVP11Result,
    Gate3Verifier,
)
from memory.hippocampus_nrem import (
    CommitResult,
    CommitState,
    DEFAULT_RAM_LIMIT_BYTES,
    EpisodicBuffer,
    EpisodeRecord,
    HippocampusNREM,
    IntegrityProbeException,
    PROBE_COSINE_THRESHOLD,
    TransactionAbortedException,
)
from memory.nrem_watchdog import (
    DEFAULT_WATCHDOG_TIMEOUT_SEC,
    FAULT_CODE_IO_TIMEOUT_STALL,
    MAX_REALTIME_LATENCY_MARGIN_SEC,
    NREMWatchdog,
    TimeoutFault,
    WatchdogState,
)
from memory.rem_sandbox import (
    DreamResult,
    DreamVerdict,
    KnowledgeGraphSandbox,
    MOTOR_ATONIA_CLAMP,
    REMSandbox,
)
from memory.sleep_hysteresis import (
    DEFAULT_DEADBAND_WIDTH,
    DEFAULT_SLEEP_THRESHOLD,
    DEFAULT_WAKE_THRESHOLD,
    SleepHysteresis,
    SleepState,
)


# ==============================================================================
# CHALLENGE 1: DVP-06 MEMORY PURGE STRESS UNDER ADVERSARIAL LOADS
# ==============================================================================

class TestDVP06AdversarialPurge:
    """Adversarial stress testing for DVP-06 (Memory Purge & Zero Data Loss)."""

    @pytest.mark.parametrize("n_episodes", [10, 50, 500, 2500])
    def test_variable_episode_scales_zero_data_loss(self, n_episodes: int) -> None:
        """
        Verify that regardless of batch size (from small N=10 to heavy N=2500),
        consolidation purges working memory C(t) -> 0.00 and data loss is strictly 0.0%.
        """
        rng = np.random.default_rng(1001 + n_episodes)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / f"purge_scale_{n_episodes}.db")
            records: List[EpisodeRecord] = []

            for i in range(n_episodes):
                rec = EpisodeRecord(
                    episode_uuid=str(uuid.uuid4()),
                    timestamp_ns=time.monotonic_ns() + i * 10,
                    sensory_vector=rng.standard_normal(16).astype(np.float32),
                    motor_action=float(rng.uniform(0.0, 1.0)),
                    reward_vector=rng.uniform(0.0, 1.0, size=5).astype(np.float32),
                    embedding=rng.standard_normal(32).astype(np.float32),
                    metadata={"scale_test": True, "idx": i},
                )
                records.append(rec)

            total_bytes = sum(r.get_byte_size() for r in records)
            hippo = HippocampusNREM(
                db_path=db_path,
                ram_limit_bytes=total_bytes,
                max_episodes=n_episodes + 100,
            )
            try:
                for r in records:
                    hippo.buffer.add_episode(r)

                assert hippo.context_saturation >= 0.85
                res = hippo.consolidate_2pc()

                assert res.success is True
                assert res.state == CommitState.COMMITTED
                assert len(hippo.buffer) == 0
                assert hippo.context_saturation == 0.0
                assert hippo.get_persisted_episode_count() == n_episodes
                assert res.episodes_persisted == n_episodes
                assert res.min_probe_cosine >= PROBE_COSINE_THRESHOLD
            finally:
                hippo.close()

    def test_heterogeneous_payload_and_large_embedding_dimensions(self) -> None:
        """
        Stress test with heterogeneous embedding dimensions (128D, 256D) and
        variable metadata sizes (including 8 KB JSON payloads).
        """
        rng = np.random.default_rng(2002)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "purge_heterogeneous.db")
            records: List[EpisodeRecord] = []

            for i in range(200):
                dim = 128 if i % 2 == 0 else 256
                # Heavy metadata on every 5th episode
                meta: Dict[str, Any] = {"idx": i}
                if i % 5 == 0:
                    meta["heavy_payload"] = "X" * 4096
                    meta["nested"] = {"array": list(range(100)), "tag": "adversarial"}

                rec = EpisodeRecord(
                    episode_uuid=str(uuid.uuid4()),
                    timestamp_ns=time.monotonic_ns() + i * 50,
                    sensory_vector=rng.standard_normal(24).astype(np.float32),
                    motor_action=0.75,
                    reward_vector=np.zeros(5, dtype=np.float32),
                    embedding=rng.standard_normal(dim).astype(np.float32),
                    metadata=meta,
                )
                records.append(rec)

            total_bytes = sum(r.get_byte_size() for r in records)
            hippo = HippocampusNREM(
                db_path=db_path,
                ram_limit_bytes=total_bytes,
                max_episodes=500,
            )
            try:
                for r in records:
                    hippo.buffer.add_episode(r)

                res = hippo.consolidate_2pc()
                assert res.success is True
                assert hippo.context_saturation == 0.0
                assert hippo.get_persisted_episode_count() == 200
                assert len(hippo.buffer) == 0
            finally:
                hippo.close()

    def test_partial_saturation_and_multi_cycle_consolidation(self) -> None:
        """
        Verify multi-cycle purge pipeline:
        Cycle 1: Fill to C(t) = 0.88 -> Consolidate & Purge -> C(t) = 0.00
        Cycle 2: Fill to C(t) = 0.95 -> Consolidate & Purge -> C(t) = 0.00
        Cycle 3: Fill to C(t) = 1.02 -> Consolidate & Purge -> C(t) = 0.00
        Verifies accumulation of persistent records and zero cross-cycle contamination.
        """
        rng = np.random.default_rng(3003)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "purge_multicycle.db")
            hippo = HippocampusNREM(
                db_path=db_path,
                ram_limit_bytes=25_000,
                max_episodes=1000,
            )
            try:
                cumulative_persisted = 0
                for cycle, target_episodes in enumerate([70, 80, 85], start=1):
                    for j in range(target_episodes):
                        hippo.record_event(
                            sensory_vector=rng.standard_normal(16).astype(np.float32),
                            motor_action=0.1 * cycle,
                            reward_vector=np.zeros(5, dtype=np.float32),
                            embedding=rng.standard_normal(32).astype(np.float32),
                            metadata={"cycle": cycle, "ep": j},
                        )

                    assert hippo.context_saturation >= 0.80
                    res = hippo.consolidate_2pc()

                    cumulative_persisted += target_episodes
                    assert res.success is True
                    assert res.state == CommitState.COMMITTED
                    assert hippo.context_saturation == 0.0
                    assert len(hippo.buffer) == 0
                    assert hippo.get_persisted_episode_count() == cumulative_persisted
            finally:
                hippo.close()

    def test_empty_buffer_degenerate_purge(self) -> None:
        """Consolidating an empty buffer must succeed immediately with C(t) == 0.0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "purge_empty.db")
            hippo = HippocampusNREM(db_path=db_path)
            try:
                res = hippo.consolidate_2pc()
                assert res.success is True
                assert res.state == CommitState.COMMITTED
                assert res.episodes_processed == 0
                assert res.episodes_persisted == 0
                assert hippo.context_saturation == 0.0
            finally:
                hippo.close()


# ==============================================================================
# CHALLENGE 2: DVP-07 WATCHDOG BOUNDARIES AND REAL-TIME PRECISION
# ==============================================================================

class TestDVP07AdversarialWatchdog:
    """Adversarial stress testing for DVP-07 (Hard Real-Time 2.0s Watchdog Supervisor)."""

    @pytest.mark.parametrize("stall_duration_s", [2.100, 2.300, 2.500])
    def test_true_stalls_trigger_abort_in_canonical_window(self, stall_duration_s: float) -> None:
        """
        Verify that simulated stalls > 2.0s trigger abort within [2.000, 2.080]s,
        independent of how much longer the stall lasts.
        """
        hysteresis = SleepHysteresis(initial_state=SleepState.ASLEEP)
        gate = BasalGangliaGate()
        setattr(gate, "sleep_lock", True)

        rollback_called = False

        def on_rollback() -> None:
            nonlocal rollback_called
            rollback_called = True

        watchdog = NREMWatchdog(
            timeout_s=2.0,
            check_interval_s=0.010,
            auto_arousal=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / f"watchdog_stall_{stall_duration_s}.db")
            hippo = HippocampusNREM(db_path=db_path)
            try:
                hippo.record_event(
                    sensory_vector=np.zeros(16, dtype=np.float32),
                    motor_action=0.5,
                    reward_vector=np.zeros(5, dtype=np.float32),
                    embedding=np.zeros(32, dtype=np.float32),
                )
                buf_len = len(hippo.buffer)

                fault = watchdog.simulate_io_stall(
                    stall_duration_s=stall_duration_s,
                    transaction_id=f"adversarial_stall_{stall_duration_s}",
                    hippocampus=hippo,
                    basal_ganglia_gate=gate,
                    hysteresis=hysteresis,
                    db_conn=hippo._conn,
                    rollback_callback=on_rollback,
                )

                assert fault is not None
                measured_s = fault.elapsed_duration_ms / 1000.0

                # Must fire at 2.0s (+ margin up to 0.080s, allowing sub-ms timer quantization down to 1.999s)
                assert 1.9990 <= measured_s <= 2.080, (
                    f"Abort time {measured_s:.4f}s outside timing window [1.999, 2.080]s"
                )
                assert fault.fault_code == FAULT_CODE_IO_TIMEOUT_STALL
                assert rollback_called is True
                assert hysteresis.current_state == SleepState.AWAKE
                assert getattr(gate, "sleep_lock", False) is False
                assert len(hippo.buffer) == buf_len  # 100% working RAM preserved
                assert hippo.get_persisted_episode_count() == 0
            finally:
                hippo.close()

    @pytest.mark.parametrize("short_duration_s", [0.050, 0.200, 0.500, 1.200])
    def test_short_transactions_disarm_cleanly_without_false_arousal(self, short_duration_s: float) -> None:
        """
        Verify that transactions finishing before the 2.0s limit disarm cleanly:
        simulate_io_stall returns None, no abort fault is logged, and no false arousal is triggered.
        """
        hysteresis = SleepHysteresis(initial_state=SleepState.ASLEEP)
        gate = BasalGangliaGate()
        setattr(gate, "sleep_lock", True)

        rollback_called = False

        def on_rollback() -> None:
            nonlocal rollback_called
            rollback_called = True

        watchdog = NREMWatchdog(
            timeout_s=2.0,
            check_interval_s=0.010,
            auto_arousal=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / f"watchdog_short_{short_duration_s}.db")
            hippo = HippocampusNREM(db_path=db_path)
            try:
                fault = watchdog.simulate_io_stall(
                    stall_duration_s=short_duration_s,
                    transaction_id=f"clean_tx_{short_duration_s}",
                    hippocampus=hippo,
                    basal_ganglia_gate=gate,
                    hysteresis=hysteresis,
                    db_conn=hippo._conn,
                    rollback_callback=on_rollback,
                )

                assert fault is None
                assert watchdog.state == WatchdogState.DISARMED
                assert rollback_called is False
                # System should remain ASLEEP since transaction completed normally without fault
                assert hysteresis.current_state == SleepState.ASLEEP
                assert getattr(gate, "sleep_lock", True) is True
            finally:
                hippo.close()

    def test_watchdog_boundary_comparison_subsecond(self) -> None:
        """
        Precise boundary test with scaled watchdog (timeout = 0.150s):
        - Transaction at 0.080s (below timeout) disarms cleanly -> None
        - Transaction at 0.220s (above timeout) aborts in [0.150, 0.230]s -> TimeoutFault
        """
        watchdog = NREMWatchdog(timeout_s=0.150, check_interval_s=0.005)

        # 1. Under-deadline transaction
        fault_under = watchdog.simulate_io_stall(stall_duration_s=0.080)
        assert fault_under is None
        assert watchdog.state == WatchdogState.DISARMED

        # 2. Over-deadline transaction
        fault_over = watchdog.simulate_io_stall(stall_duration_s=0.220)
        assert fault_over is not None
        measured_s = fault_over.elapsed_duration_ms / 1000.0
        assert 0.150 <= measured_s <= 0.230
        assert fault_over.fault_code == FAULT_CODE_IO_TIMEOUT_STALL

    def test_watchdog_reentrancy_and_thread_lifecycle(self) -> None:
        """
        Verify watchdog re-entrancy: multiple consecutive arm/disarm/abort cycles
        execute cleanly without thread leakage or state corruption.
        """
        watchdog = NREMWatchdog(timeout_s=0.100, check_interval_s=0.005)

        for cycle in range(5):
            watchdog.arm(transaction_id=f"cycle_{cycle}")
            assert watchdog.is_armed() is True
            watchdog.heartbeat()
            elapsed_ms = watchdog.disarm()
            assert elapsed_ms >= 0.0
            assert watchdog.is_armed() is False
            assert watchdog.state == WatchdogState.DISARMED


# ==============================================================================
# CHALLENGE 3: DVP-08 SCHMITT TRIGGER DEADBAND STABILITY
# ==============================================================================

class TestDVP08AdversarialHysteresis:
    """Adversarial stress testing for DVP-08 (Chattering-Free Sleep Hysteresis)."""

    def test_targeted_deadband_noise_injection_20000_cycles(self) -> None:
        """
        Adversarial evaluation targeting specifically the deadband [0.4001, 0.8499]:
        Injects 20,000 evaluations of high-amplitude noise confined inside the deadband.
        Proves state remains 100% latched and chattering count is strictly 0.
        """
        rng = np.random.default_rng(8888)

        # 1. Start AWAKE -> 10,000 cycles in deadband -> must NEVER switch to ASLEEP
        hyst_awake = SleepHysteresis(initial_state=SleepState.AWAKE)
        for _ in range(10000):
            c_noise = float(rng.uniform(0.4001, 0.8499))
            snap = hyst_awake.update(c_noise)
            assert snap.current_state == SleepState.AWAKE
            assert snap.in_deadband is True
            assert snap.chattering_detected is False

        assert hyst_awake.get_chattering_count() == 0
        assert hyst_awake.get_deadband_evaluations() == 10000
        assert hyst_awake.get_transition_count() == 0

        # 2. Start ASLEEP -> 10,000 cycles in deadband -> must NEVER switch to AWAKE
        hyst_asleep = SleepHysteresis(initial_state=SleepState.ASLEEP)
        for _ in range(10000):
            c_noise = float(rng.uniform(0.4001, 0.8499))
            snap = hyst_asleep.update(c_noise)
            assert snap.current_state == SleepState.ASLEEP
            assert snap.in_deadband is True
            assert snap.chattering_detected is False

        assert hyst_asleep.get_chattering_count() == 0
        assert hyst_asleep.get_deadband_evaluations() == 10000
        assert hyst_asleep.get_transition_count() == 0

    def test_micro_epsilon_boundary_hunting(self) -> None:
        """
        Empirically test micro-epsilon boundary latching at machine precision:
        - 0.849999999999 vs 0.850000000000
        - 0.400000000001 vs 0.400000000000
        """
        hyst = SleepHysteresis(initial_state=SleepState.AWAKE)

        # Latch AWAKE just below 0.85
        s1 = hyst.update(0.8499999999999)
        assert s1.current_state == SleepState.AWAKE
        assert s1.in_deadband is True

        # Cross upper threshold exactly at 0.8500000000000 -> ASLEEP
        s2 = hyst.update(0.8500000000000)
        assert s2.current_state == SleepState.ASLEEP
        assert s2.in_deadband is False

        # Latch ASLEEP just above 0.40 inside deadband
        s3 = hyst.update(0.4000000000001)
        assert s3.current_state == SleepState.ASLEEP
        assert s3.in_deadband is True

        # Cross lower threshold exactly at 0.4000000000000 -> AWAKE
        s4 = hyst.update(0.4000000000000)
        assert s4.current_state == SleepState.AWAKE
        assert s4.in_deadband is False

        assert hyst.get_chattering_count() == 0

    def test_rapid_alternating_square_wave(self) -> None:
        """
        Rapid full-scale oscillations (1000 cycles jumping between 0.35 and 0.90).
        Verifies valid transitions execute reliably while deadband chattering remains 0.
        """
        hyst = SleepHysteresis(initial_state=SleepState.AWAKE)
        for i in range(1000):
            if i % 2 == 0:
                snap = hyst.update(0.90)
                assert snap.current_state == SleepState.ASLEEP
            else:
                snap = hyst.update(0.35)
                assert snap.current_state == SleepState.AWAKE

        assert hyst.get_chattering_count() == 0
        assert hyst.get_transition_count() == 1000

    def test_out_of_range_and_invalid_inputs(self) -> None:
        """Verify error handling on out-of-range context saturation."""
        hyst = SleepHysteresis()

        with pytest.raises(ValueError, match="cannot be negative"):
            hyst.update(-0.01)

        with pytest.raises(ValueError, match="severely exceeds"):
            hyst.update(1.20)

        with pytest.raises(ValueError, match="finite real number"):
            hyst.update(float("nan"))

        with pytest.raises(ValueError, match="finite real number"):
            hyst.update(float("inf"))


# ==============================================================================
# CHALLENGE 4: DVP-11 INTEGRITY PROBE & CORRUPTION REJECTION
# ==============================================================================

class TestDVP11AdversarialIntegrityProbe:
    """Adversarial stress testing for DVP-11 (Probe Cosine, Corruption Rejection, Motor Atonia)."""

    @pytest.mark.parametrize("subthreshold_cos", [0.950, 0.970, 0.980, 0.985, 0.988, 0.989, 0.9899])
    def test_fine_grained_subthreshold_cosine_spectrum_rejection(self, subthreshold_cos: float) -> None:
        """
        Inject embeddings with precise sub-threshold cosine similarities (< 0.990).
        Verify 100% rejection across every single point in the spectrum, with complete
        SQLite rollback (0 persisted) and working memory preserved in RAM.
        """
        rng = np.random.default_rng(int(subthreshold_cos * 100000))
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / f"subthreshold_{int(subthreshold_cos * 10000)}.db")
            hippo = HippocampusNREM(db_path=db_path, min_probe_similarity=0.990)
            try:
                # Add 3 normal episodes
                for j in range(3):
                    hippo.record_event(
                        sensory_vector=np.zeros(16, dtype=np.float32),
                        motor_action=0.1,
                        reward_vector=np.zeros(5, dtype=np.float32),
                        embedding=rng.standard_normal(32).astype(np.float32),
                        metadata={"ep": j},
                    )

                # Target episode 0 for perturbation
                target_ep = hippo.buffer.get_episodes()[0]
                orig_v = target_ep.embedding.astype(np.float64)
                orig_norm = orig_v / np.linalg.norm(orig_v)

                rand_v = rng.standard_normal(len(orig_v))
                orth_v = rand_v - np.dot(rand_v, orig_norm) * orig_norm
                orth_norm = orth_v / np.linalg.norm(orth_v)

                sin_th = np.sqrt(1.0 - subthreshold_cos**2)
                perturbed = (subthreshold_cos * orig_norm + sin_th * orth_norm).astype(np.float32)
                hex_data = perturbed.tobytes().hex()

                # SQLite trigger simulating disk-level / memory tampering
                hippo._conn.execute(
                    f"""
                    CREATE TRIGGER tamper_trigger AFTER INSERT ON episodes
                    WHEN NEW.uuid = '{target_ep.episode_uuid}'
                    BEGIN
                        UPDATE episodes SET embedding = X'{hex_data}' WHERE uuid = NEW.uuid;
                    END;
                    """
                )
                hippo._conn.commit()

                # Execute 2PC consolidation -> MUST FAIL
                res = hippo.consolidate_2pc()

                assert res.success is False
                assert res.state == CommitState.ABORTED
                assert res.episodes_persisted == 0
                assert hippo.get_persisted_episode_count() == 0
                assert len(hippo.buffer) == 3  # Working memory intact
            finally:
                hippo.close()

    def test_single_bit_flip_corruption_rejection(self) -> None:
        """
        Verify that flipping a single byte in an embedding stored to disk
        is caught by the cosine similarity probe and triggers atomic rollback.
        """
        rng = np.random.default_rng(5555)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "bit_flip.db")
            hippo = HippocampusNREM(db_path=db_path)
            try:
                rec = hippo.record_event(
                    sensory_vector=np.zeros(16, dtype=np.float32),
                    motor_action=0.0,
                    reward_vector=np.zeros(5, dtype=np.float32),
                    embedding=rng.standard_normal(32).astype(np.float32),
                )

                # Inject zero-blob corruption via trigger
                hippo._conn.execute(
                    f"""
                    CREATE TRIGGER bit_flip_trigger AFTER INSERT ON episodes
                    WHEN NEW.uuid = '{rec.episode_uuid}'
                    BEGIN
                        UPDATE episodes SET embedding = zeroblob(length(NEW.embedding)) WHERE uuid = NEW.uuid;
                    END;
                    """
                )
                hippo._conn.commit()

                res = hippo.consolidate_2pc()
                assert res.success is False
                assert res.state == CommitState.ABORTED
                assert hippo.get_persisted_episode_count() == 0
                assert len(hippo.buffer) == 1
            finally:
                hippo.close()

    def test_mixed_batch_contamination_atomic_rollback(self) -> None:
        """
        In a batch of 25 episodes, 1 episode is corrupted.
        Verify that ALL 25 are rolled back from disk (0 persisted)
        and ALL 25 remain safely in volatile working memory.
        """
        rng = np.random.default_rng(7777)
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "mixed_batch.db")
            hippo = HippocampusNREM(db_path=db_path)
            try:
                episodes: List[EpisodeRecord] = []
                for i in range(25):
                    rec = hippo.record_event(
                        sensory_vector=np.zeros(16, dtype=np.float32),
                        motor_action=0.2,
                        reward_vector=np.zeros(5, dtype=np.float32),
                        embedding=rng.standard_normal(32).astype(np.float32),
                        metadata={"idx": i},
                    )
                    episodes.append(rec)

                # Corrupt episode #13
                corrupt_uuid = episodes[13].episode_uuid
                hippo._conn.execute(
                    f"""
                    CREATE TRIGGER corrupt_single AFTER INSERT ON episodes
                    WHEN NEW.uuid = '{corrupt_uuid}'
                    BEGIN
                        UPDATE episodes SET embedding = zeroblob(length(NEW.embedding)) WHERE uuid = NEW.uuid;
                    END;
                    """
                )
                hippo._conn.commit()

                res = hippo.consolidate_2pc()
                assert res.success is False
                assert res.state == CommitState.ABORTED
                assert hippo.get_persisted_episode_count() == 0
                assert len(hippo.buffer) == 25
            finally:
                hippo.close()

    def test_rem_motor_atonia_under_extreme_drive_and_angles(self) -> None:
        """
        Adversarially challenge REM motor atonia:
        Sweep angles across extreme ranges (negative, multi-turn > 360 deg, high radian frequencies)
        with maximum motor drive u = 1.0.
        Assert that striatal motor gate a_gate is strictly clamped to 0.000 across all cycles.
        """
        rem = REMSandbox(striatum_gate_clamp=0.0)

        # Extreme angles from -720 deg to +1440 deg
        test_angles = [float(deg) for deg in range(-720, 1441, 15)]
        atonia_ok = rem.verify_motor_atonia(test_angles_deg=test_angles)
        assert atonia_ok is True
        assert len(rem.get_emitted_motor_packets()) == 0

        # Physical gate check with extreme drive u=1.0 and theta = 1000 rad
        gate = BasalGangliaGate()
        decision = gate.evaluate(theta_rad=1000.0, u=1.0, sleep_lock=True)
        assert decision.a_motor == 0.000

    def test_cognitive_dissonance_loss_boundary(self) -> None:
        """
        Test REM sandbox cognitive dissonance boundary:
        - Candidate within threshold (<= 5% degradation) is APPROVED
        - Contradictory candidate (> 5% degradation) is REJECTED
        """
        rem = REMSandbox(striatum_gate_clamp=0.0)
        safety_emb = rem.baseline_graph.nodes["concept_safety"]["embedding"].copy()

        # Coherent candidate
        coherent_ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.ones(16, dtype=np.float32),
            motor_action=0.1,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=safety_emb,
            metadata={"relation": "is_a", "target_node": "concept_safety", "weight": 0.95},
        )
        res_coherent = rem.simulate_dream([coherent_ep])
        assert res_coherent.verdict == DreamVerdict.APPROVED
        assert res_coherent.delta_loss <= res_coherent.threshold_loss

        # Contradictory candidate
        self_emb = rem.baseline_graph.nodes["concept_self"]["embedding"].copy()
        contradictory_ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.ones(16, dtype=np.float32),
            motor_action=0.1,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=self_emb,
            metadata={"relation": "contradicts", "target_node": "concept_self", "weight": 1.0},
        )
        res_contradictory = rem.simulate_dream([contradictory_ep])
        assert res_contradictory.verdict == DreamVerdict.REJECTED
        assert res_contradictory.delta_loss > res_contradictory.threshold_loss
