"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 Test Suite
Module: tests.test_hippocampus_nrem
Author: ΔP-42 Senior QA & Testing Team (Zero Downgrade Mandate)

Comprehensive unit and adversarial test suite for:
- memory.hippocampus_nrem (EpisodicBuffer, HippocampusNREM, 2PC, Cosine Similarity Probe, Rollback)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import List
import uuid

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.hippocampus_nrem import (
    BufferOverflowException,
    CommitResult,
    CommitState,
    DEFAULT_MAX_EPISODES,
    DEFAULT_RAM_LIMIT_BYTES,
    DEFAULT_SLEEP_THRESHOLD,
    EpisodeRecord,
    EpisodicBuffer,
    HippocampusNREM,
    IntegrityProbeException,
    PROBE_COSINE_THRESHOLD,
    TransactionAbortedException,
)


class TestWorkingMemoryBuffer:
    """Tests for EpisodicBuffer volatile RAM storage and context saturation C(t)."""

    def test_buffer_initialization(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=1000, max_episodes=50, sleep_threshold=0.85)
        assert buf.ram_limit_bytes == 1000
        assert buf.max_episodes == 50
        assert buf.sleep_threshold == 0.85
        assert buf.total_bytes == 0
        assert len(buf) == 0
        assert buf.context_saturation == 0.0
        assert buf.get_context_saturation() == 0.0
        assert buf.sleep_requested is False

    def test_record_episodic_event_and_saturation(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=10_000, max_episodes=100, sleep_threshold=0.85)
        rng = np.random.default_rng(42)

        sensory = rng.standard_normal(16).astype(np.float32)
        reward = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        emb = rng.standard_normal(32).astype(np.float32)

        ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=sensory,
            motor_action=0.5,
            reward_vector=reward,
            embedding=emb,
            metadata={"source": "test_agent"},
        )

        ep_id = buf.add_episode(ep)
        assert ep_id == ep.episode_uuid
        assert len(buf) == 1
        assert buf.total_bytes == ep.get_byte_size()
        expected_sat = ep.get_byte_size() / 10_000.0
        assert abs(buf.context_saturation - expected_sat) < 1e-6
        assert buf.sleep_requested is False

    def test_buffer_sleep_requested_trigger(self) -> None:
        # Construct buffer with small limit so single record triggers sleep threshold
        buf = EpisodicBuffer(ram_limit_bytes=500, max_episodes=10, sleep_threshold=0.85)
        rng = np.random.default_rng(42)

        # 64 float32 values = 256 bytes + uuid + overhead > 425 bytes (85% of 500)
        emb = rng.standard_normal(64).astype(np.float32)
        ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=rng.standard_normal(32).astype(np.float32),
            motor_action=0.7,
            reward_vector=np.ones(5, dtype=np.float32),
            embedding=emb,
        )

        sat = buf.append_episode(ep)
        assert sat >= 0.85
        assert buf.sleep_requested is True

    def test_buffer_capacity_saturation_clamping_and_overflow(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=200, max_episodes=5, sleep_threshold=0.85)
        # Attempt to add record that severely exceeds 1.05 safety margin
        huge_emb = np.zeros(200, dtype=np.float32)  # 800 bytes
        huge_ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.zeros(10, dtype=np.float32),
            motor_action=0.1,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=huge_emb,
        )

        with pytest.raises(BufferOverflowException):
            buf.add_episode(huge_ep)

    def test_buffer_invalid_record_type_rejection(self) -> None:
        buf = EpisodicBuffer()
        with pytest.raises(TypeError):
            buf.add_episode("invalid_record")  # type: ignore

    def test_uuid4_rfc4122_compliance(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=100_000_000, max_episodes=2000)
        generated_uuids = set()

        for _ in range(1000):
            ep_uuid = str(uuid.uuid4())
            parsed = uuid.UUID(ep_uuid, version=4)
            assert parsed.version == 4
            assert ep_uuid not in generated_uuids
            generated_uuids.add(ep_uuid)

        assert len(generated_uuids) == 1000

    def test_deduplication_on_same_uuid(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=10_000)
        target_uuid = str(uuid.uuid4())

        ep1 = EpisodeRecord(
            episode_uuid=target_uuid,
            timestamp_ns=1000,
            sensory_vector=np.zeros(10, dtype=np.float32),
            motor_action=0.2,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.zeros(16, dtype=np.float32),
        )
        buf.add_episode(ep1)
        bytes1 = buf.total_bytes

        # Update with larger embedding
        ep2 = EpisodeRecord(
            episode_uuid=target_uuid,
            timestamp_ns=2000,
            sensory_vector=np.zeros(10, dtype=np.float32),
            motor_action=0.5,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.zeros(32, dtype=np.float32),
        )
        buf.add_episode(ep2)
        assert len(buf) == 1
        assert buf.total_bytes == ep2.get_byte_size()
        assert buf.total_bytes > bytes1

    def test_purge_working_memory(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=10_000)
        ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.zeros(10, dtype=np.float32),
            motor_action=0.0,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.zeros(16, dtype=np.float32),
        )
        buf.add_episode(ep)
        assert len(buf) == 1
        assert buf.total_bytes > 0

        purged_bytes = buf.purge_working_memory()
        assert purged_bytes > 0
        assert buf.total_bytes == 0
        assert len(buf) == 0
        assert buf.context_saturation == 0.0
        assert buf.sleep_requested is False

    def test_clear_method_alias(self) -> None:
        buf = EpisodicBuffer(ram_limit_bytes=10_000)
        ep = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.zeros(5, dtype=np.float32),
            motor_action=0.0,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.zeros(8, dtype=np.float32),
        )
        buf.add_episode(ep)
        count = buf.clear()
        assert count == 1
        assert len(buf) == 0


class TestTwoPhaseCommitHappyPath:
    """Tests for 2PC consolidation into persistent SQLite storage."""

    def test_2pc_empty_buffer_immediate_success(self) -> None:
        hippo = HippocampusNREM(db_path=":memory:")
        res = hippo.consolidate_2pc()
        assert res.success is True
        assert res.state == CommitState.COMMITTED
        assert res.episodes_processed == 0
        assert res.episodes_persisted == 0
        assert res.min_probe_cosine == 1.0

    def test_2pc_prepare_writes_and_commits_to_persistent_store(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test_hippo.db"
        hippo = HippocampusNREM(db_path=db_file)
        rng = np.random.default_rng(101)

        # Record 10 episodes
        created_records: List[EpisodeRecord] = []
        for i in range(10):
            sensory = rng.standard_normal(16).astype(np.float32)
            rew = np.array([0.1 * i, 0.2, 0.0, 0.4, 0.5], dtype=np.float32)
            emb = rng.standard_normal(32).astype(np.float32)
            rec = hippo.record_event(
                sensory_vector=sensory,
                motor_action=0.1 * i,
                reward_vector=rew,
                embedding=emb,
                metadata={"index": i},
            )
            created_records.append(rec)

        assert len(hippo.buffer) == 10
        assert hippo.commit_state == CommitState.IDLE

        # Consolidate
        result = hippo.consolidate_2pc()
        assert result.success is True
        assert result.state == CommitState.COMMITTED
        assert result.episodes_processed == 10
        assert result.episodes_persisted == 10
        assert result.min_probe_cosine >= PROBE_COSINE_THRESHOLD
        assert result.purged_bytes > 0
        assert len(hippo.buffer) == 0
        assert hippo.context_saturation == 0.0

        # Query database directly to assert 100% zero data loss
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes;")
        count = cursor.fetchone()[0]
        assert count == 10

        # Verify readback content for each record
        for orig in created_records:
            cursor.execute("SELECT embedding, embedding_dtype, motor_action FROM episodes WHERE uuid = ?", (orig.uuid,))
            row = cursor.fetchone()
            assert row is not None
            raw_emb, dtype_str, motor = row
            readback_emb = np.frombuffer(raw_emb, dtype=np.dtype(dtype_str))
            # Cosine similarity between orig and readback
            norm_a = np.linalg.norm(orig.embedding)
            norm_b = np.linalg.norm(readback_emb)
            cos = float(np.dot(orig.embedding, readback_emb) / (norm_a * norm_b))
            assert cos >= 0.999
            assert abs(motor - orig.motor_action) < 1e-6
        conn.close()

    def test_alias_consolidate_nrem_2pc(self) -> None:
        hippo = HippocampusNREM(db_path=":memory:")
        hippo.record_event(
            sensory_vector=np.ones(8, dtype=np.float32),
            motor_action=0.5,
            reward_vector=np.ones(5, dtype=np.float32),
            embedding=np.ones(16, dtype=np.float32),
        )
        res = hippo.consolidate_nrem_2pc()
        assert res.success is True
        assert res.state == CommitState.COMMITTED
        assert len(hippo.buffer) == 0

    def test_knowledge_graph_tables_initialized(self) -> None:
        hippo = HippocampusNREM(db_path=":memory:")
        conn = hippo._conn
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        assert "episodes" in tables
        assert "knowledge_nodes" in tables
        assert "knowledge_edges" in tables


class TestTwoPhaseCommitFailureAndRollback:
    """Adversarial fault injection and rollback protection tests."""

    def test_2pc_probe_failure_triggers_abort_and_preserves_ram(self, tmp_path: Path) -> None:
        db_file = tmp_path / "test_probe_fail.db"
        # Set impossible similarity threshold (> 1.0) to force probe failure
        hippo = HippocampusNREM(db_path=db_file, min_probe_similarity=1.001)

        rng = np.random.default_rng(202)
        rec = hippo.record_event(
            sensory_vector=rng.standard_normal(8).astype(np.float32),
            motor_action=0.3,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=rng.standard_normal(16).astype(np.float32),
        )

        assert len(hippo.buffer) == 1
        initial_sat = hippo.context_saturation

        res = hippo.consolidate_2pc()
        assert res.success is False
        assert res.state == CommitState.ABORTED
        assert "Integrity probe failed" in str(res.error_message)

        # Working memory MUST be 100% preserved (zero amnesic data loss)
        assert len(hippo.buffer) == 1
        assert hippo.context_saturation == initial_sat
        assert hippo.buffer.get_episodes()[0].uuid == rec.uuid

        # Database must have 0 committed records due to rollback
        conn = sqlite3.connect(db_file)
        count = conn.execute("SELECT COUNT(*) FROM episodes;").fetchone()[0]
        assert count == 0
        conn.close()

    def test_pre_existing_abort_signal_aborts_immediately(self) -> None:
        hippo = HippocampusNREM(db_path=":memory:")
        hippo.record_event(
            sensory_vector=np.ones(4, dtype=np.float32),
            motor_action=0.1,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.ones(8, dtype=np.float32),
        )

        # Set external abort request before consolidate
        hippo.abort_transaction("External abort injected")
        res = hippo.consolidate_2pc()
        assert res.success is False
        assert res.state == CommitState.ABORTED
        assert "pre-existing abort signal" in str(res.error_message)
        # Buffer remains 100% intact
        assert len(hippo.buffer) == 1

    def test_disk_write_error_triggers_rollback(self) -> None:
        hippo = HippocampusNREM(db_path=":memory:")
        hippo.record_event(
            sensory_vector=np.ones(4, dtype=np.float32),
            motor_action=0.1,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.ones(8, dtype=np.float32),
        )

        # Close database connection to simulate severe I/O fault
        hippo._conn.close()

        res = hippo.consolidate_2pc()
        assert res.success is False
        assert res.state == CommitState.ABORTED
        # RAM buffer preserved
        assert len(hippo.buffer) == 1


class TestConcurrencyAndSnapshotImmutability:
    """Thread safety, snapshot immutability, and numerical edge cases."""

    def test_concurrent_event_recording_thread_safety(self) -> None:
        hippo = HippocampusNREM(db_path=":memory:", ram_limit_bytes=5_000_000)
        num_threads = 8
        records_per_thread = 25

        def worker(thread_idx: int) -> None:
            rng = np.random.default_rng(thread_idx + 10)
            for j in range(records_per_thread):
                hippo.record_event(
                    sensory_vector=rng.standard_normal(8).astype(np.float32),
                    motor_action=0.1 * (j % 10),
                    reward_vector=np.ones(5, dtype=np.float32),
                    embedding=rng.standard_normal(16).astype(np.float32),
                    metadata={"thread": thread_idx, "item": j},
                )

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_count = num_threads * records_per_thread
        assert len(hippo.buffer) == expected_count

        # Consolidate all concurrent records cleanly
        res = hippo.consolidate_2pc()
        assert res.success is True
        assert res.episodes_persisted == expected_count
        assert len(hippo.buffer) == 0

    def test_episodic_record_immutability(self) -> None:
        rec = EpisodeRecord(
            episode_uuid=str(uuid.uuid4()),
            timestamp_ns=12345,
            sensory_vector=np.zeros(5, dtype=np.float32),
            motor_action=0.5,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=np.zeros(8, dtype=np.float32),
        )
        with pytest.raises(FrozenInstanceError):
            rec.motor_action = 0.9  # type: ignore

    def test_cosine_similarity_edge_cases(self) -> None:
        # Test zero vectors
        v_zero = np.zeros(16, dtype=np.float32)
        v_rand = np.ones(16, dtype=np.float32)
        sim_zero = HippocampusNREM._compute_cosine_similarity(v_zero, v_rand)
        assert sim_zero == 0.0

        # Test identical vectors
        sim_id = HippocampusNREM._compute_cosine_similarity(v_rand, v_rand)
        assert abs(sim_id - 1.0) < 1e-6

        # Test opposite vectors
        sim_opp = HippocampusNREM._compute_cosine_similarity(v_rand, -v_rand)
        assert abs(sim_opp - (-1.0)) < 1e-6
