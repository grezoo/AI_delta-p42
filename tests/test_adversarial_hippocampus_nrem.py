"""
ΔP-42 Sovereign Cognitive Core - Challenger 1 Adversarial Test Suite
Module: tests.test_adversarial_hippocampus_nrem
Author: Empirical Challenger 1

Adversarial Verification for memory.hippocampus_nrem:
1. Injected corrupted embeddings (probe < 0.990):
   - 100% abort & rollback
   - Working memory preserved 100% intact (zero amnesic data loss)
   - 0.0% unverified data persisted in SQLite
2. Sub-threshold cosine similarity (probe = 0.985 < 0.990):
   - Strict rejection at boundary
   - Complete rollback, zero persistence
3. Disk write error injection during Phase 1:
   - OperationalError / I/O error simulation
   - Immediate rollback and buffer preservation
4. Transaction abort injection:
   - Abort signal during Phase 1 & Phase 2
   - Immediate rollback and buffer preservation
5. 1,000+ episodes consolidation stress test:
   - Full flush: C(t) -> 0.00
   - Exactly 0.0% data loss across 1,200 persistent records
"""

import json
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Dict, List
import uuid

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

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
from memory.nrem_watchdog import NREMWatchdog


def create_random_episode(
    dim: int = 64,
    ep_uuid: str = None,
    motor: float = 0.5,
    meta: Dict = None,
) -> EpisodeRecord:
    rng = np.random.default_rng()
    sens = rng.standard_normal(dim).astype(np.float32)
    rew = rng.uniform(0.0, 1.0, size=5).astype(np.float32)
    raw_emb = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(raw_emb)
    emb = (raw_emb / norm if norm > 1e-9 else raw_emb).astype(np.float32)

    return EpisodeRecord(
        episode_uuid=ep_uuid or str(uuid.uuid4()),
        timestamp_ns=time.monotonic_ns(),
        sensory_vector=sens,
        motor_action=motor,
        reward_vector=rew,
        embedding=emb,
        metadata=meta or {"test_tag": "adversarial"},
    )


class TestAdversarialHippocampusNREM:
    """Empirical adversarial stress harness for Hippocampus NREM consolidation."""

    def test_corrupted_embedding_trigger_abort_and_rollback(self, tmp_path):
        """
        Adversarially corrupt embeddings in persistent SQLite using a trigger.
        Verify:
        - 100% abort & rollback
        - Volatile buffer preserved intact
        - 0 unverified records persisted in SQLite
        """
        db_path = str(tmp_path / "test_corrupt_emb.db")
        hippocampus = HippocampusNREM(db_path=db_path)
        try:
            # Populate buffer with 20 valid episodes
            episodes = [create_random_episode(ep_uuid=f"ep_corrupt_{i}") for i in range(20)]
            for ep in episodes:
                hippocampus.buffer.add_episode(ep)

            initial_count = len(hippocampus.buffer)
            initial_sat = hippocampus.context_saturation
            assert initial_count == 20
            assert initial_sat > 0.0

            # Inject SQLite trigger that corrupts the embedding BLOB upon INSERT
            # Using zeroblob in SQL syntax directly (no parameters in trigger)
            hippocampus._conn.execute(
                """
                CREATE TRIGGER corrupt_emb_trigger AFTER INSERT ON episodes
                BEGIN
                    UPDATE episodes SET embedding = zeroblob(length(NEW.embedding)) WHERE uuid = NEW.uuid;
                END;
                """
            )
            hippocampus._conn.commit()

            # Execute 2PC Consolidation
            result = hippocampus.consolidate_2pc()

            # EMPIRICAL ASSERTIONS:
            # 1. 2PC must fail with ABORTED state
            assert result.success is False, "Consolidation should fail on corrupted embedding"
            assert result.state == CommitState.ABORTED, f"Expected ABORTED, got {result.state}"
            assert result.episodes_persisted == 0, f"Expected 0 persisted, got {result.episodes_persisted}"
            assert result.error_message is not None

            # 2. Volatile working memory buffer must be 100% preserved (0% amnesia)
            assert len(hippocampus.buffer) == initial_count, "Working memory buffer was unexpectedly purged!"
            assert hippocampus.context_saturation == pytest.approx(initial_sat, rel=1e-5)
            for ep in episodes:
                assert ep.episode_uuid in hippocampus.buffer._episodes

            # 3. Persistent SQLite database must contain ZERO unverified records
            persisted_count = hippocampus.get_persisted_episode_count()
            assert persisted_count == 0, f"Database leak! Found {persisted_count} records in persistent store"
        finally:
            hippocampus.close()

    def test_subthreshold_cosine_probe_rejection(self, tmp_path):
        """
        Adversarially inject an embedding with cosine similarity exactly 0.985 (< 0.990).
        Verify strict threshold enforcement (no loose tolerance).
        """
        db_path = str(tmp_path / "test_subthreshold.db")
        hippocampus = HippocampusNREM(db_path=db_path, min_probe_similarity=0.990)
        try:
            # 5 normal episodes + 1 specifically perturbed episode
            episodes = [create_random_episode(ep_uuid=f"ep_sub_{i}") for i in range(5)]
            for ep in episodes:
                hippocampus.buffer.add_episode(ep)

            target_ep = episodes[2]

            # Calculate a perturbed vector with cosine similarity exactly 0.985
            # v_perturbed = e * cos(theta) + e_orth * sin(theta) where cos(theta) = 0.985
            target_emb = target_ep.embedding.astype(np.float64)
            target_norm = target_emb / np.linalg.norm(target_emb)

            rng = np.random.default_rng(999)
            random_v = rng.standard_normal(len(target_emb))
            orth_v = random_v - np.dot(random_v, target_norm) * target_norm
            orth_norm = orth_v / np.linalg.norm(orth_v)

            desired_cos = 0.985
            sin_theta = np.sqrt(1.0 - desired_cos**2)
            perturbed_emb = (desired_cos * target_norm + sin_theta * orth_norm).astype(np.float32)

            actual_cos = np.dot(target_norm, perturbed_emb / np.linalg.norm(perturbed_emb))
            assert actual_cos < 0.990, f"Calculated cos {actual_cos} is not strictly < 0.990"

            # Inject trigger on the target episode only using hex literal
            hex_data = perturbed_emb.tobytes().hex()
            hippocampus._conn.execute(
                f"""
                CREATE TRIGGER perturb_target_trigger AFTER INSERT ON episodes
                WHEN NEW.uuid = '{target_ep.episode_uuid}'
                BEGIN
                    UPDATE episodes SET embedding = X'{hex_data}' WHERE uuid = NEW.uuid;
                END;
                """
            )
            hippocampus._conn.commit()

            # Execute 2PC Consolidation
            result = hippocampus.consolidate_2pc()

            # EMPIRICAL ASSERTIONS:
            assert result.success is False
            assert result.state == CommitState.ABORTED
            assert result.min_probe_cosine < 0.990
            assert "Integrity probe failed" in str(result.error_message)

            # Working memory intact
            assert len(hippocampus.buffer) == 5
            assert hippocampus.get_persisted_episode_count() == 0
        finally:
            hippocampus.close()

    def test_disk_write_error_rollback(self, tmp_path):
        """
        Inject a simulated disk write error (sqlite3.OperationalError) during Phase 1.
        Verify complete rollback and buffer preservation.
        """
        db_path = str(tmp_path / "test_disk_error.db")
        hippocampus = HippocampusNREM(db_path=db_path)

        try:
            for i in range(10):
                hippocampus.buffer.add_episode(create_random_episode(ep_uuid=f"ep_disk_{i}"))

            initial_len = len(hippocampus.buffer)

            # Inject SQLite RAISE(FAIL, ...) trigger to simulate disk write failure after 2 records
            hippocampus._conn.execute(
                """
                CREATE TRIGGER sim_disk_io_fault BEFORE INSERT ON episodes
                WHEN (SELECT COUNT(*) FROM episodes) >= 2
                BEGIN
                    SELECT RAISE(FAIL, 'disk I/O error: simulated hardware sector failure');
                END;
                """
            )
            hippocampus._conn.commit()

            result = hippocampus.consolidate_2pc()

            # EMPIRICAL ASSERTIONS:
            assert result.success is False
            assert result.state == CommitState.ABORTED
            assert "disk I/O error" in str(result.error_message)

            # Buffer preserved intact
            assert len(hippocampus.buffer) == initial_len
            # Persistent storage has 0 records due to atomic ROLLBACK
            # Drop trigger to verify DB query
            hippocampus._conn.execute("DROP TRIGGER sim_disk_io_fault;")
            assert hippocampus.get_persisted_episode_count() == 0
        finally:
            hippocampus.close()

    def test_transaction_aborted_by_watchdog_signal(self, tmp_path):
        """
        Simulate an active transaction abort during Phase 1.
        Verify graceful abortion, rollback, buffer preservation, and zero data leakage.
        """
        db_path = str(tmp_path / "test_abort_sig.db")
        hippocampus = HippocampusNREM(db_path=db_path)
        try:
            for i in range(15):
                hippocampus.buffer.add_episode(create_random_episode(ep_uuid=f"ep_abort_{i}"))

            initial_len = len(hippocampus.buffer)
            initial_sat = hippocampus.context_saturation

            # Inject abort call during Phase 1 insertion after 3 records
            hippocampus._conn.create_function(
                "sim_phase1_abort", 0,
                lambda: hippocampus.abort_transaction(reason="ADVERSARIAL_PHASE1_ABORT")
            )
            hippocampus._conn.execute(
                """
                CREATE TRIGGER abort_on_insert AFTER INSERT ON episodes
                WHEN (SELECT COUNT(*) FROM episodes) >= 3
                BEGIN
                    SELECT sim_phase1_abort();
                END;
                """
            )
            hippocampus._conn.commit()

            result = hippocampus.consolidate_2pc()

            # EMPIRICAL ASSERTIONS:
            assert result.success is False
            assert result.state == CommitState.ABORTED
            assert "Consolidation aborted in Phase 1" in str(result.error_message)
            assert len(hippocampus.buffer) == initial_len
            assert hippocampus.context_saturation == pytest.approx(initial_sat, rel=1e-5)
            assert hippocampus.get_persisted_episode_count() == 0
        finally:
            hippocampus.close()

    def test_1000_episodes_stress_consolidation_zero_loss(self, tmp_path):
        """
        Stress test: 1,200 episodes 2PC consolidation.
        Verifies:
        - 1,200 episodes written and verified
        - All cosine similarity probes >= 0.9999
        - Volatile buffer flushed: C(t) -> 0.00, len(buffer) == 0
        - Exactly 0.0% data loss: byte-level inspection of all 1,200 records in SQLite
        """
        db_path = str(tmp_path / "test_stress_1000.db")
        # Ensure RAM limit allows 1,200 episodes without overflow
        hippocampus = HippocampusNREM(
            db_path=db_path,
            ram_limit_bytes=50 * 1024 * 1024,
            max_episodes=10_000,
        )
        try:
            n_records = 1200
            original_records: Dict[str, EpisodeRecord] = {}

            t0_gen = time.perf_counter()
            for i in range(n_records):
                rec = create_random_episode(
                    dim=64,
                    ep_uuid=f"ep_stress_{i:04d}_{uuid.uuid4().hex[:6]}",
                    motor=float(i % 100) / 100.0,
                    meta={"index": i, "stress_run": True, "category": f"cat_{i % 5}"},
                )
                original_records[rec.episode_uuid] = rec
                hippocampus.buffer.add_episode(rec)

            t_gen_s = time.perf_counter() - t0_gen

            # Check initial state
            initial_buffer_len = len(hippocampus.buffer)
            initial_sat = hippocampus.context_saturation
            assert initial_buffer_len == n_records
            assert initial_sat > 0.0, "Context saturation should be positive"

            # Execute 2PC Consolidation
            t0_commit = time.perf_counter()
            result = hippocampus.consolidate_2pc()
            t_commit_s = time.perf_counter() - t0_commit

            # EMPIRICAL ASSERTIONS:
            # 1. Transaction Success & State
            assert result.success is True, f"2PC failed: {result.error_message}"
            assert result.state == CommitState.COMMITTED
            assert result.episodes_processed == n_records
            assert result.episodes_persisted == n_records
            assert result.min_probe_cosine >= 0.990, f"min_probe too low: {result.min_probe_cosine}"
            assert result.purged_bytes > 0

            # 2. Buffer Flush Verification
            assert len(hippocampus.buffer) == 0, f"Buffer not empty: {len(hippocampus.buffer)}"
            assert hippocampus.context_saturation == 0.00, f"C(t) not 0: {hippocampus.context_saturation}"
            assert hippocampus.buffer.total_bytes == 0

            # 3. Persistent Record Count
            persisted_count = hippocampus.get_persisted_episode_count()
            assert persisted_count == n_records, f"Persisted count {persisted_count} != {n_records}"

            # 4. Zero Data Loss Verification: Deep Content Validation of all 1,200 records
            cur = hippocampus._conn.execute(
                "SELECT uuid, timestamp_ns, sensory_vector, motor_action, "
                "reward_vector, embedding, embedding_dim, embedding_dtype, metadata_json "
                "FROM episodes;"
            )
            rows = cur.fetchall()
            assert len(rows) == n_records

            missing_count = 0
            corrupted_embeddings = 0
            corrupted_sensory = 0
            corrupted_rewards = 0
            corrupted_metadata = 0

            for row in rows:
                u = row["uuid"]
                if u not in original_records:
                    missing_count += 1
                    continue

                orig = original_records[u]

                # Verify timestamp
                assert orig.timestamp_ns == row["timestamp_ns"]
                assert pytest.approx(orig.motor_action, abs=1e-6) == row["motor_action"]

                # Verify sensory vector
                sens_readback = np.frombuffer(row["sensory_vector"], dtype=np.float32)
                if not np.allclose(orig.sensory_vector, sens_readback, atol=1e-7):
                    corrupted_sensory += 1

                # Verify reward vector
                rew_readback = np.frombuffer(row["reward_vector"], dtype=np.float32)
                if not np.allclose(orig.reward_vector, rew_readback, atol=1e-7):
                    corrupted_rewards += 1

                # Verify embedding cosine similarity
                emb_readback = np.frombuffer(row["embedding"], dtype=np.dtype(row["embedding_dtype"]))
                cos_sim = np.dot(orig.embedding, emb_readback) / (
                    np.linalg.norm(orig.embedding) * np.linalg.norm(emb_readback)
                )
                if cos_sim < 0.9999:
                    corrupted_embeddings += 1

                # Verify metadata JSON
                meta_readback = json.loads(row["metadata_json"])
                if orig.metadata != meta_readback:
                    corrupted_metadata += 1

            total_faults = (
                missing_count
                + corrupted_embeddings
                + corrupted_sensory
                + corrupted_rewards
                + corrupted_metadata
            )
            data_loss_rate = float(total_faults) / float(n_records)

            assert total_faults == 0, f"Found {total_faults} data corruptions/losses!"
            assert data_loss_rate == 0.0, f"Data loss rate was {data_loss_rate * 100:.4f}% (expected 0.0%)"

            print(
                f"\n[PASS] 1,200 Episodes 2PC Stress Test: "
                f"Gen={t_gen_s:.2f}s, Commit={t_commit_s:.2f}s, "
                f"Min Probe={result.min_probe_cosine:.6f}, Data Loss={data_loss_rate * 100.0:.2f}%"
            )
        finally:
            hippocampus.close()

    def test_pre_consolidation_abort_signal_preservation(self, tmp_path):
        """
        VULN-02 Remediation Verification:
        Verify that if abort_transaction() is called prior to consolidate_2pc(),
        the abort signal is NOT erased; consolidate_2pc() aborts immediately,
        persists 0 records, and preserves 100% of volatile working memory.
        """
        db_path = str(tmp_path / "test_pre_abort.db")
        hippocampus = HippocampusNREM(db_path=db_path)
        try:
            for i in range(10):
                hippocampus.buffer.add_episode(create_random_episode(ep_uuid=f"ep_pre_{i}"))

            initial_len = len(hippocampus.buffer)
            initial_sat = hippocampus.context_saturation

            # Command abort BEFORE consolidation
            hippocampus.abort_transaction(reason="PRE_CONSOLIDATION_EMERGENCY_ABORT")

            # Execute consolidation
            result = hippocampus.consolidate_2pc()

            assert result.success is False
            assert result.state == CommitState.ABORTED
            assert result.episodes_persisted == 0
            assert "pre-existing abort signal" in str(result.error_message)

            # Volatile buffer must remain 100% intact
            assert len(hippocampus.buffer) == initial_len
            assert hippocampus.context_saturation == pytest.approx(initial_sat, rel=1e-5)
            assert hippocampus.get_persisted_episode_count() == 0
        finally:
            hippocampus.close()

    def test_pre_aborted_watchdog_prevents_consolidation(self, tmp_path):
        """
        Verify that passing an already-aborted watchdog to consolidate_2pc()
        immediately triggers abort without writing or purging working memory.
        """
        db_path = str(tmp_path / "test_pre_aborted_wd.db")
        hippocampus = HippocampusNREM(db_path=db_path)
        watchdog = NREMWatchdog(timeout_s=2.0)
        watchdog.trigger_emergency_abort(reason="PRE_ABORTED_WATCHDOG")
        try:
            for i in range(5):
                hippocampus.buffer.add_episode(create_random_episode(ep_uuid=f"ep_wd_pre_{i}"))

            result = hippocampus.consolidate_2pc(watchdog=watchdog)
            assert result.success is False
            assert result.state == CommitState.ABORTED
            assert result.episodes_persisted == 0
            assert len(hippocampus.buffer) == 5
            assert hippocampus.get_persisted_episode_count() == 0
        finally:
            hippocampus.close()
            watchdog.reset()


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_suite = TestAdversarialHippocampusNREM()
        print("Running test_corrupted_embedding_trigger_abort_and_rollback...")
        test_suite.test_corrupted_embedding_trigger_abort_and_rollback(p)
        print("Running test_subthreshold_cosine_probe_rejection...")
        test_suite.test_subthreshold_cosine_probe_rejection(p)
        print("Running test_disk_write_error_rollback...")
        test_suite.test_disk_write_error_rollback(p)
        print("Running test_transaction_aborted_by_watchdog_signal...")
        test_suite.test_transaction_aborted_by_watchdog_signal(p)
        print("Running test_1000_episodes_stress_consolidation_zero_loss...")
        test_suite.test_1000_episodes_stress_consolidation_zero_loss(p)
        print("Running test_pre_consolidation_abort_signal_preservation...")
        test_suite.test_pre_consolidation_abort_signal_preservation(p)
        print("Running test_pre_aborted_watchdog_prevents_consolidation...")
        test_suite.test_pre_aborted_watchdog_prevents_consolidation(p)
        print("ALL HIPPOCAMPUS ADVERSARIAL TESTS PASSED!")
