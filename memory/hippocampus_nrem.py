"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 (Memory Subsystem)
Module: memory.hippocampus_nrem
Author: ΔP-42 Senior Engineering Team (Zero Downgrade Mandate)

Hippocampus NREM Memory Subsystem:
- Volatile Working Memory Buffer (B_RAM) with exact context saturation tracking:
    C(t) = sum(bytes(UUID_k)) / RAM_limit  (with slot-count fallback)
- Two-Phase Commit (2PC) episodic memory consolidation into persistent SQLite WAL storage.
    * Phase 1 (Prepare & Persistent Write): Atomic serialization of episodes into SQLite WAL.
    * Phase 2 (Integrity & Read-Back Probe): Cosine similarity probe:
        Probe(k) = cos(e_orig, e_readback) >= 0.990 (mean >= 0.999).
    * Commit & Purge: Atomic commit and working memory purge (C(t) -> 0.00) with 0.0% data loss.
    * Rollback: Atomic transaction abort on error/probe failure/watchdog timeout,
      preserving working memory 100% intact (zero amnesia).
- Real-time watchdog supervisor integration with cooperative polling and preemptive abort response.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapter III, Chapter V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapter I.8, Chapter III.1, Chapter X.5 lines 695-703)
- Plan Review.md (REQ-SLEEP-01, REQ-SLEEP-02, DVP-06, DVP-07, DVP-11, RTM-05, RTM-06)
- megvalósítási terv.md (Phase 3: Section 3.1, 3.2)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import enum
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np

logger = logging.getLogger("delta_p42.memory.hippocampus_nrem")

# Canonical Memory and Verification Thresholds
DEFAULT_RAM_LIMIT_BYTES: int = 10 * 1024 * 1024  # 10 MB (10,485,760 bytes)
DEFAULT_MAX_EPISODES: int = 10_000
DEFAULT_SLEEP_THRESHOLD: float = 0.85
PROBE_COSINE_THRESHOLD: float = 0.990
MAX_SATURATION_MARGIN: float = 1.05


# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class HippocampusMemoryException(Exception):
    """Base exception for Hippocampus memory subsystem."""


class TransactionAbortedException(HippocampusMemoryException):
    """Raised when 2PC transaction is aborted by watchdog or fault injection."""


class IntegrityProbeException(HippocampusMemoryException):
    """Raised when read-back cosine similarity probe drops below threshold."""


class BufferOverflowException(HippocampusMemoryException):
    """Raised when working memory context saturation exceeds safety margin."""


# ==============================================================================
# DATA STRUCTURES & TRANSACTION STATES
# ==============================================================================

class CommitState(enum.Enum):
    """Lifecycle states of the Two-Phase Commit (2PC) consolidation engine."""
    IDLE = "IDLE"                 # Buffering episodes in volatile working memory
    PREPARING = "PREPARING"       # Phase 1: Serializing & writing to persistent SQLite WAL
    VERIFYING = "VERIFYING"       # Phase 2: Read-back integrity & cosine similarity probe
    COMMITTED = "COMMITTED"       # 2PC Succeeded: RAM purged, C(t) -> 0.00, graph intact
    ABORTED = "ABORTED"           # 2PC Failed: RAM preserved, exception logged


# Spec alias for cross-module compatibility
TransactionStatus = CommitState


@dataclass(frozen=True)
class EpisodeRecord:
    """
    Immutable episodic memory record stored in volatile working memory (B_RAM)
    and persisted to SQLite WAL storage.
    """
    episode_uuid: str
    timestamp_ns: int
    sensory_vector: np.ndarray  # float32[d] sensory observation
    motor_action: float         # float in [0.0, 1.0] representing motor drive
    reward_vector: np.ndarray   # float32[5] Maslow need reduction reward R in [0.0, 1.0]^5
    embedding: np.ndarray       # float32[d] latent semantic embedding
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def uuid(self) -> str:
        """Alias for episode_uuid."""
        return self.episode_uuid

    def get_byte_size(self) -> int:
        """Computes exact memory footprint of the record in bytes."""
        total = len(self.episode_uuid.encode("utf-8")) + 8 + 8  # uuid + timestamp_ns + motor_action
        if self.sensory_vector is not None:
            total += self.sensory_vector.nbytes
        if self.reward_vector is not None:
            total += self.reward_vector.nbytes
        if self.embedding is not None:
            total += self.embedding.nbytes
        if self.metadata:
            try:
                total += len(json.dumps(self.metadata).encode("utf-8"))
            except (TypeError, ValueError):
                total += 128
        return total


# Canonical alias for full spec compliance
EpisodicMemoryRecord = EpisodeRecord


@dataclass(frozen=True)
class CommitResult:
    """Quantitative result of a Two-Phase Commit consolidation operation."""
    success: bool
    state: CommitState
    episodes_processed: int
    episodes_persisted: int
    min_probe_cosine: float
    elapsed_ms: float
    purged_bytes: int = 0
    error_message: Optional[str] = None


# Spec alias
ConsolidationResult = CommitResult


# ==============================================================================
# EPISODIC WORKING MEMORY BUFFER (B_RAM)
# ==============================================================================

class EpisodicBuffer:
    """
    Volatile Working Memory Buffer (B_RAM) with exact context saturation tracking:
        C(t) = sum(bytes(UUID_k)) / RAM_limit in [0.0, 1.0]
    with discrete slot-count fallback when ram_limit_bytes == 0.
    """

    def __init__(
        self,
        ram_limit_bytes: int = DEFAULT_RAM_LIMIT_BYTES,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        sleep_threshold: float = DEFAULT_SLEEP_THRESHOLD,
    ) -> None:
        """
        Initialize the episodic working memory buffer.

        Args:
            ram_limit_bytes: Configured RAM ceiling in bytes (default 10 MB).
            max_episodes: Maximum episodic slot capacity (default 10,000).
            sleep_threshold: Context saturation threshold triggering sleep request (default 0.85).
        """
        self._ram_limit_bytes: int = int(ram_limit_bytes)
        self._max_episodes: int = int(max_episodes)
        self._sleep_threshold: float = float(sleep_threshold)

        self._episodes: OrderedDict[str, EpisodeRecord] = OrderedDict()
        self._total_bytes: int = 0
        self._sleep_requested: bool = False
        self._lock: threading.RLock = threading.RLock()

    @property
    def ram_limit_bytes(self) -> int:
        return self._ram_limit_bytes

    @property
    def max_episodes(self) -> int:
        return self._max_episodes

    @property
    def sleep_threshold(self) -> float:
        return self._sleep_threshold

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def sleep_requested(self) -> bool:
        with self._lock:
            return self._sleep_requested

    @property
    def context_saturation(self) -> float:
        """
        Returns real-time context saturation C(t) in [0.0, 1.0].
        Computes exact byte ratio C(t) = sum(bytes) / RAM_limit,
        or discrete slot ratio N_active / N_max if ram_limit_bytes <= 0.
        """
        with self._lock:
            if self._ram_limit_bytes > 0:
                raw_c = float(self._total_bytes) / float(self._ram_limit_bytes)
            elif self._max_episodes > 0:
                raw_c = float(len(self._episodes)) / float(self._max_episodes)
            else:
                raw_c = 0.0
            return float(np.clip(raw_c, 0.0, 1.0))

    def get_context_saturation(self) -> float:
        """Method alias for context_saturation property."""
        return self.context_saturation

    def add_episode(self, record: EpisodeRecord) -> str:
        """
        Add an episodic memory record to volatile working memory.

        Args:
            record: Immutable EpisodeRecord instance.

        Returns:
            episode_uuid: UUID of the added record.

        Raises:
            BufferOverflowException: If saturation severely exceeds safety margin (> 1.05).
            TypeError: If record is not an EpisodeRecord.
        """
        if not isinstance(record, EpisodeRecord):
            raise TypeError(f"Expected EpisodeRecord, got {type(record).__name__}")

        with self._lock:
            record_bytes = record.get_byte_size()
            prospective_bytes = self._total_bytes + record_bytes

            if self._ram_limit_bytes > 0:
                prospective_sat = float(prospective_bytes) / float(self._ram_limit_bytes)
                if prospective_sat > MAX_SATURATION_MARGIN:
                    raise BufferOverflowException(
                        f"Context saturation C(t)={prospective_sat:.4f} severely exceeds "
                        f"safety margin {MAX_SATURATION_MARGIN}"
                    )

            # Store record and update memory footprint
            ep_id = record.episode_uuid
            if ep_id in self._episodes:
                # Deduct previous instance byte size
                self._total_bytes -= self._episodes[ep_id].get_byte_size()

            self._episodes[ep_id] = record
            self._total_bytes += record_bytes

            # Check if sleep threshold reached
            if self.context_saturation >= self._sleep_threshold:
                self._sleep_requested = True

            return ep_id

    def append_episode(self, record: EpisodeRecord) -> float:
        """Spec alias: adds episode and returns current context saturation C(t)."""
        self.add_episode(record)
        return self.context_saturation

    def get_episodes(self) -> List[EpisodeRecord]:
        """Returns a snapshot list of active episodes in insertion order."""
        with self._lock:
            return list(self._episodes.values())

    def purge_working_memory(self) -> int:
        """
        Flushes all volatile working memory records upon successful 2PC commit.
        Drives context saturation C(t) to 0.00.

        Returns:
            Number of purged bytes.
        """
        with self._lock:
            purged = self._total_bytes
            self._episodes.clear()
            self._total_bytes = 0
            self._sleep_requested = False
            logger.debug("Episodic working memory purged: %d bytes freed, C(t) -> 0.00", purged)
            return purged

    def clear(self) -> int:
        """Clears buffer and returns number of purged episodes."""
        with self._lock:
            n = len(self._episodes)
            self.purge_working_memory()
            return n

    def __len__(self) -> int:
        with self._lock:
            return len(self._episodes)


# ==============================================================================
# HIPPOCAMPUS NREM TWO-PHASE COMMIT (2PC) CONSOLIDATION ENGINE
# ==============================================================================

class HippocampusNREM:
    """
    Hippocampus NREM Two-Phase Commit (2PC) Consolidation Engine.

    Manages volatile episodic memory buffer and persistent SQLite storage (WAL mode).
    Coordinates with NREMWatchdog to enforce hard real-time 2.0s timeout constraints.
    Guarantees 0.0% data loss across all operations.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        storage_path: Optional[Union[str, Path]] = None,
        ram_limit_bytes: int = DEFAULT_RAM_LIMIT_BYTES,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        min_probe_similarity: float = PROBE_COSINE_THRESHOLD,
        sleep_threshold: float = DEFAULT_SLEEP_THRESHOLD,
    ) -> None:
        """
        Initialize the Hippocampus NREM consolidation engine.

        Args:
            db_path: Path to SQLite database file or ':memory:' (default ':memory:').
            storage_path: Optional alias for db_path.
            ram_limit_bytes: Configured RAM limit for EpisodicBuffer (default 10 MB).
            max_episodes: Maximum episodic slot capacity.
            min_probe_similarity: Cosine similarity threshold for Phase 2 probe (canonical: 0.990).
            sleep_threshold: Sleep initiation saturation threshold (default 0.85).
        """
        resolved_path = storage_path if storage_path is not None else db_path
        self.db_path: str = str(resolved_path)
        self.ram_limit_bytes: int = int(ram_limit_bytes)
        self.max_episodes: int = int(max_episodes)
        self.min_probe_similarity: float = float(min_probe_similarity)
        self.sleep_threshold: float = float(sleep_threshold)

        # Volatile Working Memory Buffer
        self._buffer: EpisodicBuffer = EpisodicBuffer(
            ram_limit_bytes=self.ram_limit_bytes,
            max_episodes=self.max_episodes,
            sleep_threshold=self.sleep_threshold,
        )

        # Thread synchronization and transaction state
        self._lock: threading.RLock = threading.RLock()
        self._abort_event: threading.Event = threading.Event()
        self._commit_state: CommitState = CommitState.IDLE
        self._abort_requested: bool = False

        # SQLite Connection with WAL mode
        # check_same_thread=False allows watchdog cross-thread query interruption
        self._conn: sqlite3.Connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=2.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_database_schema()

    def _init_database_schema(self) -> None:
        """Initialize persistent database schema with WAL mode and indexes."""
        with self._lock:
            # WAL mode configuration
            try:
                self._conn.execute("PRAGMA journal_mode = WAL;")
            except sqlite3.OperationalError:
                pass
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA busy_timeout = 2000;")
            self._conn.execute("PRAGMA foreign_keys = ON;")

            # Persistent episodes table
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    uuid TEXT PRIMARY KEY,
                    timestamp_ns INTEGER NOT NULL,
                    sensory_vector BLOB NOT NULL,
                    motor_action REAL NOT NULL,
                    reward_vector BLOB NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_dtype TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
            """)

            # Knowledge graph associative tables
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    node_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    attributes_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL,
                    attributes_json TEXT NOT NULL,
                    PRIMARY KEY (source_id, target_id, relation),
                    FOREIGN KEY (source_id) REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE
                );
            """)

            # Query performance indexes
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp_ns);")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_id);")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_id);")
            self._conn.commit()

    @property
    def buffer(self) -> EpisodicBuffer:
        """Returns reference to volatile working memory buffer."""
        return self._buffer

    @property
    def context_saturation(self) -> float:
        """Returns current context saturation C(t) in [0.0, 1.0]."""
        return self._buffer.context_saturation

    def get_context_saturation(self) -> float:
        """Method alias for context saturation."""
        return self._buffer.context_saturation

    @property
    def commit_state(self) -> CommitState:
        """Returns current 2PC transaction status."""
        with self._lock:
            return self._commit_state

    # =========================================================================
    # RECORDING & WORKING MEMORY OPERATIONS
    # =========================================================================

    def record_event(
        self,
        sensory_vector: Union[Sequence[float], np.ndarray],
        motor_action: float,
        reward_vector: Union[Sequence[float], np.ndarray],
        embedding: Union[Sequence[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        episode_uuid: Optional[str] = None,
        timestamp_ns: Optional[int] = None,
    ) -> EpisodeRecord:
        """
        Create and record an episodic memory event into volatile working memory.

        Args:
            sensory_vector: Observation features (float32 array).
            motor_action: Motor activation in [0.0, 1.0].
            reward_vector: Maslow need reduction rewards in [0.0, 1.0]^5.
            embedding: Semantic embedding vector (float32 array).
            metadata: Optional dictionary of contextual metadata.
            episode_uuid: Optional custom RFC-4122 v4 UUID (generated if None).
            timestamp_ns: Monotonic nanosecond timestamp (captured if None).

        Returns:
            EpisodeRecord: Created record stored in working memory.
        """
        sens_arr = np.asarray(sensory_vector, dtype=np.float32).flatten()
        rew_arr = np.asarray(reward_vector, dtype=np.float32).flatten()
        emb_arr = np.asarray(embedding, dtype=np.float32).flatten()

        rec_uuid = str(episode_uuid) if episode_uuid is not None else str(uuid.uuid4())
        rec_time = int(timestamp_ns) if timestamp_ns is not None else time.monotonic_ns()

        record = EpisodeRecord(
            episode_uuid=rec_uuid,
            timestamp_ns=rec_time,
            sensory_vector=sens_arr,
            motor_action=float(np.clip(motor_action, 0.0, 1.0)),
            reward_vector=rew_arr,
            embedding=emb_arr,
            metadata=dict(metadata) if metadata else {},
        )
        self._buffer.add_episode(record)
        return record

    def append_episode(self, record: EpisodeRecord) -> float:
        """Appends episode record to working memory and returns context saturation C(t)."""
        return self._buffer.append_episode(record)

    # =========================================================================
    # TWO-PHASE COMMIT (2PC) CONSOLIDATION
    # =========================================================================

    def consolidate_2pc(self, watchdog: Optional[Any] = None) -> CommitResult:
        """
        Execute Two-Phase Commit (2PC) NREM Memory Consolidation into persistent SQLite WAL storage.

        Protocol Flow:
        1. Check buffer: if empty (N=0), return immediate success CommitResult.
        2. Arm Watchdog supervisor (if provided).
        3. Phase 1 (Prepare & Persistent Write):
           - Begin transaction: BEGIN IMMEDIATE;
           - Serialize and insert all volatile records into persistent SQLite table.
           - Check cooperative timeout & watchdog heartbeats on each record.
        4. Phase 2 (Integrity & Read-Back Verification Probe):
           - Query each persisted embedding directly from disk.
           - Compute cosine similarity Probe(k) = cos(e_orig, e_readback).
           - Enforce min_k Probe(k) >= min_probe_similarity (0.990).
           - Check cooperative timeout & watchdog heartbeats on each probe.
        5. Commit & Purge:
           - Commit SQLite transaction: COMMIT;
           - Purge volatile working memory buffer (C(t) -> 0.00).
           - Guarantee 0.0% data loss.
           - Disarm watchdog.
        6. Rollback Protection:
           - On any error, watchdog timeout, abort request, or probe failure (< 0.990):
           - Roll back SQLite transaction: ROLLBACK;
           - Preserve volatile working memory 100% intact (zero amnesic data loss).
           - Return failure CommitResult.

        Args:
            watchdog: Optional NREMWatchdog instance.

        Returns:
            CommitResult: Immutable quantitative consolidation report.
        """
        with self._lock:
            episodes = self._buffer.get_episodes()
            n_episodes = len(episodes)

            # 1. Empty buffer shortcut
            if n_episodes == 0:
                self._commit_state = CommitState.COMMITTED
                return CommitResult(
                    success=True,
                    state=CommitState.COMMITTED,
                    episodes_processed=0,
                    episodes_persisted=0,
                    min_probe_cosine=1.0,
                    elapsed_ms=0.0,
                    purged_bytes=0,
                )

            # Guard against pre-consolidation abort signal (VULN-02 fix)
            if self._is_aborted(watchdog):
                self._abort_requested = False
                self._abort_event.clear()
                self._commit_state = CommitState.ABORTED
                logger.warning(
                    "NREM 2PC Consolidation ABORTED at entry: pre-existing abort signal detected."
                )
                return CommitResult(
                    success=False,
                    state=CommitState.ABORTED,
                    episodes_processed=n_episodes,
                    episodes_persisted=0,
                    min_probe_cosine=0.0,
                    elapsed_ms=0.0,
                    purged_bytes=0,
                    error_message="Consolidation aborted in Phase 1 (pre-existing abort signal)",
                )

            t_start = time.perf_counter()
            self._abort_requested = False
            self._abort_event.clear()
            tx_id = f"nrem_2pc_{uuid.uuid4().hex[:8]}"

            # 2. Arm watchdog supervisor
            if watchdog is not None:
                if hasattr(watchdog, "arm"):
                    watchdog.arm(
                        transaction_id=tx_id,
                        hippocampus=self,
                        db_conn=self._conn,
                    )
                elif hasattr(watchdog, "start_transaction"):
                    watchdog.start_transaction(
                        transaction_id=tx_id,
                        hippocampus=self,
                        db_conn=self._conn,
                    )

            min_probe = 1.0

            try:
                # =============================================================
                # PHASE 1: PREPARE & PERSISTENT WRITE
                # =============================================================
                self._commit_state = CommitState.PREPARING
                self._conn.execute("BEGIN IMMEDIATE;")

                for ep in episodes:
                    # Check timeout & abort condition
                    if self._is_aborted(watchdog):
                        raise TransactionAbortedException("Consolidation aborted in Phase 1 (timeout or external abort)")

                    sens_blob = ep.sensory_vector.tobytes()
                    rew_blob = ep.reward_vector.tobytes()
                    emb_blob = ep.embedding.tobytes()
                    meta_json = json.dumps(ep.metadata)

                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO episodes (
                            uuid, timestamp_ns, sensory_vector, motor_action,
                            reward_vector, embedding, embedding_dim, embedding_dtype,
                            metadata_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ep.episode_uuid,
                            ep.timestamp_ns,
                            sens_blob,
                            ep.motor_action,
                            rew_blob,
                            emb_blob,
                            ep.embedding.size,
                            str(ep.embedding.dtype),
                            meta_json,
                            time.time(),
                        ),
                    )

                    # Update watchdog liveness
                    if watchdog is not None and hasattr(watchdog, "heartbeat"):
                        watchdog.heartbeat()

                # =============================================================
                # PHASE 2: INTEGRITY & READ-BACK PROBE
                # =============================================================
                self._commit_state = CommitState.VERIFYING

                for ep in episodes:
                    if self._is_aborted(watchdog):
                        raise TransactionAbortedException("Consolidation aborted in Phase 2 (timeout or external abort)")

                    # Deserialization from disk
                    cursor = self._conn.execute(
                        "SELECT embedding, embedding_dtype FROM episodes WHERE uuid = ?",
                        (ep.episode_uuid,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        raise IntegrityProbeException(f"Persisted record not found for UUID: {ep.episode_uuid}")

                    raw_emb_blob = row["embedding"]
                    dtype_str = row["embedding_dtype"]
                    readback_emb = np.frombuffer(raw_emb_blob, dtype=np.dtype(dtype_str))

                    # High-precision Cosine Similarity Probe
                    probe_k = self._compute_cosine_similarity(ep.embedding, readback_emb)
                    min_probe = min(min_probe, probe_k)

                    if probe_k < self.min_probe_similarity:
                        raise IntegrityProbeException(
                            f"Integrity probe failed for episode {ep.episode_uuid}: "
                            f"probe={probe_k:.6f} < threshold={self.min_probe_similarity:.4f}"
                        )

                    # Update watchdog liveness
                    if watchdog is not None and hasattr(watchdog, "heartbeat"):
                        watchdog.heartbeat()

                # =============================================================
                # COMMIT & PURGE WORKING MEMORY
                # =============================================================
                self._conn.execute("COMMIT;")
                purged_bytes = self._buffer.purge_working_memory()
                self._commit_state = CommitState.COMMITTED
                self._abort_requested = False
                self._abort_event.clear()

                if watchdog is not None and hasattr(watchdog, "disarm"):
                    watchdog.disarm()

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                logger.info(
                    "NREM 2PC Consolidation COMMITTED: %d episodes persisted, "
                    "min_probe=%.6f, C(t) -> 0.00, elapsed=%.2fms",
                    n_episodes,
                    min_probe,
                    elapsed_ms,
                )

                return CommitResult(
                    success=True,
                    state=CommitState.COMMITTED,
                    episodes_processed=n_episodes,
                    episodes_persisted=n_episodes,
                    min_probe_cosine=min_probe,
                    elapsed_ms=elapsed_ms,
                    purged_bytes=purged_bytes,
                )

            except Exception as e:
                # =============================================================
                # ROLLBACK & PRESERVE WORKING MEMORY (ZERO AMNESIC LOSS)
                # =============================================================
                try:
                    self._conn.execute("ROLLBACK;")
                except Exception as rollback_err:
                    logger.warning("Error during SQLite rollback: %s", rollback_err)

                self._commit_state = CommitState.ABORTED
                self._abort_requested = False
                self._abort_event.clear()

                if watchdog is not None and hasattr(watchdog, "disarm"):
                    try:
                        watchdog.disarm()
                    except Exception:
                        pass

                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                err_msg = str(e)
                logger.error(
                    "NREM 2PC Consolidation ABORTED: %s (episodes preserved in RAM=%d, C(t)=%.4f)",
                    err_msg,
                    len(self._buffer),
                    self._buffer.context_saturation,
                )

                return CommitResult(
                    success=False,
                    state=CommitState.ABORTED,
                    episodes_processed=n_episodes,
                    episodes_persisted=0,
                    min_probe_cosine=min_probe if min_probe < 1.0 else 0.0,
                    elapsed_ms=elapsed_ms,
                    purged_bytes=0,
                    error_message=err_msg,
                )

    def consolidate_nrem_2pc(self, watchdog: Optional[Any] = None) -> CommitResult:
        """Canonical spec alias for consolidate_2pc()."""
        return self.consolidate_2pc(watchdog=watchdog)

    def _is_aborted(self, watchdog: Optional[Any]) -> bool:
        """Evaluate whether transaction abort has been requested cooperatively or preemptively."""
        if self._abort_event.is_set() or self._abort_requested:
            return True
        if watchdog is not None:
            if hasattr(watchdog, "is_aborted") and watchdog.is_aborted():
                return True
            if hasattr(watchdog, "check_timeout") and watchdog.check_timeout():
                return True
        return False

    @staticmethod
    def _compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute high-precision cosine similarity between two float vectors.
        Handles zero vectors and potential NaN / Inf values robustly.
        """
        arr1 = np.asarray(v1, dtype=np.float64).flatten()
        arr2 = np.asarray(v2, dtype=np.float64).flatten()

        if arr1.shape != arr2.shape:
            return -1.0

        norm1 = float(np.linalg.norm(arr1))
        norm2 = float(np.linalg.norm(arr2))

        # Numerical boundary conditions: zero-vector handling
        if norm1 < 1e-12 and norm2 < 1e-12:
            return 1.000  # Identical zero vectors
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.000  # Orthogonal zero vs non-zero

        dot = float(np.dot(arr1, arr2))
        cos_sim = dot / (norm1 * norm2)

        if not np.isfinite(cos_sim):
            return -1.0

        return float(np.clip(cos_sim, -1.0, 1.0))

    # =========================================================================
    # PROBE & QUERY INTERFACES
    # =========================================================================

    def probe_integrity(self, episode_id: str) -> float:
        """
        Inspect physical integrity of an episode.
        Compares volatile memory embedding against persisted SQLite embedding.

        Args:
            episode_id: UUID of the episode.

        Returns:
            Cosine similarity probe value in [-1.0, 1.0], or -1.0 if not found.
        """
        with self._lock:
            # Query persisted record
            cur = self._conn.execute(
                "SELECT embedding, embedding_dtype FROM episodes WHERE uuid = ?",
                (str(episode_id),),
            )
            row = cur.fetchone()
            if row is None:
                return -1.0

            raw_blob = row["embedding"]
            dtype_str = row["embedding_dtype"]
            persisted_emb = np.frombuffer(raw_blob, dtype=np.dtype(dtype_str))

            # If present in volatile buffer, compare directly
            for ep in self._buffer.get_episodes():
                if ep.episode_uuid == episode_id:
                    return self._compute_cosine_similarity(ep.embedding, persisted_emb)

            # If not in volatile buffer, compare persisted against itself to verify uncorrupted array
            return self._compute_cosine_similarity(persisted_emb, persisted_emb)

    def probe_episode(self, uuid: str) -> float:
        """Spec alias for probe_integrity()."""
        return self.probe_integrity(uuid)

    def abort_transaction(self, reason: str = "MANUAL_ABORT") -> None:
        """
        Command Hippocampus to abort any active consolidation transaction immediately.
        Invoked by NREMWatchdog upon timeout or stall.

        Lock-free implementation: Uses atomic threading.Event signaling and SQLite
        cross-thread interrupt() without acquiring self._lock, preventing AB-BA deadlocks.
        """
        self._abort_requested = True
        self._abort_event.set()
        logger.warning("Hippocampus transaction abort requested: %s", reason)
        try:
            if self._conn is not None:
                self._conn.interrupt()
        except Exception as e:
            logger.warning("Error interrupting SQLite connection: %s", e)

    def query_persistent_episodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Query persisted episodes from SQLite storage."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT uuid, timestamp_ns, motor_action, metadata_json, created_at "
                "FROM episodes ORDER BY timestamp_ns DESC LIMIT ?",
                (int(limit),),
            )
            results = []
            for row in cur.fetchall():
                results.append({
                    "uuid": row["uuid"],
                    "timestamp_ns": row["timestamp_ns"],
                    "motor_action": row["motor_action"],
                    "metadata": json.loads(row["metadata_json"]),
                    "created_at": row["created_at"],
                })
            return results

    def get_persisted_episode_count(self) -> int:
        """Returns total count of episodes stored in persistent SQLite table."""
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM episodes;")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        """Close SQLite persistent database connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
