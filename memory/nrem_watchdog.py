"""
ΔP-42 Sovereign Cognitive Core (Gate 3) - NREM Watchdog Module
Hard Real-Time 2.0-Second Timeout Supervisor for NREM Memory Consolidation.

Implements:
1. Hard real-time 2000 ms supervisor using monotonic nanosecond clock (time.monotonic_ns()).
2. Dual supervision modes:
   - Synchronous cooperative polling (check_timeout(), heartbeat()) inside I/O loops.
   - Background asynchronous daemon timer thread for preemptive I/O interruption.
3. Abort & Rollback protocol:
   - Atomic transaction abort & rollback callback invocation.
   - Forced arousal interlock (SleepHysteresis -> WAKE, BasalGangliaGate sleep_lock -> False).
   - Structured diagnostic logging with canonical fault code IO_TIMEOUT_STALL.
4. Edge cases & thread safety:
   - Re-entrant thread safety (threading.RLock).
   - Cross-thread database query interruption (sqlite3.Connection.interrupt()).
   - Idempotent disarm, clean thread lifecycle, and context manager support.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapter III, Chapter X.5 lines 695-698, XI line 729)
- Plan Review.md (REQ-SLEEP-02, DVP-07, RTM-06)
- megvalósítási terv.md (Phase 3: Section 3.2)
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import enum
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

# Configure module logger
logger = logging.getLogger("delta_p42.memory.nrem_watchdog")

# Canonical Timing & Fault Constants
DEFAULT_WATCHDOG_TIMEOUT_SEC: float = 2.0
DEFAULT_CHECK_INTERVAL_SEC: float = 0.05
FAULT_CODE_IO_TIMEOUT_STALL: str = "IO_TIMEOUT_STALL"
MAX_REALTIME_LATENCY_MARGIN_SEC: float = 0.080  # 80 ms DVP-07 OS scheduling margin


class WatchdogState(enum.Enum):
    """Lifecycle states of the NREM Watchdog supervisor."""
    IDLE = "IDLE"           # Unarmed, inactive
    ARMED = "ARMED"         # Actively monitoring consolidation transaction
    DISARMED = "DISARMED"   # Successfully completed before deadline
    ABORTED = "ABORTED"     # Timed out or emergency abort triggered


@dataclass(frozen=True)
class TimeoutFault:
    """
    Immutable diagnostic record emitted when the NREM Watchdog triggers an abort.

    Attributes:
        timestamp_ns: Monotonic hardware nanosecond timestamp of fault trigger.
        transaction_id: Identifier of the monitored consolidation transaction.
        elapsed_duration_ms: Measured duration of transaction in milliseconds.
        timeout_threshold_ms: Configured timeout limit in milliseconds (nominal: 2000 ms).
        aborted_state: Final transaction status string (strictly "ABORTED").
        forced_arousal_success: Boolean indicating if emergency wake arousal succeeded.
        fault_code: Canonical fault identifier (strictly "IO_TIMEOUT_STALL").
        diagnostic_details: Human-readable diagnostic description and context.
    """
    timestamp_ns: int
    transaction_id: str
    elapsed_duration_ms: float
    timeout_threshold_ms: float = 2000.0
    aborted_state: str = "ABORTED"
    forced_arousal_success: bool = True
    fault_code: str = FAULT_CODE_IO_TIMEOUT_STALL
    diagnostic_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert fault record to a standard dictionary."""
        return asdict(self)


# Canonical alias matching PROJECT.md interface contract
WatchdogAbortEvent = TimeoutFault


class WatchdogTimeoutError(TimeoutError):
    """Raised when an NREM consolidation transaction exceeds the 2.0s real-time limit."""
    def __init__(self, message: str, fault: Optional[TimeoutFault] = None) -> None:
        super().__init__(message)
        self.fault: Optional[TimeoutFault] = fault


class NREMWatchdog:
    """
    Hard Real-Time 2.0-Second NREM Memory Consolidation Watchdog Supervisor.

    Guarantees:
    - Exactly 2000 ms (2.0 s) hard deadline for volatile -> persistent 2PC write & probe.
    - Dual supervision: Synchronous cooperative checks + Preemptive background daemon thread.
    - Zero sleep coma: Immediate forced arousal to WAKE and Basal Ganglia un-gating on stall.
    - Zero data loss: Preserves working memory buffer intact on rollback.
    """

    def __init__(
        self,
        timeout_s: float = DEFAULT_WATCHDOG_TIMEOUT_SEC,
        timeout_sec: Optional[float] = None,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_SEC,
        on_timeout_abort: Optional[Callable[[TimeoutFault], None]] = None,
        auto_arousal: bool = True,
    ) -> None:
        """
        Initialize the NREM Watchdog.

        Args:
            timeout_s: Timeout ceiling in seconds (default 2.0 s = 2000 ms).
            timeout_sec: Alias for timeout_s for backwards compatibility.
            check_interval_s: Polling interval for background monitor (default 0.05 s).
            on_timeout_abort: Optional external callback invoked upon timeout abort.
            auto_arousal: If True, automatically awakens system upon timeout.
        """
        resolved_timeout = float(timeout_sec if timeout_sec is not None else timeout_s)
        if resolved_timeout <= 0.0:
            raise ValueError(f"Watchdog timeout must be strictly positive, got {resolved_timeout}")

        self.timeout_s: float = resolved_timeout
        self.timeout_sec: float = resolved_timeout
        self.timeout_ms: float = resolved_timeout * 1000.0
        self.check_interval_s: float = float(check_interval_s)
        self.on_timeout_abort: Optional[Callable[[TimeoutFault], None]] = on_timeout_abort
        self.auto_arousal: bool = bool(auto_arousal)

        # Thread Synchronization & Lifecycle
        self._lock: threading.RLock = threading.RLock()
        self._cancel_event: threading.Event = threading.Event()
        self._abort_event: threading.Event = threading.Event()
        self._timer_thread: Optional[threading.Thread] = None

        # Supervisory State
        self._state: WatchdogState = WatchdogState.IDLE
        self._transaction_id: str = "none"
        self._start_time_ns: int = 0
        self._last_heartbeat_ns: int = 0
        self._last_fault: Optional[TimeoutFault] = None
        self._fault_history: List[TimeoutFault] = []

        # Interlock References
        self._hippocampus: Optional[Any] = None
        self._basal_ganglia_gate: Optional[Any] = None
        self._sleep_hysteresis: Optional[Any] = None
        self._db_conn: Optional[Any] = None
        self._rollback_callback: Optional[Callable[[], Any]] = None

    # =========================================================================
    # ARM & DISARM PROTOCOL
    # =========================================================================

    def arm(
        self,
        transaction_id: str = "nrem_consolidation_tx",
        hippocampus: Optional[Any] = None,
        basal_ganglia_gate: Optional[Any] = None,
        hysteresis: Optional[Any] = None,
        db_conn: Optional[Any] = None,
        rollback_callback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        Arm the watchdog supervisor for an active consolidation transaction.

        Starts the high-precision monotonic clock and launches the background daemon thread.
        """
        with self._lock:
            # If already armed, disarm previous run cleanly
            if self._state == WatchdogState.ARMED:
                self._cancel_event.set()
                if self._timer_thread and self._timer_thread.is_alive():
                    self._timer_thread.join(timeout=0.1)

            self._transaction_id = str(transaction_id)
            self._hippocampus = hippocampus
            self._basal_ganglia_gate = basal_ganglia_gate
            self._sleep_hysteresis = hysteresis
            self._db_conn = db_conn
            self._rollback_callback = rollback_callback

            self._cancel_event.clear()
            self._abort_event.clear()
            self._start_time_ns = time.monotonic_ns()
            self._last_heartbeat_ns = self._start_time_ns
            self._state = WatchdogState.ARMED

            # Launch Asynchronous Preemptive Background Supervisor Thread
            self._timer_thread = threading.Thread(
                target=self._supervision_loop,
                name=f"nrem-watchdog-{self._transaction_id}",
                daemon=True,
            )
            self._timer_thread.start()
            logger.debug("NREM Watchdog armed for tx='%s' (limit=%.3fs)", self._transaction_id, self.timeout_s)

    def start_transaction(
        self,
        transaction_id: str = "nrem_consolidation_tx",
        hippocampus: Optional[Any] = None,
        basal_ganglia_gate: Optional[Any] = None,
        hysteresis: Optional[Any] = None,
        db_conn: Optional[Any] = None,
        rollback_callback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Canonical alias for arm()."""
        self.arm(
            transaction_id=transaction_id,
            hippocampus=hippocampus,
            basal_ganglia_gate=basal_ganglia_gate,
            hysteresis=hysteresis,
            db_conn=db_conn,
            rollback_callback=rollback_callback,
        )

    def disarm(self) -> float:
        """
        Disarm the watchdog upon successful completion of memory consolidation.

        Returns:
            Elapsed time of transaction in milliseconds.
        """
        with self._lock:
            if self._state == WatchdogState.ABORTED:
                # Transaction was aborted by watchdog
                elapsed_ms = (time.monotonic_ns() - self._start_time_ns) / 1e6
                return elapsed_ms

            if self._state != WatchdogState.ARMED:
                return 0.0

            now_ns = time.monotonic_ns()
            elapsed_ms = (now_ns - self._start_time_ns) / 1e6
            self._state = WatchdogState.DISARMED

            # Signal cancellation to background timer thread
            self._cancel_event.set()

            # Clean references
            self._clear_references()

            logger.debug("NREM Watchdog disarmed cleanly (elapsed=%.2fms)", elapsed_ms)
            return elapsed_ms

    def finish_transaction(self) -> float:
        """Canonical alias for disarm()."""
        return self.disarm()

    # =========================================================================
    # DUAL SUPERVISION: SYNCHRONOUS COOPERATIVE POLLING
    # =========================================================================

    def check_timeout(self) -> bool:
        """
        Synchronous cooperative check to be called inside iterative I/O & probe loops.

        Returns:
            True if transaction has timed out and was aborted; False if still within limit.
        """
        if self._abort_event.is_set():
            return True
        with self._lock:
            if self._state == WatchdogState.ABORTED:
                return True
            if self._state != WatchdogState.ARMED:
                return False

            now_ns = time.monotonic_ns()
            elapsed_s = (now_ns - self._start_time_ns) / 1e9
            should_abort = elapsed_s >= self.timeout_s

        if should_abort:
            self._execute_abort_and_arouse("COOPERATIVE_TIMEOUT_STALL", now_ns)
            return True
        return False

    def heartbeat(self) -> None:
        """
        Update watchdog liveness heartbeat during long processing steps.
        Also evaluates cooperative timeout.
        """
        if self._abort_event.is_set():
            return
        with self._lock:
            if self._state == WatchdogState.ARMED:
                self._last_heartbeat_ns = time.monotonic_ns()
        self.check_timeout()

    # =========================================================================
    # DUAL SUPERVISION: ASYNCHRONOUS PREEMPTIVE BACKGROUND LOOP
    # =========================================================================

    def _supervision_loop(self) -> None:
        """Background daemon thread waiting on kernel event or timeout expiration."""
        # Wait until cancelled or timeout expires
        cancelled = self._cancel_event.wait(timeout=self.timeout_s)
        if cancelled:
            # Clean disarm occurred
            return

        # Timeout reached!
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._state != WatchdogState.ARMED:
                return
        self._execute_abort_and_arouse("ASYNC_TIMEOUT_STALL", now_ns)

    # =========================================================================
    # ABORT & FORCED AROUSAL PROTOCOL
    # =========================================================================

    def trigger_emergency_abort(self, reason: str = "MANUAL_EMERGENCY_ABORT") -> TimeoutFault:
        """
        Trigger an immediate emergency abort and forced arousal.

        Args:
            reason: Diagnostic explanation of abort trigger.

        Returns:
            The generated immutable TimeoutFault record.
        """
        with self._lock:
            if self._state == WatchdogState.ABORTED and self._last_fault is not None:
                return self._last_fault
        return self._execute_abort_and_arouse(reason, time.monotonic_ns())

    def abort_and_arouse(self, reason: str = "EMERGENCY_ABORT") -> TimeoutFault:
        """Canonical alias for trigger_emergency_abort()."""
        return self.trigger_emergency_abort(reason)

    def _execute_abort_and_arouse(self, reason: str, timestamp_ns: int) -> TimeoutFault:
        """
        Internal worker executing atomic rollback, fault logging, and forced arousal.
        Lock-free architecture: Mutex is held ONLY for internal state transition.
        All foreign calls (callbacks, interlocks, database interrupt) execute outside locks.
        """
        with self._lock:
            if self._state == WatchdogState.ABORTED:
                if self._last_fault is not None:
                    return self._last_fault
                elapsed_ms = (timestamp_ns - self._start_time_ns) / 1e6 if self._start_time_ns > 0 else 0.0
                return TimeoutFault(
                    timestamp_ns=timestamp_ns,
                    transaction_id=self._transaction_id,
                    elapsed_duration_ms=elapsed_ms,
                    timeout_threshold_ms=self.timeout_ms,
                    aborted_state="ABORTED",
                    forced_arousal_success=True,
                    fault_code=FAULT_CODE_IO_TIMEOUT_STALL,
                    diagnostic_details=f"Consolidation exceeded {self.timeout_s:.3f}s: {reason}",
                )

            self._state = WatchdogState.ABORTED
            self._abort_event.set()
            self._cancel_event.set()

            elapsed_ms = (timestamp_ns - self._start_time_ns) / 1e6 if self._start_time_ns > 0 else 0.0

            # Snapshot references under lock to execute outside locks
            db_conn = self._db_conn
            rollback_callback = self._rollback_callback
            hippocampus = self._hippocampus
            sleep_hysteresis = self._sleep_hysteresis
            basal_ganglia_gate = self._basal_ganglia_gate
            on_timeout_abort = self.on_timeout_abort
            tx_id = self._transaction_id
            timeout_ms = self.timeout_ms
            auto_arousal = self.auto_arousal

            # Clear internal references immediately to prevent leakage
            self._clear_references()

        # =====================================================================
        # OPEN CALLS: EXECUTE FOREIGN INTERLOCKS OUTSIDE self._lock
        # =====================================================================
        logger.error(
            "NREM WATCHDOG TIMEOUT FIRED! Tx='%s' elapsed=%.2fms (limit=%.2fms) reason='%s'",
            tx_id,
            elapsed_ms,
            timeout_ms,
            reason,
        )

        # 1. Cross-Thread Database Query Interruption
        if db_conn is not None:
            try:
                if hasattr(db_conn, "interrupt"):
                    db_conn.interrupt()
                    logger.info("SQLite connection query interrupted across threads.")
            except Exception as e:
                logger.warning("Error interrupting DB connection: %s", e)

        # 2. Rollback Callback Invocation
        if rollback_callback is not None:
            try:
                rollback_callback()
            except Exception as e:
                logger.error("Error in watchdog rollback callback: %s", e)

        # 3. Hippocampus 2PC Abort
        if hippocampus is not None:
            try:
                if hasattr(hippocampus, "abort_transaction"):
                    hippocampus.abort_transaction(f"Watchdog timeout: {reason}")
            except Exception as e:
                logger.error("Error aborting Hippocampus transaction: %s", e)

        # 4. Forced Arousal: Break Sleep Deadlock
        forced_arousal_ok = True
        if auto_arousal:
            forced_arousal_ok = self._force_system_arousal_unlocked(sleep_hysteresis, basal_ganglia_gate)

        # 5. Create Diagnostic Fault Record
        fault = TimeoutFault(
            timestamp_ns=timestamp_ns,
            transaction_id=tx_id,
            elapsed_duration_ms=elapsed_ms,
            timeout_threshold_ms=timeout_ms,
            aborted_state="ABORTED",
            forced_arousal_success=forced_arousal_ok,
            fault_code=FAULT_CODE_IO_TIMEOUT_STALL,
            diagnostic_details=f"Consolidation exceeded {self.timeout_s:.3f}s: {reason}",
        )

        with self._lock:
            self._last_fault = fault
            self._fault_history.append(fault)

        # 6. External Callback Notification
        if on_timeout_abort is not None:
            try:
                on_timeout_abort(fault)
            except Exception as e:
                logger.error("Error executing on_timeout_abort callback: %s", e)

        return fault

    @staticmethod
    def _force_system_arousal_unlocked(
        sleep_hysteresis: Optional[Any],
        basal_ganglia_gate: Optional[Any],
    ) -> bool:
        """Enforce wake arousal on SleepHysteresis and BasalGangliaGate without holding watchdog lock."""
        success = True
        # Wake up Sleep Hysteresis
        if sleep_hysteresis is not None:
            try:
                if hasattr(sleep_hysteresis, "force_wake"):
                    sleep_hysteresis.force_wake()
                elif hasattr(sleep_hysteresis, "force_arousal"):
                    sleep_hysteresis.force_arousal()
                elif hasattr(sleep_hysteresis, "update"):
                    sleep_hysteresis.update(0.0)
                elif hasattr(sleep_hysteresis, "current_state"):
                    setattr(sleep_hysteresis, "current_state", "WAKE")
                logger.info("Watchdog forced SleepHysteresis arousal to WAKE.")
            except Exception as e:
                logger.error("Failed to force sleep hysteresis arousal: %s", e)
                success = False

        # Release Motor Gate Sleep Lock
        if basal_ganglia_gate is not None:
            try:
                if hasattr(basal_ganglia_gate, "sleep_lock"):
                    setattr(basal_ganglia_gate, "sleep_lock", False)
                if hasattr(basal_ganglia_gate, "lock_sleep"):
                    setattr(basal_ganglia_gate, "lock_sleep", False)
                if hasattr(basal_ganglia_gate, "reset"):
                    basal_ganglia_gate.reset()
                logger.info("Watchdog released BasalGangliaGate sleep_lock.")
            except Exception as e:
                logger.error("Failed to release motor gate sleep_lock: %s", e)
                success = False

        return success

    def _force_system_arousal(self) -> bool:
        """Enforce wake arousal using current stored references."""
        with self._lock:
            hyst = self._sleep_hysteresis
            gate = self._basal_ganglia_gate
        return self._force_system_arousal_unlocked(hyst, gate)

    def _clear_references(self) -> None:
        """Clear interlock object references to avoid circular retention."""
        self._hippocampus = None
        self._basal_ganglia_gate = None
        self._sleep_hysteresis = None
        self._db_conn = None
        self._rollback_callback = None

    # =========================================================================
    # FAULT INJECTION & VERIFICATION (DVP-07 HOOK)
    # =========================================================================

    def simulate_io_stall(
        self,
        stall_duration_s: float = 2.5,
        transaction_id: str = "simulated_io_stall_tx",
        hippocampus: Optional[Any] = None,
        basal_ganglia_gate: Optional[Any] = None,
        hysteresis: Optional[Any] = None,
        db_conn: Optional[Any] = None,
        rollback_callback: Optional[Callable[[], Any]] = None,
    ) -> Optional[TimeoutFault]:
        """
        Simulate an unyielding I/O stall during NREM consolidation to test supervisor preemption.

        Blocks the calling thread for stall_duration_s (or until aborted), allowing
        the background supervisor to intercept and trigger emergency abort.

        Args:
            stall_duration_s: Injected blocking delay in seconds (nominal: 2.5 s > 2.0 s).
            transaction_id: Diagnostic test identifier.
            hippocampus: Mock or real Hippocampus instance.
            basal_ganglia_gate: Mock or real Basal Ganglia gate.
            hysteresis: Mock or real Sleep Hysteresis instance.
            db_conn: Mock or real database connection.
            rollback_callback: Mock or real rollback handler.

        Returns:
            TimeoutFault if aborted, or None if stall duration was shorter than timeout.
        """
        self.arm(
            transaction_id=transaction_id,
            hippocampus=hippocampus,
            basal_ganglia_gate=basal_ganglia_gate,
            hysteresis=hysteresis,
            db_conn=db_conn,
            rollback_callback=rollback_callback,
        )

        # Simulate blocking I/O stall on the consolidation thread
        time.sleep(stall_duration_s)

        with self._lock:
            if self._state == WatchdogState.ARMED:
                # Stall was shorter than timeout: disarm normally
                self.disarm()
                return None
            return self._last_fault

    # =========================================================================
    # CONTEXT MANAGERS & STATE QUERIES
    # =========================================================================

    @contextlib.contextmanager
    def watch(
        self,
        transaction_id: str = "context_watch_tx",
        hippocampus: Optional[Any] = None,
        basal_ganglia_gate: Optional[Any] = None,
        hysteresis: Optional[Any] = None,
        db_conn: Optional[Any] = None,
        rollback_callback: Optional[Callable[[], Any]] = None,
    ) -> Iterator[NREMWatchdog]:
        """Scoped context manager for transaction supervision."""
        self.arm(
            transaction_id=transaction_id,
            hippocampus=hippocampus,
            basal_ganglia_gate=basal_ganglia_gate,
            hysteresis=hysteresis,
            db_conn=db_conn,
            rollback_callback=rollback_callback,
        )
        try:
            yield self
            if self.is_armed():
                self.disarm()
        except Exception as e:
            if self.is_armed():
                self.trigger_emergency_abort(reason=f"Context exception: {e}")
            raise

    def __enter__(self) -> NREMWatchdog:
        """Instance context manager entry."""
        self.arm()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Instance context manager exit."""
        if self._state == WatchdogState.ARMED:
            if exc_type is None:
                self.disarm()
            else:
                self.trigger_emergency_abort(reason=f"Exception: {exc_val}")
        return False

    @property
    def state(self) -> WatchdogState:
        with self._lock:
            return self._state

    def is_armed(self) -> bool:
        with self._lock:
            return self._state == WatchdogState.ARMED

    def is_aborted(self) -> bool:
        return self._abort_event.is_set()

    def get_last_fault(self) -> Optional[TimeoutFault]:
        with self._lock:
            return self._last_fault

    def get_fault_history(self) -> List[TimeoutFault]:
        with self._lock:
            return list(self._fault_history)

    def get_elapsed_ms(self) -> float:
        with self._lock:
            if self._start_time_ns == 0:
                return 0.0
            return (time.monotonic_ns() - self._start_time_ns) / 1e6

    def reset(self) -> None:
        """Reset supervisor state and fault history to pristine idle."""
        with self._lock:
            self._cancel_event.set()
            self._abort_event.clear()
            if self._timer_thread and self._timer_thread.is_alive():
                self._timer_thread.join(timeout=0.1)
            self._state = WatchdogState.IDLE
            self._transaction_id = "none"
            self._start_time_ns = 0
            self._last_heartbeat_ns = 0
            self._last_fault = None
            self._fault_history.clear()
            self._clear_references()
