"""
ΔP-42 Sovereign Cognitive Core - Challenger 1 Adversarial Test Suite
Module: tests.test_adversarial_nrem_watchdog
Author: Empirical Challenger 1

Adversarial Verification for memory.nrem_watchdog:
1. I/O hang simulation (2.5s stall vs 2.0s timeout):
   - Measure abort timing (nominal: 2000 ms +- 80 ms, range [1920, 2080] ms)
   - Verify emergency rollback callback invocation
   - Verify fault code is IO_TIMEOUT_STALL
   - Verify forced arousal: SleepHysteresis -> WAKE/AWAKE, BasalGangliaGate sleep_lock -> False
2. Empirical Deadlock Detection (Consolidation + Watchdog):
   - Demonstrates lock inversion between HippocampusNREM._lock and NREMWatchdog._lock
   - Evaluates thread safety during slow I/O
3. Clean disarm and lifecycle idempotence.
"""

from pathlib import Path
import sys
import threading
import time
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from core.basal_ganglia_gate import BasalGangliaGate
from memory.hippocampus_nrem import CommitState, HippocampusNREM
from memory.nrem_watchdog import (
    DEFAULT_WATCHDOG_TIMEOUT_SEC,
    FAULT_CODE_IO_TIMEOUT_STALL,
    MAX_REALTIME_LATENCY_MARGIN_SEC,
    NREMWatchdog,
    TimeoutFault,
    WatchdogState,
)
from memory.sleep_hysteresis import SleepHysteresis, SleepState


class TestAdversarialNREMWatchdog:
    """Empirical adversarial stress harness for NREM Watchdog."""

    def test_watchdog_io_stall_2s_timing_and_arousal(self):
        """
        Verify that an I/O hang of 2.5s is intercepted at 2.0s (+-80ms margin),
        triggers emergency rollback, logs IO_TIMEOUT_STALL, and forces system arousal.
        """
        timeout_s = 2.0
        stall_duration_s = 2.5
        margin_ms = MAX_REALTIME_LATENCY_MARGIN_SEC * 1000.0  # 80.0 ms

        # Mock / Real sub-components
        hysteresis = SleepHysteresis(initial_state=SleepState.ASLEEP)
        # Precondition: verify system is asleep
        assert hysteresis.current_state == SleepState.ASLEEP

        gate = BasalGangliaGate()
        setattr(gate, "sleep_lock", True)
        assert getattr(gate, "sleep_lock") is True

        rollback_called = []
        callback_time_ns = []

        def on_rollback():
            callback_time_ns.append(time.monotonic_ns())
            rollback_called.append(True)

        abort_fault_received = []

        def on_abort(fault: TimeoutFault):
            abort_fault_received.append(fault)

        watchdog = NREMWatchdog(
            timeout_s=timeout_s,
            on_timeout_abort=on_abort,
            auto_arousal=True,
        )

        t_start_test = time.monotonic_ns()
        fault = watchdog.simulate_io_stall(
            stall_duration_s=stall_duration_s,
            transaction_id="dvp07_adversarial_stall_tx",
            basal_ganglia_gate=gate,
            hysteresis=hysteresis,
            rollback_callback=on_rollback,
        )
        t_end_test = time.monotonic_ns()

        # EMPIRICAL ASSERTIONS:
        # 1. Fault Record Integrity
        assert fault is not None, "Watchdog failed to emit a TimeoutFault on 2.5s stall!"
        assert fault.fault_code == FAULT_CODE_IO_TIMEOUT_STALL, (
            f"Expected fault code '{FAULT_CODE_IO_TIMEOUT_STALL}', got '{fault.fault_code}'"
        )
        assert fault.aborted_state == "ABORTED"
        assert fault.forced_arousal_success is True

        # 2. Timing Precision (+-80ms margin)
        # The fault's elapsed_duration_ms measures the hardware monotonic clock at the moment the timer thread fired
        measured_elapsed_ms = fault.elapsed_duration_ms
        min_allowed_ms = (timeout_s - MAX_REALTIME_LATENCY_MARGIN_SEC) * 1000.0  # 1920.0 ms
        max_allowed_ms = (timeout_s + MAX_REALTIME_LATENCY_MARGIN_SEC) * 1000.0  # 2080.0 ms

        print(
            f"\n[MEASUREMENT] Watchdog 2.0s Stall Test: "
            f"Measured={measured_elapsed_ms:.2f}ms, "
            f"Allowed Range=[{min_allowed_ms:.1f}, {max_allowed_ms:.1f}]ms, "
            f"Delta={measured_elapsed_ms - 2000.0:+.2f}ms"
        )

        assert min_allowed_ms <= measured_elapsed_ms <= max_allowed_ms, (
            f"Watchdog trigger timing violation! Measured {measured_elapsed_ms:.2f}ms "
            f"outside [{min_allowed_ms:.1f}, {max_allowed_ms:.1f}]ms (+-80ms margin)"
        )

        # 3. Emergency Rollback Callback Execution
        assert len(rollback_called) == 1, "Rollback callback was not invoked exactly once!"
        assert rollback_called[0] is True

        # Verify rollback callback timing occurred around 2.0s, NOT after the 2.5s stall
        cb_elapsed_ms = (callback_time_ns[0] - t_start_test) / 1e6
        assert min_allowed_ms <= cb_elapsed_ms <= max_allowed_ms + 20.0, (
            f"Rollback callback timing {cb_elapsed_ms:.2f}ms outside expected range"
        )

        # 4. Forced Arousal Interlock Verification
        # SleepHysteresis must be forced to WAKE / AWAKE
        assert hysteresis.current_state in (SleepState.AWAKE, SleepState.WAKE), (
            f"SleepHysteresis arousal failed! State is {hysteresis.current_state}"
        )
        assert hysteresis.is_awake() is True

        # BasalGangliaGate sleep_lock must be cleared to False
        assert getattr(gate, "sleep_lock") is False, "BasalGangliaGate sleep_lock was not cleared to False!"

        # 5. External on_timeout_abort callback verification
        assert len(abort_fault_received) == 1
        assert abort_fault_received[0].transaction_id == "dvp07_adversarial_stall_tx"

        watchdog.reset()

    def test_watchdog_clean_disarm_no_fault(self):
        """
        Verify that a normal fast transaction disarms cleanly without triggering abort or arousal.
        """
        hysteresis = SleepHysteresis(initial_state=SleepState.ASLEEP)
        gate = BasalGangliaGate()
        setattr(gate, "sleep_lock", True)

        rollback_called = []
        watchdog = NREMWatchdog(timeout_s=2.0)

        watchdog.arm(
            transaction_id="fast_tx",
            hysteresis=hysteresis,
            basal_ganglia_gate=gate,
            rollback_callback=lambda: rollback_called.append(True),
        )

        # Simulate fast transaction (20 ms)
        time.sleep(0.02)
        elapsed_ms = watchdog.disarm()

        assert 10.0 <= elapsed_ms <= 100.0
        assert watchdog.state == WatchdogState.DISARMED
        assert watchdog.get_last_fault() is None
        assert len(rollback_called) == 0
        # Arousal must NOT have been triggered
        assert hysteresis.current_state == SleepState.ASLEEP
        assert getattr(gate, "sleep_lock") is True

        watchdog.reset()

    def test_cooperative_timeout_polling(self):
        """
        Verify synchronous cooperative polling check_timeout() triggers abort and arousal.
        """
        hysteresis = SleepHysteresis(initial_state=SleepState.ASLEEP)
        watchdog = NREMWatchdog(timeout_s=0.05)  # 50 ms timeout

        watchdog.arm(transaction_id="coop_tx", hysteresis=hysteresis)

        # Sleep past timeout
        time.sleep(0.07)

        # Synchronous check
        timed_out = watchdog.check_timeout()
        assert timed_out is True
        assert watchdog.state == WatchdogState.ABORTED
        assert hysteresis.current_state == SleepState.AWAKE

        fault = watchdog.get_last_fault()
        assert fault is not None
        assert fault.fault_code == FAULT_CODE_IO_TIMEOUT_STALL

        watchdog.reset()

    def test_deadlock_vulnerability_on_slow_consolidation(self):
        """
        ADVERSARIAL STRESS TEST / VULNERABILITY REPRODUCTION:
        Verify that when Hippocampus consolidation executes while Watchdog fires,
        the current implementation deadlocks due to mutual lock inversion:
          - Consolidation thread: holds HippocampusNREM._lock, waits for NREMWatchdog._lock
          - Watchdog thread: holds NREMWatchdog._lock, waits for HippocampusNREM._lock

        We test this with a timeout monitor to detect deadlock deterministically without freezing the test suite.
        """
        hippocampus = HippocampusNREM(db_path=":memory:")

        for i in range(10):
            hippocampus.record_event(np.ones(64), 0.5, np.zeros(5), np.ones(64))

        # Add a delay function to SQLite to simulate slow disk persistence (30ms per row)
        hippocampus._conn.create_function("sim_slow_write", 1, lambda x: time.sleep(0.03))
        hippocampus._conn.execute(
            "CREATE TRIGGER slow_disk_write AFTER INSERT ON episodes BEGIN SELECT sim_slow_write(1); END;"
        )

        # Arm watchdog with 40ms timeout (will fire during the 2nd row insertion)
        watchdog = NREMWatchdog(timeout_s=0.04)

        consolidation_thread_done = threading.Event()
        error_container = []

        def run_consolidation():
            try:
                hippocampus.consolidate_2pc(watchdog=watchdog)
            except Exception as e:
                error_container.append(e)
            finally:
                consolidation_thread_done.set()

        t_thread = threading.Thread(target=run_consolidation, daemon=True)
        t_thread.start()

        # Wait with a 2.0s deadline
        completed = consolidation_thread_done.wait(timeout=2.0)

        try:
            if not completed:
                # Empirically verify that both locks are held and frozen
                hip_locked = not hippocampus._lock.acquire(blocking=False)
                wd_locked = not watchdog._lock.acquire(blocking=False)
                print(
                    f"\n[DEADLOCK CONFIRMED] Hippocampus consolidate_2pc deadlocked with NREMWatchdog! "
                    f"hippocampus._lock held={hip_locked}, watchdog._lock held={wd_locked}"
                )
                assert False, (
                    "CRITICAL DEADLOCK DETECTED: consolidate_2pc and NREMWatchdog deadlocked "
                    "due to AB-BA lock inversion between HippocampusNREM._lock and NREMWatchdog._lock!"
                )
            assert completed
        finally:
            if completed:
                hippocampus.close()
                watchdog.reset()
            else:
                # Unblock watchdog cancel event if possible
                watchdog._cancel_event.set()


if __name__ == "__main__":
    suite = TestAdversarialNREMWatchdog()
    print("Running test_watchdog_io_stall_2s_timing_and_arousal...")
    suite.test_watchdog_io_stall_2s_timing_and_arousal()
    print("Running test_watchdog_clean_disarm_no_fault...")
    suite.test_watchdog_clean_disarm_no_fault()
    print("Running test_cooperative_timeout_polling...")
    suite.test_cooperative_timeout_polling()
    print("Running test_deadlock_vulnerability_on_slow_consolidation...")
    try:
        suite.test_deadlock_vulnerability_on_slow_consolidation()
    except AssertionError as e:
        print(f"EXPECTED ADVERSARIAL FINDING: {e}")
    print("ALL WATCHDOG TESTS EXECUTED!")
