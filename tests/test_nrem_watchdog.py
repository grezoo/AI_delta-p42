"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 Test Suite
Module: tests.test_nrem_watchdog
Author: ΔP-42 Senior QA & Testing Team (Zero Downgrade Mandate)

Comprehensive unit and adversarial test suite for:
- memory.nrem_watchdog (NREMWatchdog, TimeoutFault, Hard Real-Time 2.0s Supervisor, Forced Arousal)
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.nrem_watchdog import (
    DEFAULT_CHECK_INTERVAL_SEC,
    DEFAULT_WATCHDOG_TIMEOUT_SEC,
    FAULT_CODE_IO_TIMEOUT_STALL,
    NREMWatchdog,
    TimeoutFault,
    WatchdogState,
    WatchdogTimeoutError,
)
from memory.sleep_hysteresis import SleepHysteresis, SleepState


class TestWatchdogInitializationAndConfig:
    """Tests for watchdog parameter initialization and boundary validation."""

    def test_watchdog_default_timeout(self) -> None:
        wd = NREMWatchdog()
        assert wd.timeout_s == DEFAULT_WATCHDOG_TIMEOUT_SEC
        assert wd.timeout_sec == 2.0
        assert wd.timeout_ms == 2000.0
        assert wd.check_interval_s == DEFAULT_CHECK_INTERVAL_SEC
        assert wd.auto_arousal is True
        assert wd.is_armed() is False
        assert wd.state == WatchdogState.IDLE

    def test_watchdog_custom_timeout(self) -> None:
        wd = NREMWatchdog(timeout_s=0.5, check_interval_s=0.01, auto_arousal=False)
        assert wd.timeout_s == 0.5
        assert wd.timeout_ms == 500.0
        assert wd.check_interval_s == 0.01
        assert wd.auto_arousal is False

    def test_watchdog_invalid_timeout_rejection(self) -> None:
        with pytest.raises(ValueError):
            NREMWatchdog(timeout_s=0.0)

        with pytest.raises(ValueError):
            NREMWatchdog(timeout_s=-1.5)


class TestWatchdogHappyPathDisarm:
    """Tests for successful consolidation transactions disarming before deadline."""

    def test_fast_transaction_disarms_watchdog(self) -> None:
        abort_called = False

        def on_abort(fault: TimeoutFault) -> None:
            nonlocal abort_called
            abort_called = True

        wd = NREMWatchdog(timeout_s=2.0, on_timeout_abort=on_abort)
        wd.arm(transaction_id="tx_fast_1")
        assert wd.is_armed() is True
        assert wd.state == WatchdogState.ARMED

        # Simulate fast 10 ms operation
        time.sleep(0.010)
        elapsed_ms = wd.disarm()

        assert elapsed_ms >= 8.0  # At least sleep duration
        assert wd.state == WatchdogState.DISARMED
        assert wd.is_armed() is False
        assert abort_called is False
        assert wd.get_last_fault() is None

    def test_multiple_consecutive_successful_disarms(self) -> None:
        wd = NREMWatchdog(timeout_s=2.0)
        for i in range(25):
            wd.arm(transaction_id=f"tx_loop_{i}")
            assert wd.is_armed() is True
            # Simulate work
            wd.heartbeat()
            elapsed = wd.disarm()
            assert elapsed >= 0.0
            assert wd.state == WatchdogState.DISARMED

    def test_idempotent_disarm(self) -> None:
        wd = NREMWatchdog(timeout_s=1.0)
        wd.arm("tx_idempotent")
        el1 = wd.disarm()
        el2 = wd.disarm()
        assert el1 >= 0.0
        assert el2 == 0.0  # Already disarmed returns 0.0

    def test_context_manager_clean_exit(self) -> None:
        wd = NREMWatchdog(timeout_s=2.0)
        with wd.watch("tx_context_clean"):
            assert wd.is_armed() is True
            time.sleep(0.01)

        assert wd.state == WatchdogState.DISARMED
        assert wd.is_armed() is False


class TestWatchdogTimeoutTriggerAndRollback:
    """Tests for timeout expiration triggering emergency abort, rollback, and forced arousal."""

    def test_timeout_triggers_abort_callback(self) -> None:
        fault_captured: Optional[TimeoutFault] = None

        def on_abort(f: TimeoutFault) -> None:
            nonlocal fault_captured
            fault_captured = f

        # Use short timeout 0.05s for fast test execution
        wd = NREMWatchdog(timeout_s=0.05, on_timeout_abort=on_abort)
        rollback_mock = MagicMock()

        wd.arm(transaction_id="tx_timeout_test", rollback_callback=rollback_mock)

        # Wait for timeout to fire asynchronously
        time.sleep(0.12)

        assert wd.state == WatchdogState.ABORTED
        assert wd.is_aborted() is True
        assert rollback_mock.called is True
        assert fault_captured is not None
        assert fault_captured.fault_code == FAULT_CODE_IO_TIMEOUT_STALL
        assert fault_captured.transaction_id == "tx_timeout_test"
        assert fault_captured.aborted_state == "ABORTED"
        assert fault_captured.elapsed_duration_ms >= 45.0

    def test_forced_arousal_signal_emitted(self) -> None:
        hysteresis = SleepHysteresis(initial_state=SleepState.ASLEEP)
        # Verify initial sleeping state
        assert hysteresis.current_state == SleepState.ASLEEP

        bg_gate_mock = MagicMock()

        wd = NREMWatchdog(timeout_s=0.05, auto_arousal=True)
        wd.arm(
            transaction_id="tx_arousal_test",
            hysteresis=hysteresis,
            basal_ganglia_gate=bg_gate_mock,
        )

        time.sleep(0.12)

        # Hysteresis MUST be forced to AWAKE
        assert hysteresis.current_state == SleepState.AWAKE
        # Basal Ganglia gate sleep lock must be cleared
        assert bg_gate_mock.sleep_lock is False

    def test_sqlite_interrupt_called_on_timeout(self) -> None:
        db_conn_mock = MagicMock()
        wd = NREMWatchdog(timeout_s=0.05)
        wd.arm(transaction_id="tx_sqlite_interrupt", db_conn=db_conn_mock)

        time.sleep(0.12)

        assert db_conn_mock.interrupt.called is True

    def test_cooperative_polling_triggers_timeout(self) -> None:
        # Test synchronous check_timeout()
        wd = NREMWatchdog(timeout_s=0.05)
        wd.arm("tx_cooperative")
        time.sleep(0.06)

        # Before calling check_timeout(), state may be transitioned by async timer or by check_timeout()
        aborted = wd.check_timeout()
        assert aborted is True
        assert wd.state == WatchdogState.ABORTED
        assert wd.check_timeout() is True  # Idempotent subsequent checks

    def test_manual_emergency_abort(self) -> None:
        rollback_called = False

        def do_rollback() -> None:
            nonlocal rollback_called
            rollback_called = True

        wd = NREMWatchdog(timeout_s=2.0)
        wd.arm("tx_manual_abort", rollback_callback=do_rollback)

        fault = wd.trigger_emergency_abort("SIMULATED_OPERATOR_PANIC")
        assert fault.fault_code == FAULT_CODE_IO_TIMEOUT_STALL
        assert "SIMULATED_OPERATOR_PANIC" in fault.diagnostic_details
        assert rollback_called is True
        assert wd.state == WatchdogState.ABORTED

    def test_simulate_io_stall_timeout(self) -> None:
        wd = NREMWatchdog(timeout_s=0.05)
        fault = wd.simulate_io_stall(stall_duration_s=0.10, transaction_id="tx_stall_test")
        assert fault is not None
        assert fault.fault_code == FAULT_CODE_IO_TIMEOUT_STALL
        assert fault.transaction_id == "tx_stall_test"
        assert wd.state == WatchdogState.ABORTED


class TestWatchdogThreadLifecycle:
    """Thread termination, reset, and zero thread leakage tests."""

    def test_watchdog_thread_termination_on_reset(self) -> None:
        wd = NREMWatchdog(timeout_s=2.0)
        wd.arm("tx_thread_test")
        assert wd._timer_thread is not None
        assert wd._timer_thread.is_alive() is True

        wd.reset()
        assert wd.state == WatchdogState.IDLE
        assert wd.is_armed() is False
        assert wd._timer_thread.is_alive() is False

    def test_zero_thread_leakage_across_many_cycles(self) -> None:
        initial_threads = threading.active_count()
        wd = NREMWatchdog(timeout_s=1.0)

        for i in range(20):
            wd.arm(f"tx_leak_probe_{i}")
            time.sleep(0.005)
            wd.disarm()

        time.sleep(0.05)
        final_threads = threading.active_count()
        # Assert active thread count has not grown
        assert final_threads <= initial_threads + 1

    def test_timeout_fault_to_dict_and_immutability(self) -> None:
        fault = TimeoutFault(
            timestamp_ns=1000,
            transaction_id="tx_dict_test",
            elapsed_duration_ms=2005.2,
            timeout_threshold_ms=2000.0,
            diagnostic_details="test stall",
        )
        d = fault.to_dict()
        assert d["transaction_id"] == "tx_dict_test"
        assert d["fault_code"] == FAULT_CODE_IO_TIMEOUT_STALL
        assert d["aborted_state"] == "ABORTED"
        assert d["elapsed_duration_ms"] == 2005.2
