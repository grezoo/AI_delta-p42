"""
ΔP-42 Sovereign Cognitive Core (Stage-Gate 3) - Test Suite
Module: tests.test_gate3_verification
Author: ΔP-42 Senior QA & Testing Team (Zero Downgrade Mandate)

Comprehensive unit and edge-case test suite for:
- core.gate3_verification (DVP-06, DVP-07, DVP-08, DVP-11, Master Runner & Standalone CLI)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import io
import json
from pathlib import Path
import sys
import unittest.mock as mock

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.gate3_verification import (
    DVP06Result,
    DVP07Result,
    DVP08Result,
    DVP11Result,
    Gate3Results,
    Gate3Verifier,
    main,
)


class TestGate3VerifierStructureAndContracts:
    """Tests for result dataclass contracts, frozen immutability, and report formatting."""

    def test_dvp06_result_immutability(self) -> None:
        res = DVP06Result(
            passed=True,
            initial_episodes=2000,
            persisted_episodes=2000,
            initial_saturation=1.00,
            final_saturation=0.00,
            data_loss_pct=0.00,
        )
        assert res.passed is True
        assert res.initial_episodes == 2000
        assert res.final_saturation == 0.00
        with pytest.raises(FrozenInstanceError):
            res.passed = False  # type: ignore

    def test_dvp07_result_immutability(self) -> None:
        res = DVP07Result(
            passed=True,
            timeout_configured_s=2.0,
            measured_abort_time_s=2.008,
            timing_window_valid=True,
            fault_code="IO_TIMEOUT_STALL",
            fault_code_verified=True,
            rollback_invoked=True,
            forced_arousal_verified=True,
        )
        assert res.passed is True
        assert res.timing_window_valid is True
        with pytest.raises(FrozenInstanceError):
            res.measured_abort_time_s = 5.0  # type: ignore

    def test_dvp08_result_immutability(self) -> None:
        res = DVP08Result(
            passed=True,
            total_cycles=10000,
            chattering_count=0,
            latching_fidelity_pct=100.0,
        )
        assert res.passed is True
        assert res.chattering_count == 0
        with pytest.raises(FrozenInstanceError):
            res.chattering_count = 1  # type: ignore

    def test_dvp11_result_immutability(self) -> None:
        res = DVP11Result(
            passed=True,
            total_episodes_probed=500,
            min_probe_cosine=0.999,
            mean_probe_cosine=1.000,
            corruption_rejection_rate_pct=100.0,
            motor_atonia_verified=True,
            motor_leakage_count=0,
        )
        assert res.passed is True
        assert res.motor_leakage_count == 0
        with pytest.raises(FrozenInstanceError):
            res.corruption_rejection_rate_pct = 50.0  # type: ignore

    def test_gate3_results_aggregation(self) -> None:
        d06 = DVP06Result(True, 100, 100, 1.0, 0.0, 0.0)
        d07 = DVP07Result(True, 2.0, 2.1, 2.005, 2005.0, True, "IO_TIMEOUT_STALL", True, True, True, True, True)
        d08 = DVP08Result(True, 10000, 0, 5000, 2, 100.0, 0.85, 0.40, 0.45, True)
        d11 = DVP11Result(True, 500, 1.0, 0.990, 1.0, 0.999, 20, 0, 100.0, True, True, 0, True)

        all_ok = Gate3Results(dvp06=d06, dvp07=d07, dvp08=d08, dvp11=d11, all_passed=True)
        assert all_ok.all_passed is True

        # If any single DVP fails, all_passed should be False
        d11_fail = DVP11Result(False, 500, 0.85, 0.990, 0.90, 0.999, 20, 5, 75.0, True, True, 0, False)
        some_fail = Gate3Results(
            dvp06=d06,
            dvp07=d07,
            dvp08=d08,
            dvp11=d11_fail,
            all_passed=bool(d06.passed and d07.passed and d08.passed and d11_fail.passed),
        )
        assert some_fail.all_passed is False

    def test_gate3_results_summary_immutability(self) -> None:
        d06 = DVP06Result(True, 100, 100, 1.0, 0.0, 0.0)
        d07 = DVP07Result(True, 2.0, 2.1, 2.005, 2005.0, True, "IO_TIMEOUT_STALL", True, True, True, True, True)
        d08 = DVP08Result(True, 10000, 0, 5000, 2, 100.0, 0.85, 0.40, 0.45, True)
        d11 = DVP11Result(True, 500, 1.0, 0.990, 1.0, 0.999, 20, 0, 100.0, True, True, 0, True)

        g3 = Gate3Results(dvp06=d06, dvp07=d07, dvp08=d08, dvp11=d11, all_passed=True, summary={"all_passed": True, "dvp06": {"passed": True}})

        # Direct mutation must raise TypeError
        with pytest.raises(TypeError):
            g3.summary["all_passed"] = False  # type: ignore

        # Nested mutation must raise TypeError
        with pytest.raises(TypeError):
            g3.summary["dvp06"]["passed"] = False  # type: ignore

        # Hash must succeed
        assert isinstance(hash(g3), int)

        # to_dict produces mutable copy
        mutable_dict = g3.to_dict()
        mutable_dict["all_passed"] = False
        assert g3.summary["all_passed"] is True

    def test_format_report_ascii_borders_and_content(self) -> None:
        verifier = Gate3Verifier(verbose=False)
        d06 = DVP06Result(True, 2000, 2000, 1.0, 0.0, 0.0, 0.0, 800000, 150.0, 1.0)
        d07 = DVP07Result(True, 2.0, 2.1, 2.005, 2005.0, True, "IO_TIMEOUT_STALL", True, True, True, True, True)
        d08 = DVP08Result(True, 10000, 0, 5000, 2, 100.0, 0.85, 0.40, 0.45, True)
        d11 = DVP11Result(True, 500, 1.0, 0.990, 1.0, 0.999, 20, 0, 100.0, True, True, 0, True)

        results = Gate3Results(dvp06=d06, dvp07=d07, dvp08=d08, dvp11=d11, all_passed=True)
        report = verifier.format_report(results)

        assert "STAGE-GATE 3: QUANTITATIVE DVP VERIFICATION REPORT" in report
        assert "DVP-06: NREM Memory Purge Stress Test" in report
        assert "DVP-07: Hard Real-Time 2.0s Watchdog Abort" in report
        assert "DVP-08: Chattering-Free Sleep Hysteresis" in report
        assert "DVP-11: Transactional & Knowledge Graph Integrity Probe" in report
        assert "GATE 3 APPROVED - PASS" in report
        assert "[FAIL]" not in report
        assert report.startswith("+--------------------------------------------------------------------------------+")
        assert report.endswith("+--------------------------------------------------------------------------------+")


class TestGate3IndividualDVPRunners:
    """Tests for each quantitative verification method in quick/accelerated mode."""

    def test_verify_dvp06_memory_purge_quick(self) -> None:
        verifier = Gate3Verifier(verbose=False, seed=101)
        res = verifier.verify_dvp06_memory_purge(n_episodes=50, quick=True)
        assert res.passed is True
        assert res.initial_episodes == 50
        assert res.persisted_episodes == 50
        assert res.initial_saturation >= 0.85
        assert res.final_saturation == 0.00
        assert res.data_loss_pct == 0.00
        assert res.min_probe_cosine >= 0.990
        assert res.purged_bytes > 0

    def test_verify_dvp07_watchdog_timeout_quick(self) -> None:
        verifier = Gate3Verifier(verbose=False, seed=102)
        res = verifier.verify_dvp07_watchdog_timeout(
            timeout_s=0.080,
            stall_duration_s=0.160,
            timing_margin_s=0.080,
            quick=True,
        )
        assert res.passed is True
        assert res.timing_window_valid is True
        assert res.fault_code == "IO_TIMEOUT_STALL"
        assert res.fault_code_verified is True
        assert res.rollback_invoked is True
        assert res.forced_arousal_verified is True
        assert res.zero_amnesia_verified is True

    def test_verify_dvp08_sleep_hysteresis_quick(self) -> None:
        verifier = Gate3Verifier(verbose=False, seed=103)
        res = verifier.verify_dvp08_sleep_hysteresis(cycles=500, quick=True)
        assert res.passed is True
        assert res.chattering_count == 0
        assert res.latching_fidelity_pct == 100.0
        assert res.boundary_epsilon_verified is True

    def test_verify_dvp11_integrity_probe_quick(self) -> None:
        verifier = Gate3Verifier(verbose=False, seed=104)
        res = verifier.verify_dvp11_integrity_probe(
            n_episodes=20,
            corrupt_count=4,
            quick=True,
        )
        assert res.passed is True
        assert res.min_probe_cosine >= 0.990
        assert res.mean_probe_cosine >= 0.999
        assert res.corruption_rejection_rate_pct == 100.0
        assert res.corrupted_episodes_persisted == 0
        assert res.rollback_buffer_preserved is True
        assert res.motor_atonia_verified is True
        assert res.motor_leakage_count == 0
        assert res.dissonance_resolution_verified is True


class TestGate3EdgeCasesAndRobustness:
    """Tests for edge cases, custom directories, seeding, and partial failures."""

    def test_verifier_deterministic_seeding(self) -> None:
        v1 = Gate3Verifier(verbose=False, seed=777)
        v2 = Gate3Verifier(verbose=False, seed=777)

        res1 = v1.verify_dvp08_sleep_hysteresis(cycles=300, quick=True)
        res2 = v2.verify_dvp08_sleep_hysteresis(cycles=300, quick=True)

        assert res1.total_cycles == res2.total_cycles
        assert res1.deadband_evaluations == res2.deadband_evaluations
        assert res1.total_transitions == res2.total_transitions

    def test_custom_temp_db_directory(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom_gate3_scratch"
        custom_dir.mkdir(parents=True, exist_ok=True)

        verifier = Gate3Verifier(verbose=False, seed=55, db_dir=custom_dir)
        res = verifier.verify_dvp06_memory_purge(n_episodes=20, quick=True)
        assert res.passed is True
        # Verify temporary directories are cleaned up
        remaining_files = list(custom_dir.glob("*.db"))
        assert len(remaining_files) == 0

    def test_partial_failure_detection(self) -> None:
        verifier = Gate3Verifier(verbose=False)
        d06 = DVP06Result(False, 2000, 1900, 1.0, 0.05, 5.0, 0.0, 750000, 150.0, 0.95)
        d07 = DVP07Result(True, 2.0, 2.1, 2.005, 2005.0, True, "IO_TIMEOUT_STALL", True, True, True, True, True)
        d08 = DVP08Result(True, 10000, 0, 5000, 2, 100.0, 0.85, 0.40, 0.45, True)
        d11 = DVP11Result(True, 500, 1.0, 0.990, 1.0, 0.999, 20, 0, 100.0, True, True, 0, True)

        results = Gate3Results(dvp06=d06, dvp07=d07, dvp08=d08, dvp11=d11, all_passed=False)
        report = verifier.format_report(results)

        assert "[FAIL]" in report
        assert "GATE 3 REJECTED - FAIL" in report

    def test_run_all_with_custom_overrides(self) -> None:
        verifier = Gate3Verifier(verbose=False, seed=42)
        res = verifier.run_all(
            quick=True,
            dvp06_episodes=25,
            dvp08_cycles=200,
            dvp11_probes=10,
        )
        assert res.all_passed is True
        assert res.dvp06.initial_episodes == 25
        assert res.dvp08.total_cycles == 200
        assert res.dvp11.total_episodes_probed == 10


class TestGate3CLIAndExitCodes:
    """Tests for standalone CLI arguments, JSON emission, and exit codes."""

    def test_cli_quick_mode_exit_0(self) -> None:
        ret = main(["--quick", "--quiet"])
        assert ret == 0

    def test_cli_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["--quick", "--quiet", "--json"])
        assert ret == 0

        captured = capsys.readouterr()
        # Find JSON payload in stdout
        json_str = captured.out.strip()
        data = json.loads(json_str)

        assert data["all_passed"] is True
        assert "dvp06" in data
        assert "dvp07" in data
        assert "dvp08" in data
        assert "dvp11" in data
        assert data["dvp06"]["passed"] is True
        assert data["dvp07"]["passed"] is True
        assert data["dvp08"]["passed"] is True
        assert data["dvp11"]["passed"] is True

    def test_cli_failure_exit_1(self) -> None:
        failing_d06 = DVP06Result(False, 100, 80, 1.0, 0.2, 20.0, 0.0, 0, 0.0, 0.8)
        with mock.patch.object(Gate3Verifier, "verify_dvp06_memory_purge", return_value=failing_d06):
            ret = main(["--quick", "--quiet"])
            assert ret == 1

    def test_cli_help_option(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "DELTA P-42 Sovereign Cognitive Core" in captured.out
