"""
Unit and Acceptance Tests for Stage-Gate 2 Verification Engine (core/gate2_verification.py).
Verifies:
1. Gate2Verifier initialization and parameter configuration.
2. DVP-03 Amygdala emergency brake and Pavlovian conditioning verification runner.
3. DVP-13 PFC Airplane Principle goal selection verification runner.
4. DVP-14 Cerebellum timing jitter and Purkinje LTD plasticity verification runner.
5. DVP-15 Sensorimotor proprioception latency and analytical Jacobian verification runner.
6. DVP-16 CDC Ring Buffer burst test verification runner.
7. Gate2Results aggregation and report formatting.
8. CLI main() entrypoint returns exit code 0.
9. Immutability of all quantitative result dataclasses.
"""

from __future__ import annotations

import math
from typing import Dict
import numpy as np
import pytest

from core.gate2_verification import (
    DVP03Result,
    DVP13Result,
    DVP14Result,
    DVP15Result,
    DVP16Result,
    Gate2Results,
    Gate2Verifier,
    main,
)


class TestGate2VerifierStructure:
    """Verifies class initialization, dataclasses, and report generation."""

    def test_verifier_initialization(self) -> None:
        v = Gate2Verifier(verbose=False, seed=123)
        assert v.verbose is False
        assert v.seed == 123

    def test_dataclasses_frozen_immutability(self) -> None:
        res03 = DVP03Result(passed=True, wcet_ms=0.15)
        with pytest.raises(Exception):
            res03.passed = False  # type: ignore

        res13 = DVP13Result(passed=True, mismatch_count=0)
        with pytest.raises(Exception):
            res13.mismatch_count = 5  # type: ignore

        res14 = DVP14Result(passed=True, error_reduction_pct=90.0)
        with pytest.raises(Exception):
            res14.error_reduction_pct = 50.0  # type: ignore

        res15 = DVP15Result(passed=True, wcet_latency_ms=0.5)
        with pytest.raises(Exception):
            res15.wcet_latency_ms = 4.0  # type: ignore

        res16 = DVP16Result(passed=True, total_packets=1000)
        with pytest.raises(Exception):
            res16.total_packets = 0  # type: ignore

    def test_format_report_rendering(self) -> None:
        v = Gate2Verifier(verbose=False, seed=42)
        res03 = DVP03Result(passed=True, wcet_ms=0.12, post_conditioning_firing=True)
        res13 = DVP13Result(passed=True, mismatch_count=0, mismatch_rate=0.0)
        res14 = DVP14Result(passed=True, timing_jitter_max_ms=0.35, error_reduction_pct=91.5)
        res15 = DVP15Result(passed=True, wcet_latency_ms=0.45, jacobian_fd_max_error=1e-9)
        res16 = DVP16Result(passed=True, total_packets=1000, packets_received=1000, throughput_packets_per_sec=50000.0)

        results = Gate2Results(
            dvp03=res03,
            dvp13=res13,
            dvp14=res14,
            dvp15=res15,
            dvp16=res16,
            all_passed=True,
        )

        report = v.format_report(results)
        assert "DELTA P-42 STAGE-GATE 2: QUANTITATIVE DVP VERIFICATION REPORT" in report
        assert "DVP-03" in report
        assert "DVP-13" in report
        assert "DVP-14" in report
        assert "DVP-15" in report
        assert "DVP-16" in report
        assert "GATE 2 APPROVED - PASS" in report


class TestGate2IndividualDVPRunners:
    """Verifies individual DVP quantitative runner methods with representative workloads."""

    def test_dvp03_amygdala_runner(self) -> None:
        v = Gate2Verifier(verbose=False, seed=42)
        # Execute DVP-03 over 200 trials for fast test execution
        res = v.verify_dvp03_amygdala(total_trials=200)

        assert isinstance(res, DVP03Result)
        assert res.passed is True
        assert res.wcet_ms <= res.wcet_threshold_ms
        assert res.wcet_ms <= 2.0
        assert res.unconditioned_kill_passed is True
        assert res.pre_conditioning_firing is False
        assert res.post_conditioning_firing is True
        assert res.pairings_completed == 3
        assert "DIRECT_TRAUMA" in res.alarm_sources_detected
        assert "CONDITIONED_REFLEX" in res.alarm_sources_detected

    def test_dvp13_pfc_goal_selection_runner(self) -> None:
        v = Gate2Verifier(verbose=False, seed=42)
        # Execute DVP-13 over 200 evaluations for fast test execution
        res = v.verify_dvp13_pfc_goal_selection(total_evaluations=200)

        assert isinstance(res, DVP13Result)
        assert res.passed is True
        assert res.mismatch_count == 0
        assert res.mismatch_rate < res.mismatch_threshold
        assert res.zero_deadlock_verified is True
        assert res.zero_floating_verified is True
        assert res.airplane_emergence_passed is True

    def test_dvp14_cerebellum_ltd_runner(self) -> None:
        v = Gate2Verifier(verbose=False, seed=42)
        # Execute DVP-14 over 50 cycles for fast test execution
        res = v.verify_dvp14_cerebellum_ltd(cycles=50)

        assert isinstance(res, DVP14Result)
        assert res.passed is True
        assert res.timing_jitter_max_ms < res.timing_jitter_threshold_ms
        assert res.timing_jitter_max_ms < 0.5
        assert res.error_reduction_pct >= res.error_reduction_threshold_pct
        assert res.error_reduction_pct >= 85.0
        assert res.max_overshoot_pct == 0.0
        assert res.reservoir_sparsity_min >= 0.90
        assert res.interlock_passed is True

    def test_dvp15_sensorimotor_latency_runner(self) -> None:
        v = Gate2Verifier(verbose=False, seed=42)
        # Execute DVP-15 over 200 cycles for fast test execution
        res = v.verify_dvp15_sensorimotor_latency(total_packets=200)

        assert isinstance(res, DVP15Result)
        assert res.passed is True
        assert res.wcet_latency_ms <= res.wcet_threshold_ms
        assert res.wcet_latency_ms <= 3.0
        assert res.corruption_count == 0
        assert res.joint_limit_violations == 0
        assert res.jacobian_fd_max_error < res.jacobian_fd_threshold
        assert res.jacobian_fd_max_error < 1e-6
        assert res.payload_tensor_valid is True
        assert res.failsafe_guard_verified is True

    def test_dvp16_cdc_ring_buffer_runner(self) -> None:
        v = Gate2Verifier(verbose=False, seed=42)
        # Execute DVP-16 over 50,000 packets for fast test execution
        res = v.verify_dvp16_cdc_ring_buffer(total_packets=50000)

        assert isinstance(res, DVP16Result)
        assert res.passed is True
        assert res.packets_received == 50000
        assert res.packet_loss_count == 0
        assert res.crc_error_count == 0
        assert res.uuid_collision_count == 0
        assert res.throughput_packets_per_sec > 1000.0
        assert res.crc_poly_verified is True


class TestGate2MainCLI:
    """Verifies full execution and exit code handling."""

    def test_main_cli_returns_0_on_success(self, monkeypatch) -> None:
        # Patch verify methods with smaller counts to ensure fast test execution
        v = Gate2Verifier(verbose=False)

        def mock_run_all(self):
            res03 = self.verify_dvp03_amygdala(total_trials=100)
            res13 = self.verify_dvp13_pfc_goal_selection(total_evaluations=100)
            res14 = self.verify_dvp14_cerebellum_ltd(cycles=30)
            res15 = self.verify_dvp15_sensorimotor_latency(total_packets=100)
            res16 = self.verify_dvp16_cdc_ring_buffer(total_packets=10000)
            all_passed = bool(res03.passed and res13.passed and res14.passed and res15.passed and res16.passed)
            return Gate2Results(
                dvp03=res03,
                dvp13=res13,
                dvp14=res14,
                dvp15=res15,
                dvp16=res16,
                all_passed=all_passed,
            )

        monkeypatch.setattr(Gate2Verifier, "run_all", mock_run_all)
        exit_code = main()
        assert exit_code == 0
