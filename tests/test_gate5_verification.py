"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 5
Test Suite: tests/test_gate5_verification.py
Author: Stage-Gate 5 AI Nursery Team

Pytest wrappers verifying all Stage-Gate 5 DVP protocols:
- DVP-17: GGUF Cortical Chorus VRAM & Latency Compliance.
- DVP-18: Long-Horizon Autonomous Ethological Stability.
- DVP-19: Multimodal Sensory Ingestion & CDC Ring Buffer Courier.
- DVP-20: Caretaker Closed-Loop Pedagogical Progression & SHA-256 Audit Seal.
- Full verification suite orchestrator (run_gate5_verification).
"""

from __future__ import annotations

import pytest

from core.gate5_verification import (
    run_gate5_verification,
    verify_dvp17,
    verify_dvp18,
    verify_dvp19,
    verify_dvp20,
)


def test_dvp17_verification() -> None:
    res = verify_dvp17(quick=True)
    assert res.passed is True
    assert res.measured_metrics["vram_used_mb"] <= 2900.0
    assert res.measured_metrics["num_subsystems"] == 6


def test_dvp18_verification(tmp_path) -> None:
    res = verify_dvp18(quick=True, scratch_dir=tmp_path)
    assert res.passed is True
    assert res.measured_metrics["cognitive_deadlocks_count"] == 0
    assert res.measured_metrics["min_kuramoto_r"] >= 0.15
    assert res.measured_metrics["phase_amplitude_cv"] <= 0.60


def test_dvp19_verification() -> None:
    res = verify_dvp19(quick=True)
    assert res.passed is True
    assert res.measured_metrics["packet_drop_rate"] == 0.000
    assert res.measured_metrics["crc_errors"] == 0


def test_dvp20_verification(tmp_path) -> None:
    res = verify_dvp20(quick=True, scratch_dir=tmp_path)
    assert res.passed is True
    assert res.measured_metrics["threat_override_active"] is True
    assert res.measured_metrics["threat_etiquette_deg"] == 0.0
    assert len(res.measured_metrics["sha256_audit_seal"]) == 64


def test_gate5_full_verification_orchestration() -> None:
    report = run_gate5_verification(quick=True)
    assert report.all_passed is True
    assert len(report.tests) == 4
    for t in report.tests:
        assert t.passed is True
