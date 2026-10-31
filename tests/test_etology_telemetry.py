"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 5
Test Suite: tests/test_etology_telemetry.py
Author: Stage-Gate 5 AI Nursery Team

Verifications for EthologyTelemetryRecorder & Observability Subsystem:
1. Multithreaded Asynchronous Sample Recording.
2. Dual-Substrate Persistence: SQLite WAL table + JSONL stream.
3. Cryptographic SHA-256 Audit Seal Integrity.
4. Statistical Stability Evaluation (DVP-18): Kuramoto r(t), phase CV, deadlocks = 0.
5. Displacement / Vacuum Sublimation (Kuka-teszt) frequency tracking.
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest

from nursery.etology_telemetry import (
    EthologySample,
    EthologyStabilitySummary,
    EthologyTelemetryRecorder,
)


# ==============================================================================
# 1. RECORDER LIFECYCLE & DUAL PERSISTENCE
# ==============================================================================

def test_telemetry_recorder_persistence(tmp_path) -> None:
    """Verifies that samples are flushed into both SQLite WAL and JSONL files."""
    recorder = EthologyTelemetryRecorder(output_dir=tmp_path)
    recorder.start()

    # Record 10 samples
    for i in range(10):
        theta = 15.0 + float(i) * 0.2
        r = 0.85 - float(i) * 0.02
        v = [0.10 + i * 0.01, 0.12, 0.20, 0.05, 0.15]
        recorder.record_sample(
            theta_deg=theta,
            kuramoto_r=r,
            deficit_vector=v,
            state_label="AWAKE" if i < 8 else "NREM_CONSOLIDATING",
            pedagogical_stage=1,
            surrogate_action_active=(i == 5),
            dominant_need_idx=2,
            cognitive_deadlock=False,
        )

    # Allow writer thread to flush
    recorder.stop(timeout=2.0)

    # 1. Verify SQLite database
    db_file = tmp_path / "etology_longitudinal.db"
    assert db_file.exists()

    with sqlite3.connect(str(db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ethology_samples;")
        count = cursor.fetchone()[0]
        assert count == 10

        cursor.execute("SELECT theta_deg, kuramoto_r, surrogate_action_active FROM ethology_samples WHERE step_index = 6;")
        row = cursor.fetchone()
        assert row is not None
        assert pytest.approx(row[0], rel=1e-3) == 16.0
        assert row[2] == 1  # surrogate active on i=5 (step 6)

    # 2. Verify JSONL stream
    jsonl_file = tmp_path / "etology_stream.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 10

    # 3. Verify SHA-256 seal
    seal = recorder.compute_sha256_audit_seal()
    assert len(seal) == 64
    assert seal != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ==============================================================================
# 2. STATISTICAL STABILITY SYNTHESIS (DVP-18)
# ==============================================================================

def test_stability_summary_evaluation(tmp_path) -> None:
    """Verifies quantitative stability metrics for long-horizon autonomous runs."""
    recorder = EthologyTelemetryRecorder(output_dir=tmp_path)
    recorder.start()

    # Generate 50 realistic periodic cognitive samples
    base_time_ns = 1700000000000000000
    for step in range(50):
        t_s = step * 0.1
        theta = 15.0 + 2.5 * np.sin(t_s * 2.0 * np.pi * 0.5)
        r = 0.70 + 0.15 * np.cos(t_s * 2.0 * np.pi * 0.5)
        v = [0.15, 0.10, 0.25, 0.10, 0.30]

        is_sublimation = (theta > 17.0)
        is_sleep = (step >= 40)

        recorder.record_sample(
            theta_deg=float(theta),
            kuramoto_r=float(r),
            deficit_vector=v,
            state_label="NREM_CONSOLIDATING" if is_sleep else "AWAKE",
            pedagogical_stage=2,
            surrogate_action_active=is_sublimation,
            cognitive_deadlock=False,
            timestamp_ns=base_time_ns + int(t_s * 1e9),
        )

    recorder.stop()

    summary = recorder.compute_stability_summary()
    assert summary.total_samples == 50
    assert summary.duration_seconds > 0.0
    assert 0.50 <= summary.mean_kuramoto_r <= 1.0
    assert summary.min_kuramoto_r >= 0.20
    assert summary.phase_amplitude_cv < 0.60
    assert summary.cognitive_deadlocks_count == 0
    assert summary.sleep_cycles_count == 10
    assert summary.hypothesis_stable is True
