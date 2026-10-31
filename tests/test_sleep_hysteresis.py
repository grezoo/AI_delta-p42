"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 Test Suite
Module: tests.test_sleep_hysteresis
Author: ΔP-42 Senior QA & Testing Team (Zero Downgrade Mandate)

Comprehensive unit and adversarial test suite for:
- memory.sleep_hysteresis (SleepHysteresis, SleepState, Schmitt Trigger, 10,000-Cycle Noise Stability)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from memory.sleep_hysteresis import (
    C_DEADBAND_WIDTH,
    C_SLEEP_THRESHOLD,
    C_WAKE_THRESHOLD,
    DEFAULT_DEADBAND_WIDTH,
    DEFAULT_SLEEP_THRESHOLD,
    DEFAULT_WAKE_THRESHOLD,
    HysteresisSnapshot,
    SleepHysteresis,
    SleepState,
)


class TestSleepHysteresisThresholds:
    """Tests for canonical Schmitt trigger physical switching thresholds (0.85 / 0.40)."""

    def test_initial_state_awake(self) -> None:
        h = SleepHysteresis()
        assert h.current_state == SleepState.AWAKE
        assert h.is_awake() is True
        assert h.is_sleeping() is False
        assert h.sleep_threshold == DEFAULT_SLEEP_THRESHOLD
        assert h.wake_threshold == DEFAULT_WAKE_THRESHOLD
        assert h.get_transition_count() == 0
        assert h.get_chattering_count() == 0

    def test_custom_initial_state_and_thresholds(self) -> None:
        h = SleepHysteresis(sleep_threshold=0.80, wake_threshold=0.30, initial_state=SleepState.ASLEEP)
        assert h.current_state == SleepState.ASLEEP
        assert h.is_sleeping() is True
        assert h.sleep_threshold == 0.80
        assert h.wake_threshold == 0.30

    def test_sleep_initiation_threshold_85(self) -> None:
        h = SleepHysteresis()
        snap = h.update(0.85)
        assert snap.current_state == SleepState.ASLEEP
        assert snap.state_changed is True
        assert h.is_sleeping() is True
        assert h.get_transition_count() == 1

    def test_sub_sleep_threshold_rejection(self) -> None:
        h = SleepHysteresis()
        # Exactly below 0.85 (e.g. 0.849999) must NOT trigger sleep
        snap = h.update(0.849999)
        assert snap.current_state == SleepState.AWAKE
        assert snap.state_changed is False
        assert h.is_awake() is True
        assert h.get_transition_count() == 0

    def test_wake_threshold_40(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.ASLEEP)
        snap = h.update(0.40)
        assert snap.current_state == SleepState.AWAKE
        assert snap.state_changed is True
        assert h.is_awake() is True
        assert h.get_transition_count() == 1

    def test_sub_wake_threshold_rejection(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.ASLEEP)
        # Exactly above 0.40 (e.g. 0.400001) must NOT trigger wake
        snap = h.update(0.400001)
        assert snap.current_state == SleepState.ASLEEP
        assert snap.state_changed is False
        assert h.is_sleeping() is True
        assert h.get_transition_count() == 0


class TestSleepHysteresisDeadband:
    """Tests for deadband latching dynamics across [0.40, 0.85]."""

    def test_deadband_stability_when_awake(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.AWAKE)
        # Vary C(t) inside deadband [0.41, 0.84]
        for sat in [0.45, 0.50, 0.60, 0.70, 0.80, 0.84]:
            snap = h.update(sat)
            assert snap.current_state == SleepState.AWAKE
            assert snap.in_deadband is True
            assert snap.state_changed is False

        assert h.get_transition_count() == 0
        assert h.get_deadband_evaluations() == 6
        assert h.get_chattering_count() == 0

    def test_deadband_stability_when_asleep(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.ASLEEP)
        # Vary C(t) inside deadband [0.41, 0.84]
        for sat in [0.84, 0.75, 0.65, 0.55, 0.45, 0.41]:
            snap = h.update(sat)
            assert snap.current_state == SleepState.ASLEEP
            assert snap.in_deadband is True
            assert snap.state_changed is False

        assert h.get_transition_count() == 0
        assert h.get_deadband_evaluations() == 6
        assert h.get_chattering_count() == 0

    def test_hysteresis_trajectory_loop(self) -> None:
        """
        Complete sweep:
        0.0 -> 0.90 (triggers ASLEEP)
            -> 0.50 (latched ASLEEP)
            -> 0.80 (latched ASLEEP)
            -> 0.35 (triggers AWAKE)
            -> 0.60 (latched AWAKE)
            -> 0.0  (latched AWAKE)
        """
        h = SleepHysteresis(initial_state=SleepState.AWAKE)

        assert h.update(0.0).current_state == SleepState.AWAKE
        assert h.update(0.50).current_state == SleepState.AWAKE  # in deadband
        assert h.update(0.90).current_state == SleepState.ASLEEP  # C >= 0.85 -> sleep
        assert h.update(0.50).current_state == SleepState.ASLEEP  # latched sleep
        assert h.update(0.80).current_state == SleepState.ASLEEP  # latched sleep
        assert h.update(0.35).current_state == SleepState.AWAKE   # C <= 0.40 -> wake
        assert h.update(0.60).current_state == SleepState.AWAKE   # latched wake
        assert h.update(0.0).current_state == SleepState.AWAKE

        assert h.get_transition_count() == 2
        assert h.get_chattering_count() == 0

    def test_snapshot_duck_typing(self) -> None:
        h = SleepHysteresis()
        snap = h.update(0.2)
        assert snap == SleepState.AWAKE
        assert snap == "AWAKE"
        assert snap.is_awake is True
        assert snap.is_sleeping is False


class TestDVP08NoiseStress10000Cycles:
    """Rigorous empirical validation of DVP-08: zero chattering across 10,000 cycles."""

    def test_10000_cycles_deadband_zero_chattering_uniform_noise(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.AWAKE)
        rng = np.random.default_rng(12345)

        # 10,000 random samples strictly inside the deadband [0.45, 0.80]
        noise_stream = rng.uniform(0.45, 0.80, size=10_000)

        t0 = time.perf_counter()
        for sat in noise_stream:
            snap = h.update(float(sat))
            assert snap.current_state == SleepState.AWAKE

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert h.get_transition_count() == 0
        assert h.get_chattering_count() == 0
        assert h.get_deadband_evaluations() == 10_000
        # Performance: 10,000 cycles should complete quickly (< 100 ms)
        assert elapsed_ms < 150.0

    def test_10000_cycles_sinusoidal_oscillation_with_noise(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.ASLEEP)
        rng = np.random.default_rng(6789)

        # High-frequency oscillation centered at 0.625 (deadband center) with amplitude 0.15 + noise
        t = np.linspace(0, 100 * np.pi, 10_000)
        oscillation = 0.625 + 0.15 * np.sin(t) + rng.normal(0.0, 0.015, size=10_000)
        # Ensure values stay in [0.42, 0.82]
        oscillation = np.clip(oscillation, 0.42, 0.82)

        for sat in oscillation:
            snap = h.update(float(sat))
            assert snap.current_state == SleepState.ASLEEP

        assert h.get_transition_count() == 0
        assert h.get_chattering_count() == 0
        assert h.get_deadband_evaluations() == 10_000

    def test_formal_dvp08_noise_stability_evaluation(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.AWAKE)
        t0 = time.perf_counter()
        passed, chattering_count, stats = h.evaluate_noise_stability(cycles=10_000, sigma=0.03, seed=42)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert passed is True
        assert chattering_count == 0
        assert stats["total_cycles"] == 10_000
        assert stats["chattering_count"] == 0
        assert stats["latching_fidelity_pct"] == 100.0
        assert elapsed_ms < 100.0


class TestEdgeAndPathologicalInputs:
    """Pathological, boundary, and forced arousal tests."""

    def test_nan_saturation_handling(self) -> None:
        h = SleepHysteresis()
        with pytest.raises(ValueError):
            h.update(float("nan"))

    def test_inf_saturation_handling(self) -> None:
        h = SleepHysteresis()
        with pytest.raises(ValueError):
            h.update(float("inf"))

        with pytest.raises(ValueError):
            h.update(float("-inf"))

    def test_negative_saturation_handling(self) -> None:
        h = SleepHysteresis()
        with pytest.raises(ValueError):
            h.update(-0.01)

    def test_extreme_out_of_range_handling(self) -> None:
        h = SleepHysteresis()
        with pytest.raises(ValueError):
            h.update(1.20)  # Severely exceeds 1.05 safety margin

    def test_slight_saturation_margin_clamping(self) -> None:
        h = SleepHysteresis()
        # 1.02 is within the [1.0, 1.05] safety margin -> clamped to 1.0 -> triggers ASLEEP
        snap = h.update(1.02)
        assert snap.current_state == SleepState.ASLEEP
        assert snap.context_saturation == 1.0

    def test_forced_arousal_support(self) -> None:
        h = SleepHysteresis(initial_state=SleepState.ASLEEP)
        assert h.is_sleeping() is True

        h.force_arousal()
        assert h.current_state == SleepState.AWAKE
        assert h.is_awake() is True

    def test_invalid_thresholds_constructor(self) -> None:
        # wake >= sleep is invalid
        with pytest.raises(ValueError):
            SleepHysteresis(sleep_threshold=0.40, wake_threshold=0.85)

        with pytest.raises(ValueError):
            SleepHysteresis(sleep_threshold=0.50, wake_threshold=0.50)
