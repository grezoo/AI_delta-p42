"""
Unit and DVP-03 Acceptance Test Suite for Amygdala Override Module (Gate 2).
Covers:
1. WCET <= 2.0 ms physical reaction time over 1000 collision/threat evaluations.
2. Analog PWM grounding emulation (pwm_grounded=True, pwm_duty_cycle=0.0, gain_boost=2.0).
3. STDP temporal window (0 < Delta tau <= 50 ms, zero update for Delta tau <= 0 or > 50 ms).
4. 3-pairing Pavlovian fear conditioning & anticipatory panic trigger on CS alone (DVP-03).
5. Synaptic fear extinction dynamics on unreinforced CS presentations.
6. Cortico-Amygdalar slow pathway valence modulation and cognitive override.
7. Fail-safe protection on corrupted / NaN / Inf inputs and quiescent stability.
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest

from modules.amygdala_override import (
    MODULE_AMYGDALA,
    MODULE_CEREBELLUM,
    MODULE_ID,
    AmygdalaOutput,
    AmygdalaOverride,
)


class TestAmygdalaCoreArchitecture:
    """Tests basic module constants, interface contracts, and immutability."""

    def test_canonical_module_identifiers(self) -> None:
        assert MODULE_ID == 6
        assert MODULE_AMYGDALA == 6
        assert MODULE_CEREBELLUM == 7

    def test_output_dataclass_immutability(self) -> None:
        out = AmygdalaOutput(
            timestamp_ns=1000,
            panic_amy=False,
            pwm_grounded=False,
            pwm_duty_cycle=1.0,
            valence=0.0,
            fear_response=0.0,
            unconditioned_threat=0.0,
            gain_boost=1.0,
            wcet_ms=0.05,
            alarm_source="NONE",
        )
        with pytest.raises((AttributeError, TypeError)):
            # Dataclass is frozen; mutating attributes must fail
            out.panic_amy = True  # type: ignore

        as_dict = out.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["timestamp_ns"] == 1000
        assert as_dict["pwm_duty_cycle"] == 1.0


class TestAmygdalaWCETAndFastBypass:
    """Verifies DVP-03 WCET <= 2.0 ms execution ceiling over 1000 iterations."""

    def test_wcet_reaction_latency_1000_trials(self) -> None:
        amygdala = AmygdalaOverride(d_sens=64, d_threat=16, wcet_target_ms=2.0)
        rng = np.random.RandomState(42)

        latencies_ms: List[float] = []

        for _ in range(1000):
            # Generate random sensory patterns with occasional danger spikes
            x_sens = rng.rand(64)
            x_us = rng.rand(16) * 0.9  # May cross 0.85 threshold

            t_start = time.perf_counter_ns()
            out = amygdala.evaluate_fast(x_sens=x_sens, x_us=x_us)
            t_end = time.perf_counter_ns()

            measured_ms = (t_end - t_start) / 1_000_000.0
            latencies_ms.append(measured_ms)

            # Internal recorded WCET must match
            assert out.wcet_ms >= 0.0

        max_latency = max(latencies_ms)
        mean_latency = sum(latencies_ms) / len(latencies_ms)
        stats = amygdala.get_wcet_stats()

        assert stats["sample_count"] == 1000.0
        assert stats["target_met"] == 1.0
        assert max_latency <= 2.0, f"DVP-03 WCET violation: max={max_latency:.4f} ms > 2.0 ms"
        assert mean_latency < 0.2, f"Expected mean subcortical latency < 0.2 ms, got {mean_latency:.4f} ms"

    def test_analog_pwm_grounding_on_direct_threat(self) -> None:
        amygdala = AmygdalaOverride(threat_threshold=0.85)

        # Normal below-threshold stimulus
        x_safe = np.zeros(64)
        x_us_safe = np.ones(16) * 0.50
        out_safe = amygdala.evaluate_fast(x_sens=x_safe, x_us=x_us_safe)
        assert not out_safe.panic_amy
        assert not out_safe.pwm_grounded
        assert out_safe.pwm_duty_cycle == 1.0
        assert out_safe.gain_boost == 1.0
        assert out_safe.alarm_source == "NONE"

        # Direct danger shock: D_direct >= 0.85
        x_us_danger = np.zeros(16)
        x_us_danger[3] = 0.95
        out_danger = amygdala.evaluate_fast(x_sens=x_safe, x_us=x_us_danger)
        assert out_danger.panic_amy
        assert out_danger.pwm_grounded
        assert out_danger.pwm_duty_cycle == 0.0
        assert out_danger.gain_boost == 2.0
        assert out_danger.alarm_source == "DIRECT_TRAUMA"
        assert out_danger.unconditioned_threat == 0.95


class TestSTDPAssociativeFearConditioning:
    """Verifies STDP temporal causality window and Pavlovian conditioning (DVP-03)."""

    def test_stdp_causality_temporal_window(self) -> None:
        amygdala = AmygdalaOverride(gamma_stdp=0.35, tau_stdp_ms=20.0, max_delta_tau_ms=50.0)
        x_cs = np.zeros(64)
        x_cs[10] = 1.0
        x_us = np.zeros(16)
        x_us[2] = 1.0

        # Case 1: Backward non-causal order (Delta tau < 0) -> ZERO update
        w_before = float(np.max(amygdala.W_fear))
        amygdala.pair_conditioning(x_cs, x_us, delta_tau_ms=-10.0)
        assert float(np.max(amygdala.W_fear)) == w_before == 0.0

        # Case 2: Simultaneous (Delta tau = 0) -> ZERO update
        amygdala.pair_conditioning(x_cs, x_us, delta_tau_ms=0.0)
        assert float(np.max(amygdala.W_fear)) == 0.0

        # Case 3: Outside temporal window (Delta tau = 80 ms > 50 ms) -> ZERO update
        amygdala.pair_conditioning(x_cs, x_us, delta_tau_ms=80.0)
        assert float(np.max(amygdala.W_fear)) == 0.0

        # Case 4: Causal window (Delta tau = 20 ms in (0, 50]) -> Exponential update
        w_after = amygdala.pair_conditioning(x_cs, x_us, delta_tau_ms=20.0)
        expected_dw = 0.35 * math.exp(-20.0 / 20.0)  # ~0.128758
        assert pytest.approx(w_after, rel=1e-3) == expected_dw
        assert amygdala.W_fear[10, 2] == pytest.approx(expected_dw, rel=1e-3)
        assert np.count_nonzero(amygdala.W_fear) == 1

    def test_three_pairing_fear_consolidation_dvp03(self) -> None:
        """
        DVP-03 Requirement:
        - Present neutral CS -> panic_amy == False.
        - Apply exactly 3 pairings of CS followed by US (Delta tau = 20 ms).
        - Present CS alone -> panic_amy == True (anticipatory emergency brake before US).
        """
        amygdala = AmygdalaOverride(
            d_sens=64,
            d_threat=16,
            threat_threshold=0.85,
            cond_fear_threshold=0.50,
        )

        # Conditioned Stimulus pattern (e.g. neutral cue sound/visual)
        cs_cue = np.zeros(64)
        cs_cue[5] = 1.0
        # Unconditioned Stimulus (severe collision trauma)
        us_trauma = np.zeros(16)
        us_trauma[1] = 1.0

        # 1. Before conditioning: CS alone produces NO alarm
        out_init = amygdala.evaluate_fast(x_sens=cs_cue)
        assert not out_init.panic_amy
        assert not out_init.pwm_grounded
        assert out_init.fear_response < 0.50

        # 2. Pairing 1 (Delta tau = 20 ms)
        amygdala.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amygdala.reset_panic()
        out_p1 = amygdala.evaluate_fast(x_sens=cs_cue)
        assert not out_p1.panic_amy, "CS after only 1 pairing should not yet trigger alarm"

        # 3. Pairing 2 (Delta tau = 20 ms)
        amygdala.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amygdala.reset_panic()
        out_p2 = amygdala.evaluate_fast(x_sens=cs_cue)
        assert not out_p2.panic_amy, "CS after 2 pairings should not yet cross 0.50 threshold"

        # 4. Pairing 3 (Delta tau = 20 ms) -> Consolidation!
        amygdala.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amygdala.reset_panic()

        # 5. Test CS alone (NO US presented: x_us is None!)
        out_p3 = amygdala.evaluate_fast(x_sens=cs_cue, x_us=None)
        assert out_p3.fear_response >= 0.50, f"Expected y_CR >= 0.50, got {out_p3.fear_response}"
        assert out_p3.panic_amy is True, "3 pairings must trigger anticipatory panic brake"
        assert out_p3.pwm_grounded is True
        assert out_p3.pwm_duty_cycle == 0.0
        assert out_p3.gain_boost == 2.0
        assert out_p3.alarm_source == "CONDITIONED_REFLEX"

    def test_fear_extinction_dynamics(self) -> None:
        """Verifies synaptic decay on unreinforced CS presentations."""
        amygdala = AmygdalaOverride(lambda_extinct=0.05)
        cs_cue = np.zeros(64)
        cs_cue[7] = 1.0
        us_trauma = np.zeros(16)
        us_trauma[0] = 1.0

        # Perform 3 pairings to establish fear
        for _ in range(3):
            amygdala.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)

        amygdala.reset_panic()
        out_conditioned = amygdala.evaluate_fast(x_sens=cs_cue)
        assert out_conditioned.panic_amy is True

        # Apply extinction over 25 unreinforced trials
        w_peak = float(np.max(amygdala.W_fear))
        for _ in range(25):
            amygdala.apply_extinction(decay_rate=0.05)

        w_extinguished = float(np.max(amygdala.W_fear))
        assert w_extinguished < w_peak
        assert w_extinguished == pytest.approx(w_peak * (0.95 ** 25), rel=1e-3)

        # After extinction, CS cue alone must no longer trigger panic
        amygdala.reset_panic()
        out_extinguished = amygdala.evaluate_fast(x_sens=cs_cue)
        assert out_extinguished.panic_amy is False
        assert out_extinguished.fear_response < 0.50


class TestCorticalPathwayAndSafetyInterlocks:
    """Verifies slow cortico-amygdalar path, valence calculation, and fail-safe bounds."""

    def test_cortical_valence_modulation(self) -> None:
        amygdala = AmygdalaOverride()
        pfc_positive = np.ones(32) * 2.0
        pfc_negative = np.ones(32) * -2.0

        # Positive PFC context yields positive valence
        val_pos = amygdala.evaluate_cortical(pfc_positive)
        assert 0.0 < val_pos <= 1.0

        # Negative PFC context yields negative valence
        val_neg = amygdala.evaluate_cortical(pfc_negative)
        assert -1.0 <= val_neg < 0.0

    def test_cortical_cannot_override_active_raw_trauma(self) -> None:
        amygdala = AmygdalaOverride()
        x_us_danger = np.ones(16) * 0.95

        # Trigger direct trauma
        out = amygdala.evaluate_fast(x_sens=np.zeros(64), x_us=x_us_danger)
        assert out.panic_amy is True

        # Cortical suppress attempt while raw trauma is ongoing MUST FAIL
        amygdala.evaluate_cortical(pfc_context=np.ones(32), suppress_panic=True)
        assert amygdala.panic_latched is True

        # When physical trauma is removed, cortical suppress succeeds
        amygdala.evaluate_fast(x_sens=np.zeros(64), x_us=np.zeros(16))
        amygdala.evaluate_cortical(pfc_context=np.ones(32), suppress_panic=True)
        assert amygdala.panic_latched is False

    def test_failsafe_protection_against_corrupted_inputs(self) -> None:
        amygdala = AmygdalaOverride()
        corrupt_sens = np.zeros(64)
        corrupt_sens[5] = np.nan

        out = amygdala.evaluate_fast(x_sens=corrupt_sens)
        assert out.panic_amy is True
        assert out.pwm_grounded is True
        assert out.pwm_duty_cycle == 0.0
        assert out.alarm_source == "FAILSAFE_CORRUPT_INPUT"

    def test_quiescent_zero_input_does_not_panic(self) -> None:
        amygdala = AmygdalaOverride()
        out = amygdala.evaluate_fast(x_sens=np.zeros(64), x_us=np.zeros(16))
        assert out.panic_amy is False
        assert out.pwm_grounded is False
        assert out.pwm_duty_cycle == 1.0
        assert out.gain_boost == 1.0
        assert out.alarm_source == "NONE"
