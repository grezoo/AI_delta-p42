"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Adversarial Stress Test Suite
Target: modules/amygdala_override.py (DVP-03)
Author: Challenger 1 (Adversarial Amygdala Stress Tester)

Empirical Adversarial Challenges:
1. WCET under 5,000 rapid burst threat evaluations (latency ceiling <= 2.0 ms, zero violations).
2. Extended 12,000 evaluations testing rolling history buffer eviction (pop(0)) latency impact.
3. Pathological input robustness: NaNs, Infinities, negative numbers, extreme float magnitudes (1e20, 1e300), denormal floats (5e-324).
4. Exact STDP temporal causality boundary conditions (Delta tau = 0.0, -1.0, 50.0, 50.0001, 100.0 ms, plus +-Inf).
5. Pavlovian fear conditioning, multi-step extinction, and savings-effect re-acquisition dynamics.
6. Safety latching persistence, analog PWM grounding emulation, and cortical override constraints.
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest

from modules.amygdala_override import (
    AmygdalaOutput,
    AmygdalaOverride,
    MODULE_AMYGDALA,
    MODULE_ID,
)


class TestAdversarialAmygdalaWCETBurst:
    """
    Challenge 1: WCET Empirical Verification under 5,000+ Rapid Burst Threat Evaluations.
    Acceptance Criteria (DVP-03):
    - Max latency across 5,000 rapid evaluations must NEVER exceed 2.0 ms (WCET <= 2.0 ms).
    - Mean subcortical bypass latency must be < 0.2 ms.
    - Zero timing violations (count == 0).
    """

    def test_wcet_5000_rapid_burst_evaluations(self) -> None:
        """Execute 5,000 diverse rapid burst threat evaluations and assert WCET <= 2.0 ms."""
        amy = AmygdalaOverride(d_sens=64, d_threat=16, wcet_target_ms=2.0)
        rng = np.random.RandomState(42)

        latencies_ms: List[float] = []

        for i in range(5000):
            phase = i % 5
            if phase == 0:
                # Quiescent baseline
                x_sens = np.zeros(64, dtype=np.float64)
                x_us = None
                delta = 0.0
            elif phase == 1:
                # Normal low-intensity sensory noise
                x_sens = rng.randn(64).astype(np.float64) * 0.1
                x_us = rng.rand(16).astype(np.float64) * 0.3
                delta = 0.05
            elif phase == 2:
                # Sudden unconditioned direct trauma burst
                x_sens = rng.randn(64).astype(np.float64) * 0.5
                x_us = rng.rand(16).astype(np.float64) * 0.95
                delta = 0.85
            elif phase == 3:
                # Extreme sensory shock
                x_sens = rng.randn(64).astype(np.float64) * 50.0
                x_us = None
                delta = 0.9
            else:
                # Subnormal / borderline numerical values
                x_sens = rng.choice([0.0, 5e-324, 1e-310, 0.4, 0.84], size=64).astype(np.float64)
                x_us = rng.choice([0.0, 0.5, 0.84, 5e-324], size=16).astype(np.float64)
                delta = 0.2

            t_start = time.perf_counter_ns()
            out = amy.evaluate_fast(x_sens=x_sens, x_us=x_us, delta_dissonance=delta)
            t_end = time.perf_counter_ns()

            dt_ms = (t_end - t_start) / 1_000_000.0
            latencies_ms.append(dt_ms)

            # Assert internal record is positive and consistent
            assert out.wcet_ms >= 0.0
            assert math.isfinite(out.wcet_ms)

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        max_lat = float(np.max(lat_arr))
        mean_lat = float(np.mean(lat_arr))
        p99_lat = float(np.percentile(lat_arr, 99.0))
        p999_lat = float(np.percentile(lat_arr, 99.9))
        violations = int(np.sum(lat_arr > 2.0))

        stats = amy.get_wcet_stats()

        # Assertions
        assert violations == 0, f"DVP-03 VIOLATION: {violations} calls exceeded 2.0 ms WCET!"
        assert max_lat <= 2.0, f"DVP-03 VIOLATION: Maximum latency {max_lat:.4f} ms > 2.0 ms!"
        assert mean_lat < 0.20, f"Mean latency {mean_lat:.4f} ms exceeded 0.20 ms ceiling!"
        assert stats["sample_count"] == 5000.0
        assert stats["target_met"] == 1.0

        print(f"\n[WCET 5,000 Burst] Max: {max_lat:.4f} ms | Mean: {mean_lat:.4f} ms | p99: {p99_lat:.4f} ms | p99.9: {p999_lat:.4f} ms")

    def test_wcet_extended_buffer_eviction_12000_calls(self) -> None:
        """
        Challenge: Test rolling buffer eviction behavior when history exceeds 10,000 items.
        Verify that pop(0) in _record_wcet does not cause WCET to exceed 2.0 ms.
        """
        amy = AmygdalaOverride(d_sens=64, d_threat=16, wcet_target_ms=2.0)
        rng = np.random.RandomState(999)

        latencies_ms: List[float] = []
        for i in range(12000):
            x_sens = rng.rand(64)
            t_start = time.perf_counter_ns()
            amy.evaluate_fast(x_sens=x_sens)
            t_end = time.perf_counter_ns()
            latencies_ms.append((t_end - t_start) / 1_000_000.0)

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        max_lat = float(np.max(lat_arr))
        violations = int(np.sum(lat_arr > 2.0))

        assert violations == 0, f"WCET violation during buffer eviction: {violations} calls > 2.0 ms"
        assert max_lat <= 2.0, f"WCET violation during buffer eviction: max={max_lat:.4f} ms > 2.0 ms"
        assert len(amy._wcet_samples) == 10000


class TestAdversarialPathologicalInputs:
    """
    Challenge 2: Robustness Against Pathological Inputs.
    Verifies behavior under:
    - IEEE 754 NaNs (in sensory array, trauma vector, and dissonance).
    - IEEE 754 Infinities (+Inf, -Inf).
    - Extreme float magnitudes (1e20, 1e300, -1e20).
    - Subnormal / denormal floats (5e-324).
    - Negative sensory and trauma numbers.
    - Zero crashing, type coercion, and fail-safe panic latching.
    """

    def test_nan_sensory_triggers_failsafe_latch(self) -> None:
        """Verify single NaN or all NaNs in sensory array triggers fail-safe latch."""
        amy = AmygdalaOverride()

        # Case 1: Single NaN at position 0
        sens_nan_0 = np.zeros(64)
        sens_nan_0[0] = np.nan
        out = amy.evaluate_fast(x_sens=sens_nan_0)
        assert out.panic_amy is True
        assert out.pwm_grounded is True
        assert out.pwm_duty_cycle == 0.0
        assert out.alarm_source == "FAILSAFE_CORRUPT_INPUT"
        assert amy.panic_latched is True

        # Verify latch persists even if next frame is safe
        out_next = amy.evaluate_fast(x_sens=np.zeros(64))
        assert out_next.panic_amy is True
        assert out_next.alarm_source == "LATCHED"

        # Case 2: All NaNs
        amy.reset_panic()
        out_all_nan = amy.evaluate_fast(x_sens=np.full(64, np.nan))
        assert out_all_nan.panic_amy is True
        assert out_all_nan.alarm_source == "FAILSAFE_CORRUPT_INPUT"

    def test_nan_trauma_and_dissonance_failsafe(self) -> None:
        """Verify NaN in unconditioned trauma or delta dissonance triggers fail-safe."""
        amy = AmygdalaOverride()

        # NaN in trauma vector
        us_nan = np.zeros(16)
        us_nan[7] = np.nan
        out_us = amy.evaluate_fast(x_sens=np.zeros(64), x_us=us_nan)
        assert out_us.panic_amy is True
        assert out_us.alarm_source == "FAILSAFE_CORRUPT_INPUT"

        # NaN in delta dissonance
        amy.reset_panic()
        out_dissonance = amy.evaluate_fast(x_sens=np.zeros(64), delta_dissonance=float("nan"))
        assert out_dissonance.panic_amy is True
        assert out_dissonance.alarm_source == "FAILSAFE_CORRUPT_INPUT"

    def test_infinities_trigger_failsafe(self) -> None:
        """Verify +Inf and -Inf in inputs trigger fail-safe protection."""
        amy = AmygdalaOverride()

        # +Inf in sensory array
        sens_pinf = np.zeros(64)
        sens_pinf[3] = float("inf")
        out_pinf = amy.evaluate_fast(x_sens=sens_pinf)
        assert out_pinf.panic_amy is True
        assert out_pinf.alarm_source == "FAILSAFE_CORRUPT_INPUT"

        # -Inf in sensory array
        amy.reset_panic()
        sens_ninf = np.zeros(64)
        sens_ninf[15] = float("-inf")
        out_ninf = amy.evaluate_fast(x_sens=sens_ninf)
        assert out_ninf.panic_amy is True
        assert out_ninf.alarm_source == "FAILSAFE_CORRUPT_INPUT"

        # +Inf in delta dissonance
        amy.reset_panic()
        out_diss_inf = amy.evaluate_fast(x_sens=np.zeros(64), delta_dissonance=float("inf"))
        assert out_diss_inf.panic_amy is True
        assert out_diss_inf.alarm_source == "FAILSAFE_CORRUPT_INPUT"

    def test_extreme_float_magnitudes(self) -> None:
        """
        Verify extreme float magnitudes (1e20, -1e20, 1e300) are recognized
        as catastrophic physical threats, safely clipped, and trigger emergency brake.
        """
        amy = AmygdalaOverride()

        # Positive extreme magnitude 1e20 in x_sens
        sens_huge = np.zeros(64)
        sens_huge[0] = 1e20
        out_huge = amy.evaluate_fast(x_sens=sens_huge)
        assert out_huge.panic_amy is True
        assert out_huge.pwm_grounded is True
        assert out_huge.alarm_source == "DIRECT_TRAUMA"
        assert out_huge.unconditioned_threat == 1.0  # Clipped safely

        # Negative extreme magnitude -1e20 in x_sens (large physical shock)
        amy.reset_panic()
        sens_nhuge = np.zeros(64)
        sens_nhuge[0] = -1e20
        out_nhuge = amy.evaluate_fast(x_sens=sens_nhuge)
        assert out_nhuge.panic_amy is True
        assert out_nhuge.alarm_source == "DIRECT_TRAUMA"
        assert out_nhuge.unconditioned_threat == 1.0

        # Positive extreme magnitude 1e300 in trauma vector
        amy.reset_panic()
        us_huge = np.zeros(16)
        us_huge[5] = 1e300
        out_us_huge = amy.evaluate_fast(x_sens=np.zeros(64), x_us=us_huge)
        assert out_us_huge.panic_amy is True
        assert out_us_huge.unconditioned_threat == 1.0

    def test_denormal_subnormal_floats(self) -> None:
        """
        Verify subnormal / denormal floats (5e-324, 1e-310):
        1. Do not cause ZeroDivisionError, FloatingPointError, or process crash.
        2. Do not trigger false positive panic when danger is zero.
        3. Do not incur severe hardware timing stalls exceeding 2.0 ms.
        """
        amy = AmygdalaOverride()

        denormal_val = 5e-324
        sens_denormal = np.full(64, denormal_val)
        us_denormal = np.full(16, denormal_val)

        t_start = time.perf_counter_ns()
        out = amy.evaluate_fast(x_sens=sens_denormal, x_us=us_denormal)
        t_end = time.perf_counter_ns()

        lat_ms = (t_end - t_start) / 1_000_000.0

        # Quiescent safety: denormals are essentially zero threat
        assert out.panic_amy is False
        assert out.pwm_grounded is False
        assert out.pwm_duty_cycle == 1.0
        assert out.alarm_source == "NONE"
        assert out.unconditioned_threat < 0.01

        # WCET guarantee under denormal processing
        assert lat_ms <= 2.0, f"Denormal processing caused timing penalty: {lat_ms:.4f} ms > 2.0 ms"

    def test_negative_trauma_and_dissonance_handling(self) -> None:
        """
        Verify negative trauma and negative dissonance values do not produce
        negative unconditioned threat scores or corrupt fear response.
        """
        amy = AmygdalaOverride()

        # Negative trauma vector (e.g. -1.0)
        us_neg = np.full(16, -1.0)
        out_us_neg = amy.evaluate_fast(x_sens=np.zeros(64), x_us=us_neg)
        assert out_us_neg.panic_amy is False
        assert out_us_neg.unconditioned_threat == 0.0

        # Negative delta dissonance (e.g. -0.5)
        out_diss_neg = amy.evaluate_fast(x_sens=np.zeros(64), delta_dissonance=-0.5)
        assert out_diss_neg.panic_amy is False
        assert out_diss_neg.unconditioned_threat == 0.0


class TestAdversarialSTDPTemporalCausality:
    """
    Challenge 3: STDP Temporal Causality Window Exact Boundary Tests.
    Testing exact boundaries:
    - Delta tau = 0.0 ms (simultaneous, non-causal -> Delta W == 0).
    - Delta tau = -1.0 ms (anti-causal backward conditioning -> Delta W == 0).
    - Delta tau = 50.0 ms (exact upper causal bound -> Delta W > 0, exact formula match).
    - Delta tau = 50.0001 ms (just beyond upper bound -> Delta W == 0).
    - Delta tau = 100.0 ms (far beyond upper bound -> Delta W == 0).
    - Delta tau = -Inf, +Inf.
    """

    @pytest.fixture
    def stdp_setup(self) -> Tuple[AmygdalaOverride, np.ndarray, np.ndarray]:
        amy = AmygdalaOverride(
            d_sens=64,
            d_threat=16,
            gamma_stdp=0.35,
            tau_stdp_ms=20.0,
            max_delta_tau_ms=50.0,
        )
        cs = np.zeros(64)
        cs[12] = 1.0
        us = np.zeros(16)
        us[4] = 1.0
        return amy, cs, us

    def test_stdp_boundary_zero_delay(self, stdp_setup) -> None:
        """Delta tau = 0.0 ms: simultaneous co-activation lacks causal precedence."""
        amy, cs, us = stdp_setup
        w_max = amy.pair_conditioning(cs, us, delta_tau_ms=0.0)
        assert w_max == 0.0
        assert np.count_nonzero(amy.W_fear) == 0
        assert amy.consecutive_pairings == 0

    def test_stdp_boundary_negative_delay(self, stdp_setup) -> None:
        """Delta tau = -1.0 ms: backward anti-causal pairing produces zero update."""
        amy, cs, us = stdp_setup
        w_max = amy.pair_conditioning(cs, us, delta_tau_ms=-1.0)
        assert w_max == 0.0
        assert np.count_nonzero(amy.W_fear) == 0
        assert amy.consecutive_pairings == 0

    def test_stdp_boundary_exact_50ms(self, stdp_setup) -> None:
        """
        Delta tau = 50.0 ms: exact upper bound of causal window (0 < Delta tau <= 50.0 ms).
        Expected update: gamma * exp(-50 / 20) = 0.35 * exp(-2.5) ~= 0.0287297.
        """
        amy, cs, us = stdp_setup
        w_max = amy.pair_conditioning(cs, us, delta_tau_ms=50.0)

        expected_dw = 0.35 * math.exp(-50.0 / 20.0)
        assert pytest.approx(w_max, rel=1e-4) == expected_dw
        assert pytest.approx(amy.W_fear[12, 4], rel=1e-4) == expected_dw
        assert np.count_nonzero(amy.W_fear) == 1
        assert amy.consecutive_pairings == 1

    def test_stdp_boundary_just_above_50ms(self, stdp_setup) -> None:
        """Delta tau = 50.0001 ms: strictly exceeds 50.0 ms -> zero update."""
        amy, cs, us = stdp_setup
        w_max = amy.pair_conditioning(cs, us, delta_tau_ms=50.0001)
        assert w_max == 0.0
        assert np.count_nonzero(amy.W_fear) == 0
        assert amy.consecutive_pairings == 0

    def test_stdp_boundary_100ms(self, stdp_setup) -> None:
        """Delta tau = 100.0 ms: far outside causal window -> zero update."""
        amy, cs, us = stdp_setup
        w_max = amy.pair_conditioning(cs, us, delta_tau_ms=100.0)
        assert w_max == 0.0
        assert np.count_nonzero(amy.W_fear) == 0
        assert amy.consecutive_pairings == 0

    def test_stdp_infinite_delays(self, stdp_setup) -> None:
        """Delta tau = -inf and +inf: must safely produce zero update without crashing."""
        amy, cs, us = stdp_setup
        w_ninf = amy.pair_conditioning(cs, us, delta_tau_ms=float("-inf"))
        assert w_ninf == 0.0

        w_pinf = amy.pair_conditioning(cs, us, delta_tau_ms=float("inf"))
        assert w_pinf == 0.0


class TestAdversarialPavlovianExtinctionAndReacquisition:
    """
    Challenge 4: Pavlovian Fear Extinction and Savings-Effect Re-Acquisition Dynamics.
    Verifies:
    1. Initial 3-pairing acquisition consolidates fear and triggers anticipatory panic.
    2. Repeated unreinforced extinction trials smoothly decay fear weights.
    3. Re-acquisition demonstrates the biological SAVINGS EFFECT:
       re-learning fear requires FEWER pairings (2 pairings) than initial naive learning (3 pairings).
    4. Multi-cycle stability across repeated learning-unlearning cycles.
    """

    def test_pavlovian_acquisition_extinction_reacquisition_lifecycle(self) -> None:
        """Test complete acquisition -> extinction -> re-acquisition lifecycle."""
        amy = AmygdalaOverride(
            d_sens=64,
            d_threat=16,
            gamma_stdp=0.35,
            tau_stdp_ms=20.0,
            threat_threshold=0.85,
            cond_fear_threshold=0.50,
            lambda_extinct=0.05,
        )

        cs_cue = np.zeros(64)
        cs_cue[8] = 1.0
        us_trauma = np.zeros(16)
        us_trauma[1] = 1.0

        # Phase 1: Naive State
        out_naive = amy.evaluate_fast(x_sens=cs_cue)
        assert out_naive.panic_amy is False
        assert out_naive.fear_response == 0.0

        # Phase 2: Naive Acquisition requires exactly 3 pairings
        # Pairing 1
        amy.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amy.reset_panic()
        out_p1 = amy.evaluate_fast(x_sens=cs_cue)
        assert out_p1.panic_amy is False
        assert out_p1.fear_response < 0.50

        # Pairing 2
        amy.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amy.reset_panic()
        out_p2 = amy.evaluate_fast(x_sens=cs_cue)
        assert out_p2.panic_amy is False
        assert out_p2.fear_response < 0.50

        # Pairing 3 -> CONSOLIDATION
        amy.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amy.reset_panic()
        out_p3 = amy.evaluate_fast(x_sens=cs_cue, x_us=None)
        assert out_p3.panic_amy is True, "3 pairings must trigger anticipatory panic"
        assert out_p3.fear_response >= 0.50
        assert out_p3.alarm_source == "CONDITIONED_REFLEX"
        assert amy.is_consolidated is True

        peak_weight = float(np.max(amy.W_fear))
        assert peak_weight > 0.35

        # Phase 3: Fear Extinction over 25 trials
        for step in range(25):
            amy.apply_extinction(decay_rate=0.05)

        extinguished_weight = float(np.max(amy.W_fear))
        assert extinguished_weight < peak_weight
        expected_extinguished = peak_weight * ((1.0 - 0.05) ** 25)
        assert pytest.approx(extinguished_weight, rel=1e-3) == expected_extinguished

        # After extinction, CS cue alone must NOT trigger panic
        amy.reset_panic()
        out_ext = amy.evaluate_fast(x_sens=cs_cue, x_us=None)
        assert out_ext.panic_amy is False
        assert out_ext.fear_response < 0.50
        assert amy.is_consolidated is False

        # Phase 4: Re-acquisition Dynamics (Biological Savings Effect)
        # Because of residual synaptic trace, re-acquisition crosses threshold in 2 pairings instead of 3
        amy.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amy.reset_panic()
        out_re1 = amy.evaluate_fast(x_sens=cs_cue)
        # Pairing 1 of re-acquisition
        re1_fear = out_re1.fear_response

        # Pairing 2 of re-acquisition -> Reaches conditioned panic faster!
        amy.pair_conditioning(cs_cue, us_trauma, delta_tau_ms=20.0)
        amy.reset_panic()
        out_re2 = amy.evaluate_fast(x_sens=cs_cue, x_us=None)
        assert out_re2.panic_amy is True, "Savings effect: re-acquisition must trigger panic by pairing 2"
        assert out_re2.fear_response >= 0.50
        assert out_re2.alarm_source == "CONDITIONED_REFLEX"

    def test_multi_cycle_extinction_stability(self) -> None:
        """Verify synaptic stability over 5 complete learning/unlearning cycles."""
        amy = AmygdalaOverride(lambda_extinct=0.10)
        cs = np.zeros(64)
        cs[0] = 1.0
        us = np.zeros(16)
        us[0] = 1.0

        for cycle in range(5):
            # Learn (3 pairings)
            for _ in range(3):
                amy.pair_conditioning(cs, us, delta_tau_ms=20.0)
            amy.reset_panic()
            out_learned = amy.evaluate_fast(x_sens=cs)
            assert out_learned.panic_amy is True

            # Extinguish (30 steps)
            for _ in range(30):
                amy.apply_extinction(decay_rate=0.10)
            amy.reset_panic()
            out_ext = amy.evaluate_fast(x_sens=cs)
            assert out_ext.panic_amy is False


class TestAdversarialStructuralAndSecurityInterlocks:
    """
    Challenge 5: Structural Integrity, Interlocks, and Dimension Resilience.
    """

    def test_dimension_padding_and_truncation(self) -> None:
        """Verify sensory array dimension mismatch is padded or truncated cleanly."""
        amy = AmygdalaOverride(d_sens=64, d_threat=16)

        # Under-dimensioned: 32 elements (must pad with zeros to 64)
        sens_short = np.ones(32) * 0.1
        out_short = amy.evaluate_fast(x_sens=sens_short)
        assert out_short.panic_amy is False

        # Over-dimensioned: 128 elements (must truncate to 64)
        sens_long = np.ones(128) * 0.1
        out_long = amy.evaluate_fast(x_sens=sens_long)
        assert out_long.panic_amy is False

        # Empty array: 0 elements
        sens_empty = np.array([])
        out_empty = amy.evaluate_fast(x_sens=sens_empty)
        assert out_empty.panic_amy is False

    def test_cortical_override_safety_lock(self) -> None:
        """Verify cortex cannot override panic while raw unconditioned trauma persists."""
        amy = AmygdalaOverride(threat_threshold=0.85)

        # Active physical trauma (e.g. D_direct = 0.95 >= 0.85)
        out = amy.evaluate_fast(x_sens=np.zeros(64), x_us=np.full(16, 0.95))
        assert out.panic_amy is True
        assert amy.panic_latched is True

        # Cortical reset attempt while trauma is active MUST BE BLOCKED
        amy.evaluate_cortical(pfc_context=np.ones(32), suppress_panic=True)
        assert amy.panic_latched is True, "Cortex must NOT override active unconditioned trauma"

        # Trauma clears: D_direct = 0.0
        amy.evaluate_fast(x_sens=np.zeros(64), x_us=np.zeros(16))
        # Now cortical reset is permitted
        amy.evaluate_cortical(pfc_context=np.ones(32), suppress_panic=True)
        assert amy.panic_latched is False, "Cortex should be permitted to reset latch after danger clears"

    def test_analog_circuit_registers_consistency(self) -> None:
        """Verify electrical analog emulation registers (pwm_pin_volts, pull_down_active)."""
        amy = AmygdalaOverride()

        # Quiescent: 3.3 V high, pull_down inactive
        amy.evaluate_fast(x_sens=np.zeros(64))
        assert amy.pwm_pin_volts == 3.3
        assert amy.pull_down_active is False

        # Trigger alarm: 0.0 V short to GND, pull_down active
        amy.evaluate_fast(x_sens=np.zeros(64), x_us=np.ones(16))
        assert amy.pwm_pin_volts == 0.0
        assert amy.pull_down_active is True

        # Reset panic: restores 3.3 V high
        amy.reset_panic()
        assert amy.pwm_pin_volts == 3.3
        assert amy.pull_down_active is False
