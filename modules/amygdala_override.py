"""
ΔP-42 Sovereign Cognitive Core (Gate 2) - Amygdala Override Module
LeDoux Dual Pathway Architecture:
1. Ultragyors Talamikus Útvonal (Thalamo-Amygdalar Shortcut, 1–5 ms, WCET <= 2.0 ms)
   - Közvetlen analóg nyitott kollektoros/lebegő PWM földelés emuláció: pwm_grounded=True, pwm_duty_cycle=0.0
   - Hardveres reakcióidő mérés nanomásodperces pontossággal (t_alarm -> t_kill <= 2.0 ms)
   - Adrenalin talamikus érzékenység-erősítés (gain_boost = 2.0)
2. Lassú Kérgi Útvonal (Cortico-Amygdalar Pathway, ~200 ms)
   - Prefrontális kéreg (PFC) top-down kontextus kiértékelése
   - Érzelmi valencia moduláció: A_valencia in [-1.0, 1.0]
   - Kognitív felülbírálás és biztonságos vészfék feloldás
3. STDP Asszociatív Félelmi Kondicionálás (Spike-Timing-Dependent Plasticity)
   - Delta W_fear,ij = gamma * x_CS,i(t - Delta tau) * x_US,j(t) * exp(-Delta tau / tau_stdp) (0 < Delta tau <= 50 ms)
   - Szigorú kauzalitás: Delta tau <= 0 vagy Delta tau > 50 ms esetén zéró módosulás
   - 3 párosításos konszolidáció: 3 egymást követő inger-trauma után a feltételes inger önmagában kiváltja a pánikot
4. Félelmi Kihalás (Fear Extinction):
   - W_fear(t+1) = W_fear(t) * (1 - lambda_extinct) unreinforced ingerek esetén

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapter I.6 lines 197-222, Chapter V.3 line 409)
- Plan Review.md (REQ-ANAT-02, DVP-03)
- megvalósítási terv.md (Gate 2 Subcortical Specification)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Canonical Module Identifiers
MODULE_ID: int = 6
MODULE_AMYGDALA: int = 6
MODULE_CEREBELLUM: int = 7


@dataclass(frozen=True)
class AmygdalaOutput:
    """
    Immutable atomic output packet from the Amygdala subcortical override engine.

    Attributes:
        timestamp_ns: Monotonic nanosecond timestamp of the evaluation.
        panic_amy: Boolean hardware emergency override signal.
        pwm_grounded: Boolean simulation of physical analog PWM pull-down line.
        pwm_duty_cycle: Continuous PWM duty cycle (0.0 when grounded, 1.0 in pass-through).
        valence: Continuous emotional valence A_valencia in [-1.0, 1.0].
        fear_response: Conditioned fear response magnitude y_CR in [0.0, 1.0].
        unconditioned_threat: Unconditioned direct trauma score D_direct in [0.0, 1.0].
        gain_boost: Thalamic sensory amplification factor in [1.0, 2.0].
        wcet_ms: Measured execution latency of the fast pathway in milliseconds.
        alarm_source: Diagnostic origin ('NONE', 'DIRECT_TRAUMA', 'CONDITIONED_REFLEX',
                      'PFC_LOCK', 'LATCHED', 'FAILSAFE_CORRUPT_INPUT').
    """
    timestamp_ns: int
    panic_amy: bool
    pwm_grounded: bool
    pwm_duty_cycle: float
    valence: float
    fear_response: float
    unconditioned_threat: float
    gain_boost: float
    wcet_ms: float
    alarm_source: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert output dataclass to a plain dictionary."""
        return asdict(self)


class AmygdalaOverride:
    """
    Amygdala Associative Fear Conditioning & Fast Hardware Override Engine.
    Implements LeDoux Dual Pathway, STDP Pavlovian conditioning, and analog PWM grounding.
    """

    def __init__(
        self,
        d_sens: int = 64,
        d_threat: int = 16,
        gamma_stdp: float = 0.35,
        tau_stdp_ms: float = 20.0,
        max_delta_tau_ms: float = 50.0,
        threat_threshold: float = 0.85,
        cond_fear_threshold: float = 0.50,
        wcet_target_ms: float = 2.0,
        lambda_extinct: float = 0.05,
    ) -> None:
        """
        Initialize the Amygdala subcortical engine.

        Args:
            d_sens: Sensory input feature vector dimensionality (default 64).
            d_threat: Unconditioned threat / trauma vector dimensionality (default 16).
            gamma_stdp: STDP synaptic learning rate (canonical: 0.35).
            tau_stdp_ms: Biological STDP temporal decay constant in ms (canonical: 20.0 ms).
            max_delta_tau_ms: Maximum causal temporal associative window in ms (canonical: 50.0 ms).
            threat_threshold: Direct unconditioned danger threshold Theta_threat (canonical: 0.85).
            cond_fear_threshold: Conditioned fear trigger threshold Theta_cond (canonical: 0.50).
            wcet_target_ms: Worst-Case Execution Time ceiling in ms (canonical: 2.0 ms).
            lambda_extinct: Synaptic decay rate during unreinforced extinction (canonical: 0.05).
        """
        self.d_sens: int = int(d_sens)
        self.d_threat: int = int(d_threat)
        self.gamma_stdp: float = float(gamma_stdp)
        self.tau_stdp_ms: float = float(tau_stdp_ms)
        self.max_delta_tau_ms: float = float(max_delta_tau_ms)
        self.threat_threshold: float = float(threat_threshold)
        self.cond_fear_threshold: float = float(cond_fear_threshold)
        self.wcet_target_ms: float = float(wcet_target_ms)
        self.lambda_extinct: float = float(lambda_extinct)

        # Associative fear synaptic weight matrix W_fear in [0.0, 1.0]^(d_sens x d_threat)
        self.W_fear: np.ndarray = np.zeros((self.d_sens, self.d_threat), dtype=np.float64)

        # 3-Pairing Pavlovian Consolidation Tracker (DVP-03)
        self.consecutive_pairings: int = 0
        self.consolidation_threshold: int = 3
        self.is_consolidated: bool = False

        # Physical Analog Open-Drain PWM Pull-Down Emulation Registers (0.0 V / 3.3 V)
        self.pwm_pin_volts: float = 3.3         # Normal 3.3 V high; 0.0 V when grounded
        self.pull_down_active: bool = False
        self.gain_boost: float = 1.0

        # State Variables
        self.valence: float = 0.0
        self.panic_latched: bool = False
        self.last_alarm_source: str = "NONE"
        self.last_unconditioned_threat: float = 0.0
        self.last_fear_response: float = 0.0

        # WCET Diagnostics & Realtime Monitoring
        self._wcet_samples: List[float] = []
        self._max_samples: int = 10000
        self._wcet_violation_count: int = 0

    def evaluate_fast(
        self,
        x_sens: np.ndarray,
        x_us: Optional[np.ndarray] = None,
        delta_dissonance: float = 0.0,
        timestamp_ns: Optional[int] = None,
    ) -> AmygdalaOutput:
        """
        Execute the ultra-fast thalamo-amygdalar subcortical bypass path (1–5 ms).
        Monitors raw sensory shock and associative fear, measuring WCET in nanoseconds.

        If threat exceeds threshold, immediately emulates analog PWM grounding:
            pwm_grounded = True, pwm_duty_cycle = 0.0, panic_amy = True, gain_boost = 2.0.

        Args:
            x_sens: Raw sensory vector of shape (d_sens,).
            x_us: Optional unconditioned trauma vector of shape (d_threat,).
            delta_dissonance: Normalized cognitive dissonance delta(t) in [0.0, 1.0].
            timestamp_ns: Optional reference nanosecond timestamp.

        Returns:
            Immutable AmygdalaOutput record.
        """
        t_start = time.perf_counter_ns()
        curr_ts = timestamp_ns if timestamp_ns is not None else t_start

        # 1. Input Validation and Fail-Safe NaN / Inf Interlock
        sens_arr = np.asarray(x_sens, dtype=np.float64)
        has_invalid_num = not np.all(np.isfinite(sens_arr))

        us_arr: Optional[np.ndarray] = None
        if x_us is not None:
            us_arr = np.asarray(x_us, dtype=np.float64)
            if not np.all(np.isfinite(us_arr)):
                has_invalid_num = True

        if has_invalid_num or not math.isfinite(delta_dissonance):
            # Failsafe protection against numerical corruption
            self.panic_latched = True
            self.last_alarm_source = "FAILSAFE_CORRUPT_INPUT"
            self.valence = -1.0
            t_end = time.perf_counter_ns()
            wcet_ms = (t_end - t_start) / 1_000_000.0
            self._record_wcet(wcet_ms)
            return AmygdalaOutput(
                timestamp_ns=curr_ts,
                panic_amy=True,
                pwm_grounded=True,
                pwm_duty_cycle=0.0,
                valence=-1.0,
                fear_response=1.0,
                unconditioned_threat=1.0,
                gain_boost=2.0,
                wcet_ms=wcet_ms,
                alarm_source=self.last_alarm_source,
            )

        # Pad or truncate sensory array to expected d_sens
        if sens_arr.shape[0] != self.d_sens:
            padded_sens = np.zeros(self.d_sens, dtype=np.float64)
            copy_len = min(sens_arr.shape[0], self.d_sens)
            padded_sens[:copy_len] = sens_arr[:copy_len]
            sens_arr = padded_sens

        # 2. Evaluate Unconditioned Direct Threat D_direct
        d_direct: float = 0.0
        if us_arr is not None and us_arr.size > 0:
            d_direct = max(d_direct, float(np.max(us_arr)))
        elif sens_arr.size > 0:
            # Dimension-scaled sensory norm: 1/sqrt(d_sens) * ||x_sens||_inf
            scale = 1.0 / math.sqrt(float(self.d_sens))
            d_direct = max(d_direct, float(np.max(np.abs(sens_arr))) * scale)

        if delta_dissonance > 0.0:
            d_direct = max(d_direct, float(delta_dissonance))

        d_direct = float(np.clip(d_direct, 0.0, 1.0))
        self.last_unconditioned_threat = d_direct

        # 3. Evaluate Conditioned Associative Fear Response y_CR
        # z_j = sum_i (W_fear,ij * x_sens,i) -> shape (d_threat,)
        # y_CR = tanh(1.75 * max(z))
        z_associative = sens_arr @ self.W_fear
        z_max = float(np.max(z_associative)) if z_associative.size > 0 else 0.0
        if z_max > 0.0:
            y_cr = float(np.tanh(1.75 * z_max))
        else:
            y_cr = 0.0
        y_cr = float(np.clip(y_cr, 0.0, 1.0))
        self.last_fear_response = y_cr

        # 4. Comparator Logic: Threat Threshold Evaluation
        alarm_triggered = False
        source = "NONE"

        if d_direct >= self.threat_threshold:
            alarm_triggered = True
            source = "DIRECT_TRAUMA"
        elif y_cr >= self.cond_fear_threshold:
            alarm_triggered = True
            source = "CONDITIONED_REFLEX"
        elif self.panic_latched:
            alarm_triggered = True
            source = "LATCHED"

        # 5. Analog PWM Grounding & Gain Boost Emulation (Circuit Simulation)
        if alarm_triggered:
            self.panic_latched = True
            self.last_alarm_source = source
            pwm_grounded = True
            pwm_duty_cycle = 0.0
            self.pwm_pin_volts = 0.0  # Open-drain short to GND (0.0 V)
            self.pull_down_active = True
            gain_boost = 2.0
            panic_amy = True
            # Drive valence towards negative panic state
            self.valence = float(np.clip(self.valence - 0.2, -1.0, 1.0))
            if d_direct >= self.threat_threshold:
                self.valence = -1.0
        else:
            pwm_grounded = False
            pwm_duty_cycle = 1.0
            self.pwm_pin_volts = 3.3  # Pulled high by 10k resistor to 3.3 V
            self.pull_down_active = False
            gain_boost = 1.0
            panic_amy = False
            self.last_alarm_source = "NONE"

        t_end = time.perf_counter_ns()
        wcet_ms = (t_end - t_start) / 1_000_000.0
        if wcet_ms > self.wcet_target_ms:
            self._wcet_violation_count += 1
        self._record_wcet(wcet_ms)

        return AmygdalaOutput(
            timestamp_ns=curr_ts,
            panic_amy=panic_amy,
            pwm_grounded=pwm_grounded,
            pwm_duty_cycle=pwm_duty_cycle,
            valence=self.valence,
            fear_response=y_cr,
            unconditioned_threat=d_direct,
            gain_boost=gain_boost,
            wcet_ms=wcet_ms,
            alarm_source=source,
        )

    def evaluate_cortical(
        self,
        pfc_context: np.ndarray,
        suppress_panic: bool = False,
        timestamp_ns: Optional[int] = None,
    ) -> float:
        """
        Execute the slower cortico-amygdalar pathway (~200 ms).
        Integrates prefrontal cortex cognitive context, computes emotional valence,
        and provides controlled deliberate panic override when physical danger has cleared.

        Args:
            pfc_context: Cognitive context tensor from prefrontal cortex.
            suppress_panic: Boolean flag from PFC requesting panic reset.
            timestamp_ns: Optional reference timestamp.

        Returns:
            Updated emotional valence A_valencia in [-1.0, 1.0].
        """
        pfc_arr = np.asarray(pfc_context, dtype=np.float64)
        if not np.all(np.isfinite(pfc_arr)):
            # Numerical sanitization
            pfc_arr = np.nan_to_num(pfc_arr, nan=0.0, posinf=1.0, neginf=-1.0)

        # Baseline valence modulation from PFC goal alignment / emotional tone
        pfc_mean = float(np.mean(pfc_arr)) if pfc_arr.size > 0 else 0.0
        cognitive_valence = float(np.tanh(pfc_mean))

        # Modulate valence with residual conditioned fear and unconditioned danger
        total_valence = cognitive_valence - 0.7 * self.last_fear_response - 0.9 * self.last_unconditioned_threat
        self.valence = float(np.clip(total_valence, -1.0, 1.0))

        # Cortical suppress panic: allowed ONLY IF active direct trauma is below threshold
        if suppress_panic and self.last_unconditioned_threat < self.threat_threshold:
            self.panic_latched = False
            self.last_alarm_source = "NONE"

        return self.valence

    def pair_conditioning(
        self,
        x_cs: np.ndarray,
        x_us: np.ndarray,
        delta_tau_ms: float = 20.0,
    ) -> float:
        """
        Apply Spike-Timing-Dependent Plasticity (STDP) associative fear conditioning.

        Formula:
            Delta W_fear,ij = gamma * x_CS,i(t - Delta tau) * x_US,j(t) * exp(-Delta tau / tau_stdp)
            for 0 < Delta tau <= 50 ms.
            Strict causality: Delta tau <= 0 or Delta tau > 50 ms produces ZERO update.

        Args:
            x_cs: Conditioned Stimulus feature vector (shape d_sens,).
            x_us: Unconditioned Stimulus trauma vector (shape d_threat,).
            delta_tau_ms: Temporal precedence delay (t_US - t_CS) in milliseconds.

        Returns:
            Updated maximum synaptic fear weight max(W_fear) in [0.0, 1.0].
        """
        # Strict biological temporal causality window enforcement
        if delta_tau_ms <= 0.0 or delta_tau_ms > self.max_delta_tau_ms:
            return float(np.max(self.W_fear)) if self.W_fear.size > 0 else 0.0

        cs_arr = np.asarray(x_cs, dtype=np.float64)
        us_arr = np.asarray(x_us, dtype=np.float64)

        if not np.all(np.isfinite(cs_arr)) or not np.all(np.isfinite(us_arr)):
            return float(np.max(self.W_fear)) if self.W_fear.size > 0 else 0.0

        # Dimensionality alignment
        if cs_arr.shape[0] != self.d_sens:
            padded_cs = np.zeros(self.d_sens, dtype=np.float64)
            copy_cs = min(cs_arr.shape[0], self.d_sens)
            padded_cs[:copy_cs] = cs_arr[:copy_cs]
            cs_arr = padded_cs

        if us_arr.shape[0] != self.d_threat:
            padded_us = np.zeros(self.d_threat, dtype=np.float64)
            copy_us = min(us_arr.shape[0], self.d_threat)
            padded_us[:copy_us] = us_arr[:copy_us]
            us_arr = padded_us

        # STDP Exponential Temporal Weighting Factor
        exp_decay = math.exp(-delta_tau_ms / self.tau_stdp_ms)

        # Outer product Hebbian co-activation: (d_sens, 1) x (1, d_threat) -> (d_sens, d_threat)
        delta_w = self.gamma_stdp * exp_decay * np.outer(cs_arr, us_arr)

        # Update synaptic matrix with saturation boundary [0.0, 1.0]
        self.W_fear = np.clip(self.W_fear + delta_w, 0.0, 1.0)

        # 3-Pairing Pavlovian Consolidation Tracker
        self.consecutive_pairings += 1
        if self.consecutive_pairings >= self.consolidation_threshold:
            self.is_consolidated = True

        return float(np.max(self.W_fear))

    def apply_extinction(self, decay_rate: Optional[float] = None) -> None:
        """
        Apply unreinforced fear extinction dynamics to synaptic weights.
        W_fear(t+1) = W_fear(t) * (1 - lambda_extinct).

        Args:
            decay_rate: Optional override for extinction rate lambda_extinct (default 0.05).
        """
        rate = float(decay_rate) if decay_rate is not None else self.lambda_extinct
        rate = float(np.clip(rate, 0.0, 1.0))
        self.W_fear *= (1.0 - rate)

        # Also decay residual conditioned fear response
        self.last_fear_response *= (1.0 - rate)

        # Deconsolidate if fear response decays below threshold
        self.consecutive_pairings = max(0, self.consecutive_pairings - 1)
        if self.consecutive_pairings < self.consolidation_threshold:
            self.is_consolidated = False

    def reset_panic(self) -> None:
        """
        Directly reset the panic latch and restore normal motor pass-through.
        """
        self.panic_latched = False
        self.last_alarm_source = "NONE"
        self.pwm_pin_volts = 3.3
        self.pull_down_active = False

    def get_wcet_stats(self) -> Dict[str, float]:
        """
        Compute statistical distribution of execution latency for the fast pathway.

        Returns:
            Dictionary with wcet_max_ms, wcet_mean_ms, wcet_min_ms, sample_count,
            wcet_target_ms, and target_met boolean flag.
        """
        if not self._wcet_samples:
            return {
                "wcet_max_ms": 0.0,
                "wcet_mean_ms": 0.0,
                "wcet_min_ms": 0.0,
                "sample_count": 0.0,
                "wcet_target_ms": self.wcet_target_ms,
                "target_met": 1.0,
            }

        samples = np.array(self._wcet_samples, dtype=np.float64)
        max_val = float(np.max(samples))
        mean_val = float(np.mean(samples))
        min_val = float(np.min(samples))

        return {
            "wcet_max_ms": max_val,
            "wcet_mean_ms": mean_val,
            "wcet_min_ms": min_val,
            "sample_count": float(len(samples)),
            "wcet_target_ms": self.wcet_target_ms,
            "target_met": 1.0 if max_val <= self.wcet_target_ms else 0.0,
        }

    def _record_wcet(self, elapsed_ms: float) -> None:
        """Record an execution latency sample into the rolling history."""
        if len(self._wcet_samples) >= self._max_samples:
            self._wcet_samples.pop(0)
        self._wcet_samples.append(elapsed_ms)
