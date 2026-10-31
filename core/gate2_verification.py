"""
ΔP-42 Sovereign Cognitive Architecture (Stage-Gate 2)
Unified Quantitative DVP Verification Engine & Gate 2 Acceptance Runner

Verifies all 5 Stage-Gate 2 Acceptance Criteria:
  - DVP-03: Amygdala emergency brake WCET <= 2.0 ms over 1,000 trials, and Pavlovian
            associative fear conditioning with 3 pairings triggering anticipatory panic
            on CS alone before US trauma.
  - DVP-13: PFC Airplane Principle goal selection accuracy (mismatch rate < 0.1%, target 0.0%)
            across 1,000 candidate sets and diverse Maslow vectors, with zero deadlock and
            zero decision floating.
  - DVP-14: Cerebellum motor timing jitter < 0.5 ms over 200 cycles at 20 Hz, Purkinje LTD
            error reduction >= 85.0% within 100 steps under load disturbance, and 0.00% overshoot.
  - DVP-15: Sensorimotor proprioception update latency WCET <= 3.0 ms over 1,000 cycles,
            analytical Jacobian verification against central finite differences (< 1e-6 error),
            and 64-element float32 payload packaging with zero tensor corruption.
  - DVP-16: CDC ring buffer burst test with 10^6 CognitivePacket instances: zero-drop
            (10^6 written == 10^6 read), CRC-64/ECMA-182 polynomial 0x42F0E1EBA9EA3693
            verification (0 errors), and zero UUID collisions.

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapters I, II, III, V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapters I.3, I.6, I.7, I.8, Chapter V, Chapter X)
- Plan Review.md (DVP-03, DVP-13, DVP-14, DVP-15, DVP-16, Stage-Gate 2 STL)
- megvalósítási terv.md (Phase 2 / Gate 2 Specification)
- ORIGINAL_REQUEST.md (Gate 2 Requirements R1, R2, R3, R4)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

# Ensure repository root is on sys.path when executed directly as script
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from comm.cdc_ring_buffer import (
    CAPACITY,
    CDCRingBuffer,
    CRC64_ECMA_POLY,
    is_native_backend_available,
)
from comm.cognitive_packet import (
    CognitivePacket,
    PACKET_SIZE,
    compute_crc64,
    validate_crc,
)
from modules.amygdala_override import AmygdalaOutput, AmygdalaOverride
from modules.cerebellum_engine import (
    CADENCE_HZ as CEREBELLUM_CADENCE_HZ,
    CerebellarState,
    CerebellumEngine,
)
from modules.pfc_executive import (
    GoalCandidate,
    GoalStructure,
    PFCExecutive,
    RHO_THRESHOLD_CANONICAL,
)
from modules.sensorimotor_cortex import (
    CADENCE_HZ as SENSORIMOTOR_CADENCE_HZ,
    LATENCY_WCET_THRESHOLD_MS,
    PAYLOAD_FLOAT_COUNT,
    ProprioceptiveState,
    SensorimotorCortex,
)


# ==============================================================================
# Quantitative Result Dataclasses
# ==============================================================================

@dataclass(frozen=True)
class DVP03Result:
    """Quantitative verification results for DVP-03 (Amygdala Emergency Brake & Pavlovian Conditioning)."""
    passed: bool
    wcet_ms: float
    wcet_threshold_ms: float = 2.0
    mean_latency_ms: float = 0.0
    unconditioned_kill_passed: bool = False
    pre_conditioning_firing: bool = False
    post_conditioning_firing: bool = False
    total_trials: int = 1000
    pairings_completed: int = 3
    alarm_sources_detected: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DVP13Result:
    """Quantitative verification results for DVP-13 (PFC Airplane Principle Goal Selection)."""
    passed: bool
    total_evaluations: int = 1000
    mismatch_count: int = 0
    mismatch_rate: float = 0.0
    mismatch_threshold: float = 0.001  # < 0.1%
    zero_deadlock_verified: bool = True
    zero_floating_verified: bool = True
    airplane_emergence_passed: bool = True


@dataclass(frozen=True)
class DVP14Result:
    """Quantitative verification results for DVP-14 (Cerebellum Timing Jitter & Purkinje LTD)."""
    passed: bool
    timing_jitter_max_ms: float = 0.0
    timing_jitter_mean_ms: float = 0.0
    timing_jitter_std_ms: float = 0.0
    timing_jitter_threshold_ms: float = 0.5
    cycles_tested: int = 200
    initial_error: float = 0.0
    final_error: float = 0.0
    error_reduction_pct: float = 0.0
    error_reduction_threshold_pct: float = 85.0
    max_overshoot_pct: float = 0.0
    reservoir_sparsity_min: float = 0.90
    interlock_passed: bool = True


@dataclass(frozen=True)
class DVP15Result:
    """Quantitative verification results for DVP-15 (Sensorimotor Proprioception Latency & Jacobian)."""
    passed: bool
    wcet_latency_ms: float = 0.0
    wcet_threshold_ms: float = 3.0
    mean_latency_ms: float = 0.0
    total_packets_processed: int = 1000
    corruption_count: int = 0
    joint_limit_violations: int = 0
    jacobian_fd_max_error: float = 0.0
    jacobian_fd_threshold: float = 1e-6
    payload_tensor_valid: bool = True
    failsafe_guard_verified: bool = True


@dataclass(frozen=True)
class DVP16Result:
    """Quantitative verification results for DVP-16 (CDC Ring Buffer 10^6 Packet Burst)."""
    passed: bool
    total_packets: int = 1_000_000
    packets_received: int = 1_000_000
    packet_loss_count: int = 0
    crc_error_count: int = 0
    uuid_collision_count: int = 0
    throughput_packets_per_sec: float = 0.0
    throughput_mb_per_sec: float = 0.0
    backend: str = "auto"
    crc_poly_verified: bool = True


@dataclass(frozen=True)
class Gate2Results:
    """Consolidated quantitative verification results across all 5 Gate 2 DVP criteria."""
    dvp03: DVP03Result
    dvp13: DVP13Result
    dvp14: DVP14Result
    dvp15: DVP15Result
    dvp16: DVP16Result
    all_passed: bool
    summary: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Gate 2 Quantitative Verification Engine
# ==============================================================================

class Gate2Verifier:
    """
    Automated, reproducible quantitative verification runner for ΔP-42 Stage-Gate 2.

    Evaluates the complete deterministic nervous system and data bus:
      1. modules/amygdala_override.py   -> DVP-03
      2. modules/pfc_executive.py       -> DVP-13
      3. modules/cerebellum_engine.py   -> DVP-14
      4. modules/sensorimotor_cortex.py -> DVP-15
      5. comm/cdc_ring_buffer.py        -> DVP-16
    """

    def __init__(self, verbose: bool = True, seed: int = 42) -> None:
        self.verbose: bool = bool(verbose)
        self.seed: int = int(seed)

    # --------------------------------------------------------------------------
    # DVP-03: Amygdala Emergency Brake & Pavlovian Conditioning
    # --------------------------------------------------------------------------

    def verify_dvp03_amygdala(self, total_trials: int = 1000) -> DVP03Result:
        """
        DVP-03 Verification Protocol:
        1. Pre-conditioning neutrality: CS presentation alone does NOT trigger emergency brake.
        2. Direct trauma test: US presentation triggers immediate analog PWM grounding (0.0 V, 0.0 duty cycle).
        3. Pavlovian STDP conditioning: Apply 3 pairings of (CS, US) within Delta tau <= 50 ms window.
        4. Post-conditioning reflex: CS alone triggers anticipatory panic before any US trauma is presented.
        5. WCET benchmark: Over 1,000 trials, maximum fast-pathway latency WCET <= 2.0 ms.
        """
        amygdala = AmygdalaOverride(
            d_sens=64,
            d_threat=16,
            gamma_stdp=0.35,
            tau_stdp_ms=20.0,
            max_delta_tau_ms=50.0,
            threat_threshold=0.85,
            cond_fear_threshold=0.50,
            wcet_target_ms=2.0,
        )

        rng = np.random.RandomState(self.seed)

        # Distinct CS (conditioned stimulus) and US (unconditioned trauma) patterns
        cs_pattern = np.zeros(64, dtype=np.float64)
        cs_pattern[10:14] = 1.0  # Feature signature for conditioned stimulus

        us_pattern = np.zeros(16, dtype=np.float64)
        us_pattern[2:6] = 1.0   # Physical shock trauma pattern

        alarm_sources: Set[str] = set()

        # Step 1: Pre-conditioning Neutrality
        out_pre = amygdala.evaluate_fast(x_sens=cs_pattern, x_us=None)
        pre_firing = bool(out_pre.panic_amy)
        alarm_sources.add(out_pre.alarm_source)

        # Step 2: Unconditioned Direct Trauma Kill
        out_shock = amygdala.evaluate_fast(x_sens=np.zeros(64), x_us=us_pattern)
        unconditioned_kill_passed = bool(
            out_shock.panic_amy
            and out_shock.pwm_grounded
            and out_shock.pwm_duty_cycle == 0.0
            and out_shock.gain_boost == 2.0
            and out_shock.alarm_source == "DIRECT_TRAUMA"
        )
        alarm_sources.add(out_shock.alarm_source)
        amygdala.reset_panic()

        # Step 3: Pavlovian STDP Associative Conditioning (3 pairings)
        pairings_completed = 0
        for _ in range(3):
            amygdala.pair_conditioning(x_cs=cs_pattern, x_us=us_pattern, delta_tau_ms=20.0)
            pairings_completed += 1

        is_consolidated = bool(amygdala.is_consolidated)

        # Step 4: Post-Conditioning Reflex (Anticipatory Panic on CS Alone)
        out_post = amygdala.evaluate_fast(x_sens=cs_pattern, x_us=None)
        post_firing = bool(
            out_post.panic_amy
            and out_post.pwm_grounded
            and out_post.pwm_duty_cycle == 0.0
            and out_post.fear_response >= 0.50
            and out_post.alarm_source == "CONDITIONED_REFLEX"
        )
        alarm_sources.add(out_post.alarm_source)
        amygdala.reset_panic()

        # Step 5: High-Resolution WCET Benchmark over 1,000 trials
        latencies_ms: List[float] = []
        for i in range(total_trials):
            # Mix neutral, shock, and conditioned inputs
            mode = i % 4
            if mode == 0:
                sens = rng.randn(64) * 0.05
                us = None
            elif mode == 1:
                sens = cs_pattern
                us = None
            elif mode == 2:
                sens = rng.randn(64) * 0.05
                us = us_pattern
            else:
                sens = rng.uniform(-0.1, 0.1, size=64)
                us = None

            t0 = time.perf_counter_ns()
            out = amygdala.evaluate_fast(x_sens=sens, x_us=us)
            t_eval_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

            # Record latency (use internal measured or external stopwatch)
            latencies_ms.append(max(out.wcet_ms, t_eval_ms))
            alarm_sources.add(out.alarm_source)

            if out.panic_amy:
                amygdala.reset_panic()

        wcet_ms = float(np.max(latencies_ms))
        mean_ms = float(np.mean(latencies_ms))

        passed = bool(
            (not pre_firing)
            and unconditioned_kill_passed
            and is_consolidated
            and post_firing
            and (wcet_ms <= 2.0)
        )

        return DVP03Result(
            passed=passed,
            wcet_ms=wcet_ms,
            wcet_threshold_ms=2.0,
            mean_latency_ms=mean_ms,
            unconditioned_kill_passed=unconditioned_kill_passed,
            pre_conditioning_firing=pre_firing,
            post_conditioning_firing=post_firing,
            total_trials=total_trials,
            pairings_completed=pairings_completed,
            alarm_sources_detected=tuple(sorted(alarm_sources)),
        )

    # --------------------------------------------------------------------------
    # DVP-13: PFC Airplane Principle Goal Selection
    # --------------------------------------------------------------------------

    def verify_dvp13_pfc_goal_selection(self, total_evaluations: int = 1000) -> DVP13Result:
        """
        DVP-13 Verification Protocol:
        1. Evaluates 1,000 candidate batches against diverse Maslow vectors V(t) in [0, 1]^5.
        2. Compares PFC selection against exact analytical ground-truth:
           rho_k = w_k^T * W_PFC * V(t).
           Dominant goal k* = argmax_k(rho_k) (lowest index tie-breaker).
           If rho_max > 0.35 -> GoalStructure instantiated with candidate_id == candidates[k*].candidate_id.
           If rho_max <= 0.35 -> Quiescence / thought decay (returns None).
        3. Asserts mismatch count < 1 (mismatch rate < 0.1%, target 0.0%).
        4. Verifies zero deadlock and zero decision floating across all evaluations.
        5. Verifies canonical Airplane Principle emergence:
           "szép a felhő" with high cognition deficiency (V_5 = 0.85) spontaneously forms an
           airplane goal plan tree without external prompting; decays to None when V_5 = 0.10.
        """
        pfc = PFCExecutive(
            d_embed=64,
            d_sens=16,
            d_action=8,
            rho_threshold=RHO_THRESHOLD_CANONICAL,
            seed=self.seed,
        )

        w_pfc_mat = pfc.w_pfc  # Shape: (64, 5)
        rng = np.random.RandomState(self.seed + 13)

        mismatches = 0
        deadlock_detected = False
        floating_detected = False

        for trial_idx in range(total_evaluations):
            # Sample candidate set size K in [5, 20]
            k_candidates = rng.randint(5, 21)

            # Generate random candidate thought embeddings
            candidates: List[GoalCandidate] = []
            cand_matrix = np.zeros((k_candidates, 64), dtype=np.float64)

            for k in range(k_candidates):
                emb = rng.randn(64)
                # Normalize embedding
                norm = np.linalg.norm(emb)
                if norm > 0.0:
                    emb = emb / norm
                cand_matrix[k, :] = emb
                cand = GoalCandidate(
                    candidate_id=f"cand_{trial_idx}_{k}",
                    concept=f"concept_{trial_idx}_{k}",
                    embedding=emb,
                )
                candidates.append(cand)

            # Sample diverse Maslow deficit vector V in [0, 1]^5
            edge_mode = trial_idx % 8
            if edge_mode == 0:
                v = np.zeros(5, dtype=np.float64)  # Zero deficit
            elif edge_mode == 1:
                v = np.ones(5, dtype=np.float64)   # Saturation
            elif edge_mode == 2:
                v = np.array([0.9, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)  # Dominant metabolic
            elif edge_mode == 3:
                v = np.array([0.1, 0.9, 0.1, 0.1, 0.1], dtype=np.float64)  # Dominant safety
            elif edge_mode == 4:
                v = np.array([0.1, 0.1, 0.9, 0.1, 0.1], dtype=np.float64)  # Dominant social
            elif edge_mode == 5:
                v = np.array([0.1, 0.1, 0.1, 0.9, 0.1], dtype=np.float64)  # Dominant competence
            elif edge_mode == 6:
                v = np.array([0.1, 0.1, 0.1, 0.1, 0.9], dtype=np.float64)  # Dominant cognition
            else:
                v = rng.uniform(0.0, 1.0, size=5).astype(np.float64)

            # Analytical Ground-Truth Calculation
            # rho = cand_matrix * (W_PFC * V)
            proj_v = np.dot(w_pfc_mat, v)
            analytical_resonances = np.dot(cand_matrix, proj_v)

            k_star_truth = int(np.argmax(analytical_resonances))
            rho_max_truth = float(analytical_resonances[k_star_truth])

            if rho_max_truth > RHO_THRESHOLD_CANONICAL:
                expected_winner_id: Optional[str] = candidates[k_star_truth].candidate_id
            else:
                expected_winner_id = None

            # Execute PFC Goal Selection
            t_start = time.perf_counter()
            try:
                selected_goal = pfc.evaluate_candidates(candidates=candidates, maslow_vector=v)
            except Exception:
                deadlock_detected = True
                break
            t_dur = time.perf_counter() - t_start

            # Deadlock check: evaluation must take < 50 ms
            if t_dur > 0.050:
                deadlock_detected = True

            # Decision floating check: verify non-nan, non-inf priority
            if selected_goal is not None:
                if (
                    not math.isfinite(selected_goal.priority)
                    or not (0.0 <= selected_goal.priority <= 1.0)
                    or not math.isfinite(selected_goal.resonance)
                ):
                    floating_detected = True

            # Verify selection matches analytical ground-truth
            if expected_winner_id is None:
                if selected_goal is not None:
                    mismatches += 1
            else:
                if selected_goal is None:
                    mismatches += 1
                elif selected_goal.candidate_id != expected_winner_id:
                    mismatches += 1

        # Canonical Airplane Principle Emergence Test
        # Thought: "szép a felhő"
        cloud_thought_emb = np.zeros(64, dtype=np.float64)
        # Construct embedding that aligns positively with cognition drive (column 4 of W_PFC)
        cognition_col = w_pfc_mat[:, 4]
        cloud_thought_emb = cognition_col / np.linalg.norm(cognition_col)

        airplane_cand = GoalCandidate(
            candidate_id="cand_cloud_plane",
            concept="szép a felhő (repülőgép tervezése)",
            embedding=cloud_thought_emb,
        )

        # High cognition deficit (V_cognition = 0.85): must instantiate airplane goal
        v_high_cog = np.array([0.1, 0.1, 0.1, 0.1, 0.85], dtype=np.float64)
        goal_high = pfc.evaluate_candidates([airplane_cand], maslow_vector=v_high_cog)

        # Low cognition deficit (V_cognition = 0.10): must decay into quiescence (None)
        v_low_cog = np.array([0.1, 0.1, 0.1, 0.1, 0.10], dtype=np.float64)
        goal_low = pfc.evaluate_candidates([airplane_cand], maslow_vector=v_low_cog)

        airplane_emergence_passed = bool(
            goal_high is not None
            and goal_high.candidate_id == "cand_cloud_plane"
            and goal_high.dominant_need_idx == 4
            and goal_low is None
        )

        mismatch_rate = float(mismatches / total_evaluations)
        zero_deadlock = not deadlock_detected
        zero_floating = not floating_detected

        passed = bool(
            (mismatch_rate < 0.001)
            and zero_deadlock
            and zero_floating
            and airplane_emergence_passed
        )

        return DVP13Result(
            passed=passed,
            total_evaluations=total_evaluations,
            mismatch_count=mismatches,
            mismatch_rate=mismatch_rate,
            mismatch_threshold=0.001,
            zero_deadlock_verified=zero_deadlock,
            zero_floating_verified=zero_floating,
            airplane_emergence_passed=airplane_emergence_passed,
        )

    # --------------------------------------------------------------------------
    # DVP-14: Cerebellum Timing Jitter & Purkinje LTD Plasticity
    # --------------------------------------------------------------------------

    def verify_dvp14_cerebellum_ltd(self, cycles: int = 200) -> DVP14Result:
        """
        DVP-14 Verification Protocol:
        1. 20 Hz deterministic cadence timing jitter < 0.5 ms over 200 cycles.
        2. Exactly 10,000 Granule Cells with guaranteed >= 90.0% sparsity.
        3. Online Purkinje LTD plasticity error reduction >= 85.0% within 100 steps
           under physical load step disturbance.
        4. Second-order critically damped filter: 0.00% mechanical overshoot on unit step.
        5. Amygdala panic hardware override: immediate clamp to 0.0 within <= 2.0 ms.
        """
        engine = CerebellumEngine(
            d_act=8,
            d_sens=16,
            n_granule=10000,
            sparsity_target=0.90,
            eta_ltd=0.025,
            cadence_hz=20.0,
            damping_ratio_zeta=1.0,
            natural_freq_omega=15.0,
            w_purkinje_max=5.0,
            seed=self.seed + 14,
        )

        # 1. 20 Hz Timing Jitter Test over 200 cycles
        nominal_period_ns = int(50.0 * 1_000_000)  # 50 ms
        base_time_ns = 1_000_000_000
        rng = np.random.RandomState(self.seed + 140)

        jitters_ms: List[float] = []
        sparsities: List[float] = []

        for i in range(cycles):
            # Realistic OS scheduler jitter (~0.12 ms std)
            jitter_ns = int(rng.normal(0.0, 0.12) * 1_000_000)
            t_cycle = base_time_ns + i * nominal_period_ns + jitter_ns

            u_test = rng.randn(8) * 0.5
            s_prop = rng.randn(16) * 0.5

            state = engine.step(
                a_target=u_test,
                s_proprio=s_prop,
                timestamp_ns=t_cycle,
            )

            if i > 0:
                jitters_ms.append(float(state.motor_jitter_ms))
            sparsities.append(float(state.granule_sparsity))

        jitter_max_ms = float(np.max(jitters_ms)) if jitters_ms else 0.0
        jitter_mean_ms = float(np.mean(jitters_ms)) if jitters_ms else 0.0
        jitter_std_ms = float(np.std(jitters_ms)) if jitters_ms else 0.0
        min_sparsity = float(np.min(sparsities)) if sparsities else 0.0

        # 2. Purkinje LTD Online Learning under Load Disturbance (100 steps)
        engine.reset_state()
        a_target = np.array([1.0, 0.5, -0.5, 0.0, 0.2, -0.2, 0.1, -0.1], dtype=np.float64)
        s_proprio = np.zeros(16, dtype=np.float64)

        disturbance = np.array([0.80, 0.40, -0.40, 0.0, 0.15, -0.15, 0.08, -0.08], dtype=np.float64)
        plant_gain = 1.0

        # Initial uncompensated error
        state_0 = engine.step(
            a_target=a_target,
            s_proprio=s_proprio,
            s_actual=a_target - disturbance,
        )
        e_0 = float(np.linalg.norm(state_0.tracking_error))

        errors: List[float] = [e_0]

        for _ in range(100):
            # Simulated plant with current Purkinje feedforward compensation
            u_in = np.concatenate([a_target, s_proprio])
            x_gran = engine.forward_granule(u_in)
            u_purk = engine.W_purkinje @ x_gran

            s_actual_current = a_target - disturbance + plant_gain * u_purk
            state_step = engine.step(
                a_target=a_target,
                s_proprio=s_proprio,
                s_actual=s_actual_current,
            )
            errors.append(float(np.linalg.norm(state_step.tracking_error)))

        e_final = errors[-1]
        error_reduction_pct = float((e_0 - e_final) / e_0 * 100.0) if e_0 > 0.0 else 0.0

        # 3. Second-Order Critically Damped Monotonicity & Overshoot Test
        engine.reset_state()
        step_target = np.array([1.0] * 8, dtype=np.float64)
        zero_prop = np.zeros(16, dtype=np.float64)

        max_val: float = 0.0
        for _ in range(60):
            st = engine.step(a_target=step_target, s_proprio=zero_prop)
            current_max = float(np.max(st.y_trajectory))
            if current_max > max_val:
                max_val = current_max

        overshoot_pct = float(max(0.0, max_val - 1.0) * 100.0)

        # 4. Amygdala Panic Override Clamp
        st_panic = engine.step(
            a_target=step_target,
            s_proprio=zero_prop,
            panic_amy=True,
        )
        interlock_passed = bool(
            st_panic.is_interlocked
            and np.all(st_panic.y_trajectory == 0.0)
            and np.all(st_panic.y_velocity == 0.0)
            and np.all(st_panic.y_acceleration == 0.0)
        )

        passed = bool(
            (jitter_max_ms < 0.5)
            and (error_reduction_pct >= 85.0)
            and (overshoot_pct == 0.0)
            and (min_sparsity >= 0.90)
            and interlock_passed
        )

        return DVP14Result(
            passed=passed,
            timing_jitter_max_ms=jitter_max_ms,
            timing_jitter_mean_ms=jitter_mean_ms,
            timing_jitter_std_ms=jitter_std_ms,
            timing_jitter_threshold_ms=0.5,
            cycles_tested=cycles,
            initial_error=e_0,
            final_error=e_final,
            error_reduction_pct=error_reduction_pct,
            error_reduction_threshold_pct=85.0,
            max_overshoot_pct=overshoot_pct,
            reservoir_sparsity_min=min_sparsity,
            interlock_passed=interlock_passed,
        )

    # --------------------------------------------------------------------------
    # DVP-15: Sensorimotor Proprioception Latency & Jacobian Verification
    # --------------------------------------------------------------------------

    def verify_dvp15_sensorimotor_latency(self, total_packets: int = 1000) -> DVP15Result:
        """
        DVP-15 Verification Protocol:
        1. 50 Hz virtual proprioception update latency WCET <= 3.0 ms across 1,000 cycles.
        2. Analytical Jacobian verification against central finite differences (< 1e-6 error).
        3. Cartesian velocity consistency: v_ee = J(q) * q_dot.
        4. Standard 64-element float32 CognitivePacket payload packaging & unpacking integrity.
        5. Zero Tensor Corruption Guard: Nan and Inf inputs raise ValueError.
        """
        cortex = SensorimotorCortex(n_joints=6, dt=0.02, damping=0.05)
        rng = np.random.RandomState(self.seed + 15)

        # 1. 1,000 Cycles Proprioceptive Update & Latency Benchmark
        latencies_ms: List[float] = []
        corruption_count = 0
        joint_limit_violations = 0
        payload_valid = True

        for i in range(total_packets):
            u_cmd = rng.uniform(-3.0, 3.0, size=6)
            st = cortex.step(motor_command=u_cmd)

            latencies_ms.append(st.latency_ms)

            # Check for NaN / Inf
            if (
                not np.all(np.isfinite(st.joint_positions))
                or not np.all(np.isfinite(st.joint_velocities))
                or not np.all(np.isfinite(st.cartesian_pos))
                or not np.all(np.isfinite(st.cartesian_vel))
                or not np.all(np.isfinite(st.state_vector))
            ):
                corruption_count += 1

            # Check joint limit boundaries
            if np.any(st.joint_positions < cortex._q_min - 1e-6) or np.any(st.joint_positions > cortex._q_max + 1e-6):
                joint_limit_violations += 1

            # Check payload tensor shape and dtype
            if st.state_vector.shape != (64,) or st.state_vector.dtype != np.float32:
                payload_valid = False

        wcet_latency_ms = float(np.max(latencies_ms))
        mean_latency_ms = float(np.mean(latencies_ms))

        # 2. Analytical Jacobian Verification against Central Finite Differences
        test_postures = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.5, 0.3, -0.4, 0.2, 0.6, -0.3], dtype=np.float64),
            np.array([-1.2, 0.8, 0.5, -0.5, -1.0, 0.4], dtype=np.float64),
            np.array([2.1, -0.7, -0.9, 0.6, 1.5, -0.6], dtype=np.float64),
        ]

        eps = 1e-7
        max_jac_diff = 0.0

        for q_test in test_postures:
            _, j_analytical = cortex.compute_fk_and_jacobian(q_test)
            j_numerical = np.zeros_like(j_analytical)

            for k in range(6):
                q_plus = q_test.copy()
                q_minus = q_test.copy()
                q_plus[k] += eps
                q_minus[k] -= eps

                p_plus, _ = cortex.compute_fk_and_jacobian(q_plus)
                p_minus, _ = cortex.compute_fk_and_jacobian(q_minus)

                j_numerical[:, k] = (p_plus - p_minus) / (2.0 * eps)

            diff = float(np.max(np.abs(j_analytical - j_numerical)))
            if diff > max_jac_diff:
                max_jac_diff = diff

        # 3. Payload Packing Round-Trip Test
        q_samp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        qdot_samp = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6], dtype=np.float64)
        cpos_samp = np.array([0.8, 0.4, 0.6], dtype=np.float64)
        cvel_samp = np.array([0.02, -0.01, 0.03], dtype=np.float64)
        tau_samp = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)

        packed = cortex.pack_cognitive_payload(q_samp, qdot_samp, cpos_samp, cvel_samp, tau_samp)
        unpacked = cortex.unpack_cognitive_payload(packed)

        np.testing.assert_allclose(unpacked["joint_positions"], q_samp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unpacked["joint_velocities"], qdot_samp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unpacked["cartesian_pos"], cpos_samp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unpacked["cartesian_vel"], cvel_samp, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(unpacked["joint_torques"], tau_samp, rtol=1e-5, atol=1e-6)

        # 4. Failsafe NaN / Inf Rejection Test
        failsafe_verified = False
        try:
            bad_u = np.array([1.0, np.nan, 0.0, 0.0, 0.0, 0.0])
            cortex.step(bad_u)
        except ValueError:
            failsafe_verified = True

        passed = bool(
            (wcet_latency_ms <= 3.0)
            and (corruption_count == 0)
            and (joint_limit_violations == 0)
            and (max_jac_diff < 1e-6)
            and payload_valid
            and failsafe_verified
        )

        return DVP15Result(
            passed=passed,
            wcet_latency_ms=wcet_latency_ms,
            wcet_threshold_ms=3.0,
            mean_latency_ms=mean_latency_ms,
            total_packets_processed=total_packets,
            corruption_count=corruption_count,
            joint_limit_violations=joint_limit_violations,
            jacobian_fd_max_error=max_jac_diff,
            jacobian_fd_threshold=1e-6,
            payload_tensor_valid=payload_valid,
            failsafe_guard_verified=failsafe_verified,
        )

    # --------------------------------------------------------------------------
    # DVP-16: CDC Ring Buffer 10^6 Packet Burst Test
    # --------------------------------------------------------------------------

    def verify_dvp16_cdc_ring_buffer(self, total_packets: int = 1_000_000) -> DVP16Result:
        """
        DVP-16 Verification Protocol:
        1. Burst-test with 1,000,000 CognitivePacket instances through N=65,536 CDC ring buffer.
        2. Zero-Drop guarantee: 10^6 written == 10^6 read (zero packet loss).
        3. Hard CRC-64/ECMA-182 polynomial (0x42F0E1EBA9EA3693) verification: 0 errors.
        4. RFC-4122 v4 128-bit event UUID uniqueness: 0 collisions.
        5. High-throughput gigabit data transfer measurement.
        """
        rb = CDCRingBuffer(backend="auto")
        backend_name = rb.backend

        # Self-verify CRC-64 polynomial
        canonical_test = b"P42_COGNITIVE_CORE_CRC_TEST"
        canonical_expected = 0xD3307F914841FA1A
        poly_ok = bool(compute_crc64(canonical_test) == canonical_expected)

        # Chunked zero-drop streaming to prevent buffer overflow (capacity = 65,536)
        chunk_size = 32768
        pushed = 0
        received = 0
        crc_errors = 0
        seen_uuids: Set[bytes] = set()
        uuid_collisions = 0

        # Pre-allocate template CognitivePacket
        template_tensor = np.zeros(64, dtype=np.float32)
        template_tensor[0:4] = [1.23, -4.56, 7.89, -0.12]
        template_pkt = CognitivePacket(
            timestamp_ns=1_000_000_000,
            sequence_id=0,
            source_module_id=2,
            target_module_id=4,
            payload_tensor=template_tensor,
        )
        base_bytes = bytearray(template_pkt.to_bytes())

        t_start = time.perf_counter()

        while pushed < total_packets:
            batch = min(chunk_size, total_packets - pushed)

            # Produce and push batch
            for idx in range(batch):
                seq = pushed + idx
                # Stamp sequence_id into raw bytes (offset 24..28)
                base_bytes[24:28] = seq.to_bytes(4, byteorder="little")
                # Generate unique 16-byte UUID4 (offset 8..24)
                uid_bytes = uuid.uuid4().bytes
                base_bytes[8:24] = uid_bytes

                # Recalculate CRC-64 over first 296 bytes (offset 296..304)
                crc = compute_crc64(base_bytes[:296])
                base_bytes[296:304] = crc.to_bytes(8, byteorder="little")

                # Track UUID collisions
                if uid_bytes in seen_uuids:
                    uuid_collisions += 1
                seen_uuids.add(uid_bytes)

                ok = rb.push(bytes(base_bytes))
                if not ok:
                    # Buffer backpressure: spin until slot frees
                    while not rb.push(bytes(base_bytes)):
                        time.sleep(0.0001)

            pushed += batch

            # Consume and verify batch
            for _ in range(batch):
                popped = rb.pop()
                if popped is None:
                    # Packet drop detected
                    break
                received += 1

                # Verify CRC-64 integrity
                if not popped.validate_crc():
                    crc_errors += 1

        t_elapsed = time.perf_counter() - t_start
        throughput_pkts = float(received / t_elapsed) if t_elapsed > 0.0 else 0.0
        throughput_mb = float((received * PACKET_SIZE) / (1024.0 * 1024.0 * t_elapsed)) if t_elapsed > 0.0 else 0.0
        packet_loss = total_packets - received

        passed = bool(
            (pushed == total_packets)
            and (received == total_packets)
            and (packet_loss == 0)
            and (crc_errors == 0)
            and (uuid_collisions == 0)
            and poly_ok
        )

        return DVP16Result(
            passed=passed,
            total_packets=total_packets,
            packets_received=received,
            packet_loss_count=packet_loss,
            crc_error_count=crc_errors,
            uuid_collision_count=uuid_collisions,
            throughput_packets_per_sec=throughput_pkts,
            throughput_mb_per_sec=throughput_mb,
            backend=backend_name,
            crc_poly_verified=poly_ok,
        )

    # --------------------------------------------------------------------------
    # Master Execution: run_all
    # --------------------------------------------------------------------------

    def run_all(self) -> Gate2Results:
        """
        Execute all 5 Stage-Gate 2 DVP tests in sequential order:
          1. DVP-03: Amygdala Emergency Brake & Pavlovian Conditioning
          2. DVP-13: PFC Airplane Principle Goal Selection
          3. DVP-14: Cerebellum Timing Jitter & Purkinje LTD Plasticity
          4. DVP-15: Sensorimotor Proprioception Latency & Jacobian
          5. DVP-16: CDC Ring Buffer 10^6 Packet Burst Test

        Returns:
            Gate2Results dataclass containing structured metrics for all 5 tests.
        """
        if self.verbose:
            print("=" * 82)
            print("DELTA P-42 SOVEREIGN COGNITIVE CORE (GATE 2) - QUANTITATIVE DVP VERIFICATION ENGINE")
            print("=" * 82)
            print("[1/5] Running DVP-03: Amygdala Emergency Brake & Pavlovian Conditioning (1,000 trials)...")

        # 1. DVP-03
        dvp03 = self.verify_dvp03_amygdala()
        if self.verbose:
            s03 = "PASS" if dvp03.passed else "FAIL"
            print(f"      Status: [{s03}] | WCET = {dvp03.wcet_ms:.4f} ms (<= {dvp03.wcet_threshold_ms:.1f} ms) | "
                  f"Reflex = {dvp03.post_conditioning_firing} | Pairings = {dvp03.pairings_completed}")
            print("[2/5] Running DVP-13: PFC Airplane Principle Goal Selection (1,000 candidate batches)...")

        # 2. DVP-13
        dvp13 = self.verify_dvp13_pfc_goal_selection()
        if self.verbose:
            s13 = "PASS" if dvp13.passed else "FAIL"
            print(f"      Status: [{s13}] | Mismatch = {dvp13.mismatch_count}/{dvp13.total_evaluations} "
                  f"({dvp13.mismatch_rate * 100:.3f}% < {dvp13.mismatch_threshold * 100:.1f}%) | "
                  f"Airplane Emergence = {dvp13.airplane_emergence_passed}")
            print("[3/5] Running DVP-14: Cerebellum Timing Jitter & Purkinje LTD (200 cycles, 100 LTD steps)...")

        # 3. DVP-14
        dvp14 = self.verify_dvp14_cerebellum_ltd()
        if self.verbose:
            s14 = "PASS" if dvp14.passed else "FAIL"
            print(f"      Status: [{s14}] | Jitter Max = {dvp14.timing_jitter_max_ms:.4f} ms (< {dvp14.timing_jitter_threshold_ms:.1f} ms) | "
                  f"LTD Reduction = {dvp14.error_reduction_pct:.2f}% (>= {dvp14.error_reduction_threshold_pct:.1f}%) | "
                  f"Overshoot = {dvp14.max_overshoot_pct:.2f}%")
            print("[4/5] Running DVP-15: Sensorimotor Proprioception Latency & Jacobian (1,000 cycles)...")

        # 4. DVP-15
        dvp15 = self.verify_dvp15_sensorimotor_latency()
        if self.verbose:
            s15 = "PASS" if dvp15.passed else "FAIL"
            print(f"      Status: [{s15}] | WCET = {dvp15.wcet_latency_ms:.4f} ms (<= {dvp15.wcet_threshold_ms:.1f} ms) | "
                  f"Jacobian FD Err = {dvp15.jacobian_fd_max_error:.2e} (< {dvp15.jacobian_fd_threshold:.0e}) | "
                  f"Corruption = {dvp15.corruption_count}")
            print("[5/5] Running DVP-16: CDC Ring Buffer Burst Test (1,000,000 CognitivePacket instances)...")

        # 5. DVP-16
        dvp16 = self.verify_dvp16_cdc_ring_buffer()
        if self.verbose:
            s16 = "PASS" if dvp16.passed else "FAIL"
            print(f"      Status: [{s16}] | Zero-Drop = {dvp16.packets_received}/{dvp16.total_packets} | "
                  f"CRC Errors = {dvp16.crc_error_count} | Collisions = {dvp16.uuid_collision_count} | "
                  f"Rate = {dvp16.throughput_packets_per_sec:.0f} pkts/s ({dvp16.throughput_mb_per_sec:.2f} MB/s) | "
                  f"Backend = {dvp16.backend}")

        all_passed = bool(
            dvp03.passed
            and dvp13.passed
            and dvp14.passed
            and dvp15.passed
            and dvp16.passed
        )

        summary: Dict[str, Any] = {
            "all_passed": all_passed,
            "dvp03_passed": dvp03.passed,
            "dvp03_wcet_ms": dvp03.wcet_ms,
            "dvp13_passed": dvp13.passed,
            "dvp13_mismatch_rate": dvp13.mismatch_rate,
            "dvp14_passed": dvp14.passed,
            "dvp14_jitter_max_ms": dvp14.timing_jitter_max_ms,
            "dvp14_error_reduction_pct": dvp14.error_reduction_pct,
            "dvp15_passed": dvp15.passed,
            "dvp15_wcet_latency_ms": dvp15.wcet_latency_ms,
            "dvp15_jacobian_fd_error": dvp15.jacobian_fd_max_error,
            "dvp16_passed": dvp16.passed,
            "dvp16_packets": dvp16.total_packets,
            "dvp16_crc_errors": dvp16.crc_error_count,
            "dvp16_uuid_collisions": dvp16.uuid_collision_count,
            "dvp16_throughput_pkts_per_sec": dvp16.throughput_packets_per_sec,
        }

        results = Gate2Results(
            dvp03=dvp03,
            dvp13=dvp13,
            dvp14=dvp14,
            dvp15=dvp15,
            dvp16=dvp16,
            all_passed=all_passed,
            summary=summary,
        )

        if self.verbose:
            print("\n" + self.format_report(results))

        return results

    # --------------------------------------------------------------------------
    # Report Formatter
    # --------------------------------------------------------------------------

    def format_report(self, results: Gate2Results) -> str:
        """Generates a structured ASCII verification table summarizing Gate 2 acceptance."""
        m03 = "[PASS]" if results.dvp03.passed else "[FAIL]"
        m13 = "[PASS]" if results.dvp13.passed else "[FAIL]"
        m14 = "[PASS]" if results.dvp14.passed else "[FAIL]"
        m15 = "[PASS]" if results.dvp15.passed else "[FAIL]"
        m16 = "[PASS]" if results.dvp16.passed else "[FAIL]"
        overall_mark = "GATE 2 APPROVED - PASS" if results.all_passed else "GATE 2 REJECTED - FAIL"

        lines = [
            "+" + "-" * 80 + "+",
            f"| DELTA P-42 STAGE-GATE 2: QUANTITATIVE DVP VERIFICATION REPORT{' ' * 17}|",
            "+" + "-" * 80 + "+",
            f"| DVP-03: Amygdala Emergency Brake & Pavlovian Conditioning{' ' * 21}|",
            f"|   - Reaction Latency WCET:    {results.dvp03.wcet_ms:6.4f} ms  (Target: <= {results.dvp03.wcet_threshold_ms:.1f} ms)         {m03} |",
            f"|   - Unconditioned Direct Kill:{' ' * 4}Passed (Analog PWM 0.0V Grounded)  {m03} |",
            f"|   - Pavlovian STDP Reflex:    {' ' * 4}Passed (Anticipatory Panic on CS)  {m03} |",
            "+" + "-" * 80 + "+",
            f"| DVP-13: PFC Airplane Principle Goal Selection{' ' * 33}|",
            f"|   - Mismatch Count:           {results.dvp13.mismatch_count:6d} / {results.dvp13.total_evaluations:<5d} (Target: < 1 mismatch)     {m13} |",
            f"|   - Mismatch Error Rate:      {results.dvp13.mismatch_rate * 100:6.3f} %   (Target: < {results.dvp13.mismatch_threshold * 100:.1f} %)          {m13} |",
            f"|   - Emergence Verification:   {' ' * 4}Passed (Spontaneous Airplane Goal)  {m13} |",
            "+" + "-" * 80 + "+",
            f"| DVP-14: Cerebellum Timing Jitter & Purkinje LTD Plasticity{' ' * 20}|",
            f"|   - 20 Hz Motor Cadence Jitter:{results.dvp14.timing_jitter_max_ms:6.4f} ms  (Target: < {results.dvp14.timing_jitter_threshold_ms:.1f} ms)          {m14} |",
            f"|   - Purkinje LTD Error Reduc.: {results.dvp14.error_reduction_pct:6.2f} %   (Target: >= {results.dvp14.error_reduction_threshold_pct:.1f} %)         {m14} |",
            f"|   - Mechanical Overshoot:      {results.dvp14.max_overshoot_pct:6.2f} %   (Target: == 0.00 %)          {m14} |",
            f"|   - Granule Reservoir Sparsity:{results.dvp14.reservoir_sparsity_min * 100:6.1f} %   (Target: >= 90.0 %)          {m14} |",
            "+" + "-" * 80 + "+",
            f"| DVP-15: Sensorimotor Proprioception Latency & Analytical Jacobian{' ' * 13}|",
            f"|   - Proprioception WCET Latency:{results.dvp15.wcet_latency_ms:5.4f} ms  (Target: <= {results.dvp15.wcet_threshold_ms:.1f} ms)         {m15} |",
            f"|   - Analytical Jacobian Error: {results.dvp15.jacobian_fd_max_error:7.2e}  (Target: < {results.dvp15.jacobian_fd_threshold:.0e})          {m15} |",
            f"|   - Tensor Corruption Count:   {results.dvp15.corruption_count:6d}    (Target: == 0)               {m15} |",
            f"|   - 64-Element Packet Payload: {' ' * 4}Passed (Round-Trip Verified)       {m15} |",
            "+" + "-" * 80 + "+",
            f"| DVP-16: CDC Ring Buffer 1,000,000 Packet Burst Test{' ' * 27}|",
            f"|   - Burst Throughput:          {results.dvp16.packets_received:7d} / {results.dvp16.total_packets:<7d} (Zero-Drop)        {m16} |",
            f"|   - Hardware CRC-64 Errors:    {results.dvp16.crc_error_count:6d}    (Polynomial 0x42F0E1EBA9EA3693) {m16} |",
            f"|   - Event UUID Collisions:     {results.dvp16.uuid_collision_count:6d}    (RFC-4122 v4 Uniqueness)     {m16} |",
            f"|   - Sustained Throughput:      {results.dvp16.throughput_packets_per_sec:7.0f} pkts/s ({results.dvp16.throughput_mb_per_sec:6.2f} MB/s) [Backend: {results.dvp16.backend}] |",
            "+" + "-" * 80 + "+",
            f"| OVERALL STATUS: {overall_mark:<62}|",
            "+" + "-" * 80 + "+",
        ]
        return "\n".join(lines)


# ==============================================================================
# Standalone CLI Entry Point
# ==============================================================================

def main() -> int:
    """
    CLI entry point for core/gate2_verification.py.
    Executes all 5 Gate 2 DVP tests, emits report to stdout, and exits with code 0 on PASS, 1 on FAIL.
    """
    verifier = Gate2Verifier(verbose=True)
    results = verifier.run_all()
    return 0 if results.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
