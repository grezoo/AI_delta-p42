"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Adversarial Stress Test Suite
Target: modules/pfc_executive.py (DVP-13)
Author: Challenger 1 (Adversarial PFC Stress Tester)

Empirical Adversarial Challenges:
1. DVP-13 quantitative benchmark under 5,000 competing candidate batches with
   diverse embeddings, near-identical resonance scores (differences < 1e-8 down to 1e-12),
   and pathological Maslow vectors (all zeros, all ones, single need dominant, NaN/Inf rejection).
   Strict verification: mismatch rate < 0.1% (target 0.0%) and zero decision floating.
2. Deterministic tie-breaking under exact duplicate resonance scores across varied candidate
   pool sizes, tied candidate positions, and ordering permutations.
3. Sub-nanoscale threshold gating sensitivity around boundary rho = 0.3500000001 vs 0.3499999999,
   monotonicity sweep across 1,000 points spanning [0.3499, 0.3501], and machine-epsilon boundaries.
4. Memory stability and execution latency across long continuous thought streams (10,000 continuous
   iterations): tracemalloc heap leak verification (net growth < 50 KB) and latency percentiles
   (mean < 0.5 ms, p99 < 1.0 ms, max < 2.0 ms).
5. Pathological input robustness and adversarial boundary fuzzing (NaNs, Infinities, dimension
   mismatches, empty batches, invalid concept types, denormal and extreme float values).
6. State lifecycle, rapid multi-drive switching, and cognitive quiescence transitions.
"""

from __future__ import annotations

import math
import time
import tracemalloc
from typing import List, Sequence, Tuple
import uuid

import numpy as np
import pytest

from modules.pfc_executive import (
    DEFAULT_D_ACTION,
    DEFAULT_D_EMBED,
    DEFAULT_D_SENS,
    MASLOW_DRIVE_COUNT,
    MASLOW_DRIVES,
    RHO_THRESHOLD_CANONICAL,
    GoalCandidate,
    GoalStructure,
    PFCExecutive,
)


class TestAdversarialPFC5000Benchmark:
    """
    Challenge 1: DVP-13 Quantitative Benchmark under 5,000 Competing Candidate Batches.
    Acceptance Criteria (DVP-13):
    - Mismatch rate against analytical float64 ground truth must be strictly < 0.1% (target 0.0%).
    - Zero decision floating / deadlock (count == 0).
    - Robustness across diverse embeddings, near-identical resonance differences (< 1e-8),
      and pathological Maslow vectors (all zeros, all ones, single need dominant).
    """

    def test_dvp13_5000_competing_candidate_batches(self) -> None:
        """
        Execute 5,000 adversarial candidate evaluation batches:
        - 1,000 trials with near-identical resonance scores (|rho_i - rho_j| < 1e-8 down to 1e-12).
        - 1,000 trials with pathological Maslow vectors (all zeros, all ones, single-need dominant).
        - 1,500 trials with diverse embeddings (sparse 1-hot, orthogonal, opposing, normalized).
        - 1,500 trials with varying candidate batch sizes (K between 5 and 30).
        """
        pfc = PFCExecutive(d_embed=DEFAULT_D_EMBED, rho_threshold=RHO_THRESHOLD_CANONICAL, seed=999)
        rng = np.random.RandomState(42)

        total_trials = 5000
        mismatches = 0
        deadlocks = 0

        for trial in range(total_trials):
            phase = trial % 5

            if phase == 0:
                # Sub-batch 1: Near-identical resonance scores (|rho_1 - rho_0| < 1e-8 down to 1e-12)
                v = rng.uniform(0.1, 0.9, size=MASLOW_DRIVE_COUNT).astype(np.float64)
                u = np.dot(pfc.w_pfc, v)
                norm_u_sq = float(np.dot(u, u))
                if norm_u_sq < 1e-9:
                    norm_u_sq = 1.0

                k_cand = rng.randint(5, 16)
                # Base candidate placed to have resonance ~0.50
                target_rho_base = 0.50
                base_w = (target_rho_base / norm_u_sq) * u

                candidates: List[GoalCandidate] = []
                # Candidate 0 is base
                candidates.append(GoalCandidate("cand_base", "base_thought", base_w))

                # Candidate 1 has delta rho in [1e-12, 1e-9]
                delta_rho = 10.0 ** (-rng.uniform(9.0, 12.0))
                w_near = base_w + (delta_rho / norm_u_sq) * u
                candidates.append(GoalCandidate("cand_near_higher", "near_thought", w_near))

                # Remaining candidates have lower resonance
                for j in range(2, k_cand):
                    lower_rho = target_rho_base - rng.uniform(0.05, 0.3)
                    w_low = (lower_rho / norm_u_sq) * u
                    candidates.append(GoalCandidate(f"cand_low_{j}", f"low_thought_{j}", w_low))

                # Shuffle candidates randomly to challenge index bias
                rng.shuffle(candidates)

            elif phase == 1:
                # Sub-batch 2: Pathological Maslow deficit vectors
                pathology_sub = (trial // 5) % 7
                if pathology_sub == 0:
                    # All zeros: [0, 0, 0, 0, 0]
                    v = np.zeros(MASLOW_DRIVE_COUNT, dtype=np.float64)
                elif pathology_sub == 1:
                    # All ones: [1, 1, 1, 1, 1] (saturation)
                    v = np.ones(MASLOW_DRIVE_COUNT, dtype=np.float64)
                else:
                    # Single need dominant
                    drive_idx = pathology_sub - 2  # 0..4
                    v = np.zeros(MASLOW_DRIVE_COUNT, dtype=np.float64)
                    v[drive_idx] = 1.0
                    # Add tiny noise to non-dominant drives
                    for d in range(MASLOW_DRIVE_COUNT):
                        if d != drive_idx:
                            v[d] = 0.001

                k_cand = rng.randint(5, 20)
                candidates = []
                for j in range(k_cand):
                    emb = rng.randn(DEFAULT_D_EMBED).astype(np.float64)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-9:
                        emb /= norm
                    candidates.append(GoalCandidate(f"cand_path_{trial}_{j}", f"concept_{j}", emb))

            elif phase == 2:
                # Sub-batch 3: Diverse structured embeddings (sparse 1-hot, 2-hot, orthogonal)
                v = rng.uniform(0.05, 0.95, size=MASLOW_DRIVE_COUNT).astype(np.float64)
                k_cand = rng.randint(6, 22)
                candidates = []
                for j in range(k_cand):
                    emb = np.zeros(DEFAULT_D_EMBED, dtype=np.float64)
                    if j % 3 == 0:
                        # 1-hot sparse
                        emb[j % DEFAULT_D_EMBED] = 1.0
                    elif j % 3 == 1:
                        # 2-hot sparse
                        emb[j % DEFAULT_D_EMBED] = 0.7071
                        emb[(j + 1) % DEFAULT_D_EMBED] = 0.7071
                    else:
                        # dense Gaussian
                        emb = rng.randn(DEFAULT_D_EMBED).astype(np.float64)
                        emb /= np.linalg.norm(emb)
                    candidates.append(GoalCandidate(f"cand_struct_{trial}_{j}", f"struct_{j}", emb))

            elif phase == 3:
                # Sub-batch 4: Near-boundary threshold cases (max resonance in [0.3499, 0.3501])
                v = rng.uniform(0.2, 0.8, size=MASLOW_DRIVE_COUNT).astype(np.float64)
                u = np.dot(pfc.w_pfc, v)
                norm_u_sq = float(np.dot(u, u))
                if norm_u_sq < 1e-9:
                    norm_u_sq = 1.0

                # Either slightly above or slightly below threshold
                target_rho = 0.35 + rng.choice([-1.0, 1.0]) * 10.0 ** (-rng.uniform(5.0, 9.0))
                lead_w = (target_rho / norm_u_sq) * u
                candidates = [GoalCandidate(f"cand_boundary_lead_{trial}", "boundary_lead", lead_w)]

                # Competitors with lower resonance
                for j in range(1, rng.randint(5, 15)):
                    comp_rho = 0.35 - rng.uniform(0.01, 0.2)
                    comp_w = (comp_rho / norm_u_sq) * u
                    candidates.append(GoalCandidate(f"cand_comp_{trial}_{j}", f"comp_{j}", comp_w))
                rng.shuffle(candidates)

            else:
                # Sub-batch 5: Large candidate pool (K in [20, 35]) under random drives
                v = rng.uniform(0.0, 1.0, size=MASLOW_DRIVE_COUNT).astype(np.float64)
                k_cand = rng.randint(20, 36)
                candidates = [
                    GoalCandidate(
                        f"cand_large_{trial}_{j}",
                        f"large_concept_{j}",
                        rng.randn(DEFAULT_D_EMBED).astype(np.float64),
                    )
                    for j in range(k_cand)
                ]

            # --- Analytical Ground Truth Oracle Calculation ---
            emb_matrix = np.stack([c.embedding for c in candidates])  # (K, d_embed)
            proj_drive = np.dot(pfc.w_pfc, v)
            analytical_resonances = np.dot(emb_matrix, proj_drive)  # (K,)
            k_true = int(np.argmax(analytical_resonances))
            rho_true_max = float(analytical_resonances[k_true])

            # --- Module Execution ---
            goal = pfc.evaluate_candidates(candidates, v, timestamp_ns=1000000000 + trial)

            # --- Decision Verification ---
            if rho_true_max > RHO_THRESHOLD_CANONICAL:
                if goal is None:
                    # Floating error: failed to instantiate goal above threshold
                    mismatches += 1
                else:
                    # Winner candidate check:
                    if goal.candidate_id != candidates[k_true].candidate_id:
                        # Allow float64 machine-epsilon ties where resonances are identical
                        cand_chosen_idx = next(
                            i for i, c in enumerate(candidates) if c.candidate_id == goal.candidate_id
                        )
                        diff = abs(analytical_resonances[cand_chosen_idx] - rho_true_max)
                        if diff > 1e-12:
                            mismatches += 1

                    # Decision floating checks:
                    if not (0.0 <= goal.priority <= 1.0):
                        deadlocks += 1
                    if not np.isfinite(goal.resonance):
                        deadlocks += 1
                    if not (0 <= goal.dominant_need_idx < MASLOW_DRIVE_COUNT):
                        deadlocks += 1
                    if goal.dominant_need_idx != int(np.argmax(v)):
                        deadlocks += 1
                    if not np.all(np.isfinite(goal.intent_vector)):
                        deadlocks += 1
                    if not np.all(np.isfinite(goal.predictive_state)):
                        deadlocks += 1
            else:
                # Thought must decay into quiescence (None)
                if goal is not None:
                    # Floating error: instantiated goal at or below threshold
                    mismatches += 1

        mismatch_rate = mismatches / float(total_trials)
        assert deadlocks == 0, f"Decision deadlocks/floating detected: count={deadlocks}"
        assert mismatches == 0, f"Decision mismatches detected: count={mismatches}, rate={mismatch_rate:.6f}"
        assert mismatch_rate < 0.001, f"Mismatch rate {mismatch_rate} exceeds 0.1% ceiling"


class TestAdversarialPFCDuplicateTieBreaking:
    """
    Challenge 2: Deterministic Tie-Breaking under Exact Duplicate Resonance Scores.
    Acceptance Criteria:
    - When multiple candidates have bitwise identical maximum resonance scores,
      PFCExecutive must ALWAYS deterministically select the candidate with the lowest index.
    - Zero stochasticity, zero oscillation under permutation or repeated evaluation.
    """

    def test_exact_tie_all_candidates_identical_resonance(self) -> None:
        """Verify that when ALL candidates have identical resonance, index 0 is always chosen."""
        d_embed = 16
        w_pfc = np.zeros((d_embed, 5), dtype=np.float64)
        w_pfc[0, 0] = 1.0  # Drive 0 projects directly to feature 0
        pfc = PFCExecutive(d_embed=d_embed, w_pfc=w_pfc, rho_threshold=0.2)

        v = np.array([0.6, 0.0, 0.0, 0.0, 0.0])

        # 1-hot embedding on feature 0 guarantees exact bitwise resonance across all rows
        w = np.zeros(d_embed, dtype=np.float64)
        w[0] = 1.0  # rho = 1.0 * 0.6 = 0.6 > 0.2

        for pool_size in [2, 3, 5, 10, 20]:
            candidates = [
                GoalCandidate(f"cand_{i}", f"concept_{i}", w.copy())
                for i in range(pool_size)
            ]
            # Verify resonances are bitwise identical
            res = pfc.compute_resonances(candidates, v)
            assert all(r == res[0] for r in res), f"Resonances not bitwise equal for pool_size={pool_size}"

            goal = pfc.evaluate_candidates(candidates, v)
            assert goal is not None
            assert goal.candidate_id == "cand_0", (
                f"Tie-break failed for pool_size={pool_size}: expected 'cand_0', got '{goal.candidate_id}'"
            )

    def test_exact_tie_at_interior_indices(self) -> None:
        """
        Verify that when a subset of candidates share the maximum resonance,
        the one with the lowest index among the winners is deterministically selected.
        """
        d_embed = 16
        w_pfc = np.zeros((d_embed, 5), dtype=np.float64)
        w_pfc[0, 0] = 1.0
        pfc = PFCExecutive(d_embed=d_embed, w_pfc=w_pfc, rho_threshold=0.2)

        v = np.array([0.6, 0.0, 0.0, 0.0, 0.0])

        # High resonance candidate: rho = 0.60 > 0.20
        w_high = np.zeros(d_embed, dtype=np.float64)
        w_high[0] = 1.0
        # Low resonance candidate: rho = 0.0 <= 0.20
        w_low = np.zeros(d_embed, dtype=np.float64)

        # Scenario A: Tie at indices 2 and 4, indices 0, 1, 3 have lower resonance
        candidates_a = [
            GoalCandidate("cand_0", "low", w_low),
            GoalCandidate("cand_1", "low", w_low),
            GoalCandidate("cand_2", "high_first", w_high.copy()),
            GoalCandidate("cand_3", "low", w_low),
            GoalCandidate("cand_4", "high_second", w_high.copy()),
        ]
        res_a = pfc.compute_resonances(candidates_a, v)
        assert res_a[2] == res_a[4] == 0.6
        goal_a = pfc.evaluate_candidates(candidates_a, v)
        assert goal_a is not None
        assert goal_a.candidate_id == "cand_2"

        # Scenario B: Tie at indices 1, 3, 5, index 0 is low
        candidates_b = [
            GoalCandidate("cand_0", "low", w_low),
            GoalCandidate("cand_1", "high_first", w_high.copy()),
            GoalCandidate("cand_2", "low", w_low),
            GoalCandidate("cand_3", "high_second", w_high.copy()),
            GoalCandidate("cand_4", "low", w_low),
            GoalCandidate("cand_5", "high_third", w_high.copy()),
        ]
        res_b = pfc.compute_resonances(candidates_b, v)
        assert res_b[1] == res_b[3] == res_b[5] == 0.6
        goal_b = pfc.evaluate_candidates(candidates_b, v)
        assert goal_b is not None
        assert goal_b.candidate_id == "cand_1"

    def test_tie_breaking_consistency_under_100_evaluations(self) -> None:
        """Verify 100% deterministic reproducibility across 100 consecutive evaluations."""
        d_embed = 8
        w_pfc = np.zeros((d_embed, 5), dtype=np.float64)
        w_pfc[0, 0] = 1.0
        pfc = PFCExecutive(d_embed=d_embed, w_pfc=w_pfc, rho_threshold=0.1)

        v = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        w = np.zeros(d_embed, dtype=np.float64)
        w[0] = 1.0

        candidates = [
            GoalCandidate("first", "concept", w.copy()),
            GoalCandidate("second", "concept", w.copy()),
            GoalCandidate("third", "concept", w.copy()),
        ]

        results = [pfc.evaluate_candidates(candidates, v).candidate_id for _ in range(100)]
        assert all(cid == "first" for cid in results), "Tie-breaking exhibited non-deterministic output!"


class TestAdversarialPFCBoundarySensitivity:
    """
    Challenge 3: Sub-Nanoscale Threshold Gating Sensitivity (rho = 0.3500000001 vs 0.3499999999).
    Acceptance Criteria:
    - Absolute boundary resolution: rho = 0.3500000001 MUST instantiate GoalStructure.
    - rho = 0.3499999999 and rho = 0.3500000000 MUST decay to None (quiescence).
    - Monotonic step function across 1,000 sub-nanoscale points spanning [0.3499, 0.3501].
    - Zero hysteresis, zero decision floating.
    """

    def test_canonical_boundary_exact_values(self) -> None:
        """Direct test of rho = 0.3500000001 vs 0.3499999999."""
        pfc = PFCExecutive(d_embed=64, rho_threshold=RHO_THRESHOLD_CANONICAL)
        v = np.array([0.3, 0.2, 0.5, 0.1, 0.4], dtype=np.float64)
        u = np.dot(pfc.w_pfc, v)
        norm_u_sq = float(np.dot(u, u))

        # 1. Just above boundary: rho = 0.3500000001
        rho_above = 0.3500000001
        w_above = (rho_above / norm_u_sq) * u
        cand_above = GoalCandidate("cand_above", "above_threshold", w_above)
        goal_above = pfc.evaluate_candidates([cand_above], v)

        assert goal_above is not None, "rho = 0.3500000001 unexpectedly decayed to None!"
        assert isinstance(goal_above, GoalStructure)
        assert math.isclose(goal_above.resonance, rho_above, rel_tol=1e-9)
        assert math.isclose(goal_above.priority, rho_above, rel_tol=1e-9)
        assert goal_above.candidate_id == "cand_above"

        # 2. Just below boundary: rho = 0.3499999999
        rho_below = 0.3499999999
        w_below = (rho_below / norm_u_sq) * u
        cand_below = GoalCandidate("cand_below", "below_threshold", w_below)
        goal_below = pfc.evaluate_candidates([cand_below], v)

        assert goal_below is None, "rho = 0.3499999999 unexpectedly created a GoalStructure!"

        # 3. Exact boundary: rho = 0.3500000000 (condition is rho_max <= rho_threshold -> decay)
        rho_exact = 0.3500000000
        w_exact = (rho_exact / norm_u_sq) * u
        cand_exact = GoalCandidate("cand_exact", "exact_threshold", w_exact)
        goal_exact = pfc.evaluate_candidates([cand_exact], v)

        assert goal_exact is None, "rho = 0.3500000000 exact unexpectedly created a GoalStructure!"

    def test_machine_epsilon_boundary_sensitivity(self) -> None:
        """Test boundary at machine epsilon scales (eps down to 1e-14)."""
        pfc = PFCExecutive(d_embed=32, rho_threshold=0.35)
        v = np.array([0.5, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        u = np.dot(pfc.w_pfc, v)
        norm_u_sq = float(np.dot(u, u))

        epsilons = [1e-8, 1e-10, 1e-12, 1e-14]
        for eps in epsilons:
            w_hi = ((0.35 + eps) / norm_u_sq) * u
            c_hi = GoalCandidate("c_hi", "above", w_hi)
            goal_hi = pfc.evaluate_candidates([c_hi], v)
            assert goal_hi is not None, f"Failed at eps=+{eps}"

            w_lo = ((0.35 - eps) / norm_u_sq) * u
            c_lo = GoalCandidate("c_lo", "below", w_lo)
            goal_lo = pfc.evaluate_candidates([c_lo], v)
            assert goal_lo is None, f"Failed at eps=-{eps}"

    def test_monotonic_boundary_sweep_1000_points(self) -> None:
        """
        Sweep 1,000 finely spaced points across [0.34990, 0.35010] in steps of 2e-7.
        Verify that gating is strictly monotonic:
        - Points <= 0.350000000000 always return None.
        - Points > 0.350000000000 always return GoalStructure.
        """
        pfc = PFCExecutive(d_embed=16, rho_threshold=0.35)
        v = np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        u = np.dot(pfc.w_pfc, v)
        norm_u_sq = float(np.dot(u, u))

        rhos = np.linspace(0.34990, 0.35010, 1000, dtype=np.float64)

        for rho_val in rhos:
            w = (rho_val / norm_u_sq) * u
            cand = GoalCandidate("c_sweep", "sweep", w)
            goal = pfc.evaluate_candidates([cand], v)

            if rho_val <= 0.3500000000000000:
                assert goal is None, f"Failed quiescence for rho={rho_val:.12f}"
            else:
                assert goal is not None, f"Failed activation for rho={rho_val:.12f}"
                assert math.isclose(goal.resonance, rho_val, abs_tol=1e-11)


class TestAdversarialPFCContinuousMemoryAndLatency:
    """
    Challenge 4: Memory and Execution Latency across Long Continuous Thought Streams.
    Acceptance Criteria:
    - 10,000 continuous sequential evaluations on a single PFC instance.
    - Zero memory leak: net heap growth tracked via tracemalloc must be < 50 KB.
    - Mean execution latency < 0.5 ms (well below 20 ms / 50 Hz subcortical cycle).
    - 99th percentile latency (p99) < 1.0 ms.
    - Max latency < 2.0 ms across all 10,000 calls.
    - Zero timing violations.
    """

    def test_10000_continuous_iterations_memory_and_latency(self) -> None:
        """
        Execute two-phase continuous stream test:
        Phase 1: Pure execution latency across 10,000 iterations (zero profiler instrumentation overhead).
        Phase 2: Heap memory growth tracking across 10,000 iterations via tracemalloc.
        """
        pfc = PFCExecutive(d_embed=DEFAULT_D_EMBED, rho_threshold=RHO_THRESHOLD_CANONICAL, seed=42)
        rng = np.random.RandomState(1337)

        # Pre-instantiate candidate thought pool (size K=12)
        candidate_pool = [
            GoalCandidate(f"stream_cand_{i}", f"concept_{i}", rng.randn(DEFAULT_D_EMBED).astype(np.float64))
            for i in range(12)
        ]

        iterations = 10000
        v_pool = [np.clip(rng.rand(MASLOW_DRIVE_COUNT), 0.05, 0.95).astype(np.float64) for _ in range(500)]

        # --- Phase 1: Pure Execution Latency Profiling (10,000 iterations) ---
        latencies_ms = np.zeros(iterations, dtype=np.float64)
        for it in range(iterations):
            v = v_pool[it % 500]
            t0 = time.perf_counter()
            goal = pfc.evaluate_candidates(candidate_pool, v, timestamp_ns=1000000 + it)
            t1 = time.perf_counter()
            latencies_ms[it] = (t1 - t0) * 1000.0

        lat_mean = float(np.mean(latencies_ms))
        lat_median = float(np.median(latencies_ms))
        lat_p95 = float(np.percentile(latencies_ms, 95))
        lat_p99 = float(np.percentile(latencies_ms, 99))
        lat_max = float(np.max(latencies_ms))

        # Latency constraints:
        assert lat_mean < 0.5, f"Mean latency {lat_mean:.4f} ms exceeds 0.5 ms ceiling"
        assert lat_p99 < 1.0, f"p99 latency {lat_p99:.4f} ms exceeds 1.0 ms ceiling"
        assert lat_max < 2.0, f"Max latency {lat_max:.4f} ms exceeds 2.0 ms ceiling"

        # --- Phase 2: Memory Leak Tracking via tracemalloc (10,000 iterations) ---
        tracemalloc.start()
        snap_initial = tracemalloc.take_snapshot()

        for it in range(iterations):
            v = v_pool[it % 500]
            goal = pfc.evaluate_candidates(candidate_pool, v, timestamp_ns=2000000 + it)

        snap_final = tracemalloc.take_snapshot()
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        diff_stats = snap_final.compare_to(snap_initial, "lineno")
        net_heap_growth_kb = sum(stat.size_diff for stat in diff_stats if stat.size_diff > 0) / 1024.0

        metrics = pfc.get_metrics()
        assert metrics["total_evaluations"] == iterations * 2
        assert metrics["goals_created"] + metrics["quiescent_cycles"] == iterations * 2

        # Memory leak constraint: net heap growth must be < 50 KB
        assert net_heap_growth_kb < 50.0, (
            f"Possible memory leak detected: net heap growth {net_heap_growth_kb:.2f} KB exceeds 50 KB ceiling"
        )


class TestAdversarialPFCPathologicalInputRobustness:
    """
    Challenge 5: Pathological Input Robustness and Adversarial Boundary Fuzzing.
    Acceptance Criteria:
    - NaNs and Infinities in candidate embeddings or Maslow vectors must strictly raise ValueError.
    - Dimension mismatches must strictly raise ValueError.
    - Empty candidate sequence must strictly raise ValueError.
    - Denormal floats and extreme magnitudes must not trigger NaN/Inf propagation or crashes.
    """

    def test_nan_in_candidate_embedding_rejected(self) -> None:
        """Candidate embedding with NaN anywhere must raise ValueError upon instantiation."""
        w_nan = np.zeros(DEFAULT_D_EMBED)
        w_nan[3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            GoalCandidate("bad_nan", "concept", w_nan)

    def test_inf_in_candidate_embedding_rejected(self) -> None:
        """Candidate embedding with +-Inf must raise ValueError upon instantiation."""
        w_inf = np.ones(DEFAULT_D_EMBED)
        w_inf[0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            GoalCandidate("bad_inf", "concept", w_inf)

        w_neginf = np.ones(DEFAULT_D_EMBED)
        w_neginf[0] = -np.inf
        with pytest.raises(ValueError, match="non-finite"):
            GoalCandidate("bad_neginf", "concept", w_neginf)

    def test_nan_inf_in_maslow_vector_rejected(self) -> None:
        """Maslow vector with NaN or Inf at any of the 5 positions must raise ValueError."""
        pfc = PFCExecutive()
        cand = GoalCandidate("valid", "concept", np.ones(DEFAULT_D_EMBED))

        for pos in range(MASLOW_DRIVE_COUNT):
            v_nan = np.ones(MASLOW_DRIVE_COUNT)
            v_nan[pos] = np.nan
            with pytest.raises(ValueError, match="non-finite"):
                pfc.evaluate_candidates([cand], v_nan)

            v_inf = np.ones(MASLOW_DRIVE_COUNT)
            v_inf[pos] = np.inf
            with pytest.raises(ValueError, match="non-finite"):
                pfc.evaluate_candidates([cand], v_inf)

    def test_empty_candidate_list_rejected(self) -> None:
        """Empty sequence of candidates must raise ValueError."""
        pfc = PFCExecutive()
        with pytest.raises(ValueError, match="cannot be empty"):
            pfc.evaluate_candidates([], np.ones(5))

    def test_dimension_mismatch_rejected(self) -> None:
        """Candidates with wrong embedding dimension or Maslow vectors with wrong shape must be rejected."""
        pfc = PFCExecutive(d_embed=64)

        # Candidate with dimension 32 instead of 64
        c_short = GoalCandidate("short", "concept", np.ones(32))
        with pytest.raises(ValueError, match="embedding dimension"):
            pfc.evaluate_candidates([c_short], np.ones(5))

        # Candidate with dimension 128 instead of 64
        c_long = GoalCandidate("long", "concept", np.ones(128))
        with pytest.raises(ValueError, match="embedding dimension"):
            pfc.evaluate_candidates([c_long], np.ones(5))

        # Maslow vector with shape (4,) instead of (5,)
        c_valid = GoalCandidate("valid", "concept", np.ones(64))
        with pytest.raises(ValueError, match="shape"):
            pfc.evaluate_candidates([c_valid], np.ones(4))

        # Maslow vector with shape (6,) instead of (5,)
        with pytest.raises(ValueError, match="shape"):
            pfc.evaluate_candidates([c_valid], np.ones(6))

    def test_pathological_candidate_metadata(self) -> None:
        """Invalid candidate_id or non-string concept must raise ValueError."""
        w = np.ones(DEFAULT_D_EMBED)

        # Empty candidate_id
        with pytest.raises(ValueError, match="candidate_id"):
            GoalCandidate("", "concept", w)

        # Non-string candidate_id
        with pytest.raises(ValueError, match="candidate_id"):
            GoalCandidate(123, "concept", w)  # type: ignore

        # Non-string concept
        with pytest.raises(ValueError, match="concept"):
            GoalCandidate("c1", None, w)  # type: ignore

    def test_denormal_and_extreme_magnitudes(self) -> None:
        """Denormal floats (1e-310) and large magnitude vectors (1e20) must not crash."""
        pfc = PFCExecutive(d_embed=16)
        v = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        u = np.dot(pfc.w_pfc, v)
        norm_u_sq = float(np.dot(u, u))

        # Denormal floats
        w_denormal = np.full(16, 1e-310, dtype=np.float64)
        c_denormal = GoalCandidate("denormal", "concept", w_denormal)
        goal_denormal = pfc.evaluate_candidates([c_denormal], v)
        assert goal_denormal is None  # Effectively zero resonance

        # Positive large finite magnitude (rho = +1e20)
        w_large_pos = (1e20 / norm_u_sq) * u
        c_large_pos = GoalCandidate("large_pos", "concept", w_large_pos)
        goal_large_pos = pfc.evaluate_candidates([c_large_pos], v)
        assert goal_large_pos is not None
        assert goal_large_pos.priority == 1.0  # Clipped safely in [0, 1]
        assert np.isfinite(goal_large_pos.resonance)

        # Negative large finite magnitude (rho = -1e20)
        w_large_neg = (-1e20 / norm_u_sq) * u
        c_large_neg = GoalCandidate("large_neg", "concept", w_large_neg)
        goal_large_neg = pfc.evaluate_candidates([c_large_neg], v)
        assert goal_large_neg is None  # Negative resonance <= 0.35 decays safely to None


class TestAdversarialPFCStateLifecycleAndEmergence:
    """
    Challenge 6: State Lifecycle, Rapid Multi-Drive Switching, and Emergence.
    Acceptance Criteria:
    - Reset cleans active goal and metrics.
    - Quiescence transition clears active_goal.
    - Dominant drive tracking across rapid 5-drive cycling.
    - Emergence plan tree generation for canonical concepts ('szép a felhő', 'barát', 'veszély', 'töltés').
    """

    def test_state_lifecycle_and_reset(self) -> None:
        """Verify full lifecycle: activation -> quiescence -> reset."""
        pfc = PFCExecutive(d_embed=16, rho_threshold=0.2)
        v = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        w = np.dot(pfc.w_pfc, v)
        w /= np.linalg.norm(w)
        c = GoalCandidate("c1", "test", w)

        # 1. Activation
        g1 = pfc.evaluate_candidates([c], v)
        assert g1 is not None
        assert pfc.active_goal is not None

        # 2. Transition to quiescence
        g2 = pfc.evaluate_candidates([c], np.zeros(5))
        assert g2 is None
        assert pfc.active_goal is None

        # 3. Reset
        pfc.reset()
        metrics = pfc.get_metrics()
        assert metrics["total_evaluations"] == 0
        assert metrics["goals_created"] == 0
        assert metrics["quiescent_cycles"] == 0
        assert pfc.active_goal is None

    def test_rapid_drive_cycling_tracking(self) -> None:
        """Verify that dominant_need_idx accurately tracks as dominant drive cycles through 0..4."""
        pfc = PFCExecutive(d_embed=16, rho_threshold=0.1)
        w = np.ones(16, dtype=np.float64)
        c = GoalCandidate("cand", "cycling", w)

        for drive in range(MASLOW_DRIVE_COUNT):
            v = np.zeros(MASLOW_DRIVE_COUNT, dtype=np.float64)
            v[drive] = 0.95
            goal = pfc.evaluate_candidates([c], v)
            if goal is not None:
                assert goal.dominant_need_idx == drive

    def test_emergence_plan_trees_all_paradigms(self) -> None:
        """Verify plan tree generation for all four canonical paradigms plus generic fallback."""
        pfc = PFCExecutive()

        # 1. Airplane / Cloud emergence (Cognition drive 4)
        plan_cloud_cog = pfc.generate_plan_tree("szép a felhő", need_idx=4)
        assert "design_aerodynamic_glider_profile" in plan_cloud_cog

        # 2. Cloud observation (Non-cognition drive)
        plan_cloud_other = pfc.generate_plan_tree("szép a felhő", need_idx=1)
        assert "scan_sky_for_clouds" in plan_cloud_other

        # 3. Social paradigm
        plan_social = pfc.generate_plan_tree("emberi barát", need_idx=2)
        assert "locate_interaction_partner" in plan_social

        # 4. Danger paradigm
        plan_danger = pfc.generate_plan_tree("veszélyes fék", need_idx=1)
        assert "engage_emergency_posture" in plan_danger

        # 5. Metabolic recharge paradigm
        plan_recharge = pfc.generate_plan_tree("akkumulátor töltés alvás", need_idx=0)
        assert "dock_physical_power_interface" in plan_recharge

        # 6. Generic fallback
        plan_generic = pfc.generate_plan_tree("kvantum_számítás", need_idx=4)
        assert len(plan_generic) == 3
        assert plan_generic[0] == "analyze_kvantum_számítás"
