"""
ΔP-42 Sovereign Cognitive Core (Gate 2) - PFC Executive Unit Tests
Verifies:
1. Airplane Principle (Repülőgép-elv) mathematical formulation: rho_k = w_k^T * W_PFC * V(t).
2. DVP-13 quantitative benchmark over 1,000 trials (mismatch rate < 0.1%, zero decision floating).
3. Threshold gating (rho > 0.35 -> GoalStructure, rho <= 0.35 -> quiescence/None).
4. Top-down sensory prediction c_pred and motor action intent u_intent synthesis.
5. Cloud / Airplane emergence paradigm (DELTA_P42_MUKODESI_ELVEK.md Chapter I.3).
6. Robustness against non-finite values, dimension mismatches, and boundary conditions.
"""

import math
import numpy as np
import pytest

from modules.pfc_executive import (
    DEFAULT_D_ACTION,
    DEFAULT_D_EMBED,
    DEFAULT_D_SENS,
    MASLOW_DRIVE_COUNT,
    RHO_THRESHOLD_CANONICAL,
    GoalCandidate,
    GoalStructure,
    PFCExecutive,
)


def test_pfc_airplane_principle_formula():
    """Verifies that scalar resonance matches analytical dot product exactly: rho_k = w_k^T * W_PFC * V."""
    d_embed = 4
    w_pfc = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.0, -0.1],
        [0.0, 0.0, 0.0, 1.0, 0.8],
    ], dtype=np.float64)

    pfc = PFCExecutive(d_embed=d_embed, w_pfc=w_pfc, rho_threshold=0.35)

    c1 = GoalCandidate("c1", "test1", np.array([1.0, 0.0, 0.0, 0.0]))
    c2 = GoalCandidate("c2", "test2", np.array([0.0, 0.0, 0.0, 1.0]))

    v = np.array([0.2, 0.1, 0.0, 0.4, 0.9], dtype=np.float64)

    # Manual calculation:
    # W_PFC * V:
    # row 0: 1*0.2 + 0.5*0.9 = 0.2 + 0.45 = 0.65
    # row 1: 1*0.1 + 0.2*0.9 = 0.1 + 0.18 = 0.28
    # row 2: 1*0.0 - 0.1*0.9 = -0.09
    # row 3: 1*0.4 + 0.8*0.9 = 0.4 + 0.72 = 1.12
    # c1: [1, 0, 0, 0] . [0.65, 0.28, -0.09, 1.12] = 0.65
    # c2: [0, 0, 0, 1] . [0.65, 0.28, -0.09, 1.12] = 1.12
    resonances = pfc.compute_resonances([c1, c2], v)
    assert len(resonances) == 2
    assert math.isclose(resonances[0], 0.65, abs_tol=1e-10)
    assert math.isclose(resonances[1], 1.12, abs_tol=1e-10)


def test_pfc_threshold_gating_and_quiescence():
    """Verifies that rho > rho_threshold creates GoalStructure and rho <= rho_threshold returns None."""
    pfc = PFCExecutive(d_embed=4, rho_threshold=0.35)

    c1 = GoalCandidate("c1", "low_interest", np.zeros(4))
    v = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

    # Zero embedding produces zero resonance <= 0.35
    goal = pfc.evaluate_candidates([c1], v)
    assert goal is None
    assert pfc.active_goal is None

    # High resonance candidate
    # Fabricate W_PFC and candidate so rho = 0.75 > 0.35
    w_pfc = np.eye(4, 5, dtype=np.float64)
    pfc2 = PFCExecutive(d_embed=4, w_pfc=w_pfc, rho_threshold=0.35)
    c_high = GoalCandidate("c_high", "critical_need", np.array([1.0, 0.0, 0.0, 0.0]))
    v_high = np.array([0.8, 0.1, 0.1, 0.1, 0.1])

    goal2 = pfc2.evaluate_candidates([c_high], v_high)
    assert goal2 is not None
    assert isinstance(goal2, GoalStructure)
    assert goal2.candidate_id == "c_high"
    assert math.isclose(goal2.resonance, 0.8, abs_tol=1e-10)
    assert math.isclose(goal2.priority, 0.8, abs_tol=1e-10)
    assert goal2.dominant_need_idx == 0
    assert pfc2.active_goal == goal2


def test_pfc_zero_maslow_vector_quiescence():
    """Verifies that completely zero Maslow deficit V = 0 always decays to None."""
    pfc = PFCExecutive()
    candidates = [
        GoalCandidate("c1", "eat", np.random.randn(DEFAULT_D_EMBED)),
        GoalCandidate("c2", "play", np.random.randn(DEFAULT_D_EMBED)),
    ]
    v_zero = np.zeros(MASLOW_DRIVE_COUNT)
    goal = pfc.evaluate_candidates(candidates, v_zero)
    assert goal is None


def test_pfc_dvp13_1000_trials_benchmark():
    """Formal DVP-13 Benchmark: 1,000 trials with competing candidate sets, mismatch rate < 0.1%."""
    pfc = PFCExecutive(seed=123)
    results = pfc.run_dvp13_benchmark(n_trials=1000, seed=42)

    assert results["passed"] is True
    assert results["total_evaluations"] == 1000
    assert results["mismatch_count"] == 0
    assert results["mismatch_rate"] < 0.001
    assert results["zero_deadlock_verified"] is True


def test_pfc_deterministic_tie_breaking():
    """Verifies that identical maximum resonance resolves deterministically to the lowest index."""
    pfc = PFCExecutive(d_embed=4, rho_threshold=0.1)
    w = np.array([1.0, 0.0, 0.0, 0.0])
    c1 = GoalCandidate("cand_0", "item_0", w)
    c2 = GoalCandidate("cand_1", "item_1", w)
    c3 = GoalCandidate("cand_2", "item_2", w)

    v = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
    goal = pfc.evaluate_candidates([c1, c2, c3], v)
    assert goal is not None
    # Lowest index cand_0 must win
    assert goal.candidate_id == "cand_0"


def test_pfc_intent_and_predictive_synthesis():
    """Verifies synthesis of u_intent in [-1, 1]^d_action and c_pred in [0, 1]^d_sens."""
    pfc = PFCExecutive(d_embed=16, d_sens=8, d_action=4)
    emb = np.random.randn(16)
    x_ext = np.random.uniform(0.0, 1.0, 8)

    u_intent, c_pred = pfc.synthesize_intent_and_prediction(emb, x_ext=x_ext)

    assert u_intent.shape == (4,)
    assert c_pred.shape == (8,)
    assert np.all(u_intent >= -1.0) and np.all(u_intent <= 1.0)
    assert np.all(c_pred >= 0.0) and np.all(c_pred <= 1.0)


def test_pfc_emergence_airplane_cloud():
    """
    Verifies the canonical emergence test (DELTA_P42_MUKODESI_ELVEK.md Chapter I.3):
    'szép a felhő' with V_megismerés > 0.8 instantiates an airplane design goal tree;
    low deficit V_megismerés = 0.1 decays to quiescence.
    """
    # Design W_PFC so that the cloud embedding has strong affinity with drive 4 (cognition)
    d_embed = 16
    w_pfc = np.zeros((d_embed, 5), dtype=np.float64)
    # Cognition column has 1.0 for feature 0
    w_pfc[0, 4] = 1.0

    pfc = PFCExecutive(d_embed=d_embed, w_pfc=w_pfc, rho_threshold=0.35)

    cloud_emb = np.zeros(d_embed)
    cloud_emb[0] = 1.0  # Feature 0 corresponds to aerial/cloud concept
    cloud_candidate = GoalCandidate("cloud_thought", "szép a felhő", cloud_emb)

    # 1. High cognitive deficit (V_cognition = 0.9)
    v_high_cognition = np.array([0.0, 0.0, 0.0, 0.0, 0.9])
    goal_active = pfc.evaluate_candidates([cloud_candidate], v_high_cognition)

    assert goal_active is not None
    assert goal_active.dominant_need_idx == 4
    assert goal_active.resonance >= 0.89
    assert len(goal_active.subtasks) >= 3
    assert any("glider" in st or "cloud" in st for st in goal_active.subtasks)

    # 2. Low cognitive deficit (V_cognition = 0.1)
    v_low_cognition = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    goal_decay = pfc.evaluate_candidates([cloud_candidate], v_low_cognition)
    assert goal_decay is None


def test_pfc_edge_cases_and_input_validation():
    """Verifies that invalid inputs, NaNs, and empty candidate sequences raise ValueError."""
    pfc = PFCExecutive(d_embed=8)

    # Empty candidates
    with pytest.raises(ValueError, match="cannot be empty"):
        pfc.evaluate_candidates([], np.zeros(5))

    # Invalid Maslow vector shape
    c1 = GoalCandidate("c1", "test", np.zeros(8))
    with pytest.raises(ValueError, match="shape"):
        pfc.evaluate_candidates([c1], np.zeros(4))

    # Maslow vector with NaN
    v_nan = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
    with pytest.raises(ValueError, match="non-finite"):
        pfc.evaluate_candidates([c1], v_nan)

    # Candidate with Inf
    with pytest.raises(ValueError, match="non-finite"):
        GoalCandidate("c_bad", "bad", np.array([np.inf] * 8))

    # Maximum saturation Maslow vector [1, 1, 1, 1, 1]
    v_sat = np.ones(5)
    c_sat = GoalCandidate("c_sat", "sat", np.ones(8) * 10.0)
    goal_sat = pfc.evaluate_candidates([c_sat], v_sat)
    if goal_sat is not None:
        assert 0.0 <= goal_sat.priority <= 1.0


def test_pfc_reset_and_metrics():
    """Verifies that reset clears active goal and metrics are tracked accurately."""
    pfc = PFCExecutive(rho_threshold=0.2)
    v = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    aligned_emb = np.dot(pfc.w_pfc, v)
    c = GoalCandidate("c1", "explore", aligned_emb / np.linalg.norm(aligned_emb))

    goal = pfc.evaluate_candidates([c], v)
    assert goal is not None
    assert pfc.active_goal is not None

    metrics = pfc.get_metrics()
    assert metrics["total_evaluations"] == 1
    assert metrics["has_active_goal"] is True

    pfc.reset()
    assert pfc.active_goal is None
    reset_metrics = pfc.get_metrics()
    assert reset_metrics["total_evaluations"] == 0
    assert reset_metrics["has_active_goal"] is False
