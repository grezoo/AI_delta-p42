"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Test Suite: tests/test_three_arm_experiment.py
Author: Stage-Gate 4 Milestone 2 Specialist

Verifies:
1. Long-Horizon 3-Arm Comparative Ontogenesis Simulator (DVP-12) specifications.
2. Arm A (Full Sovereign Closed Loop) thermodynamic drive, limit cycle, and endogenous initiation (R_init >= 0.85).
3. Arm B (Open-Loop Reactive) hyperactivity, uncoordinated thrashing, and metabolic exhaustion.
4. Arm C (Passive LLM Control Baseline) digital coma, prompt-dependence, and lack of endogenous drive (R_init == 0.0).
5. Statistical hypothesis falsification:
   - Adaptation ratio: Adaptation(Arm A) / Adaptation(Arm C) >= 3.50.
   - Welch's two-sample t-test p-value < 0.01 (Arm A > Arm C).
   - Mann-Whitney U test p-value < 0.01 (Arm A > Arm C).
   - Cohen's d standardized effect size > 1.50.
   - Overall DVP-12 verdict: HYPOTHESIS_CONFIRMED.
6. ArmResult, ArmTelemetry, and ExperimentSummary data structure invariants.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nursery.three_arm_experiment import (
    DVP12_MAX_P_VALUE,
    DVP12_MIN_ADAPTATION_RATIO,
    MIN_TRIALS_FOR_SIGNIFICANCE,
    R_INIT_EXPECTED_ARM_C,
    R_INIT_THRESHOLD_ARM_A,
    W_EFF,
    W_HOMEO,
    W_TASK,
    ArmResult,
    ArmTelemetry,
    ArmType,
    ExperimentSummary,
    FalsificationReport,
    ThreeArmExperiment,
)


# ==============================================================================
# 1. CANONICAL CONSTANTS & ENUMS
# ==============================================================================

def test_three_arm_constants_and_enums() -> None:
    assert DVP12_MIN_ADAPTATION_RATIO == 3.50
    assert DVP12_MAX_P_VALUE == 0.01
    assert R_INIT_THRESHOLD_ARM_A == 0.85
    assert R_INIT_EXPECTED_ARM_C == 0.05
    assert MIN_TRIALS_FOR_SIGNIFICANCE == 30

    assert pytest.approx(W_TASK + W_HOMEO + W_EFF, rel=1e-6) == 1.00
    assert W_TASK == 0.50
    assert W_HOMEO == 0.35
    assert W_EFF == 0.15

    assert ArmType.ARM_A_SOVEREIGN.value == "Arm A (Sovereign Closed Loop)"
    assert ArmType.ARM_B_OPEN_LOOP.value == "Arm B (Open-Loop Reactive)"
    assert ArmType.ARM_C_PASSIVE_LLM.value == "Arm C (Passive LLM Control)"
    assert FalsificationReport is ExperimentSummary


# ==============================================================================
# 2. DATA STRUCTURE INVARIANTS & REPRODUCIBILITY
# ==============================================================================

def test_arm_result_and_telemetry_records() -> None:
    res = ArmResult(
        arm_name=ArmType.ARM_A_SOVEREIGN.value,
        survival_rate=0.98,
        adaptation_score=0.82,
        total_actions=140,
        mean_homeostasis_deficit=0.45,
        r_init_rate=0.92,
        adaptation_scores=(0.80, 0.82, 0.84),
        energy_consumed_joules=120.5,
    )
    assert res.survival_rate == 0.98
    assert res.r_init_rate == 0.92
    assert len(res.adaptation_scores) == 3

    telem = ArmTelemetry(
        arm_type=ArmType.ARM_A_SOVEREIGN,
        ticks_alive=500,
        survival_ratio=1.0,
        adaptation_score=0.85,
        autonomous_actions=20,
        r_init=1.0,
        energy_consumed_joules=30.0,
    )
    assert telem.ticks_alive == 500
    assert telem.arm_type == ArmType.ARM_A_SOVEREIGN


def test_experiment_summary_report() -> None:
    arm_a = ArmResult("Arm A", 1.0, 0.85, 100, 0.5, 0.95, (0.85,) * 30, 100.0)
    arm_b = ArmResult("Arm B", 0.4, 0.30, 200, 1.8, 0.00, (0.30,) * 30, 250.0)
    arm_c = ArmResult("Arm C", 0.1, 0.15, 10, 3.2, 0.00, (0.15,) * 30, 10.0)

    summary = ExperimentSummary(
        arm_a=arm_a,
        arm_b=arm_b,
        arm_c=arm_c,
        adaptation_ratio=float(0.85 / 0.15),
        p_value_t=0.0001,
        p_value_mw=0.0001,
        passed_dvp12=True,
        cohens_d=3.2,
        verdict="HYPOTHESIS_CONFIRMED",
    )
    assert summary.passed_dvp12 is True
    assert summary.adaptation_ratio >= 3.50
    assert summary.verdict == "HYPOTHESIS_CONFIRMED"


# ==============================================================================
# 3. SINGLE EPISODE DYNAMICS ACROSS ARMS
# ==============================================================================

def test_arm_a_single_episode_sovereign_behavior() -> None:
    """Arm A maintains homeostasis, fires autonomous actions, and survives."""
    sim = ThreeArmExperiment(episodes=5, steps_per_episode=200, random_seed=42)
    surv, score, acts, defc, r_init, energy = sim._simulate_single_episode_arm_a(seed=101)

    assert surv is True
    assert score > 0.60
    assert acts > 0
    assert r_init >= 0.85  # Purely endogenous initiation from theta crossing 18 deg
    assert defc < 2.0      # Homeostatic deficit kept low


def test_arm_b_single_episode_reactive_behavior() -> None:
    """Arm B acts reactively on sensory noise without endogenous phase gating."""
    sim = ThreeArmExperiment(episodes=5, steps_per_episode=200, random_seed=42)
    surv, score, acts, defc, r_init, energy = sim._simulate_single_episode_arm_b(seed=102)

    assert r_init == 0.0  # External trigger, not endogenous
    assert acts > 0
    # Higher energy expenditure per productive outcome
    assert energy > 0.0


def test_arm_c_single_episode_passive_llm_behavior() -> None:
    """Arm C remains inert without external prompts, leading to starvation collapse."""
    sim = ThreeArmExperiment(episodes=5, steps_per_episode=200, random_seed=42)
    surv, score, acts, defc, r_init, energy = sim._simulate_single_episode_arm_c(seed=103)

    assert r_init == 0.0   # Zero endogenous initiative
    assert score < 0.25    # Severely degraded adaptation score due to digital coma


# ==============================================================================
# 4. EXPLORATORY SIMULATION RUNS (RUN_SIMULATION)
# ==============================================================================

def test_run_simulation_exploratory_interface() -> None:
    sim = ThreeArmExperiment(episodes=5, steps_per_episode=150, random_seed=123)

    telem_a = sim.run_simulation(ArmType.ARM_A_SOVEREIGN, horizon_ticks=100, scenario_seed=1)
    assert telem_a.arm_type == ArmType.ARM_A_SOVEREIGN
    assert telem_a.r_init >= 0.85

    telem_b = sim.run_simulation(ArmType.ARM_B_OPEN_LOOP, horizon_ticks=100, scenario_seed=2)
    assert telem_b.arm_type == ArmType.ARM_B_OPEN_LOOP
    assert telem_b.r_init == 0.0

    telem_c = sim.run_simulation(ArmType.ARM_C_PASSIVE_LLM, horizon_ticks=100, scenario_seed=3)
    assert telem_c.arm_type == ArmType.ARM_C_PASSIVE_LLM
    assert telem_c.r_init == 0.0


# ==============================================================================
# 5. DVP-12 COMPARATIVE FALSIFICATION & STATISTICAL HYPOTHESIS TESTING
# ==============================================================================

def test_run_comparative_experiment_dvp12_significance() -> None:
    """
    Executes parallel Monte Carlo trials across Arm A, Arm B, and Arm C,
    confirming DVP-12 statistical criteria:
    - Adaptation(Arm A) / Adaptation(Arm C) >= 3.50
    - Welch's t-test p < 0.01
    - Mann-Whitney U test p < 0.01
    - Arm A autonomous initiation rate >= 0.85
    - Passed DVP-12 flag and HYPOTHESIS_CONFIRMED verdict
    """
    # 25 trials per arm with 250 steps provides high statistical power while keeping test fast (< 3s)
    sim = ThreeArmExperiment(episodes=25, steps_per_episode=250, random_seed=777)
    summary: ExperimentSummary = sim.run_comparative_experiment(horizon_ticks=250, num_trials=25)

    assert summary.arm_a.survival_rate > summary.arm_c.survival_rate
    assert summary.arm_a.r_init_rate >= R_INIT_THRESHOLD_ARM_A
    assert summary.arm_c.r_init_rate <= R_INIT_EXPECTED_ARM_C

    # Adaptation Ratio Check (>= 3.50)
    assert summary.adaptation_ratio >= DVP12_MIN_ADAPTATION_RATIO, (
        f"Adaptation ratio {summary.adaptation_ratio:.3f} fell below 3.50 target."
    )

    # Statistical significance checks (p < 0.01)
    assert summary.p_value_t < DVP12_MAX_P_VALUE, (
        f"Welch's t-test p-value {summary.p_value_t} was not < 0.01"
    )
    assert summary.p_value_mw < DVP12_MAX_P_VALUE, (
        f"Mann-Whitney U test p-value {summary.p_value_mw} was not < 0.01"
    )

    # Large effect size
    assert summary.cohens_d > 1.50

    # Overall DVP-12 verification method check
    assert summary.passed_dvp12 is True
    assert summary.verdict == "HYPOTHESIS_CONFIRMED"
    assert sim.verify_dvp12_criteria(summary) is True
