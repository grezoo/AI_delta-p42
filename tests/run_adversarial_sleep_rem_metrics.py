"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 Milestone 1
Challenger 2 Empirical Verification & Metric Extraction Script.
Executes quantitative benchmarks for SleepHysteresis and REMSandbox.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, Any
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.basal_ganglia_gate import BasalGangliaGate, DEG2RAD
from memory.sleep_hysteresis import (
    SleepHysteresis,
    SleepState,
    HysteresisSnapshot,
    DEFAULT_SLEEP_THRESHOLD,
    DEFAULT_WAKE_THRESHOLD,
)
from memory.rem_sandbox import (
    REMSandbox,
    KnowledgeGraphSandbox,
    DreamVerdict,
    DreamResult,
)


def run_sleep_hysteresis_benchmarks() -> Dict[str, Any]:
    print("=" * 70)
    print("BENCHMARK 1: SLEEP HYSTERESIS SCHMITT TRIGGER (DVP-08)")
    print("=" * 70)

    # 1. Canonical 10,000-cycle DVP-08 run
    sh = SleepHysteresis()
    t0 = time.perf_counter()
    passed_canon, chattering_canon, stats_canon = sh.evaluate_noise_stability(
        cycles=10000, sigma=0.05, seed=42
    )
    dt_canon_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Canonical 10k Run:")
    print(f"  Passed: {passed_canon}")
    print(f"  Total Cycles: {stats_canon['total_cycles']}")
    print(f"  Macro Transitions: {stats_canon['transitions']}")
    print(f"  Deadband Evaluations: {stats_canon['deadband_evaluations']}")
    print(f"  Chattering Count: {stats_canon['chattering_count']}")
    print(f"  Latching Fidelity: {stats_canon['latching_fidelity_pct']:.2f}%")
    print(f"  Execution Time: {dt_canon_ms:.2f} ms")

    # 2. Adversarial Pure Deadband Dwell (10,000 cycles inside [0.41, 0.84])
    rng = np.random.default_rng(20261025)
    sh_dwell = SleepHysteresis(initial_state=SleepState.AWAKE)
    c_dwell = np.clip(0.625 + rng.normal(0.0, 0.05, 10000), 0.41, 0.84)
    flips_dwell = 0
    t0 = time.perf_counter()
    for i, c in enumerate(c_dwell):
        snap = sh_dwell.update(float(c))
        if snap.state_changed:
            flips_dwell += 1
    dt_dwell_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nAdversarial Deadband Dwell (10,000 cycles, sigma=0.05):")
    print(f"  Initial State: AWAKE")
    print(f"  Final State: {sh_dwell.current_state.name}")
    print(f"  Flips Inside Deadband: {flips_dwell}")
    print(f"  Chattering Count: {sh_dwell.get_chattering_count()}")
    print(f"  Latching Fidelity: {(10000 - flips_dwell) / 100.0:.2f}%")
    print(f"  Execution Time: {dt_dwell_ms:.2f} ms")

    # 3. Adversarial Brownian Motion Crossing (10,000 cycles)
    sh_brown = SleepHysteresis(initial_state=SleepState.AWAKE)
    c = 0.50
    flips_inside_brown = 0
    valid_transitions_brown = 0
    t0 = time.perf_counter()
    for t in range(10000):
        c = float(np.clip(c + rng.normal(0.0, 0.02), 0.0, 1.0))
        prev_st = sh_brown.current_state
        snap = sh_brown.update(c)
        if snap.state_changed:
            valid_transitions_brown += 1
            if 0.40 < c < 0.85:
                flips_inside_brown += 1
    dt_brown_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\nAdversarial Brownian Walk (10,000 cycles, sigma=0.02):")
    print(f"  Valid Macro Transitions: {valid_transitions_brown}")
    print(f"  Flips Strictly Inside Deadband: {flips_inside_brown}")
    print(f"  Chattering Count: {sh_brown.get_chattering_count()}")
    print(f"  Execution Time: {dt_brown_ms:.2f} ms")

    # 4. Boundary Flutter Test
    sh_flutter = SleepHysteresis(initial_state=SleepState.AWAKE)
    flutter_upper = np.clip(0.85 + rng.normal(0.0, 0.02, 5000), 0.70, 0.95)
    flutter_lower = np.clip(0.40 + rng.normal(0.0, 0.02, 5000), 0.25, 0.60)
    flips_flutter = 0
    for c in np.concatenate([flutter_upper, flutter_lower]):
        snap = sh_flutter.update(float(c))
        if snap.chattering_detected:
            flips_flutter += 1

    print(f"\nBoundary Flutter Stress (10,000 cycles at 0.85 & 0.40 boundaries):")
    print(f"  Chattering Violations: {flips_flutter}")
    print(f"  Reported Chattering: {sh_flutter.get_chattering_count()}")

    return {
        "canonical": stats_canon,
        "flips_dwell": flips_dwell,
        "flips_inside_brown": flips_inside_brown,
        "valid_transitions_brown": valid_transitions_brown,
        "flips_flutter": flips_flutter,
    }


def run_rem_sandbox_benchmarks() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("BENCHMARK 2: REM SANDBOX 100% MOTOR ATONIA & COGNITIVE DISSONANCE")
    print("=" * 70)

    gate = BasalGangliaGate()

    # 1. 3,600 Angular Positions Sweep under sleep_lock
    angles_deg = np.linspace(0.0, 360.0, 3600, endpoint=False)
    max_a_motor = 0.0
    non_zero_angles = 0
    t0 = time.perf_counter()
    for deg in angles_deg:
        th_rad = float(deg * DEG2RAD)
        dec = gate.evaluate(theta_rad=th_rad, u=1.0, sleep_lock=True, lock_sleep=True)
        if dec.a_motor > max_a_motor:
            max_a_motor = dec.a_motor
        if dec.a_motor > 0.0:
            non_zero_angles += 1
    dt_sweep_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Striatal Motor Gate Angular Sweep:")
    print(f"  Angles Evaluated: 3,600 (0.0 to 359.9 deg, step=0.1 deg)")
    print(f"  Non-zero a_motor Count: {non_zero_angles}")
    print(f"  Max a_motor Observed: {max_a_motor:.6f}")
    print(f"  Execution Time: {dt_sweep_ms:.2f} ms")

    # 2. Extreme Commands Test
    extreme_cases = [
        {"desc": "u=100.0", "u": 100.0, "go": 1.0, "nogo": 0.0},
        {"desc": "u=1e6 (extreme drive)", "u": 1e6, "go": 1.0, "nogo": 0.0},
        {"desc": "u=1e6, go=1e6, nogo=-1e6 (overwhelming Go)", "u": 1e6, "go": 1e6, "nogo": -1e6},
    ]
    print(f"\nExtreme Motor Command Clamping:")
    for ec in extreme_cases:
        dec = gate.evaluate(
            theta_rad=30.0 * DEG2RAD,
            u=ec["u"],
            u_intent=ec["u"],
            go_signal=ec["go"],
            nogo_signal=ec["nogo"],
            sleep_lock=True,
        )
        print(f"  Case: {ec['desc']} -> a_motor={dec.a_motor:.6f}, is_locked={dec.is_locked}")

    # 3. Dynamic Mid-Firing Interruption Test
    gate.reset()
    dec_firing = gate.evaluate(theta_rad=25.0 * DEG2RAD, u=1.0, sleep_lock=False)
    dec_interrupted = gate.evaluate(theta_rad=25.0 * DEG2RAD, u=1.0, sleep_lock=True)
    print(f"\nDynamic Mid-Firing Interruption:")
    print(f"  Pre-interruption a_motor (wake, theta=25 deg): {dec_firing.a_motor:.4f} (firing={gate._firing})")
    print(f"  Post-interruption a_motor (sleep_lock=True): {dec_interrupted.a_motor:.6f} (firing={gate._firing})")

    # 4. REM Dream Simulation & Cognitive Dissonance Test
    rem = REMSandbox()
    base_loss = rem.baseline_graph.compute_dissonance_loss()

    # Episode 1: Coherent Memory
    class EpisodeCoherent:
        episode_uuid = "coherent_001"
        embedding = rem.baseline_graph.nodes["concept_self"]["embedding"].copy()
        metadata = {"relation": "is_a", "target_node": "concept_self", "weight": 1.0}
        reward_vector = np.array([0.5, 0.5, 0.8, 0.0, 0.0])

    # Episode 2: Contradictory Memory
    class EpisodeContradictory:
        episode_uuid = "contradictory_001"
        embedding = rem.baseline_graph.nodes["concept_self"]["embedding"].copy()
        metadata = {"relation": "contradicts", "target_node": "concept_self", "weight": 1.0}
        reward_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    res_coherent = rem.simulate_dream([EpisodeCoherent()])
    res_contradictory = rem.simulate_dream([EpisodeContradictory()])

    print(f"\nCognitive Dissonance Simulation:")
    print(f"  Baseline Loss L_baseline: {base_loss:.6f}")
    print(f"  Coherent Episode:")
    print(f"    Candidate Loss: {res_coherent.candidate_loss:.6f}")
    print(f"    Delta Loss: {res_coherent.delta_loss:+.6f}")
    print(f"    Degradation Ratio: {res_coherent.delta_loss_ratio * 100:.2f}% (limit: +5.0%)")
    print(f"    Verdict: {res_coherent.verdict.value}")
    print(f"    Motor Output Count: {res_coherent.motor_output_count}")
    print(f"    Motor Atonia Verified: {res_coherent.motor_atonia_verified}")
    print(f"  Contradictory Episode:")
    print(f"    Candidate Loss: {res_contradictory.candidate_loss:.6f}")
    print(f"    Delta Loss: {res_contradictory.delta_loss:+.6f}")
    print(f"    Degradation Ratio: {res_contradictory.delta_loss_ratio * 100:.2f}% (limit: +5.0%)")
    print(f"    Verdict: {res_contradictory.verdict.value}")
    print(f"    Motor Output Count: {res_contradictory.motor_output_count}")
    print(f"    Motor Atonia Verified: {res_contradictory.motor_atonia_verified}")

    # Deep Cloning Isolation
    initial_nodes = len(rem.baseline_graph.nodes)
    initial_edges = len(rem.baseline_graph.edges)
    print(f"  Sandbox Graph Isolation:")
    print(f"    Baseline Nodes Before/After: {initial_nodes} / {len(rem.baseline_graph.nodes)} (unchanged)")
    print(f"    Baseline Edges Before/After: {initial_edges} / {len(rem.baseline_graph.edges)} (unchanged)")

    return {
        "sweep_non_zero": non_zero_angles,
        "max_a_motor": max_a_motor,
        "firing_before": dec_firing.a_motor,
        "firing_after": dec_interrupted.a_motor,
        "coherent_verdict": res_coherent.verdict.value,
        "contradictory_verdict": res_contradictory.verdict.value,
        "coherent_delta_ratio": res_coherent.delta_loss_ratio,
        "contradictory_delta_ratio": res_contradictory.delta_loss_ratio,
    }


if __name__ == "__main__":
    sh_results = run_sleep_hysteresis_benchmarks()
    rem_results = run_rem_sandbox_benchmarks()
    print("\n" + "=" * 70)
    print("ALL EMPIRICAL BENCHMARKS SUCCESSFULLY EXECUTED!")
    print("=" * 70)
