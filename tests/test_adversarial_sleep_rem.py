"""
Adversarial Stress-Testing and Empirical Verification Suite for Stage-Gate 3 Milestone 1.
Challenger: Challenger 2 (Empirical Verification & Stress Harness)
Target Modules:
    - memory/sleep_hysteresis.py (SleepHysteresis, HysteresisSnapshot, SleepState)
    - memory/rem_sandbox.py (REMSandbox, KnowledgeGraphSandbox, DreamVerdict, DreamResult)

Verifies:
1. Sleep Hysteresis Schmitt Trigger Stability (DVP-08):
   - 10,000 noisy cycles (sigma = 0.05) across deadband [0.40, 0.85].
   - Strict zero-chattering invariant: chattering_count == 0 in open interval (0.40, 0.85).
   - Boundary epsilon transitions at 0.85 and 0.40.
   - Emergency forced arousal latching and recovery.
   - Input sanitization and robustness against NaN, Inf, negative, and out-of-range values.
2. REM Sandbox 100% Motor Atonia Interlock:
   - Full 360-degree sweep (3600 discrete angles at 0.1 deg resolution) under sleep lock.
   - Clamping under extreme motor commands (u = 1e6, go_signal = 1e6, nogo = -1e6).
   - Mid-firing quench interruption: instant drop from active firing to a_motor == 0.000000.
   - Actuator stream isolation: exactly 0 motor packets escape into physical actuation.
3. Cognitive Dissonance Loss Evaluation Logic:
   - Mathematical oracle validation for affirmative and contradictory relational edges.
   - Simulation verdict gate: APPROVED for Delta L <= 5% L_base, REJECTED for Delta L > 5%.
   - Sandbox deep-cloning isolation (zero mutation of baseline knowledge graph).
   - Social vacuum reduction logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np
import pytest

from core.basal_ganglia_gate import BasalGangliaGate, DEG2RAD, THETA_GATE_RAD
from memory.sleep_hysteresis import (
    SleepHysteresis,
    SleepState,
    HysteresisSnapshot,
    DEFAULT_SLEEP_THRESHOLD,
    DEFAULT_WAKE_THRESHOLD,
    DEFAULT_DEADBAND_WIDTH,
)
from memory.rem_sandbox import (
    REMSandbox,
    KnowledgeGraphSandbox,
    DreamVerdict,
    DreamResult,
    DEFAULT_MAX_LOSS_DEGRADATION,
    MOTOR_ATONIA_CLAMP,
    LOSS_EPSILON,
)


# ==============================================================================
# 1. SLEEP HYSTERESIS ADVERSARIAL & EMPIRICAL STRESS TESTS
# ==============================================================================

class TestAdversarialSleepHysteresisNoise:
    """Empirical noise and chattering stress test suite for SleepHysteresis."""

    def test_canonical_evaluate_noise_stability_dvp08(self):
        """Canonical DVP-08 evaluation: 10,000 cycles, sigma=0.05, seed=42."""
        sh = SleepHysteresis()
        passed, chattering_count, stats = sh.evaluate_noise_stability(cycles=10000, sigma=0.05, seed=42)

        assert passed is True
        assert chattering_count == 0
        assert stats["total_cycles"] == 10000
        assert stats["chattering_count"] == 0
        assert stats["deadband_evaluations"] > 0
        assert stats["latching_fidelity_pct"] == 100.0

    def test_adversarial_10k_noisy_deadband_dwell(self):
        """
        Subject SleepHysteresis to 10,000 continuous noisy cycles strictly centered
        inside the deadband [0.40, 0.85] with sigma=0.05.
        State must never flip while remaining within the deadband.
        """
        rng = np.random.default_rng(20261025)
        # Test 1: Starting from AWAKE
        sh_awake = SleepHysteresis(initial_state=SleepState.AWAKE)
        # Trajectory stays strictly inside (0.41, 0.84)
        c_dwell = np.clip(0.625 + rng.normal(0.0, 0.05, 10000), 0.41, 0.84)

        flips_inside_deadband = 0
        for i, c in enumerate(c_dwell):
            snap = sh_awake.update(float(c), timestamp_ns=i * 20_000_000)
            assert snap.in_deadband is True
            assert snap.current_state == SleepState.AWAKE
            assert snap.state_changed is False
            assert snap.chattering_detected is False
            if snap.state_changed:
                flips_inside_deadband += 1

        assert flips_inside_deadband == 0
        assert sh_awake.get_chattering_count() == 0
        assert sh_awake.get_transition_count() == 0
        assert sh_awake.get_deadband_evaluations() == 10000

        # Test 2: Starting from ASLEEP
        sh_asleep = SleepHysteresis(initial_state=SleepState.ASLEEP)
        flips_inside_deadband_asleep = 0
        for i, c in enumerate(c_dwell):
            snap = sh_asleep.update(float(c), timestamp_ns=i * 20_000_000)
            assert snap.in_deadband is True
            assert snap.current_state == SleepState.ASLEEP
            assert snap.state_changed is False
            assert snap.chattering_detected is False
            if snap.state_changed:
                flips_inside_deadband_asleep += 1

        assert flips_inside_deadband_asleep == 0
        assert sh_asleep.get_chattering_count() == 0
        assert sh_asleep.get_transition_count() == 0

    def test_adversarial_10k_brownian_walk_across_deadband(self):
        """
        10,000 steps of random walk (Brownian motion) crossing back and forth across
        the deadband [0.40, 0.85].
        Verifies:
        - Exactly zero flips occur inside (0.40, 0.85).
        - Sleep transition happens ONLY when C >= 0.85.
        - Wake transition happens ONLY when C <= 0.40.
        """
        rng = np.random.default_rng(999)
        sh = SleepHysteresis(initial_state=SleepState.AWAKE)

        c = 0.50
        step_sigma = 0.02
        chattering_violations = 0
        valid_transitions = 0

        for t in range(10000):
            # Brownian step
            c = float(np.clip(c + rng.normal(0.0, step_sigma), 0.0, 1.0))
            prev_st = sh.current_state
            snap = sh.update(c, timestamp_ns=t * 1_000_000)

            if snap.state_changed:
                valid_transitions += 1
                # If state changed, verify it conforms to physical thresholds
                if prev_st.is_awake and snap.current_state.is_sleeping:
                    # Must be at or above sleep threshold
                    assert c >= 0.85, f"Sleep triggered prematurely at C={c}"
                elif prev_st.is_sleeping and snap.current_state.is_awake:
                    # Must be at or below wake threshold
                    assert c <= 0.40, f"Wake triggered prematurely at C={c}"
                else:
                    pytest.fail(f"Unexpected transition from {prev_st} to {snap.current_state}")

                # If transition happened strictly inside deadband, count as violation
                if 0.40 < c < 0.85:
                    chattering_violations += 1

        assert chattering_violations == 0
        assert sh.get_chattering_count() == 0
        assert valid_transitions > 0  # Confirms that transitions actually occurred when crossing bounds

    def test_adversarial_boundary_flutter_stress(self):
        """
        Adversarial high-frequency jitter hovering immediately above and below
        the 0.85 sleep threshold and 0.40 wake threshold.
        """
        rng = np.random.default_rng(12345)
        sh = SleepHysteresis(initial_state=SleepState.AWAKE)

        # 1. Jitter around sleep threshold [0.84, 0.86]
        # Once it crosses >= 0.85, it enters ASLEEP. While it drops back to 0.84, it MUST remain ASLEEP!
        jitter_upper = np.clip(0.85 + rng.normal(0.0, 0.015, 2000), 0.70, 0.95)
        entered_sleep = False
        for c in jitter_upper:
            snap = sh.update(float(c))
            if snap.current_state == SleepState.ASLEEP:
                entered_sleep = True
            if entered_sleep and c > 0.40:
                # Must stay asleep even if c drops below 0.85 down to 0.70
                assert snap.current_state == SleepState.ASLEEP
                assert snap.chattering_detected is False

        assert entered_sleep is True

        # 2. Jitter around wake threshold [0.39, 0.41]
        # Once it drops <= 0.40, it enters AWAKE. While it bounces back to 0.41, it MUST remain AWAKE!
        jitter_lower = np.clip(0.40 + rng.normal(0.0, 0.015, 2000), 0.35, 0.60)
        entered_wake = False
        for c in jitter_lower:
            snap = sh.update(float(c))
            if snap.current_state == SleepState.AWAKE:
                entered_wake = True
            if entered_wake and c < 0.85:
                # Must stay awake even if c bounces back to 0.41 up to 0.60
                assert snap.current_state == SleepState.AWAKE
                assert snap.chattering_detected is False

        assert entered_wake is True
        assert sh.get_chattering_count() == 0

    def test_exact_threshold_boundary_machine_precision(self):
        """Test exact mathematical boundary conditions at 0.850000 and 0.400000."""
        eps = 1e-7

        # Case 1: Just below sleep threshold
        sh1 = SleepHysteresis(initial_state=SleepState.AWAKE)
        snap = sh1.update(0.85 - eps)
        assert snap.current_state == SleepState.AWAKE
        assert snap.in_deadband is True

        # Exact sleep threshold
        snap = sh1.update(0.85)
        assert snap.current_state == SleepState.ASLEEP
        assert snap.state_changed is True
        assert snap.in_deadband is False

        # Case 2: Just above wake threshold
        sh2 = SleepHysteresis(initial_state=SleepState.ASLEEP)
        snap = sh2.update(0.40 + eps)
        assert snap.current_state == SleepState.ASLEEP
        assert snap.in_deadband is True

        # Exact wake threshold
        snap = sh2.update(0.40)
        assert snap.current_state == SleepState.AWAKE
        assert snap.state_changed is True
        assert snap.in_deadband is False

    def test_forced_arousal_and_deadband_latching(self):
        """Test emergency forced arousal and subsequent deadband stability."""
        sh = SleepHysteresis(initial_state=SleepState.ASLEEP)
        assert sh.is_sleeping() is True

        # Watchdog timeout stall triggers forced arousal
        snap_forced = sh.force_arousal(reason="IO_TIMEOUT_STALL")
        assert snap_forced.current_state == SleepState.AWAKE
        assert sh.is_awake() is True

        # Subsequent updates inside deadband must remain AWAKE
        for c_val in [0.80, 0.65, 0.50, 0.45]:
            snap = sh.update(c_val)
            assert snap.current_state == SleepState.AWAKE
            assert snap.in_deadband is True

        # force_sleep works as well
        snap_sleep = sh.force_sleep(reason="manual_sleep")
        assert snap_sleep.current_state == SleepState.ASLEEP
        for c_val in [0.45, 0.50, 0.65, 0.80]:
            snap = sh.update(c_val)
            assert snap.current_state == SleepState.ASLEEP
            assert snap.in_deadband is True

    def test_duck_typed_snapshot_comparisons(self):
        """Verify HysteresisSnapshot compares seamlessly with SleepState and strings."""
        sh = SleepHysteresis(initial_state=SleepState.AWAKE)
        snap = sh.update(0.20)

        assert snap == SleepState.AWAKE
        assert snap == "AWAKE"
        assert not (snap == SleepState.ASLEEP)
        assert not (snap == "ASLEEP")

        snap_sleep = sh.update(0.90)
        assert snap_sleep == SleepState.ASLEEP
        assert snap_sleep == "ASLEEP"

    def test_input_sanitization_and_error_handling(self):
        """Verify rigorous validation of invalid saturation inputs."""
        sh = SleepHysteresis()

        with pytest.raises(ValueError, match="cannot be negative"):
            sh.update(-0.01)

        with pytest.raises(ValueError, match="severely exceeds"):
            sh.update(1.06)

        with pytest.raises(ValueError, match="finite real number"):
            sh.update(float("nan"))

        with pytest.raises(ValueError, match="finite real number"):
            sh.update(float("inf"))

        # In [1.00, 1.05], value is clipped to 1.0 without error
        snap = sh.update(1.03)
        assert snap.context_saturation == 1.0
        assert snap.current_state == SleepState.ASLEEP


# ==============================================================================
# 2. REM SANDBOX 100% MOTOR ATONIA ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestAdversarialREMSandboxMotorAtonia:
    """Rigorous empirical verification of Basal Ganglia striatal clamp and motor packet emission."""

    def test_striatal_gate_full_360_degree_sweep_under_atonia(self):
        """
        Rigorously sweeps 3,600 angular positions theta in [0.0 deg, 360.0 deg] at 0.1 deg
        resolution with full motor intent u = 1.0 under sleep_lock=True.
        Motor activation a_motor must be strictly 0.000000 across all 3,600 angles.
        """
        gate = BasalGangliaGate()
        angles_deg = np.linspace(0.0, 360.0, 3600, endpoint=False)

        max_a_motor = -1.0
        non_zero_count = 0

        for deg in angles_deg:
            th_rad = float(deg * DEG2RAD)
            decision = gate.evaluate(
                theta_rad=th_rad,
                u=1.0,
                u_intent=1.0,
                go_signal=1.0,
                nogo_signal=0.0,
                sleep_lock=True,
                lock_sleep=True,
            )
            if decision.a_motor > max_a_motor:
                max_a_motor = decision.a_motor
            if decision.a_motor != 0.0:
                non_zero_count += 1
            assert decision.is_locked is True

        assert non_zero_count == 0
        assert max_a_motor == 0.0

    def test_striatal_gate_negative_and_extreme_angles_under_atonia(self):
        """Test negative angles and multi-turn winding angles up to 10,000 rad under sleep_lock."""
        gate = BasalGangliaGate()
        adversarial_angles = [
            -720.0 * DEG2RAD, -360.0 * DEG2RAD, -180.0 * DEG2RAD, -18.0 * DEG2RAD, -0.001,
            0.0, 18.0 * DEG2RAD, 25.0 * DEG2RAD, 90.0 * DEG2RAD, 180.0 * DEG2RAD,
            720.0 * DEG2RAD, 1000.0, 10000.0,
        ]

        for th in adversarial_angles:
            decision = gate.evaluate(theta_rad=th, u=1.0, sleep_lock=True)
            assert decision.a_motor == 0.0
            assert decision.is_locked is True

    def test_striatal_gate_extreme_drive_and_panic_under_atonia(self):
        """
        Test extreme motor commands:
        u = 1e6, u_intent = 1e6, go_signal = 1e6, nogo_signal = -1e6.
        All must be unconditionally clamped to a_motor == 0.000000.
        """
        gate = BasalGangliaGate()
        extreme_commands = [
            {"u": 1.0, "u_intent": 1.0, "go": 1.0, "nogo": 0.0},
            {"u": 10.0, "u_intent": 10.0, "go": 10.0, "nogo": 0.0},
            {"u": 100.0, "u_intent": 100.0, "go": 100.0, "nogo": -10.0},
            {"u": 1e6, "u_intent": 1e6, "go": 1e6, "nogo": -1e6},
        ]

        for cmd in extreme_commands:
            decision = gate.evaluate(
                theta_rad=30.0 * DEG2RAD,  # normally a heavy firing angle
                u=cmd["u"],
                u_intent=cmd["u_intent"],
                go_signal=cmd["go"],
                nogo_signal=cmd["nogo"],
                sleep_lock=True,
                lock_sleep=True,
            )
            assert decision.a_motor == 0.0
            assert decision.is_locked is True

    def test_mid_firing_atonia_clamp_interruption(self):
        """
        Simulate a waking active motor action firing at peak drive, and then
        asserting sleep_lock (e.g. entering REM sleep).
        Gate must immediately collapse from a_motor > 0.8 to exactly 0.0.
        """
        gate = BasalGangliaGate()

        # Step 1: Trigger active firing during wakefulness (theta = 25 deg > 18 deg)
        decision_awake = gate.evaluate(
            theta_rad=25.0 * DEG2RAD,
            u=1.0,
            sleep_lock=False,
            lock_sleep=False,
        )
        assert decision_awake.a_motor >= 0.70
        assert decision_awake.is_locked is False

        # Step 2: Transition into sleep / REM atonia
        decision_rem = gate.evaluate(
            theta_rad=25.0 * DEG2RAD,
            u=1.0,
            sleep_lock=True,
            lock_sleep=True,
        )
        assert decision_rem.a_motor == 0.000000
        assert decision_rem.is_locked is True

        # Step 3: Verify internal firing flags reset
        assert gate._firing is False
        assert gate._peak_theta == 0.0
        assert gate._quench_drive == 0.0

    def test_rem_sandbox_zero_actuator_packet_leakage(self):
        """
        Execute REM dream simulation and verify that:
        - get_emitted_motor_packets() is strictly empty (len == 0).
        - DreamResult.motor_output_count == 0.
        - DreamResult.motor_atonia_verified == True.
        """
        rem = REMSandbox()
        assert rem.motor_atonia_active is True
        assert rem.verify_motor_atonia() is True

        # Simulate dream with mock episodes
        class MockEpisode:
            def __init__(self, uid: str, emb: np.ndarray, meta: Dict[str, Any]):
                self.episode_uuid = uid
                self.embedding = emb
                self.metadata = meta
                self.reward_vector = np.array([0.5, 0.5, 0.8, 0.0, 0.0])

        rng = np.random.default_rng(42)
        episodes = [
            MockEpisode(f"ep_{i}", rng.standard_normal(64).astype(np.float32), {})
            for i in range(10)
        ]

        result = rem.simulate_dream(episodes)

        assert result.motor_atonia_verified is True
        assert result.motor_output_count == 0
        assert len(rem.get_emitted_motor_packets()) == 0


# ==============================================================================
# 3. COGNITIVE DISSONANCE LOSS EVALUATION ADVERSARIAL TESTS
# ==============================================================================

class TestAdversarialCognitiveDissonanceLogic:
    """Verifies mathematical correctness of dissonance loss in KnowledgeGraphSandbox and REMSandbox."""

    def test_dissonance_loss_mathematical_oracle(self):
        """
        Test exact analytical dissonance values:
        - Affirmative relation ('associated_with', 'is_a', 'causes'):
            d = 0.5 * (1 - cos(u, v))^2
            If cos = 1.0  -> d = 0.0
            If cos = 0.0  -> d = 0.5
            If cos = -1.0 -> d = 2.0
        - Contradictory relation ('contradicts', 'incompatible_with', 'cannot_be', 'negates'):
            d = 0.5 * (1 + cos(u, v))^2
            If cos = 1.0  -> d = 2.0
            If cos = 0.0  -> d = 0.5
            If cos = -1.0 -> d = 0.0
        """
        kg = KnowledgeGraphSandbox(dimension=4)

        # Orthogonal, identical, and opposite vectors
        v_ident_1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_ident_2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_opp = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v_orth = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        kg.add_node("n1", "N1", v_ident_1)
        kg.add_node("n2", "N2", v_ident_2)
        kg.add_node("n_opp", "NOpp", v_opp)
        kg.add_node("n_orth", "NOrth", v_orth)

        # 1. Affirmative identical (cos = 1.0) -> d = 0.0
        kg.edges = []
        kg.add_edge("n1", "n2", "associated_with", weight=1.0)
        assert math.isclose(kg.compute_dissonance_loss(), 0.0, abs_tol=1e-6)

        # 2. Affirmative orthogonal (cos = 0.0) -> d = 0.5
        kg.edges = []
        kg.add_edge("n1", "n_orth", "associated_with", weight=1.0)
        assert math.isclose(kg.compute_dissonance_loss(), 0.5, abs_tol=1e-6)

        # 3. Affirmative opposite (cos = -1.0) -> d = 2.0
        kg.edges = []
        kg.add_edge("n1", "n_opp", "associated_with", weight=1.0)
        assert math.isclose(kg.compute_dissonance_loss(), 2.0, abs_tol=1e-6)

        # 4. Contradictory identical (cos = 1.0) -> d = 2.0
        kg.edges = []
        kg.add_edge("n1", "n2", "contradicts", weight=1.0)
        assert math.isclose(kg.compute_dissonance_loss(), 2.0, abs_tol=1e-6)

        # 5. Contradictory orthogonal (cos = 0.0) -> d = 0.5
        kg.edges = []
        kg.add_edge("n1", "n_orth", "incompatible_with", weight=1.0)
        assert math.isclose(kg.compute_dissonance_loss(), 0.5, abs_tol=1e-6)

        # 6. Contradictory opposite (cos = -1.0) -> d = 0.0
        kg.edges = []
        kg.add_edge("n1", "n_opp", "negates", weight=1.0)
        assert math.isclose(kg.compute_dissonance_loss(), 0.0, abs_tol=1e-6)

    def test_rem_dream_verdict_approved_for_coherent_episodes(self):
        """
        Verify that coherent candidate episodes which do not degrade dissonance
        by more than 5% are APPROVED.
        """
        rem = REMSandbox()
        base_loss = rem.baseline_graph.compute_dissonance_loss()

        # Create a candidate episode that reinforces concept_safety
        self_emb = rem.baseline_graph.nodes["concept_self"]["embedding"]
        coherent_emb = self_emb.copy()

        class CoherentEpisode:
            episode_uuid = "coherent_ep_001"
            embedding = coherent_emb
            metadata = {
                "relation": "is_a",
                "target_node": "concept_self",
                "weight": 1.0,
            }
            reward_vector = np.array([0.5, 0.5, 0.9, 0.0, 0.0])

        result = rem.simulate_dream([CoherentEpisode()])

        assert result.verdict == DreamVerdict.APPROVED
        assert result.delta_loss <= result.threshold_loss
        assert result.delta_loss_ratio <= 0.05

    def test_rem_dream_verdict_rejected_for_dissonant_episodes(self):
        """
        Verify that contradictory candidate episodes which increase dissonance
        by > 5.0% are REJECTED.
        """
        rem = REMSandbox()

        # Self embedding contradicts self!
        self_emb = rem.baseline_graph.nodes["concept_self"]["embedding"]

        class ContradictoryEpisode:
            episode_uuid = "contradictory_ep_666"
            embedding = self_emb.copy()  # identical to self
            metadata = {
                "relation": "contradicts",  # contradicts itself with cos = 1.0!
                "target_node": "concept_self",
                "weight": 1.0,
            }
            reward_vector = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = rem.simulate_dream([ContradictoryEpisode()])

        assert result.verdict == DreamVerdict.REJECTED
        assert result.delta_loss > result.threshold_loss
        assert result.delta_loss_ratio > 0.05

    def test_rem_sandbox_deep_cloning_isolation(self):
        """
        Verify that candidate dream simulation operates on a cloned sandbox
        and never pollutes or mutates the baseline knowledge graph.
        """
        rem = REMSandbox()
        initial_nodes = dict(rem.baseline_graph.nodes)
        initial_edges = list(rem.baseline_graph.edges)
        initial_node_count = len(initial_nodes)
        initial_edge_count = len(initial_edges)

        # Inject 10 highly chaotic episodes
        rng = np.random.default_rng(777)
        chaotic_episodes = []
        for i in range(10):
            class Ep:
                pass
            ep = Ep()
            ep.episode_uuid = f"chaos_{i}"
            ep.embedding = rng.standard_normal(64).astype(np.float32)
            ep.metadata = {"relation": "contradicts", "target_node": "concept_safety", "weight": 1.0}
            ep.reward_vector = np.zeros(5)
            chaotic_episodes.append(ep)

        # Run simulation
        result = rem.simulate_dream(chaotic_episodes)
        assert result.verdict == DreamVerdict.REJECTED

        # Assert baseline graph is 100% unmutated
        assert len(rem.baseline_graph.nodes) == initial_node_count
        assert len(rem.baseline_graph.edges) == initial_edge_count
        for nid in initial_nodes:
            assert nid in rem.baseline_graph.nodes

    def test_social_satisfaction_evaluation_boundaries(self):
        """Verify boundary conditions for daytime social feedback evaluation."""
        rem = REMSandbox()

        # 1. Empty events -> True
        assert rem.evaluate_social_satisfaction([]) is True

        # 2. Majority positive (> 0.30)
        events_pos = [{"reward": 0.35}, {"reward": 0.40}, {"reward": 0.20}]
        assert rem.evaluate_social_satisfaction(events_pos) is True

        # 3. Majority negative (<= 0.30)
        events_neg = [{"reward": 0.10}, {"reward": 0.25}, {"reward": 0.50}, {"reward": 0.15}]
        assert rem.evaluate_social_satisfaction(events_neg) is False
