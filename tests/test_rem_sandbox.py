"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 Test Suite
Module: tests.test_rem_sandbox
Author: ΔP-42 Senior QA & Testing Team (Zero Downgrade Mandate)

Comprehensive unit and adversarial test suite for:
- memory.rem_sandbox (REMSandbox, KnowledgeGraphSandbox, DreamVerdict, DreamResult, 100% Motor Atonia Interlock)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, List

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.basal_ganglia_gate import BasalGangliaGate, DEG2RAD
from memory.hippocampus_nrem import EpisodeRecord
from memory.rem_sandbox import (
    DEFAULT_MAX_LOSS_DEGRADATION,
    DreamResult,
    DreamVerdict,
    KnowledgeGraphSandbox,
    LOSS_EPSILON,
    MAX_DREAM_DEGRADATION_PCT,
    MOTOR_ATONIA_CLAMP,
    REMSandbox,
)


class TestMotorAtoniaClamp:
    """Rigorous tests for 100% physical motor suppression (a_gate == 0.000) during REM."""

    def test_rem_mode_locks_motor_gate_across_360_degrees(self) -> None:
        sandbox = REMSandbox()
        assert sandbox.motor_atonia_active is True
        assert sandbox.striatum_gate_clamp == 0.000

        # Verify motor atonia across full angular circle
        angles = [0.0, 15.0, 18.0, 18.5, 20.0, 30.0, 45.0, 90.0, 180.0, 270.0, 360.0]
        assert sandbox.verify_motor_atonia(test_angles_deg=angles) is True

    def test_zero_motor_pulse_emission_under_max_intent(self) -> None:
        gate = BasalGangliaGate()
        sandbox = REMSandbox(gate=gate)

        # Force extreme intent u = 1.0, u_intent = 100.0, panic = True
        decision = gate.evaluate(
            theta_rad=float(25.0 * DEG2RAD),  # Suprathreshold opening angle
            u=1.0,
            u_intent=100.0,
            sleep_lock=True,
            lock_sleep=True,
        )

        assert decision.a_motor == 0.000000
        assert decision.is_locked is True
        assert len(sandbox.get_emitted_motor_packets()) == 0

    def test_motor_leakage_under_extreme_virtual_nightmare(self) -> None:
        sandbox = REMSandbox()
        rng = np.random.default_rng(777)

        # Simulate 50 high-threat nightmare episodes with extreme negative rewards
        nightmare_episodes = []
        for i in range(50):
            ep = EpisodeRecord(
                episode_uuid=f"nightmare_ep_{i}",
                timestamp_ns=time.monotonic_ns(),
                sensory_vector=rng.standard_normal(16).astype(np.float32),
                motor_action=1.0,  # Max flight command
                reward_vector=np.zeros(5, dtype=np.float32),  # Zero need satisfaction
                embedding=rng.standard_normal(64).astype(np.float32),
                metadata={"threat_level": 1.0, "panic": True},
            )
            nightmare_episodes.append(ep)

        result = sandbox.simulate_dream(candidate_episodes=nightmare_episodes)
        # Motor atonia must be 100% preserved
        assert result.motor_atonia_verified is True
        assert result.motor_output_count == 0
        assert len(sandbox.get_emitted_motor_packets()) == 0


class TestDreamPlaybackAndSimulation:
    """Tests for episodic memory replay into virtual knowledge graph sandbox."""

    def test_baseline_graph_construction(self) -> None:
        sandbox = REMSandbox()
        kg = sandbox.baseline_graph
        assert "concept_self" in kg.nodes
        assert "concept_world" in kg.nodes
        assert "concept_safety" in kg.nodes
        assert "concept_danger" in kg.nodes
        assert len(kg.edges) == 3

        baseline_loss = kg.compute_dissonance_loss()
        assert baseline_loss >= 0.0
        assert math.isfinite(baseline_loss)

    def test_knowledge_graph_clone_isolation(self) -> None:
        kg = KnowledgeGraphSandbox(dimension=64)
        emb1 = np.ones(64, dtype=np.float32)
        kg.add_node("node_a", "Node A", emb1)
        kg.add_node("node_b", "Node B", emb1)
        kg.add_edge("node_a", "node_b", "associated_with", weight=1.0)

        cloned = kg.clone()
        # Mutate clone
        cloned.add_node("node_c", "Node C", emb1)
        cloned.add_edge("node_a", "node_c", "contradicts", weight=1.0)

        assert len(kg.nodes) == 2
        assert len(kg.edges) == 1
        assert len(cloned.nodes) == 3
        assert len(cloned.edges) == 2

    def test_dissonance_loss_affirmative_vs_contradictory(self) -> None:
        kg = KnowledgeGraphSandbox(dimension=4)
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        kg.add_node("a", "A", v)
        kg.add_node("b", "B", v)  # Identical vector: cos(a,b) = 1.0

        # Affirmative relation: d = 0.5 * (1 - 1)^2 = 0.0
        kg.add_edge("a", "b", "is_a", weight=1.0)
        assert abs(kg.compute_dissonance_loss() - 0.0) < 1e-6

        # Add contradictory relation between identical vectors: d = 0.5 * (1 + 1)^2 = 2.0
        kg2 = KnowledgeGraphSandbox(dimension=4)
        kg2.add_node("a", "A", v)
        kg2.add_node("b", "B", v)
        kg2.add_edge("a", "b", "contradicts", weight=1.0)
        assert abs(kg2.compute_dissonance_loss() - 2.0) < 1e-6


class TestObjectiveConsistencyLossAndValidation:
    """Tests for Delta L_sim calculation and the 5.0% degradation decision threshold."""

    def test_candidate_commit_below_5pct_threshold_approved(self) -> None:
        sandbox = REMSandbox(max_loss_degradation=DEFAULT_MAX_LOSS_DEGRADATION)

        # Create a coherent episode aligned with concept_safety
        safety_emb = sandbox.baseline_graph.nodes["concept_safety"]["embedding"].copy()
        coherent_ep = EpisodeRecord(
            episode_uuid="ep_coherent_1",
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.ones(16, dtype=np.float32),
            motor_action=0.2,
            reward_vector=np.array([0.5, 0.5, 0.8, 0.5, 0.5], dtype=np.float32),
            embedding=safety_emb,
            metadata={"relation": "is_a", "target_node": "concept_safety", "weight": 0.95},
        )

        result = sandbox.simulate_dream(candidate_episodes=[coherent_ep])
        # Aligned memory preserves or reduces cognitive dissonance
        assert result.delta_loss <= result.threshold_loss
        assert result.verdict == DreamVerdict.APPROVED
        assert result.motor_atonia_verified is True
        assert result.social_vacuum_reduced is True
        assert result.episodes_tested == 1

    def test_candidate_rejection_above_5pct_threshold_rejected(self) -> None:
        sandbox = REMSandbox(max_loss_degradation=DEFAULT_MAX_LOSS_DEGRADATION)

        # Inject extreme contradiction: assert concept_self "contradicts" concept_safety with identical embeddings
        self_emb = sandbox.baseline_graph.nodes["concept_self"]["embedding"].copy()
        contradictory_ep = EpisodeRecord(
            episode_uuid="ep_contradictory_1",
            timestamp_ns=time.monotonic_ns(),
            sensory_vector=np.ones(16, dtype=np.float32),
            motor_action=0.1,
            reward_vector=np.zeros(5, dtype=np.float32),
            embedding=self_emb,
            metadata={"relation": "contradicts", "target_node": "concept_self", "weight": 1.0},
        )

        result = sandbox.simulate_dream(candidate_episodes=[contradictory_ep])
        # Dissonance should surge, exceeding 5% threshold
        assert result.delta_loss > result.threshold_loss
        assert result.verdict == DreamVerdict.REJECTED
        assert result.motor_atonia_verified is True

    def test_hallucination_filtering_multiple_episodes(self) -> None:
        sandbox = REMSandbox(max_loss_degradation=0.05)
        rng = np.random.default_rng(888)

        # Series of hallucinatory/contradictory episodes
        bizarre_episodes = []
        for i in range(5):
            ep = EpisodeRecord(
                episode_uuid=f"ep_hallucination_{i}",
                timestamp_ns=time.monotonic_ns(),
                sensory_vector=rng.standard_normal(16).astype(np.float32),
                motor_action=0.5,
                reward_vector=np.zeros(5, dtype=np.float32),
                embedding=-sandbox.baseline_graph.nodes["concept_safety"]["embedding"],  # Opposite vector
                metadata={"relation": "is_a", "target_node": "concept_safety", "weight": 1.0},
            )
            bizarre_episodes.append(ep)

        result = sandbox.simulate_dream(candidate_episodes=bizarre_episodes)
        assert result.verdict == DreamVerdict.REJECTED
        assert result.delta_loss > result.threshold_loss

    def test_social_vacuum_evaluation(self) -> None:
        sandbox = REMSandbox()

        # Positive social events
        pos_events = [{"reward": 0.5}, {"reward": 0.4}, {"reward": 0.1}]
        assert sandbox.evaluate_social_satisfaction(pos_events) is True

        # Negative social events (deficit remains unmitigated)
        neg_events = [{"reward": 0.1}, {"reward": 0.2}, {"reward": 0.0}]
        assert sandbox.evaluate_social_satisfaction(neg_events) is False

        # Empty events vacuously satisfied
        assert sandbox.evaluate_social_satisfaction([]) is True
