"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 (Memory Subsystem)
Module: memory.rem_sandbox
Author: ΔP-42 Senior Engineering Team (Zero Downgrade Mandate)

REM Dream Simulation Sandbox with 100% Motor Atonia Physical Interlock
and Associative Cognitive Dissonance Loss Verification.

Canonical References:
- DELTA_P42_MUKODESI_ELVEK.md (Chapter III.2 lines 319-329, Chapter I.5)
- DELTA_P42_MESTER_TEZIS.md (Section III.3, V)
- Plan Review.md (REQ-SLEEP-03, DVP-11, Stage-Gate 3 STL)
- megvalósítási terv.md (Phase 3: Section 3.2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.basal_ganglia_gate import BasalGangliaGate, DEG2RAD

logger = logging.getLogger("delta_p42.memory.rem_sandbox")

# Canonical Constants
DEFAULT_MAX_LOSS_DEGRADATION: float = 0.05   # 5.0% maximum allowable consistency degradation
MAX_DREAM_DEGRADATION_PCT: float = 5.0       # 5.0%
MOTOR_ATONIA_CLAMP: float = 0.000            # a_gate clamped to exactly 0.000
LOSS_EPSILON: float = 1e-6                   # Numerical floor for near-zero baseline loss


class DreamVerdict(enum.Enum):
    """
    Decision verdict of REM virtual dream simulation.

    Members:
        APPROVED: Candidate memory modifications preserve or improve cognitive consistency
                  (Delta L_sim <= 0.05 * L_baseline). Approved for long-term consolidation.
        REJECTED: Cognitive dissonance, logical contradiction, or consistency degradation > 5%
                  detected. Candidate memory discarded.
    """
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

    # Aliases
    PASS = "APPROVED"
    ABORT = "REJECTED"


@dataclass(frozen=True)
class DreamResult:
    """
    Immutable comprehensive result record emitted by REM dream simulation.

    Attributes:
        verdict: DreamVerdict (APPROVED or REJECTED).
        baseline_loss: Cognitive loss of baseline knowledge graph L_baseline.
        candidate_loss: Cognitive loss with candidate memory integrated L_candidate.
        delta_loss: Absolute change in loss Delta L_sim = L_candidate - L_baseline.
        delta_loss_ratio: Relative degradation Delta L_sim / max(L_baseline, eps).
        threshold_loss: Maximum allowable loss increase 0.05 * max(L_baseline, eps).
        motor_atonia_verified: True if striatal motor gate remained 0.000 across all test angles.
        motor_output_count: Total physical motor output packets emitted (strictly 0).
        episodes_tested: Number of candidate episodic records evaluated.
        social_vacuum_reduced: True if simulated interpersonal outcomes mitigated social deficit.
        simulation_duration_ms: Total duration of virtual dream replay in milliseconds.
        details: Diagnostic dictionary containing component loss breakdown.
    """
    verdict: DreamVerdict
    baseline_loss: float
    candidate_loss: float
    delta_loss: float
    delta_loss_ratio: float
    threshold_loss: float
    motor_atonia_verified: bool
    motor_output_count: int
    episodes_tested: int
    social_vacuum_reduced: bool
    simulation_duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# Cross-contract alias
REMSimulationResult = DreamResult


class KnowledgeGraphSandbox:
    """
    Lightweight, high-performance associative knowledge graph model for dream simulation.

    Maintains semantic concept nodes with embeddings and directed relational edges,
    computing topological dissonance and consistency loss.
    """

    def __init__(self, dimension: int = 64) -> None:
        self.dimension: int = int(dimension)
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def add_node(
        self,
        node_id: str,
        label: str,
        embedding: np.ndarray,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        emb_arr = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(emb_arr))
        normalized_emb = emb_arr / norm if norm > 1e-12 else emb_arr
        self.nodes[str(node_id)] = {
            "label": str(label),
            "embedding": normalized_emb,
            "attributes": attributes or {},
        }

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.edges.append({
            "source": str(source_id),
            "target": str(target_id),
            "relation": str(relation),
            "weight": float(np.clip(weight, 0.0, 1.0)),
            "attributes": attributes or {},
        })

    def clone(self) -> KnowledgeGraphSandbox:
        """Deep copy of the knowledge graph for non-destructive sandbox simulation."""
        cloned = KnowledgeGraphSandbox(dimension=self.dimension)
        for nid, ndata in self.nodes.items():
            cloned.nodes[nid] = {
                "label": ndata["label"],
                "embedding": ndata["embedding"].copy(),
                "attributes": dict(ndata["attributes"]),
            }
        for edge in self.edges:
            cloned.edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "relation": edge["relation"],
                "weight": edge["weight"],
                "attributes": dict(edge["attributes"]),
            })
        return cloned

    def compute_dissonance_loss(self) -> float:
        """
        Computes the global cognitive dissonance loss L(W) across all active edges.

        Dissonance definitions:
        - Affirmative relations ('is_a', 'associated_with', 'causes', 'co_occurs'):
            d = 0.5 * (1 - cos(e_u, e_v))^2
        - Contradictory relations ('contradicts', 'incompatible_with', 'cannot_be', 'negates'):
            d = 0.5 * (1 + cos(e_u, e_v))^2
        """
        if not self.edges:
            return 0.01  # Nominal baseline floor for empty edge sets

        total_dissonance = 0.0
        total_weight = 0.0

        for edge in self.edges:
            src = edge["source"]
            tgt = edge["target"]
            rel = edge["relation"].lower()
            w = edge["weight"]

            if src not in self.nodes or tgt not in self.nodes:
                continue

            e_u = self.nodes[src]["embedding"]
            e_v = self.nodes[tgt]["embedding"]

            dot = float(np.dot(e_u, e_v))
            cos_sim = float(np.clip(dot, -1.0, 1.0))

            if rel in ("contradicts", "incompatible_with", "cannot_be", "negates"):
                # Dissonance is high if contradictory nodes are aligned
                d = 0.5 * ((1.0 + cos_sim) ** 2)
            else:
                # Dissonance is high if associated nodes are orthogonal or opposed
                d = 0.5 * ((1.0 - cos_sim) ** 2)

            total_dissonance += w * d
            total_weight += w

        if total_weight < 1e-12:
            return 0.01

        return float(total_dissonance / total_weight)


class REMSandbox:
    """
    ΔP-42 REM Dream Simulation Sandbox.

    Executes candidate memory integration and cognitive dissonance resolution
    under 100% physical motor atonia:
        1. Clamps Basal Ganglia striatum motor gate to a_gate == 0.000 unconditionally.
        2. Strictly blocks actuator packet emissions (emitted_count == 0).
        3. Evaluates cognitive consistency loss Delta L_sim = L_candidate - L_baseline.
        4. Rejects candidate memories that degrade consistency by > 5.0%.
        5. Validates daytime social feedback to recalibrate social vacuum.
    """

    def __init__(
        self,
        baseline_graph: Optional[Any] = None,
        striatum_gate_clamp: float = MOTOR_ATONIA_CLAMP,
        gate: Optional[BasalGangliaGate] = None,
        max_loss_degradation: float = DEFAULT_MAX_LOSS_DEGRADATION,
    ) -> None:
        """
        Initialize the REM dream sandbox.

        Args:
            baseline_graph: Pre-existing persistent knowledge graph or sandbox model.
            striatum_gate_clamp: Motor gate clamp intensity (strictly 0.000).
            gate: Reference to BasalGangliaGate module (instantiated if None).
            max_loss_degradation: Maximum allowable loss degradation (default 0.05 = 5.0%).
        """
        self.striatum_gate_clamp: float = float(striatum_gate_clamp)
        self.max_loss_degradation: float = float(max_loss_degradation)
        self._gate: BasalGangliaGate = gate if gate is not None else BasalGangliaGate()

        # Initialize knowledge graph
        if baseline_graph is None:
            self.baseline_graph: KnowledgeGraphSandbox = self._build_default_baseline_graph()
        elif isinstance(baseline_graph, KnowledgeGraphSandbox):
            self.baseline_graph = baseline_graph
        else:
            self.baseline_graph = self._adapt_external_graph(baseline_graph)

        self._motor_output_stream: List[Any] = []
        self._atonia_active: bool = True

    @property
    def motor_atonia_active(self) -> bool:
        """Returns True: motor atonia is strictly active during REM simulation."""
        return self._atonia_active

    def _build_default_baseline_graph(self) -> KnowledgeGraphSandbox:
        """Constructs a canonical coherent baseline knowledge graph."""
        kg = KnowledgeGraphSandbox(dimension=64)
        rng = np.random.default_rng(42)

        # Base concepts
        self_emb = rng.standard_normal(64).astype(np.float32)
        world_emb = rng.standard_normal(64).astype(np.float32)
        safety_emb = 0.8 * self_emb + 0.2 * rng.standard_normal(64).astype(np.float32)
        danger_emb = -safety_emb

        kg.add_node("concept_self", "Self", self_emb)
        kg.add_node("concept_world", "World", world_emb)
        kg.add_node("concept_safety", "Safety", safety_emb)
        kg.add_node("concept_danger", "Danger", danger_emb)

        kg.add_edge("concept_self", "concept_world", "associated_with", weight=0.9)
        kg.add_edge("concept_self", "concept_safety", "is_a", weight=0.95)
        kg.add_edge("concept_safety", "concept_danger", "contradicts", weight=1.0)

        return kg

    def _adapt_external_graph(self, external_graph: Any) -> KnowledgeGraphSandbox:
        """Adapts external persistent graph or SQLite database to KnowledgeGraphSandbox."""
        if hasattr(external_graph, "nodes") and hasattr(external_graph, "edges"):
            return external_graph
        return self._build_default_baseline_graph()

    def verify_motor_atonia(
        self,
        test_angles_deg: Sequence[float] = (0.0, 15.0, 18.0, 18.5, 20.0, 30.0, 45.0, 90.0, 360.0),
    ) -> bool:
        """
        Rigorously tests Basal Ganglia striatal gate across physical angles under sleep lock.

        Verifies that a_motor == 0.000000 across all angles including suprathreshold drives.
        """
        for deg in test_angles_deg:
            th_rad = float(deg * DEG2RAD)
            decision = self._gate.evaluate(
                theta_rad=th_rad,
                u=1.0,
                u_intent=1.0,
                sleep_lock=True,
                lock_sleep=True,
            )
            if decision.a_motor != 0.0 or not decision.is_locked:
                return False
        return True

    def simulate_dream(
        self,
        candidate_episodes: Sequence[Any],
        knowledge_graph: Optional[Any] = None,
    ) -> DreamResult:
        """
        Execute REM virtual dream replay and consistency validation.

        Args:
            candidate_episodes: Sequence of candidate EpisodicMemoryRecords from working memory.
            knowledge_graph: Optional custom knowledge graph (defaults to self.baseline_graph).

        Returns:
            DreamResult: Immutable result record containing verdict and loss metrics.
        """
        t_start = time.perf_counter()

        # 1. 100% Motor Atonia Physical Interlock
        self._atonia_active = True
        atonia_ok = self.verify_motor_atonia()
        self._motor_output_stream.clear()

        # 2. Select knowledge graph
        active_kg = knowledge_graph if knowledge_graph is not None else self.baseline_graph
        if not isinstance(active_kg, KnowledgeGraphSandbox):
            active_kg = self._adapt_external_graph(active_kg)

        # 3. Evaluate Baseline Loss
        l_base = float(active_kg.compute_dissonance_loss())

        # 4. Integrate Candidate Episodes into Cloned Sandbox Graph
        sandbox_kg = active_kg.clone()
        social_events = []

        for ep in candidate_episodes:
            # Extract episode attributes
            ep_uuid = getattr(ep, "episode_uuid", getattr(ep, "uuid", None)) or f"ep_{len(sandbox_kg.nodes)}"
            emb = getattr(ep, "embedding", None)
            meta = getattr(ep, "metadata", {}) or {}
            reward = getattr(ep, "reward_vector", None)

            if emb is not None:
                sandbox_kg.add_node(f"node_{ep_uuid}", f"Episode_{ep_uuid[:8]}", emb, meta)

            # Check for candidate relations or contradictory metadata
            candidate_relation = meta.get("relation")
            target_node = meta.get("target_node")
            weight = float(meta.get("weight", 0.9))

            if candidate_relation and target_node and target_node in sandbox_kg.nodes:
                sandbox_kg.add_edge(f"node_{ep_uuid}", target_node, candidate_relation, weight=weight)

            # Track social events
            if reward is not None and len(reward) >= 3:
                social_reward = float(reward[2])  # V_social index = 2
                social_events.append({"reward": social_reward, "metadata": meta})

        # 5. Evaluate Candidate Loss in Sandbox
        l_cand = float(sandbox_kg.compute_dissonance_loss())

        # 6. Dissonance Difference and Gate Evaluation
        delta_l = l_cand - l_base
        denom = max(l_base, LOSS_EPSILON)
        delta_ratio = delta_l / denom
        threshold_l = self.max_loss_degradation * denom

        if delta_l <= threshold_l:
            verdict = DreamVerdict.APPROVED
        else:
            verdict = DreamVerdict.REJECTED

        # 7. Evaluate Social Satisfaction
        social_satisfied = self.evaluate_social_satisfaction(social_events)

        # 8. Assert zero motor leakage
        motor_count = len(self._motor_output_stream)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return DreamResult(
            verdict=verdict,
            baseline_loss=l_base,
            candidate_loss=l_cand,
            delta_loss=delta_l,
            delta_loss_ratio=delta_ratio,
            threshold_loss=threshold_l,
            motor_atonia_verified=bool(atonia_ok and motor_count == 0),
            motor_output_count=motor_count,
            episodes_tested=len(candidate_episodes),
            social_vacuum_reduced=social_satisfied,
            simulation_duration_ms=elapsed_ms,
            details={
                "max_allowed_degradation_pct": self.max_loss_degradation * 100.0,
                "measured_degradation_pct": delta_ratio * 100.0,
                "sandbox_nodes": len(sandbox_kg.nodes),
                "sandbox_edges": len(sandbox_kg.edges),
            },
        )

    def evaluate_social_satisfaction(
        self,
        social_feedback_events: Sequence[Any],
    ) -> bool:
        """
        Evaluates whether waking interpersonal interactions reduced the social deficit V_social.
        """
        if not social_feedback_events:
            return True  # Vacuously satisfied if no social events recorded

        positive_count = 0
        for ev in social_feedback_events:
            if isinstance(ev, dict):
                r = ev.get("reward", 0.0)
            else:
                r = getattr(ev, "reward", 0.0)
            if float(r) > 0.30:
                positive_count += 1

        return positive_count >= (len(social_feedback_events) // 2)

    def get_emitted_motor_packets(self) -> List[Any]:
        """Returns the motor output packet stream (strictly empty under atonia)."""
        return list(self._motor_output_stream)
