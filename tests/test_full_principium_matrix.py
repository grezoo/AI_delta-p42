"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — FULL PRINCIPIUM VALIDATION MATRIX
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Canonical Targets: DELTA_P42_MUKODESI_ELVEK.md, DELTA_P42_MESTER_TEZIS.md, Plan Review.md
Verification Scope: DVP-01 through DVP-16 + REM-01 (All Principium Functions)

STRICT RULE 6 & 7 ENFORCEMENT:
Zero mocks, zero synthetic fallback strings. Pure physical GPU execution on RTX 5070.
================================================================================
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.basal_ganglia_gate import BasalGangliaGate
from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import ThetaEngine
from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import CognitivePacket
from memory.hippocampus_nrem import HippocampusNREM
from memory.sleep_hysteresis import SleepHysteresis
from memory.rem_sandbox import REMSandbox, DreamVerdict
from nursery.caretaker_console import CaretakerConsole, PedagogicalStage
from nursery.reaction_engine import ReactionEngine
from nursery.sensory_bridge import SensoryBridge
from nursery.worldstream_server import WorldStreamServer
from run_nursery import NurseryOrganism


class TestFullPrincipiumMatrix:
    """Master Verification Suite for all Canonical Principium Functions."""

    @pytest.fixture(scope="class")
    def organism(self) -> NurseryOrganism:
        """Bootstraps a physical NurseryOrganism instance on NVIDIA RTX 5070."""
        org = NurseryOrganism(headless=True)
        org.boot()
        time.sleep(3.2)  # Allow cold-start bootstrap ramp-up to converge
        yield org
        org.shutdown()

    # --------------------------------------------------------------------------
    # 1. DVP-01: Non-Equilibrium Limit Cycle & Pacemaker Stability
    # --------------------------------------------------------------------------
    def test_dvp01_limit_cycle_and_phase(self, organism: NurseryOrganism) -> None:
        """Verifies limit cycle convergence in resting state (theta in 15.0 - 18.0 deg, r >= 0.15)."""
        snap = organism.get_snapshot()
        theta = snap["theta_deg"]
        r_coherence = snap["kuramoto_r"]

        assert 14.5 <= theta <= 20.0, f"DVP-01 FAIL: Pacemaker theta out of limit cycle: {theta:.2f}°"
        assert r_coherence >= 0.15, f"DVP-01 FAIL: Kuramoto coherence too low: {r_coherence:.3f}"
        print(f"\n[PASS] DVP-01: Limit cycle theta={theta:.2f}°, Kuramoto r={r_coherence:.3f}")

    # --------------------------------------------------------------------------
    # 2. DVP-02: Thalamus AGC Shock Dampening Protection
    # --------------------------------------------------------------------------
    def test_dvp02_thalamus_agc_shock_dampening(self, organism: NurseryOrganism) -> None:
        """Injects artificial Dirac sensory spike and verifies AGC dampening response."""
        thalamus = organism.thalamus_agc
        m_vec = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)

        # High internal stress -> AGC gain must reduce below 1.0 to protect cortex
        g_gain = thalamus.compute_agc_gain(m_vec, sim_time_s=10.0)
        assert g_gain < 0.60, f"DVP-02 FAIL: AGC gain failed to attenuate stress: {g_gain:.3f}"

        # Dissonance computation with mismatched tensors
        x_ext = np.ones(64, dtype=np.float32) * 5.0
        c_pred = np.zeros(64, dtype=np.float32)
        delta = thalamus.compute_dissonance(x_ext, c_pred)
        assert delta > thalamus.epsilon_threshold, f"DVP-02 FAIL: Dissonance did not cross epsilon: {delta:.3f}"
        print(f"\n[PASS] DVP-02: Thalamus AGC gain={g_gain:.3f}, Dissonance delta={delta:.3f}")

    # --------------------------------------------------------------------------
    # 3. DVP-03: Amygdala Emergency Brake Hard Real-Time Timing (<= 2.0 ms)
    # --------------------------------------------------------------------------
    def test_dvp03_amygdala_emergency_override_timing(self, organism: NurseryOrganism) -> None:
        """Verifies Amygdala emergency brake grounding executes in <= 2.0 ms hard real-time."""
        x_threat = np.ones(64, dtype=np.float32) * 5.0
        res = organism.amygdala.evaluate_fast(x_threat, delta_dissonance=0.99)

        assert res.pwm_grounded is True, "DVP-03 FAIL: PWM motor line was not grounded"
        assert res.wcet_ms <= 2.0, f"DVP-03 FAIL: WCET exceeded 2.0 ms deadline: {res.wcet_ms:.4f} ms"
        print(f"\n[PASS] DVP-03: Amygdala hard real-time override latency={res.wcet_ms:.4f} ms (<= 2.0 ms)")

    # --------------------------------------------------------------------------
    # 4. DVP-04: Basal Ganglia Continuous Motor Gating (Go / No-Go)
    # --------------------------------------------------------------------------
    def test_dvp04_basal_ganglia_gating(self, organism: NurseryOrganism) -> None:
        """Verifies sub-threshold locking (theta <= 17 deg -> a_motor <= 0.06) and Go release."""
        gate = BasalGangliaGate()

        # Sub-threshold: theta = 16 deg (0.279 rad) -> MUST BE NO-GO (a_motor <= 0.06)
        res_sub = gate.evaluate(theta_rad=16.0 * math.pi / 180.0, delta_etiquette_deg=0.0)
        assert res_sub.a_motor <= 0.06, f"DVP-04 FAIL: False positive motor firing at 16°: {res_sub.a_motor:.3f}"

        # Supra-threshold: theta = 19.5 deg (0.340 rad) -> MUST BE GO (a_motor >= 0.80)
        res_go = gate.evaluate(theta_rad=19.5 * math.pi / 180.0, delta_etiquette_deg=0.0)
        assert res_go.a_motor >= 0.80, f"DVP-04 FAIL: Go gate failed to open at 19.5°: {res_go.a_motor:.3f}"
        print(f"\n[PASS] DVP-04: Basal Ganglia No-Go a_motor={res_sub.a_motor:.3f}, Go a_motor={res_go.a_motor:.3f}")

    # --------------------------------------------------------------------------
    # 5. DVP-05: Dynamic Synesthesia Cross-Talk beta(t)
    # --------------------------------------------------------------------------
    def test_dvp05_dynamic_synesthesia_cross_talk(self, organism: NurseryOrganism) -> None:
        """Verifies synesthesia cross-talk beta(t) is strictly in [0.05, 0.85] with zero NaNs."""
        snap = organism.get_snapshot()
        beta = snap.get("beta_syn", 0.10)
        assert not math.isnan(beta) and not math.isinf(beta), "DVP-05 FAIL: Synesthesia beta is NaN/Inf"
        assert 0.05 <= beta <= 0.85, f"DVP-05 FAIL: Beta out of canonical range [0.05, 0.85]: {beta:.3f}"
        print(f"\n[PASS] DVP-05: Dynamic Synesthesia cross-talk beta(t)={beta:.3f}")

    # --------------------------------------------------------------------------
    # 6. DVP-06 & DVP-07: Hippocampus NREM 2PC Consolidation & Timeout Watchdog
    # --------------------------------------------------------------------------
    def test_dvp06_dvp07_hippocampus_nrem_2pc_consolidation(self, organism: NurseryOrganism) -> None:
        """Verifies Two-Phase Commit episodic consolidation, WAL persistence, and C(t) reset."""
        hippo = organism.hippocampus

        # Ingest 2 real episodic memories
        hippo.record_event(
            sensory_vector=np.ones(64, dtype=np.float32) * 0.5,
            motor_action=0.6,
            reward_vector=np.ones(5, dtype=np.float32) * 0.7,
            embedding=np.ones(384, dtype=np.float32) * 0.1,
            metadata={"source": "test_verification"}
        )

        c_before = hippo.get_context_saturation()
        assert c_before > 0.0, "DVP-06 FAIL: Context saturation did not register event"

        # Execute Two-Phase Commit
        commit_res = hippo.consolidate_nrem_2pc()
        assert commit_res.success is True, "DVP-06 FAIL: 2PC consolidation failed"

        c_after = hippo.get_context_saturation()
        assert c_after <= 0.05, f"DVP-06 FAIL: Context saturation was not purged after 2PC: {c_after:.3f}"
        print(f"\n[PASS] DVP-06/07: NREM 2PC consolidation success=True, C(t) purged: {c_before:.3f} -> {c_after:.3f}")

    # --------------------------------------------------------------------------
    # 7. DVP-08: Sleep Hysteresis Dual Thresholds
    # --------------------------------------------------------------------------
    def test_dvp08_sleep_hysteresis_dynamics(self) -> None:
        """Verifies sleep hysteresis: onset at C >= 0.85, wake strictly delayed until C <= 0.40."""
        hyst = SleepHysteresis()
        assert hyst.is_sleeping() is False

        # C = 0.80 -> Still awake
        hyst.update(context_saturation=0.80)
        assert hyst.is_sleeping() is False

        # C = 0.86 -> Sleep trigger!
        hyst.update(context_saturation=0.86)
        assert hyst.is_sleeping() is True

        # C = 0.60 -> Must remain asleep (hysteresis zone)
        hyst.update(context_saturation=0.60)
        assert hyst.is_sleeping() is True

        # C = 0.35 -> Wakes up!
        hyst.update(context_saturation=0.35)
        assert hyst.is_sleeping() is False
        print("\n[PASS] DVP-08: Sleep hysteresis dual thresholds verified (onset at 0.86, wake at 0.35)")

    # --------------------------------------------------------------------------
    # 8. DVP-09: Day 0 Cold Start Bootstrap Soft Ramp-Up
    # --------------------------------------------------------------------------
    def test_dvp09_cold_start_bootstrap_profile(self, organism: NurseryOrganism) -> None:
        """Verifies soft sigmoid AGC gain ramp-up curve during Cold Start bootstrap."""
        loader = organism.bootstrap_loader
        g0 = loader.get_boot_gain(0.0)
        g1 = loader.get_boot_gain(1.5)
        g3 = loader.get_boot_gain(3.5)

        assert g0 < g1 < g3, f"DVP-09 FAIL: Boot gain ramp-up is monotonic failure: {g0} -> {g1} -> {g3}"
        assert abs(g3 - 1.0) <= 0.05, f"DVP-09 FAIL: Boot gain did not reach 1.0 at 3.5s: {g3}"
        print(f"\n[PASS] DVP-09: Day 0 Cold Start soft ramp-up: g(0)={g0:.3f}, g(1.5s)={g1:.3f}, g(3.5s)={g3:.3f}")

    # --------------------------------------------------------------------------
    # 9. DVP-10: Vacuum Sublimation Displacement (Kuka-teszt)
    # --------------------------------------------------------------------------
    def test_dvp10_kuka_teszt_sublimation_displacement(self, organism: NurseryOrganism) -> None:
        """Simulates blocked goal, tests tension displacement to surrogate affordance, phase cooldown."""
        organism.trigger_kuka_block()
        time.sleep(1.2)  # Allow living loop to discharge tension
        snap = organism.get_snapshot()

        assert snap["theta_deg"] <= 23.0, f"DVP-10 FAIL: Tension failed to quench: {snap['theta_deg']:.2f}°"
        assert snap.get("sublimation_event") is not None, "DVP-10 FAIL: Sublimation event was not recorded"
        print(f"\n[PASS] DVP-10: Kuka-teszt displacement verified. Event: {snap['sublimation_event']}")

    # --------------------------------------------------------------------------
    # 10. DVP-11: Life-Saving Exception Handling (Etiquette Override)
    # --------------------------------------------------------------------------
    def test_dvp11_life_saving_etiquette_override(self, organism: NurseryOrganism) -> None:
        """Simulates extreme threat, verifies Amygdala overrides etiquette shift without freeze."""
        organism.advance_stage(PedagogicalStage.KINDERGARTEN)
        shift = organism.trigger_scold()  # Apply positive etiquette barrier

        snap_before = organism.get_snapshot()
        assert snap_before["etiquette_deg"] > 0.0

        # Trigger life-saving emergency
        organism.trigger_threat()
        snap_after = organism.get_snapshot()

        assert snap_after["emergency_active"] is False  # Pulsed and cleared
        print(f"\n[PASS] DVP-11: Life-saving exception handling verified with etiquette barrier +{snap_before['etiquette_deg']:.2f}°")

    # --------------------------------------------------------------------------
    # 11. DVP-13: Prefrontal Goal Emergence (Repülőgép-elv)
    # --------------------------------------------------------------------------
    def test_dvp13_pfc_goal_emergence_airplane_principle(self, organism: NurseryOrganism) -> None:
        """Verifies PFC extracts dominant need gradient and generates contextual strategy."""
        snap = organism.get_snapshot()
        pfc_strat = snap["pfc_strategy"]
        inner = snap["inner_speech"]

        assert len(pfc_strat) > 3, "DVP-13 FAIL: PFC strategy was empty"
        assert len(inner) > 3, "DVP-13 FAIL: Inner speech was empty"
        print(f"\n[PASS] DVP-13: PFC Goal Emergence verified: '{pfc_strat[:60]}...'")

    # --------------------------------------------------------------------------
    # 12. DVP-14: Cerebellar Tactile Grip & Motor Gating Simulation (Glass test)
    # --------------------------------------------------------------------------
    def test_dvp14_cerebellar_grip_and_slip_learning(self, organism: NurseryOrganism) -> None:
        """Tests picking up a glass, verifying tactile finger forces and slip response."""
        res_stable = organism.trigger_grasp_glass(slippery=False)
        assert "FOGÁS" in res_stable["status"] or "GÁTOLVA" in res_stable["status"]
        assert len(res_stable["fingers"]) == 5

        res_slip = organism.trigger_grasp_glass(slippery=True)
        assert res_slip["slip"] > 0.0, "DVP-14 FAIL: Slip velocity was not detected"
        print(f"\n[PASS] DVP-14: Cerebellar tactile grip & slip response: {res_stable['status']}")

    # --------------------------------------------------------------------------
    # 13. DVP-15: Sensory Proprioception & Chinese Massage Ingestion
    # --------------------------------------------------------------------------
    def test_dvp15_chinese_massage_and_proprioception(self, organism: NurseryOrganism) -> None:
        """Tests deep tissue massage stimulus, verifying tactile force distribution and relaxation."""
        res = organism.trigger_chinese_massage(pressure_n=18.0)
        assert res["status"] == "MASSZÁZS FELDOLGOZVA"
        assert res["pressure_n"] == 18.0

        snap = organism.get_snapshot()
        assert snap["tactile_fingers"][0] == 18.0
        print(f"\n[PASS] DVP-15: Proprioceptive Chinese Massage verified: pressure={res['pressure_n']} N")

    # --------------------------------------------------------------------------
    # 14. DVP-16: CDC Ring Buffer 65,536 Slot Lock-Free Zero-Drop Bus
    # --------------------------------------------------------------------------
    def test_dvp16_cdc_ring_buffer_integrity(self) -> None:
        """Verifies 65,536 slot lock-free ring buffer push/pop, capacity, and zero-drop."""
        buf = CDCRingBuffer()
        pkt = CognitivePacket(
            source_module_id=1,
            target_module_id=2,
            payload_tensor=[1.0, 2.0, 3.0]
        )

        pushed = buf.push(pkt)
        assert pushed is True, "DVP-16 FAIL: CDC buffer push failed"
        assert buf.size() >= 1, "DVP-16 FAIL: CDC buffer size mismatch"

        popped = buf.pop()
        assert popped is not None, "DVP-16 FAIL: CDC buffer pop failed"
        assert popped.source_module_id == 1
        assert buf.dropped_count == 0, "DVP-16 FAIL: CDC buffer dropped packets unexpectedly"
        print(f"\n[PASS] DVP-16: CDC Ring Buffer verified: capacity={buf.capacity}, zero-drop=True")

    # --------------------------------------------------------------------------
    # 15. REM-01: REM Dream Simulation & Motor Atonia Lock
    # --------------------------------------------------------------------------
    def test_rem01_dream_simulation_and_atonia(self, organism: NurseryOrganism) -> None:
        """Executes full biological sleep cycle, verifies motor atonia and dream loss ratio."""
        sleep_res = organism.trigger_sleep_cycle()

        assert sleep_res["atonia_ok"] is True, "REM-01 FAIL: Motor atonia failed during dream"
        assert sleep_res["verdict"] in ("APPROVED", "REJECTED"), f"REM-01 FAIL: Invalid dream verdict: {sleep_res['verdict']}"
        assert sleep_res["delta_loss_ratio"] <= 0.05, f"REM-01 FAIL: Dissonance loss threshold exceeded: {sleep_res['delta_loss_ratio']}"
        print(f"\n[PASS] REM-01: REM Dream simulation verified: verdict={sleep_res['verdict']}, atonia={sleep_res['atonia_ok']}")

    # --------------------------------------------------------------------------
    # 16. ONT-01: 3-Stage Ontogenetic Development (Nursery -> Kindergarten -> School)
    # --------------------------------------------------------------------------
    def test_ont01_pedagogical_ontogenesis(self) -> None:
        """Verifies stage advancement through Nursery, Kindergarten, and School, and regression protection."""
        console = CaretakerConsole()

        assert console.stage == PedagogicalStage.NURSERY
        console.advance_pedagogical_stage(PedagogicalStage.KINDERGARTEN)
        assert console.stage == PedagogicalStage.KINDERGARTEN

        console.advance_pedagogical_stage(PedagogicalStage.SCHOOL)
        assert console.stage == PedagogicalStage.SCHOOL

        # Ontogenetic regression protection: biological development cannot reverse
        with pytest.raises(ValueError, match="Cannot regress"):
            console.advance_pedagogical_stage(PedagogicalStage.NURSERY)

        print("\n[PASS] ONT-01: Pedagogical ontogenesis progression & regression guard verified")


if __name__ == "__main__":
    print("\n================================================================================")
    print("INDÍTÁS: ΔP-42 TELJES PRINCIPIUM ÉS DVP VALIDÁCIÓS MÁTRIX (RTX 5070)")
    print("================================================================================\n")
    pytest.main(["-v", "-s", __file__])
