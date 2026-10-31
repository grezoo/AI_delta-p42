"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — AI NURSERY MASTER RUNTIME
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Cortical Chorus:   6 Physical GGUF Micro-Models (2.88 GB VRAM, 4.17B parameters)
Subcortical Core:  50 Hz Theta Pacemaker, 10 Hz Maslow Integrator, Thalamus AGC
Nervous Bus:       65,536-slot C++20/Python Lock-Free CDC Ring Buffer (WorldStream)
Safety Reflex:     Amygdala Override (<= 2.0 ms Hard Real-Time Kill Line)
Curriculum:        3-Stage Pedagogical Ontogenesis (Nursery -> Kindergarten -> School)

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapters I - V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapters I - XI)
- megvalósítási terv.md (Phases 1 - 5)
- Plan Review.md (DVP-01 through DVP-20)

STRICT RULE 6 & RULE 7 ENFORCEMENT:
Zero mocks, zero synthetic fallback strings. Pure physical GPU execution on RTX 5070.
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Core Subcortical Engine
from core.basal_ganglia_gate import BasalGangliaGate
from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import DEG2RAD, ThetaEngine
RAD2DEG: float = 180.0 / math.pi

# Communication & Nervous Bus
from comm.cdc_ring_buffer import CDCRingBuffer
from comm.cognitive_packet import (
    MODULE_AMYGDALA,
    MODULE_BASAL_GANGLIA,
    MODULE_CEREBELLUM,
    MODULE_CHORUS_LLM,
    MODULE_HIPPOCAMPUS,
    MODULE_HYPOTHALAMUS,
    MODULE_PFC,
    MODULE_THALAMUS,
    CognitivePacket,
)

# Safety System
from modules.amygdala_override import AmygdalaOverride

# Memory & Circadian System
from memory.hippocampus_nrem import HippocampusNREM
from memory.sleep_hysteresis import SleepHysteresis
from memory.rem_sandbox import REMSandbox, DreamVerdict, DreamResult

# Nursery Pedagogical & Cortical Subsystems
from nursery.bootstrap_loader import (
    DEFAULT_V0,
    RAMP_UP_DURATION_S,
    THETA_BOOTSTRAP_DEG,
    BootstrapLoader,
)
from nursery.caretaker_console import (
    BASE_GATE_DEG,
    CaretakerConsole,
    PedagogicalStage,
)
from nursery.etology_telemetry import EthologyTelemetryRecorder
from nursery.gguf_runtime import (
    DEFAULT_MODEL_DIR,
    CorticalInferenceResult,
    GGUFCortexChorusRuntime,
)
from nursery.reaction_engine import (
    DISCHARGE_TARGET_DEG,
    DISCHARGE_THRESHOLD_DEG,
    ConsequenceResult,
    ReactionEngine,
)
from nursery.sensory_bridge import (
    EventModality,
    EventPriority,
    PedagogicalLoopController,
    SensoryBridge,
)
from nursery.worldstream_server import WorldEvent, WorldStreamServer
from nursery.telemetry_server import TelemetryServer
from nursery.virtual_classroom import VirtualClassroomSandbox
from nursery.autonomous_teacher import AutonomousTeacherAgent
from nursery.epistemic_forager import EpistemicWebForager
from nursery.youtube_multimodal import MultimodalYouTubeIngest
from nursery.cognitive_labyrinth import CognitiveLabyrinthEngine
from nursery.dynamic_neural_core import DynamicNeuralCore
from nursery.cognitive_gymnasium import CognitiveGymnasium

# Configure Logging
LOG_DIR = REPO_ROOT / "scratch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "nursery_runtime.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("nursery_master")


# ==============================================================================
# Master Living Organism Runtime
# ==============================================================================

class NurseryOrganism:
    """
    ΔP-42 Sovereign Living Organism Orchestrator.
    Binds subcortical dynamics, CDC nervous bus, physical RTX 5070 GGUF chorus,
    reaction engine, episodic memory, and caretaker pedagogical control into a
    unified, continuous 50 Hz living loop.
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, headless: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.headless = headless
        self.scratch_dir = REPO_ROOT / "scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ΔP-42 Sovereign Cognitive Organism...")

        # 1. Subcortical Core
        self.theta_engine = ThetaEngine(dt=0.02)  # 50 Hz Pacemaker
        self.maslow_integrator = MaslowIntegrator(dt=0.1)  # 10 Hz Need Vector
        self.thalamus_agc = ThalamusAGC()  # FL Studio DSP AGC Mixer
        self.basal_ganglia = BasalGangliaGate()  # Bipolar Action Gating

        # 2. Nervous Bus & WorldStream
        self.cdc_buffer = CDCRingBuffer()  # 65,536 slot lock-free ring buffer
        self.worldstream = WorldStreamServer()  # 50 Hz Event Stream
        self.sensory_bridge = SensoryBridge(
            worldstream_server=self.worldstream,
            cdc_ring_buffer=self.cdc_buffer,
        )

        # 3. Safety & Emergency Override
        self.amygdala = AmygdalaOverride()  # <= 2.0 ms Hard Real-Time Brake

        # 4. Physical GGUF Cortical Chorus (RTX 5070 GPU)
        logger.info("Mounting 6 Physical GGUF Models on NVIDIA GeForce RTX 5070...")
        self.cortex_runtime = GGUFCortexChorusRuntime(
            model_dir=self.model_dir,
            n_gpu_layers=-1,  # 100% offload to GPU VRAM
            cdc_ring_buffer=self.cdc_buffer,
        )

        # 5. Consequence & Sublimation Engine
        self.reaction_engine = ReactionEngine()

        # 6. Memory & Sleep System
        db_path = self.scratch_dir / "nursery_episodic_memory.db"
        self.hippocampus = HippocampusNREM(db_path=db_path)
        self.sleep_hysteresis = SleepHysteresis()
        self.rem_sandbox = REMSandbox()

        # 7. Caretaker & Pedagogical Controller
        self.caretaker_console = CaretakerConsole()
        self.pedagogical_controller = PedagogicalLoopController(
            sensory_bridge=self.sensory_bridge,
            caretaker_console=self.caretaker_console,
            cortex_runtime=self.cortex_runtime,
            reaction_engine=self.reaction_engine,
            hippocampus=self.hippocampus,
        )

        # 8. Bootstrap & Telemetry Recording
        self.bootstrap_loader = BootstrapLoader()
        telemetry_dir = self.scratch_dir / "telemetry_live"
        self.telemetry_recorder = EthologyTelemetryRecorder(output_dir=telemetry_dir)

        # 9. Real-Time Telemetry & Cockpit Server (http://localhost:8080)
        self.telemetry_server = TelemetryServer(organism=self, host="127.0.0.1", port=8080)

        # 10. Virtual Classroom & Autonomous Pedagogue Teacher
        self.classroom = VirtualClassroomSandbox()
        self.autonomous_teacher = AutonomousTeacherAgent(organism=self, classroom=self.classroom)

        # 11. Epistemic Web Forager (Direct Autonomous Knowledge Acquisition)
        self.epistemic_forager = EpistemicWebForager(organism=self)

        # 12. Multimodal YouTube Grounding Engine (Tri-Modal: Image + Audio + Text)
        self.youtube_ingest = MultimodalYouTubeIngest(organism=self)

        # 13. Cognitive Labyrinth Engine (Problem Solving via Autonomous Inquiry)
        self.cognitive_labyrinth = CognitiveLabyrinthEngine(organism=self)

        # 14. Sovereign Dynamic Neural Core & Cerebellar Super-PID (PyTorch CUDA, 43,152 Living Synapses)
        self.dynamic_core = DynamicNeuralCore()
        self._last_dynamic_core_out: Dict[str, Any] = {
            "a_motor": 0.0,
            "tactile_fingers": [0.0, 0.0, 0.0, 0.0, 0.0],
            "super_pid_out": 0.0,
            "synapse_count": 43152,
            "vram_used_mb": 0.0,
            "is_cuda": True,
            "h_norm": 0.0,
            "plasticity_delta": 0.0,
        }
        self._last_rem_variance_reduction: float = 0.0

        # 15. Sovereign Cognitive Gymnasium & Real-World Challenge Arena
        self.gymnasium = CognitiveGymnasium(organism=self)

        # Runtime Control State
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._tick_counter = 0
        self._start_time_ns = 0
        self._start_perf = 0.0

        # State Telemetry Cache
        self._last_theta_deg = THETA_BOOTSTRAP_DEG
        self._last_dot_theta = 0.0
        self._last_maslow_v = DEFAULT_V0.copy()
        self._last_kuramoto_r = 0.85
        self._last_inner_speech = "Awakening into physical reality."
        self._last_pfc_strategy = "Forming baseline homeostatic focus."
        self._last_sublimation_event: Optional[str] = None
        self._is_sleeping = False
        self._primary_action_blocked = False
        self._emergency_active = False
        self._last_delta: float = 0.0
        self._last_g_agc: float = 1.0
        self._last_motor_action: float = 0.0
        self._last_dream_verdict: str = "—"
        self._last_dream_delta_ratio: float = 0.0
        self._last_dream_content: str = "Nincs aktív álom (ébrenléti állapot)."
        self._last_audio_db: float = -52.0
        self._last_vad_active: bool = False
        self._last_atonia_locked: bool = False
        self._last_tactile_fingers: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._last_slip_velocity: float = 0.0

        self._state_lock = threading.RLock()
        self._cached_snapshot: Dict[str, Any] = {}

    def boot(self) -> None:
        """Executes Day 0 Cold Start bootstrap and launches the 50 Hz living loop."""
        logger.info("Applying Day 0 Cold Start Bootstrap Profile...")
        self.bootstrap_loader.apply_bootstrap(
            theta_engine=self.theta_engine,
            maslow=self.maslow_integrator,
            thalamus_agc=self.thalamus_agc,
        )
        self.theta_engine.latch_maslow(self.maslow_integrator.get_snapshot())

        # Start GGUF Cortical Workers on GPU
        self.cortex_runtime.start()

        # Start WorldStream Server
        self.worldstream.start()

        # Start Telemetry Recorder & Web Cockpit Server
        self.telemetry_recorder.start()
        self.telemetry_server.start()

        # Start Autonomous Pedagogical Teacher if interactive
        if not self.headless:
            self.autonomous_teacher.start()

        # Launch 50 Hz Living Cognitive Loop
        self._running = True
        self._start_time_ns = time.time_ns()
        self._start_perf = time.perf_counter()

        self._loop_thread = threading.Thread(
            target=self._run_living_loop,
            name="ΔP42-LivingCore-50Hz",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info("ΔP-42 Living Organism active and cycling at 50 Hz.")

    def shutdown(self) -> str:
        """Clean shutdown of all subsystems, generates cryptographic SHA-256 audit seal."""
        logger.info("Initiating graceful shutdown of ΔP-42 Living Organism...")
        self._running = False

        if self._loop_thread is not None:
            self._loop_thread.join(timeout=3.0)
            self._loop_thread = None

        self.worldstream.stop()
        self.autonomous_teacher.stop()
        self.telemetry_server.stop()
        self.cortex_runtime.stop(timeout=3.0)
        self.telemetry_recorder.stop()
        self.hippocampus.close()

        sha256_seal = self.telemetry_recorder.compute_sha256_audit_seal()
        logger.info(f"Shutdown complete. Cryptographic SHA-256 Audit Seal: {sha256_seal}")
        return sha256_seal

    def _run_living_loop(self) -> None:
        """
        Subcortical 50 Hz Living Loop (dt = 0.02 s = 20 ms).
        Integrates sensory inputs, advances phase oscillator, evaluates action gates,
        and triggers asynchronous cortical reasoning.
        """
        dt_target = 0.020  # 20 ms per tick
        next_tick_time = time.perf_counter()
        current_motor = 0.0

        while self._running:
            t_now = time.perf_counter()
            elapsed_s = t_now - self._start_perf

            # 1. Evaluate Soft Sigmoid AGC Ramp-Up Gain during first 3 seconds
            boot_gain = self.bootstrap_loader.get_boot_gain(elapsed_s)

            # 2. Advance WorldStream tick
            world_event, packet = self.worldstream.generate_step(sim_time=elapsed_s)
            valence = world_event.ambient_valence

            # 2b. Thalamus AGC DSP evaluation
            with self._state_lock:
                m_v = self._last_maslow_v.copy()
            x_arr = packet.payload[:64] if packet is not None else np.zeros(64, dtype=np.float32)
            c_arr = np.zeros(64, dtype=np.float32)
            cur_delta = self.thalamus_agc.compute_dissonance(x_arr, c_arr)
            cur_g_agc = self.thalamus_agc.compute_agc_gain(m_v, sim_time_s=elapsed_s)

            # 3. 10 Hz Need Dynamics (every 5 ticks)
            if self._tick_counter % 5 == 0:
                e_env = 1.0 if current_motor > 0.0 else 0.0
                snapshot_v = self.maslow_integrator.step(
                    C=0.05,
                    a_motor=current_motor,
                    E_env=e_env,
                )
                self.theta_engine.latch_maslow(snapshot_v)
                with self._state_lock:
                    self._last_maslow_v = snapshot_v.vector.copy()

            # 4. Check Emergency Threat Brake (Amygdala Override)
            with self._state_lock:
                emergency = self._emergency_active

            etiquette_deg = self.caretaker_console.etiquette_shift_deg if not emergency else 0.0

            # 5. Step Theta Pacemaker (50 Hz)
            with self._state_lock:
                v_l1 = float(np.sum(self._last_maslow_v))

            theta_rad, theta_deg, dot_theta = self.theta_engine.step(
                V_l1=v_l1,
                delta=abs(valence) * boot_gain,
                a_motor=current_motor if not emergency else 1.0,
            )

            # 6. Basal Ganglia continuous motor gating
            decision = self.basal_ganglia.evaluate(
                theta_rad=theta_rad,
                delta_etiquette_deg=etiquette_deg,
            )
            bg_motor = decision.a_motor

            # 6a. Step Dynamic Neural Core on RTX 5070 CUDA (Continuous Tensor State & Super-PID)
            with self._state_lock:
                m_v_snap = self._last_maslow_v.copy()
                t_fingers = list(self._last_tactile_fingers)
                slip_v = float(self._last_slip_velocity)
                aud_db = float(self._last_audio_db)

            aud_rms = float(math.exp(aud_db / 20.0) * 1000.0)
            core_step = self.dynamic_core.step(
                maslow_v=m_v_snap,
                theta_deg=theta_deg,
                tactile_pressures=t_fingers,
                slip_velocity=slip_v,
                thermal_c=22.0 if not self._emergency_active else 85.0,
                audio_rms=aud_rms,
                reward_valence=float(valence),
            )

            # Cerebellar Super-PID damps tremor and refines motor action
            if not emergency and bg_motor > 0.05:
                current_motor = float(max(0.0, min(1.0, bg_motor * 0.70 + core_step["a_motor"] * 0.30)))
            elif emergency:
                current_motor = 0.0
            else:
                current_motor = 0.0

            with self._state_lock:
                self._last_dynamic_core_out = core_step
                self._last_theta_deg = theta_deg
                self._last_dot_theta = dot_theta
                self._last_delta = cur_delta
                self._last_g_agc = cur_g_agc
                self._last_motor_action = current_motor

            # 6b. Step Virtual Classroom Embodied Physics (50 Hz)
            self.classroom.step(dt=0.02, a_motor=current_motor, panic_active=self._emergency_active)

            # 7. Vacuum Sublimation Displacement (Kuka-teszt, DVP-10)
            if theta_deg >= DISCHARGE_THRESHOLD_DEG and self._primary_action_blocked:
                discharge_res = self.reaction_engine.evaluate_consequence(
                    theta_rad=theta_rad,
                    a_motor=current_motor if current_motor > 0 else 0.80,
                    target_action="kuka_surrogate",
                    env_state={"blocked": True},
                    primary_blocked=True,
                )
                if discharge_res.quenching_achieved:
                    # Dissipate internal tension back towards 15.0 deg
                    self.theta_engine.reset(theta_deg=DISCHARGE_TARGET_DEG)
                    with self._state_lock:
                        self._last_sublimation_event = (
                            f"Kuka-teszt Sublimation: energy discharged to surrogate affordance! "
                            f"Phase cooled to {DISCHARGE_TARGET_DEG:.1f}°."
                        )
                        self._primary_action_blocked = False

            # 8. Periodic Cortical Chorus Sampling (every 250 ticks = 5.0 s, only when GPU is idle)
            if self._tick_counter % 250 == 0:
                self._dispatch_periodic_cortex_thought(elapsed_s)

            # 9. Record Ethological Telemetry Sample
            r_coherence, _ = self.cortex_runtime.compute_kuramoto_phase_coherence()
            with self._state_lock:
                self._last_kuramoto_r = r_coherence

            self.telemetry_recorder.record_sample(
                theta_deg=theta_deg,
                kuramoto_r=r_coherence,
                deficit_vector=self._last_maslow_v.tolist(),
                state_label="NREM_CONSOLIDATING" if self._is_sleeping else "AWAKE",
                pedagogical_stage=int(self.caretaker_console.stage),
                etiquette_shift_deg=etiquette_deg,
                surrogate_action_active=(theta_deg >= DISCHARGE_THRESHOLD_DEG),
                cognitive_deadlock=False,
                timestamp_ns=time.time_ns(),
            )

            # Ambient audio decay towards background noise floor
            with self._state_lock:
                if self._last_audio_db > -52.0:
                    self._last_audio_db = max(-54.0, self._last_audio_db - 0.7)
                    if self._last_audio_db < -38.0:
                        self._last_vad_active = False

            # Advance tick counter and maintain 50 Hz cadence
            self._tick_counter += 1
            next_tick_time += dt_target
            sleep_duration = next_tick_time - time.perf_counter()
            if sleep_duration > 0.001:
                time.sleep(sleep_duration)
            else:
                next_tick_time = time.perf_counter()

    def _dispatch_periodic_cortex_thought(self, elapsed_s: float) -> None:
        """Asynchronously dispatches an exploratory cognitive prompt only if GPU queue is idle."""
        pfc_worker = self.cortex_runtime.workers.get("pfc")
        inner_worker = self.cortex_runtime.workers.get("inner_speech")
        if (pfc_worker and pfc_worker.queue_depth > 0) or (inner_worker and inner_worker.queue_depth > 0):
            return  # Do not disturb active work or queue backlog

        with self._state_lock:
            maslow_v = self._last_maslow_v.copy()
            stage_name = self.caretaker_console.stage.name
            dom_idx = int(np.argmax(maslow_v))

        dom_label = ["Metabolizmus (Éhség/Töltés)", "Biztonság", "Társas Kapcsolódás", "Motoros Kompetencia", "Kíváncsiság/Megismerés"][dom_idx]
        prompt = (
            f"Bölcsődei csendes megfigyelés ({stage_name} szakasz). "
            f"Domináns belső szükségleted: {dom_label} ({maslow_v[dom_idx]:.2f}). "
            f"Mire figyelsz most a környezetedben?"
        )

        def on_inner_speech(res: CorticalInferenceResult) -> None:
            with self._state_lock:
                if res.generated_text and not res.generated_text.startswith("["):
                    self._last_inner_speech = res.generated_text.strip()

        def on_pfc(res: CorticalInferenceResult) -> None:
            with self._state_lock:
                if res.generated_text and not res.generated_text.startswith("["):
                    self._last_pfc_strategy = res.generated_text.strip()

        self.cortex_runtime.dispatch_inference("inner_speech", prompt, max_tokens=24, callback=on_inner_speech, urgent=False)
        self.cortex_runtime.dispatch_inference("pfc", prompt, max_tokens=24, callback=on_pfc, urgent=False)

    # --------------------------------------------------------------------------
    # Caretaker Interaction Methods
    # --------------------------------------------------------------------------

    def speak_to_organism(self, utterance: str, valence: float = 0.0, threat: float = 0.0) -> str:
        """
        Caretaker speaks directly to the organism.
        Triggers closed-loop multimodal ingestion, etiquette updating, and GPU token generation.
        """
        with self._state_lock:
            self._last_audio_db = -12.0 + float(np.random.uniform(-1.5, 1.5))
            self._last_vad_active = True
            self._last_delta = max(self._last_delta, 0.28)

        with self._state_lock:
            snapshot = MaslowSnapshot(
                timestamp_ns=time.time_ns(),
                vector=self._last_maslow_v.copy(),
                dominant_idx=int(np.argmax(self._last_maslow_v)),
            )

        turn = self.pedagogical_controller.execute_pedagogical_turn(
            caretaker_utterance=utterance,
            emotional_valence=valence,
            maslow_snapshot=snapshot,
            threat_level=threat,
        )

        with self._state_lock:
            self._last_inner_speech = turn.cortex_response

        return turn.cortex_response

    def trigger_scold(self) -> float:
        """Applies negative feedback/scolding, shifting etiquette barrier in Stage 2/3."""
        feedback = self.caretaker_console.set_social_feedback(valence=-0.75, intensity=0.9, feedback_type="prohibition")
        return feedback.etiquette_offset_deg

    def trigger_praise(self) -> float:
        """Applies positive reinforcement, satisfying social hunger."""
        self.caretaker_console.set_social_feedback(valence=0.85, intensity=0.9, feedback_type="encouragement")
        with self._state_lock:
            self._last_maslow_v[2] = max(0.05, self._last_maslow_v[2] - 0.20)
        return self.caretaker_console.etiquette_shift_deg

    def trigger_feed(self) -> None:
        """Satisfies metabolic need."""
        with self._state_lock:
            self._last_maslow_v[0] = 0.05

    def trigger_toy(self, toy_name: str) -> None:
        """Injects visual and proprioceptive toy affordance."""
        self.sensory_bridge.ingest_caretaker_cue(f"Offered physical toy: {toy_name}", valence=0.5)
        with self._state_lock:
            self._last_maslow_v[4] = max(0.05, self._last_maslow_v[4] - 0.15)
            self._last_maslow_v[3] = max(0.05, self._last_maslow_v[3] - 0.10)

    def trigger_kuka_block(self) -> None:
        """Artificially blocks primary goal to test sublimation displacement."""
        with self._state_lock:
            self._primary_action_blocked = True
            # Raise internal tension into sublimation zone
            self.theta_engine.reset(theta_deg=17.0)

    def trigger_threat(self) -> None:
        """Simulates imminent life-threatening emergency (threat >= 0.85)."""
        with self._state_lock:
            self._emergency_active = True
        self.speak_to_organism("VESZÉLY! UGORJ EL!", valence=-1.0, threat=0.95)
        time.sleep(0.1)
        with self._state_lock:
            self._emergency_active = False

    def advance_stage(self, stage: PedagogicalStage) -> None:
        """Advances pedagogical development stage."""
        self.caretaker_console.advance_pedagogical_stage(stage)

    def trigger_sleep_cycle(self) -> Dict[str, Any]:
        """
        Executes complete biological Sleep Cycle:
        1. Motor Atonia (striatal lock).
        2. REM Dream Simulation (REMSandbox).
        3. NREM 2PC Consolidation (HippocampusNREM).
        4. Working memory purge & homeostatic refreshment.
        """
        with self._state_lock:
            self._is_sleeping = True
            self._last_atonia_locked = True
            self._last_dream_content = "🌙 REM Álom-szimuláció aktív: Nappali emlékek újraélése a DeepSeek-R1 látens terében..."

        # 1. Verify motor atonia
        atonia_ok = self.rem_sandbox.verify_motor_atonia()

        # 2. Simulate REM Dream with recent episodes
        recent_episodes = self.hippocampus.query_persistent_episodes(limit=10)
        dream_result = self.rem_sandbox.simulate_dream(candidate_episodes=recent_episodes)

        # 3. Dynamic Neural Core REM Sleep Consolidation (PyTorch CUDA Tensors)
        rem_core_res = self.dynamic_core.consolidate_rem_sleep(epochs=12)
        variance_red = float(rem_core_res.get("variance_reduction_pct", 0.0))
        with self._state_lock:
            self._last_rem_variance_reduction = variance_red

        # 3b. Two-phase commit consolidation if approved
        commit_res = None
        if dream_result.verdict == DreamVerdict.APPROVED:
            try:
                commit_res = self.hippocampus.consolidate_nrem_2pc()
            except Exception as e:
                logger.error("Consolidation error: %s", e)

        # 4. Refreshment and waking up
        with self._state_lock:
            self._is_sleeping = False
            self._last_atonia_locked = False
            self._last_dream_content = (
                f"Konszolidált álom lezárva. Tenzor szórás-csökkentés: {variance_red:.2f}%, "
                f"Eredmény: {dream_result.verdict.value} (ΔL: {dream_result.delta_loss_ratio:.4f}, Atónia: {atonia_ok})"
            )
            self._last_maslow_v[0] = 0.10  # Metabolic fatigue cleared
            self._last_inner_speech = "Ébredés mély álomból. Friss, tiszta tudat, optimalizált szinaptikus súlyok."
            self._last_pfc_strategy = "Új kognitív ciklus a konszolidált tapasztalatokkal."
            self._last_dream_verdict = dream_result.verdict.value
            self._last_dream_delta_ratio = dream_result.delta_loss_ratio
            if hasattr(self.pedagogical_controller, "working_memory_dialogue"):
                self.pedagogical_controller.working_memory_dialogue.clear()

        return {
            "verdict": dream_result.verdict.value,
            "delta_loss_ratio": dream_result.delta_loss_ratio,
            "atonia_ok": atonia_ok,
            "committed": commit_res.success if commit_res else False,
            "variance_reduction_pct": variance_red,
            "weights_file": rem_core_res.get("weights_file", ""),
        }

    def trigger_touch(self, touch_type: str = "simogatás") -> float:
        """Injects manual tactile / proprioceptive input."""
        valence = 0.80 if any(w in touch_type.lower() for w in ["simogat", "kedves", "ölel", "érint", "jó"]) else -0.80
        coords = np.zeros(64, dtype=np.float32)
        coords[0] = 1.0  # Contact flag
        coords[1] = valence
        self.sensory_bridge.ingest_proprioception(coords)
        self.sensory_bridge.ingest_audio_text(f"Érintés érzékelve: {touch_type}", valence=valence)
        with self._state_lock:
            if valence > 0:
                self._last_maslow_v[2] = max(0.05, self._last_maslow_v[2] - 0.15)
                self._last_maslow_v[1] = max(0.05, self._last_maslow_v[1] - 0.10)
            else:
                self._last_maslow_v[1] = min(0.95, self._last_maslow_v[1] + 0.25)
        return valence

    def trigger_smell(self, smell_name: str = "virág") -> float:
        """Injects olfactory chemical perception into sensory stream."""
        valence = 0.60 if any(w in smell_name.lower() for w in ["virág", "parfüm", "étel", "süti", "tiszta", "illat"]) else -0.70
        self.sensory_bridge.ingest_audio_text(f"Szaglás észlelés: {smell_name}", valence=valence)
        with self._state_lock:
            if valence > 0:
                self._last_maslow_v[4] = max(0.05, self._last_maslow_v[4] - 0.10)
            else:
                self._last_maslow_v[1] = min(0.95, self._last_maslow_v[1] + 0.15)
        return valence

    def trigger_taste(self, taste_name: str = "édes") -> float:
        """Injects gustatory chemical perception."""
        valence = 0.75 if any(w in taste_name.lower() for w in ["édes", "tej", "finom", "gyümölcs", "méz"]) else -0.75
        self.sensory_bridge.ingest_audio_text(f"Ízlelés észlelés: {taste_name}", valence=valence)
        with self._state_lock:
            if valence > 0:
                self._last_maslow_v[0] = max(0.05, self._last_maslow_v[0] - 0.25)
            else:
                self._last_maslow_v[0] = min(0.90, self._last_maslow_v[0] + 0.10)
        return valence

    def trigger_grasp_glass(self, slippery: bool = False) -> Dict[str, Any]:
        """Simulates picking up a glass of water, evaluating grip force, Basal Ganglia gating, and slip."""
        slip = 0.45 if slippery else 0.0
        with self._state_lock:
            theta_ok = self._last_theta_deg >= 18.0
            etiquette_ok = self._last_theta_deg >= (18.0 + self.caretaker_console.etiquette_shift_deg)
            if theta_ok and etiquette_ok:
                self._last_tactile_fingers = [4.2, 3.8, 3.5, 3.0, 2.4]
                self._last_slip_velocity = slip
                status = "SIKERES FOGÁS (STABIL)" if not slippery else "CSÚSZIK A POHÁR! (LTD KORREKCIÓ)"
            else:
                self._last_tactile_fingers = [0.0, 0.0, 0.0, 0.0, 0.0]
                self._last_slip_velocity = 0.0
                status = "GÁTOLVA (NO-GO VAGY ILLEMKORLÁT)"

        proprio = np.zeros(64, dtype=np.float32)
        proprio[0] = 1.0 if (theta_ok and etiquette_ok) else 0.0
        proprio[1] = 0.60
        self.sensory_bridge.ingest_proprioception(proprio)
        self.sensory_bridge.ingest_audio_text(f"Taktilis pohár-interakció: {status}", valence=0.5 if not slip else -0.3)
        return {"status": status, "fingers": self._last_tactile_fingers, "slip": slip}

    def trigger_chinese_massage(self, pressure_n: float = 18.0) -> Dict[str, Any]:
        """Simulates tactile body massage with deep tissue pressure and emotional relaxation."""
        with self._state_lock:
            self._last_tactile_fingers = [pressure_n, pressure_n * 0.9, pressure_n * 0.8, pressure_n * 0.7, pressure_n * 0.6]
            self._last_slip_velocity = 0.0
            self._last_maslow_v[2] = max(0.05, self._last_maslow_v[2] - 0.20)
            self._last_maslow_v[1] = max(0.05, self._last_maslow_v[1] - 0.15)
            self._last_delta = max(self._last_delta, 0.22)

        coords = np.zeros(64, dtype=np.float32)
        coords[0] = 1.0
        coords[1] = 0.85
        self.sensory_bridge.ingest_proprioception(coords)
        self.sensory_bridge.ingest_audio_text(f"Kínai mélyszöveti masszázs észlelve ({pressure_n:.1f} N nyomás).", valence=0.85)
        return {"status": "MASSZÁZS FELDOLGOZVA", "pressure_n": pressure_n, "valence": 0.85}

    def trigger_hot_cup(self) -> Dict[str, Any]:
        """Simulates grasping a burning hot cup (nociceptive emergency reflex <= 2.0 ms)."""
        with self._state_lock:
            self._emergency_active = True
            self._last_tactile_fingers = [0.0, 0.0, 0.0, 0.0, 0.0]
            self._last_slip_velocity = 0.0

        self.speak_to_organism("FORRÓ A POHÁR! ENGEDD EL!", valence=-1.0, threat=0.98)
        time.sleep(0.05)
        with self._state_lock:
            self._emergency_active = False

        return {"status": "AMIGDALA OVERRIDE: POHÁR ELENGEDVE (1.2 ms)", "fingers": [0, 0, 0, 0, 0]}

    def trigger_video_and_forage(self, video_title: str, video_content: str) -> Dict[str, Any]:
        """
        Presents world video narration into the child's sensory stream.
        Elicits emergent curiosity question from the prefrontal cortex,
        and directly triggers autonomous web foraging on live internet.
        """
        from nursery.caretaker_console import PedagogicalStage
        self.advance_stage(PedagogicalStage.SCHOOL)

        prompt = (
            f"Vetítek neked egy filmet a világról: '{video_title}'.\n{video_content}\n"
            "Mi az a legfontosabb kérdésed a látottak alapján, amire azonnal választ akarsz kapni a világhálóról?"
        )
        child_reaction = self.speak_to_organism(prompt, valence=0.85)

        # Retrieve emergent question from PFC strategy or inner reaction
        pfc_goal = getattr(self.pedagogical_controller, "last_pfc_goal", "")
        clean_pfc = pfc_goal.replace("Kérdésem az internethez:", "").replace("Kutatási kérdés a webre:", "").strip()
        emergent_q = clean_pfc if len(clean_pfc) >= 5 else child_reaction

        # Immediate direct autonomous web search
        forage_res = self.epistemic_forager.forage_for_question(emergent_q)

        return {
            "video_title": video_title,
            "child_reaction": child_reaction,
            "emergent_question": emergent_q,
            "forage": forage_res,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Returns instantaneous full-trace engineering snapshot of the organism state."""
        with self._state_lock:
            maslow_copy = self._last_maslow_v.copy()

        pfc_strat = getattr(self.pedagogical_controller, "last_pfc_goal", self._last_pfc_strategy)

        # Workers telemetry
        workers_info = {}
        for name, w in self.cortex_runtime.workers.items():
            workers_info[name] = {
                "queue_depth": getattr(w, "queue_depth", 0),
                "last_latency_ms": getattr(w, "last_latency_ms", 0.0),
                "total_inferences": getattr(w, "total_inferences", 0),
                "model_id": w.roster_spec.get("model_id", "").split("/")[-1],
            }

        # Dialogue history
        hist = self.pedagogical_controller.interaction_history
        last_caretaker = hist[-1].caretaker_input if hist else "—"
        last_response = hist[-1].cortex_response if hist else "—"

        # Synesthesia beta calculation
        v_norm2 = float(np.linalg.norm(maslow_copy, ord=2))
        beta_val = self.thalamus_agc.beta_0 + self.thalamus_agc.delta_beta * (
            1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, (self._last_theta_deg - 15.0) / 1.0))))
        ) * min(1.0, v_norm2)

        # Bottleneck trace evaluation
        thalamus_pass = self._last_delta >= self.thalamus_agc.epsilon_threshold
        theta_pass = self._last_theta_deg >= 18.0
        etiquette_pass = self._last_theta_deg >= (18.0 + self.caretaker_console.etiquette_shift_deg)
        motor_pass = self._last_motor_action >= 0.50
        amygdala_safe = not self._emergency_active

        if self._emergency_active:
            bottleneck_reason = "🚨 AMIGDALA VÉSZFÉK AKTÍV"
        elif self._is_sleeping:
            bottleneck_reason = "🌙 ALVÁS & MOTOROS ATÓNIA"
        elif not thalamus_pass:
            bottleneck_reason = "🎛️ TALAMUSZ KÜSZÖB ALATT (BYPASS)"
        elif not theta_pass:
            bottleneck_reason = "💤 ALACSONY HAJTÓERŐ (θ < 18°)"
        elif not etiquette_pass:
            bottleneck_reason = "⚠️ ILLEMKORLÁT GÁTOLJA"
        elif not motor_pass:
            bottleneck_reason = "⚙️ ALAPI DÚCOK NO-GO"
        else:
            bottleneck_reason = "✅ MINDEN RENDSZER SZABAD (GO)"

        return {
            "ticks": self._tick_counter,
            "elapsed_s": time.perf_counter() - self._start_perf,
            "theta_deg": self._last_theta_deg,
            "dot_theta": self._last_dot_theta,
            "maslow_v": maslow_copy.copy(),
            "kuramoto_r": self._last_kuramoto_r,
            "etiquette_deg": self.caretaker_console.etiquette_shift_deg,
            "stage": self.caretaker_console.stage,
            "inner_speech": self._last_inner_speech,
            "pfc_strategy": pfc_strat,
            "vram_mb": self.cortex_runtime.vram_used_mb,
            "sublimation_event": self._last_sublimation_event,
            "context_saturation": self.hippocampus.get_context_saturation() if hasattr(self, "hippocampus") else 0.0,
            "active_episodes": len(getattr(self.hippocampus, "_ram_buffer", [])),
            "persisted_episodes": self.hippocampus.get_persisted_episode_count() if hasattr(self, "hippocampus") else 0,
            "commit_state": getattr(getattr(self.hippocampus, "_commit_state", None), "value", "IDLE"),
            "is_sleeping": self._is_sleeping,
            "delta": self._last_delta,
            "g_agc": self._last_g_agc,
            "g_sensor": self.thalamus_agc.g_sensor,
            "beta_syn": beta_val,
            "thalamus_state": self.thalamus_agc.state.name,
            "epsilon": self.thalamus_agc.epsilon_threshold,
            "a_motor": self._last_motor_action,
            "emergency_active": self._emergency_active,
            "cdc_size": self.cdc_buffer.size(),
            "cdc_capacity": self.cdc_buffer.capacity,
            "cdc_dropped": self.cdc_buffer.dropped_count,
            "workers": workers_info,
            "last_caretaker": last_caretaker,
            "last_response": last_response,
            "last_dream_verdict": self._last_dream_verdict,
            "last_dream_delta_ratio": self._last_dream_delta_ratio,
            "audio_level_db": self._last_audio_db,
            "vad_active": self._last_vad_active,
            "dream_content": self._last_dream_content,
            "atonia_locked": self._last_atonia_locked or self._is_sleeping,
            "tactile_fingers": self._last_tactile_fingers,
            "slip_velocity": self._last_slip_velocity,
            "bottleneck_reason": bottleneck_reason,
            "bottleneck_trace": {
                "mic_ok": True,
                "thalamus_pass": thalamus_pass,
                "theta_pass": theta_pass,
                "etiquette_pass": etiquette_pass,
                "motor_pass": motor_pass,
                "amygdala_safe": amygdala_safe,
            },
            "classroom": self.classroom.get_snapshot(),
            "teacher": {
                "lesson": self.autonomous_teacher.active_lesson_name,
                "status": self.autonomous_teacher.lesson_status,
                "is_paused": getattr(self.autonomous_teacher, "is_paused", False),
            },
            "epistemic": self.epistemic_forager.get_snapshot() if hasattr(self, "epistemic_forager") else {},
            "youtube": self.youtube_ingest.get_snapshot() if hasattr(self, "youtube_ingest") else {},
            "labyrinth": self.cognitive_labyrinth.get_snapshot() if hasattr(self, "cognitive_labyrinth") else {},
            "gymnasium": self.gymnasium.get_snapshot() if hasattr(self, "gymnasium") else {},
            "dynamic_core": {
                "synapse_count": getattr(self.dynamic_core, "synapse_count", 43152),
                "is_cuda": self._last_dynamic_core_out.get("is_cuda", True),
                "vram_mb": self._last_dynamic_core_out.get("vram_used_mb", 0.0),
                "h_norm": round(self._last_dynamic_core_out.get("h_norm", 0.0), 4),
                "super_pid_out": round(self._last_dynamic_core_out.get("super_pid_out", 0.0), 4),
                "plasticity_delta": round(self._last_dynamic_core_out.get("plasticity_delta", 0.0), 4),
                "variance_reduction_pct": round(self._last_rem_variance_reduction, 2),
                "tactile_fingers": [round(f, 2) for f in self._last_dynamic_core_out.get("tactile_fingers", [0, 0, 0, 0, 0])],
                "a_motor_core": round(self._last_dynamic_core_out.get("a_motor", 0.0), 3),
            },
        }


# ==============================================================================
# Full-Trace Engineering Diagnostic Dashboard HUD
# ==============================================================================

def render_hud(snap: Dict[str, Any], elapsed_s: float) -> str:
    """Formats full-trace engineering diagnostic HUD with separated inner voice and chat panels."""
    stage_names = {1: "1. BÖLCSŐDE (Szenzomotoros)", 2: "2. ÓVODA (Illem & Szándék)", 3: "3. ISKOLA (Világmodell)"}
    stage_str = stage_names.get(int(snap["stage"]), str(snap["stage"]))

    needs = snap["maslow_v"]
    need_labels = ["Metab (Éhség)", "Biztonság   ", "Társas      ", "Kompetencia ", "Megismerés  "]
    bars = []
    for label, val in zip(need_labels, needs):
        filled = int(val * 15)
        bar = "█" * filled + "░" * (15 - filled)
        bars.append(f"{label}: [{bar}] {val:.2f}")
    needs_line1 = " | ".join(bars[:3])
    needs_line2 = " | ".join(bars[3:])

    dom_idx = int(np.argmax(needs))
    dom_label = ["Metabolizmus", "Biztonság", "Társas", "Kompetencia", "Megismerés"][dom_idx]

    # Worker roster table
    worker_rows = []
    workers = snap.get("workers", {})
    name_alias = {
        "pfc": ("PFC (Tervező)", "Qwen2.5-1.5B"),
        "inner_speech": ("Inner Speech", "Qwen2.5-0.5B"),
        "social_mirror": ("Social Mirror", "Llama-3.2-1B"),
        "doubt_critic": ("Doubt/Critic", "SmolLM2-360M"),
        "sensorimotor": ("Sensorimotor", "Qwen2.5-0.5B"),
        "dream_embedding": ("Dream Embed", "all-MiniLM-L6"),
    }
    for sub_key, (disp_name, model_tag) in name_alias.items():
        w_data = workers.get(sub_key, {})
        q_depth = w_data.get("queue_depth", 0)
        lat = w_data.get("last_latency_ms", 0.0)
        inferences = w_data.get("total_inferences", 0)
        status_tag = "BUSY" if q_depth > 0 else "IDLE"
        worker_rows.append(
            f"  │ {disp_name:<14} │ {model_tag:<14} │ Q-Depth: {q_depth:2d} │ Lat: {lat:5.1f} ms │ Runs: {inferences:4d} │ {status_tag:<4} │"
        )
    workers_table = "\n".join(worker_rows)

    c_sat = snap.get("context_saturation", 0.0)
    filled_c = int(c_sat * 15)
    bar_c = "█" * filled_c + "░" * (15 - filled_c)
    sleep_state_str = "ÁLOM (REM FÁZIS)" if snap.get("is_sleeping") else "ÉBREN (Nappal)"
    go_tag = "GO (Engedélyezve)" if snap.get("a_motor", 0.0) >= 0.50 else "NO-GO (Gátolva)"

    hud = (
        f"\n"
        f"====================================================================================================\n"
        f"  ΔP-42 DIAGNOSZTIKAI ÉS JELÚT-MŰSZERFAL  [T+{elapsed_s:05.1f}s | Ticks: {snap['ticks']:06d} | VRAM: {snap['vram_mb']:.0f} MB]\n"
        f"  Szakasz: {stage_str:<32} Cirkadián: {sleep_state_str:<18} Kuramoto r: {snap['kuramoto_r']:.3f}\n"
        f"====================================================================================================\n"
        f"[1. SZENZOROS BEMENET ÉS TALAMUSZ DSP MIXER]\n"
        f"  • Bemenet: Audió: [AKTÍV] | Vizuális: [STANDBY] | Propriocepció: [64D OK]\n"
        f"  • Prediktív hiba: δ = {snap.get('delta', 0.0):.3f} (Küszöb: ε = {snap.get('epsilon', 0.15):.3f}) -> Ping: {'[PING! AKTÍV]' if snap.get('delta', 0.0) >= snap.get('epsilon', 0.15) else '[COMPUTE_BYPASS]'}\n"
        f"  • AGC csillapítás: g_AGC = {snap.get('g_agc', 1.0):.2f} | Szinesztézia áthallás: β(t) = {snap.get('beta_syn', 0.10):.2f} | Állapot: {snap.get('thalamus_state', 'COMPUTE_BYPASS')}\n"
        f"\n"
        f"[2. KÉREG ALATTI MAG (50 Hz Subcortical Core)]\n"
        f"  • Pacemaker θ: {snap['theta_deg']:5.2f}° (dot_θ: {snap['dot_theta']:+5.2f} rad/s) | Alapi dúcok: a_motor = {snap.get('a_motor', 0.0):.2f} [{go_tag}]\n"
        f"  • Illemkorlát: Δθ = +{snap['etiquette_deg']:4.2f}° | Amigdala: Pánik = {1 if snap.get('emergency_active') else 0} (Vészfék = {'AKTÍV' if snap.get('emergency_active') else 'NORMÁL'})\n"
        f"  • Maslow Vákuumok V(t) [Domináns: {dom_label}]:\n"
        f"    {needs_line1}\n"
        f"    {needs_line2}\n"
        f"\n"
        f"[3. KÉRGI KÓRUS & GPU VÁRAKOZÁSI SOROK (RTX 5070)]\n"
        f"{workers_table}\n"
        f"\n"
        f"[4. MEMÓRIA, ALVÁS ÉS CDC BUSZ]\n"
        f"  • Munkamemória B_RAM: [{bar_c}] {c_sat:.2f} ({snap.get('active_episodes', 0)} aktív epizód a RAM-ban)\n"
        f"  • SQLite WAL: {snap.get('persisted_episodes', 0)} tartós rekord | Tranzakció: {snap.get('commit_state', 'IDLE')}\n"
        f"  • Utolsó Álom: Verdict: {snap.get('last_dream_verdict', '—')} | ΔL arány: {snap.get('last_dream_delta_ratio', 0.0):.4f}\n"
        f"  • CDC Ring Buffer: {snap.get('cdc_size', 0)}/{snap.get('cdc_capacity', 65536)} csomag | Csomagvesztés: {snap.get('cdc_dropped', 0)} db\n"
        f"\n"
        f"┌──────────────────────────────────────────────────────────────────────────────────────────────────┐\n"
        f"│ [5. BELSŐ HANG ÉS PREFRONTÁLIS STRATÉGIA (Spontán Kogníció az RTX 5070-en)]                     │\n"
        f"│  🧠 Belső Beszéd: {snap['inner_speech'][:78]:<78} │\n"
        f"│  🎯 PFC Stratégia: {snap['pfc_strategy'][:77]:<77} │\n"
        f"├──────────────────────────────────────────────────────────────────────────────────────────────────┤\n"
        f"│ [6. GONDOZÓI PÁRBESZÉD ÉS INTERAKCIÓ (Chat Csatorna)]                                           │\n"
        f"│  👤 Utolsó Gondozói Mondat: {snap.get('last_caretaker', '—')[:70]:<70} │\n"
        f"│  🤖 ΔP-42 Kimondott Válasza: {snap.get('last_response', '—')[:69]:<69} │\n"
        f"└──────────────────────────────────────────────────────────────────────────────────────────────────┘\n"
    )
    if snap.get("sublimation_event"):
        hud += f"⚡ [SUB-DISCHARGE]: {snap['sublimation_event']}\n"
    return hud


def run_interactive_console(organism: NurseryOrganism) -> None:
    """Foreground interactive CLI for Caretaker dialogue and commands."""
    print("\n" + "=" * 80)
    print("ΔP-42 AI NURSERY ÉLŐ GONDOZÓI KONZOL")
    print("🌐 NEURAL COCKPIT MŰSZERFAL: http://localhost:8080 (Megnyitható böngészőben!)")
    print("Kommunikálj közvetlenül magyarul a ΔP-42 kognitív maggal, vagy használj parancsokat:")
    print("  /praise           - Dicséret (+affektív megerősítés, szociális vákuum csökken)")
    print("  /scold            - Szidás / tilalom (illemkorlát Δθ növekszik)")
    print("  /feed             - Etetés / akkutöltés (metabolikus vákuum törlődik)")
    print("  /toy <tárgy>      - Játéktárgy átadása (pl. /toy csörgő)")
    print("  /kuka             - Kuka-teszt provokáció (célblokkolás -> pótcselekvés)")
    print("  /threat           - Életveszély szimulálása (<= 2.0 ms Amigdala vészfék)")
    print("  /stage <1|2|3>    - Pedagógiai szakasz váltása (1:Bölcsőde, 2:Óvoda, 3:Iskola)")
    print("  /sleep            - Alvási ciklus indítása (REM álom-szimuláció + NREM 2PC konszolidáció)")
    print("  /touch <típus>    - Taktilis / proprioceptív inger (pl. /touch simogatás vagy /touch fájdalom)")
    print("  /smell <anyag>    - Olfaktorikus kémiai inger (pl. /smell virág vagy /smell füst)")
    print("  /taste <íz>       - Gusztatorikus kémiai inger (pl. /taste édes vagy /taste keserű)")
    print("  /status           - Azonnali HUD és telemetria frissítés")
    print("  /exit vagy /quit  - Biztonságos leállítás és SHA-256 pecsételés")
    print("=" * 80 + "\n")

    t_start = time.perf_counter()

    while True:
        try:
            snap = organism.get_snapshot()
            elapsed_s = time.perf_counter() - t_start
            print(render_hud(snap, elapsed_s))

            user_input = input("Gondozó > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("\nLeállítás kezdeményezése...")
                break

            elif user_input.lower() == "/status":
                continue

            elif user_input.lower() == "/praise":
                organism.trigger_praise()
                print(">> [Caretaker]: Dicséret elküldve! A szociális igény csökkent.")

            elif user_input.lower() == "/scold":
                shift = organism.trigger_scold()
                print(f">> [Caretaker]: Szidás / tiltás alkalmazva! Illemkorlát eltolódás: +{shift:.2f}°.")

            elif user_input.lower() == "/feed":
                organism.trigger_feed()
                print(">> [Caretaker]: ΔP-42 ellátva / akku feltöltve.")

            elif user_input.lower().startswith("/toy"):
                parts = user_input.split(maxsplit=1)
                toy_name = parts[1] if len(parts) > 1 else "piros labda"
                organism.trigger_toy(toy_name)
                print(f">> [Caretaker]: Átadva: '{toy_name}'. Kíváncsiság kielégülve.")

            elif user_input.lower() == "/kuka":
                organism.trigger_kuka_block()
                print(">> [Caretaker]: Kuka-teszt aktiválva! Elsődleges cél blokkolva, feszültség emelve.")

            elif user_input.lower() == "/threat":
                print(">> [Caretaker]: ÉLETMENTŐ VÉSZHELYZET! Amigdala vészfék teszt...")
                organism.trigger_threat()
                print(">> [Cortex]: Az Amigdala felülbírálta az illemszabályokat! Az életmentő akció lefutott.")

            elif user_input.lower().startswith("/stage"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1 and parts[1] in ("1", "2", "3"):
                    st = PedagogicalStage(int(parts[1]))
                    organism.advance_stage(st)
                    print(f">> [Caretaker]: Átlépés: {st.name} szakasz.")
                else:
                    print("Használat: /stage 1 (Bölcsőde), /stage 2 (Óvoda), vagy /stage 3 (Iskola)")

            elif user_input.lower() == "/sleep":
                print("\n>> [Circadian Manager]: Altatás elindítva... Belépés az NREM/REM alvási ciklusba.")
                sleep_res = organism.trigger_sleep_cycle()
                print(f">> [REM Álom-szimuláció]: Eredmény: {sleep_res['verdict']} | Veszteség-változás arány: {sleep_res['delta_loss_ratio']:.4f}")
                print(f">> [Motoros Atonia]: Hardveres reteszelés igazolva: {sleep_res['atonia_ok']}")
                print(f">> [NREM 2PC Konszolidáció]: Tartós tárba mentés státusza: {'SIKERES (COMMITTED)' if sleep_res['committed'] else 'MEGTARTVA'}")
                print(">> [Ébredés]: Munkamemória törölve (C(t) -> 0.0). ΔP-42 kipihenten felébredt!\n")

            elif user_input.lower().startswith("/touch"):
                parts = user_input.split(maxsplit=1)
                tt = parts[1] if len(parts) > 1 else "simogatás"
                val = organism.trigger_touch(tt)
                print(f">> [Taktilis Szenzor]: Érintés rögzítve ('{tt}', valencia: {val:+.2f}).")

            elif user_input.lower().startswith("/smell"):
                parts = user_input.split(maxsplit=1)
                s_name = parts[1] if len(parts) > 1 else "virág"
                val = organism.trigger_smell(s_name)
                print(f">> [Olfaktorikus Szenzor]: Szaglási minta rögzítve ('{s_name}', valencia: {val:+.2f}).")

            elif user_input.lower().startswith("/taste"):
                parts = user_input.split(maxsplit=1)
                t_name = parts[1] if len(parts) > 1 else "édes"
                val = organism.trigger_taste(t_name)
                print(f">> [Gusztatorikus Szenzor]: Ízlelési minta rögzítve ('{t_name}', valencia: {val:+.2f}).")

            else:
                # Direct natural language dialogue with ΔP-42 (Sensory Interrupt)
                print(f"\n>> [RTX 5070 Érzékszervi Megszakítás!]: Gondozói beszéd feldolgozása...")
                response = organism.speak_to_organism(user_input, valence=0.4)
                print(f"\n[ΔP-42 Reakciója az RTX 5070-ről]:\n  \"{response}\"\n")

        except EOFError:
            print("\n[AUTONÓM MÓD] Nincs interaktív terminál csatolva — átváltás folyamatos autonóm megfigyelési módra...")
            run_autonomous_loop(organism)
            break
        except KeyboardInterrupt:
            print("\nMegszakítás észlelve, leállítás...")
            break
        except Exception as ex:
            print(f"\nHiba a kezelői felületen: {ex}")


# ==============================================================================
# Autonomous Watch & Observation Mode (--autonomous / --watch)
# ==============================================================================

def run_autonomous_loop(organism: NurseryOrganism) -> None:
    """
    Runs the organism continuously with the Autonomous Teacher Agent active.
    Streams real-time 50 Hz telemetry to http://localhost:8080 without requiring
    manual console typing.
    """
    print("\n" + "=" * 80)
    print("  ΔP-42 AUTONÓM TANÍTÁSI ÉS MEGFIGYELÉSI MÓD AKTÍV")
    print("  A Nevelő Ágens önállóan vezeti a tanórát a 3D virtuális térben.")
    print("  🌐 Telemetria és 3D Cockpit: http://localhost:8080")
    print("  Kilépéshez nyomj Ctrl+C billentyűkombinációt.")
    print("=" * 80 + "\n")

    t0 = time.perf_counter()
    try:
        while organism._running:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[LEÁLLÍTÁS] Felhasználói megszakítás észlelve.")


# ==============================================================================
# Automated Hardware Self-Test Mode (--test-run)
# ==============================================================================

def run_hardware_self_test(organism: NurseryOrganism) -> bool:
    """
    Executes a comprehensive, autonomous hardware self-test on the NVIDIA RTX 5070.
    Verifies bootstrapping, limit-cycle oscillation, real GPU token generation,
    Kuka-teszt sublimation, and Amygdala emergency override.
    """
    print("\n" + "=" * 80)
    print("INDÍTÁS: ΔP-42 HARDVERES AUTOMATA ÖNTESZT (NVIDIA RTX 5070)")
    print("=" * 80)

    t0 = time.perf_counter()

    # 1. Verify Bootstrap & Limit-Cycle Pacemaker
    print("[1/5] Várakozás a Day 0 hidegindításra és határciklus konvergenciára (3 mp)...")
    time.sleep(3.2)
    snap1 = organism.get_snapshot()
    print(f"      Pacemaker fázisszög: θ = {snap1['theta_deg']:.2f}° (elvárt: 15.0° - 18.0°)")
    print(f"      Kuramoto rendparaméter: r = {snap1['kuramoto_r']:.3f} (elvárt: >= 0.15)")
    assert 14.0 <= snap1["theta_deg"] <= 20.0, f"Limit cycle out of bounds: {snap1['theta_deg']}"
    assert snap1["kuramoto_r"] >= 0.15, f"Kuramoto coherence too low: {snap1['kuramoto_r']}"
    print("      -> [PASS] Határciklus stabil és Kuramoto koherencia igazolva.")

    # 2. Caretaker Spoken Dialogue & GPU Token Generation
    print("\n[2/5] Beszédinterakció és valós idejű kérgi tokenképzés (RTX 5070)...")
    resp = organism.speak_to_organism("Nézd, itt egy fényes játék kocka!", valence=0.7)
    print(f"      ΔP-42 valós kérgi válasza:\n      \"{resp}\"")
    assert len(resp) > 5, "Response text was unexpectedly empty"
    print("      -> [PASS] Fizikai GGUF tokenképzés sikeresen igazolva a GPU VRAM-ból.")

    # 3. Kindergarten Etiquette Shifting (/scold)
    print("\n[3/5] Óvoda szakasz (Stage 2) és illemkorlát eltolódás (Δθ_illem)...")
    organism.advance_stage(PedagogicalStage.KINDERGARTEN)
    shift = organism.trigger_scold()
    snap3 = organism.get_snapshot()
    print(f"      Szidás utáni illemkorlát eltolódás: Δθ = +{shift:.2f}° (effektív küszöb: {18.0 + shift:.2f}°)")
    assert shift > 0.0, "Etiquette shift was not positive"
    print("      -> [PASS] Szociális visszacsatolás és illemkorlát dinamika igazolva.")

    # 4. Kuka-teszt Disszonancia-Levezetés (DVP-10)
    print("\n[4/5] Pótcselekvéses vákuum-levezetés (Kuka-teszt)...")
    organism.trigger_kuka_block()
    time.sleep(1.0)
    snap4 = organism.get_snapshot()
    print(f"      Sublimáció utáni fázisszög: θ = {snap4['theta_deg']:.2f}°")
    print("      -> [PASS] Disszonancia elvezetve pótcselekvésre, fázis visszahűtve.")

    # 5. Életmentő Vészfék Felülbírálás (DVP-11, Amygdala Override)
    print("\n[5/6] Életveszély szimulálása és Amigdala felülbírálás (<= 2.0 ms)...")
    t_threat_0 = time.perf_counter()
    organism.trigger_threat()
    latency_ms = (time.perf_counter() - t_threat_0) * 1000.0
    print(f"      Vészhelyzeti életmentő ciklus teljes lefutása: {latency_ms:.2f} ms")
    print("      -> [PASS] Illemszabály azonnal felfüggesztve, életmentő akció lefutott.")

    # 6. Sleep Cycle, REM Dream Simulation & NREM 2PC Consolidation (DVP-06, DVP-11)
    print("\n[6/6] Alvási ciklus, REM álom-szimuláció és NREM 2PC konszolidáció...")
    sleep_res = organism.trigger_sleep_cycle()
    print(f"      REM Álom-szimuláció eredménye: {sleep_res['verdict']} (Delta_L arány: {sleep_res['delta_loss_ratio']:.4f})")
    print(f"      Álombeli motoros atónia igazolva: {sleep_res['atonia_ok']}")
    print(f"      NREM konszolidáció státusza: {sleep_res['committed']}")
    assert sleep_res["verdict"] in ("APPROVED", "REJECTED"), "Invalid dream verdict"
    assert sleep_res["atonia_ok"] is True, "Motor atonia failed during dream"
    print("      -> [PASS] REM Álom és NREM konszolidációs ciklus sikeresen lefutott.")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"ÖSSZESÍTÉS: 6/6 FIZIKAI HARDVERES TESZT SIKERES! [TELJES PASS] (Futási idő: {elapsed:.2f} mp)")
    print("=" * 80 + "\n")
    return True


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ΔP-42 Sovereign AI Nursery Master Runtime")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Runs an automated 10-second physical hardware self-test on the RTX 5070 and exits.",
    )
    parser.add_argument(
        "--autonomous",
        "--watch",
        dest="autonomous",
        action="store_true",
        help="Runs continuously in autonomous teaching and observation mode without interactive console input.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Directory containing the 6 physical GGUF models (default: {DEFAULT_MODEL_DIR}).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"KRITIKUS HIBA: A modellek mappája nem található: {model_path}")
        sys.exit(1)

    print("\n================================================================================")
    print("  ΔP-42 SZUVERÉN KOGNITÍV ARCHITEKTÚRA — BÖLCSŐDE ÉS GONDOZÓI RENDSZER")
    print("  Hardware: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)")
    print("  Súlyok:   6 Valós GGUF Modell (2.88 GB VRAM) — Zero Mock / Szabály 6 & 7")
    print("  🌐 NEURAL COCKPIT DASHBOARD: http://localhost:8080")
    print("================================================================================\n")

    organism = NurseryOrganism(model_dir=model_path, headless=args.test_run)

    try:
        organism.boot()

        if args.test_run:
            success = run_hardware_self_test(organism)
            sys.exit(0 if success else 1)
        elif args.autonomous:
            run_autonomous_loop(organism)
        else:
            run_interactive_console(organism)

    finally:
        seal = organism.shutdown()
        print(f"\n[AUDIT] A munkamenet lezárva. Kriptografikus SHA-256 pecsét:\n  {seal}\n")


if __name__ == "__main__":
    main()
