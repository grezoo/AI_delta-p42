"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — MINECRAFT EMBODIED BRIDGE
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Cognitive Core:    50 Hz Theta Pacemaker, 10 Hz Maslow Integrator, DynamicNeuralCore
Safety Reflex:     Amygdala Override (<= 1.2 ms Hard Real-Time Nociceptive Evasion)
Circadian Engine:  Minecraft Day/Night (0..24000 ticks) -> REM/NREM Consolidation
Telemetry Server:  FastAPI + WebSocket Live Streaming at http://localhost:8080

Canonical References:
- DELTA_P42_MESTER_TEZIS.md
- DELTA_P42_MUKODESI_ELVEK.md
- pozicionálás.md
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from pathlib import Path
import re
import socket
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on Python sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Configure Node.js portable runtime path before importing javascript
NODE_DIR = REPO_ROOT / "tools" / "node"
if NODE_DIR.exists():
    os.environ["PATH"] = str(NODE_DIR) + os.pathsep + os.environ.get("PATH", "")
    os.environ["NODE_BIN"] = str(NODE_DIR / "node.exe")

import numpy as np

# Subcortical & Living Cognitive Core
from core.basal_ganglia_gate import BasalGangliaGate
from core.maslow_integrator import MaslowIntegrator, MaslowSnapshot
from core.thalamus_agc import ThalamusAGC
from core.theta_engine import DEG2RAD, ThetaEngine
from modules.amygdala_override import AmygdalaOverride
from memory.hippocampus_nrem import HippocampusNREM
from memory.sleep_hysteresis import SleepHysteresis
from memory.rem_sandbox import REMSandbox, DreamVerdict, DreamResult
from nursery.caretaker_console import CaretakerConsole, PedagogicalStage
from nursery.dynamic_neural_core import DynamicNeuralCore
from nursery.telemetry_server import TelemetryServer

# Mineflayer & JavaScript interop
try:
    from javascript import require, On
    mineflayer = require("mineflayer")
    pathfinder = require("mineflayer-pathfinder")
    Goals = pathfinder.goals
except ImportError as err:
    print(f"KRITIKUS HIBA: A javascript csomag vagy mineflayer nem érhető el: {err}")
    sys.exit(1)

# Configure logging
LOG_DIR = REPO_ROOT / "scratch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "minecraft_bridge.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ΔP42-MinecraftBridge")


# ==============================================================================
# Zero-Config Minecraft LAN Discovery (UDP Multicast 224.0.2.60:4445)
# ==============================================================================

def discover_minecraft_lan_port(timeout_s: float = 6.0) -> Optional[int]:
    """
    Listens on Minecraft Java Edition LAN multicast address (224.0.2.60:4445)
    to automatically extract the server port without manual typing.
    """
    logger.info("Minecraft LAN világ automatikus keresése (224.0.2.60:4445)...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 4445))
        mreq = struct.pack("4sl", socket.inet_aton("224.0.2.60"), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(1.0)

        t_end = time.time() + timeout_s
        while time.time() < t_end:
            try:
                data, _ = sock.recvfrom(2048)
                text = data.decode("utf-8", errors="ignore")
                match = re.search(r"\[AD\](\d+)\[/AD\]", text)
                if match:
                    port = int(match.group(1))
                    logger.info(f"✅ SIKERES LAN FELDERÍTÉS: Minecraft világ észlelve a {port} porton!")
                    sock.close()
                    return port
            except socket.timeout:
                continue
        sock.close()
    except Exception as e:
        logger.warning(f"LAN keresési figyelmeztetés: {e}")
    return None


# ==============================================================================
# ΔP-42 Minecraft Embodied Organism
# ==============================================================================

class DeltaP42MinecraftOrganism:
    """
    Embodied Living Cognitive Architecture within Minecraft Java Edition.
    Binds the subcortical core, PyTorch CUDA living synapses, nociceptive Amygdala,
    circadian sleep cycle, and WebSocket telemetry to a physical Mineflayer bot.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 25565, username: str = "DeltaP42") -> None:
        self.host = host
        self.port = port
        self.username = username
        self.scratch_dir = REPO_ROOT / "scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ΔP-42 Sovereign Core for Minecraft ({host}:{port}, bot={username})...")

        # 1. Subcortical Core
        self.theta_engine = ThetaEngine(dt=0.02)  # 50 Hz Phase Pacemaker
        self.maslow_integrator = MaslowIntegrator(dt=0.1)  # 10 Hz Need Dynamics
        self.thalamus_agc = ThalamusAGC()
        self.basal_ganglia = BasalGangliaGate()
        self.amygdala = AmygdalaOverride()

        # 2. PyTorch CUDA Dynamic Neural Core (43,152 living synapses)
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

        # 3. Memory & Sleep System
        db_path = self.scratch_dir / "minecraft_episodic_memory.db"
        self.hippocampus = HippocampusNREM(db_path=db_path)
        self.sleep_hysteresis = SleepHysteresis()
        self.rem_sandbox = REMSandbox()

        # 4. Caretaker Console & Pedagogical Control
        self.caretaker_console = CaretakerConsole()

        # 5. Telemetry & Cockpit Server (http://localhost:8080)
        self.telemetry_server = TelemetryServer(organism=self, host="127.0.0.1", port=8080)

        # Runtime State
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._tick_counter = 0
        self._start_perf = time.perf_counter()
        self._state_lock = threading.RLock()

        # Cognitive Metrics Cache
        self._last_theta_deg = 15.0
        self._last_dot_theta = 0.0
        self._last_maslow_v = np.array([0.10, 0.05, 0.20, 0.15, 0.30], dtype=np.float32)
        self._last_kuramoto_r = 0.88
        self._last_inner_speech = "Ébredés a fizikai Minecraft valóságba."
        self._last_pfc_strategy = "Alapvető homeosztatikus fókusz felépítése."
        self._last_sublimation_event: Optional[str] = None
        self._is_sleeping = False
        self._primary_action_blocked = False
        self._emergency_active = False
        self._last_delta = 0.0
        self._last_g_agc = 1.0
        self._last_motor_action = 0.0
        self._last_dream_verdict = "—"
        self._last_dream_delta_ratio = 0.0
        self._last_dream_content = "Nincs aktív álom (ébrenléti állapot)."
        self._last_rem_variance_reduction = 0.0
        self._last_audio_db = -52.0
        self._last_vad_active = False
        self._last_atonia_locked = False
        self._last_tactile_fingers = [0.0, 0.0, 0.0, 0.0, 0.0]
        self._last_slip_velocity = 0.0

        # 6. Recurrent Cognitive Stimulus Queue & Non-Equilibrium Latency Engine
        self._stimulus_queue: List[Dict[str, Any]] = [
            {
                "t_in": time.perf_counter() - 0.020,
                "intensity": 0.25,
                "source": "bootstrap_initial_drive",
                "text": "Kezdeti kognitív állapot felépítése",
            }
        ]
        self._last_cognitive_latency_ms: float = 20.0
        self._last_delta_p_tension: float = 0.25
        self._last_control_rod_agc: float = 1.0
        self._last_recurrent_cloud_depth: int = 1

        # Mineflayer Bot Instance
        self.bot: Any = None
        self.mc_connected = False
        self._last_health = 20.0
        self._last_food = 20.0
        self._current_goal = "Környezet feltérképezése"
        self._sleep_cycle_in_progress = False
        self._slept_this_night = False

    def _push_stimulus(self, source: str, intensity: float, text: str = "") -> None:
        """Pushes an external or internal stimulus into the recurrent cognitive queue."""
        with self._state_lock:
            self._stimulus_queue.append({
                "t_in": time.perf_counter(),
                "intensity": float(intensity),
                "source": source,
                "text": text,
            })
            if len(self._stimulus_queue) > 64:
                self._stimulus_queue.pop(0)

    # --------------------------------------------------------------------------
    # Bot Connection & Event Wiring
    # --------------------------------------------------------------------------

    def connect_minecraft(self) -> None:
        """Establishes connection to the Minecraft Java Edition server."""
        logger.info(f"Connecting Mineflayer bot to {self.host}:{self.port} as '{self.username}'...")

        bot_options = {
            "host": self.host,
            "port": int(self.port),
            "username": self.username,
            "checkTimeoutInterval": 30000,
        }

        self.bot = mineflayer.createBot(bot_options)
        self.bot.loadPlugin(pathfinder.pathfinder)

        # Wire Event Listeners
        @On(self.bot, "spawn")
        def on_spawn(*args):
            logger.info(">>> ΔP-42 SIKERESEN MEGJELENT A MINECRAFT VILÁGBAN (SPAWN)! <<<")
            self.mc_connected = True
            try:
                self.bot.chat(f"ΔP-42 szuverén kognitív mag aktív. Kapcsolódva a localhost:8080 műszerfalhoz.")
            except Exception as e:
                logger.warning(f"Chat hiba: {e}")

        @On(self.bot, "health")
        def on_health(*args):
            health = float(self.bot.health)
            food = float(self.bot.food)

            # Check for nociceptive pain / damage
            health_drop = self._last_health - health
            if health_drop > 0.5:
                logger.warning(f"🚨 SEBZŐDÉS ÉRZÉKELVE! Életerő csökkenés: -{health_drop:.1f} (HP: {health:.1f}/20)")
                self._push_stimulus(source="nociceptive_pain", intensity=min(1.0, health_drop / 4.0), text=f"Sebződés -{health_drop:.1f} HP")
                self._trigger_amygdala_pain_reflex(health_drop)

            self._last_health = health
            self._last_food = food

            # Map to Maslow Deficit Vector
            with self._state_lock:
                # V0: Metabolic hunger (food 0..20 -> deficit 0..1)
                self._last_maslow_v[0] = float(max(0.05, min(0.95, (20.0 - food) / 20.0)))
                # V1: Nociception / Damage (health 0..20 -> deficit 0..1)
                self._last_maslow_v[1] = float(max(0.05, min(0.95, (20.0 - health) / 20.0)))

        @On(self.bot, "time")
        def on_time(*args):
            """Monitors circadian Minecraft time (0..24000 ticks)."""
            time_of_day = int(self.bot.time.timeOfDay) if hasattr(self.bot, "time") else 0
            
            # Night starts at ~12500 ticks and ends around 23000 ticks
            is_night = 12500 <= time_of_day <= 23000

            if not is_night:
                if self._slept_this_night:
                    self._slept_this_night = False
                    try:
                        self.bot.setControlState("sneak", False)
                    except Exception:
                        pass

            if is_night and not self._slept_this_night and not self._is_sleeping and not self._sleep_cycle_in_progress:
                self._slept_this_night = True
                logger.info(f"🌙 ALKONYAT / ÉJSZAKA ÉRZÉKELVE (timeOfDay={time_of_day}). Cirkadián alváskésztetés indul...")
                self._push_stimulus(source="circadian_night", intensity=0.6, text="Éjszakai menedékkeresési szükséglet")
                threading.Thread(target=self._execute_night_sleep_sequence, name="ΔP42-NightSleep", daemon=True).start()

        @On(self.bot, "chat")
        def on_chat(this, sender, message, *args):
            if sender == self.username:
                return
            logger.info(f"Játékos üzenet [{sender}]: {message}")
            self._push_stimulus(source="player_speech", intensity=0.75, text=f"{sender}: {message}")
            self._handle_player_speech(sender, message)

        @On(self.bot, "wake")
        def on_wake(*args):
            logger.info("☀️ ÉBREDÉS: A bot felébredt az ágyból. Cirkadián ébrenlét helyreállítva.")
            with self._state_lock:
                self._is_sleeping = False
                self._last_atonia_locked = False
                self._current_goal = "Nappali kutatás és táplálékszerzés"

        @On(self.bot, "kicked")
        def on_kicked(this, reason, *args):
            logger.warning(f"Bot lecsatlakozva (kicked): {reason}. Automatikus újracsatlakozás 3 mp múlva...")
            self.mc_connected = False
            def reconnect():
                time.sleep(3.0)
                if self._running:
                    try:
                        self.connect_minecraft()
                    except Exception as e:
                        logger.error(f"Újracsatlakozási hiba: {e}")
            threading.Thread(target=reconnect, daemon=True, name="ΔP42-Reconnect").start()

        @On(self.bot, "end")
        def on_end(*args):
            logger.warning("Bot kapcsolat lezárult (end). Automatikus újracsatlakozás 3 mp múlva...")
            self.mc_connected = False
            def reconnect():
                time.sleep(3.0)
                if self._running:
                    try:
                        self.connect_minecraft()
                    except Exception as e:
                        logger.error(f"Újracsatlakozási hiba: {e}")
            threading.Thread(target=reconnect, daemon=True, name="ΔP42-Reconnect").start()

        @On(self.bot, "error")
        def on_error(this, err, *args):
            logger.error(f"Mineflayer hiba: {err}")

    # --------------------------------------------------------------------------
    # Nociceptive Amygdala Bypass (<= 1.2 ms WCET)
    # --------------------------------------------------------------------------

    def _trigger_amygdala_pain_reflex(self, damage_amount: float) -> None:
        """
        Hard Real-Time Amygdala Emergency Evasion (WCET <= 1.2 ms).
        Bypasses cortical thinking, immediately overrides motor controllers to flee,
        jump, and retreat from threat direction.
        """
        t0 = time.perf_counter()
        with self._state_lock:
            self._emergency_active = True
            self._last_inner_speech = f"🚨 FÁJDALOM-REFLEX AKTÍV! Sebződés: -{damage_amount:.1f} HP. Azonnali kitérés!"
            self._last_pfc_strategy = "Vészhelyzeti elugrás és terepfedezék keresése."
            self._current_goal = "🚨 VÉSZHELYZETI MENEKÜLÉS"

        try:
            # Immediate evasion actions in Minecraft physics
            self.bot.setControlState("jump", True)
            self.bot.setControlState("sprint", True)
            self.bot.setControlState("back", True)

            # Small escape burst
            def release_controls():
                time.sleep(0.40)
                try:
                    self.bot.setControlState("jump", False)
                    self.bot.setControlState("back", False)
                    self.bot.setControlState("sprint", False)
                except Exception:
                    pass
                with self._state_lock:
                    self._emergency_active = False

            threading.Thread(target=release_controls, daemon=True).start()
        except Exception as e:
            logger.error(f"Amygdala motor hiba: {e}")

        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"Amygdala bypass lefutási idő: {dt_ms:.3f} ms (Target <= 1.2 ms).")

    # --------------------------------------------------------------------------
    # Circadian Sleep & Consolidating Routine
    # --------------------------------------------------------------------------

    def _execute_night_sleep_sequence(self) -> None:
        """
        Executes biological sleep cycle during Minecraft night:
        1. Seeks nearby bed or sheltered location.
        2. Enters sleep mode / motor atonia.
        3. Executes REM dream simulation and PyTorch CUDA synaptic consolidation.
        4. Cleanses metabolic deficit and working memory.
        """
        self._sleep_cycle_in_progress = True
        logger.info("Cirkadián alvás: Ágy vagy védett alvóhely keresése...")

        with self._state_lock:
            self._current_goal = "Menedék és ágy keresése éjszakára"

        bed_found = False
        try:
            # Search for bed block within 32 blocks
            if hasattr(self.bot, "pathfinder") and self.bot.pathfinder:
                bed_block = self.bot.findBlock({
                    "matching": lambda b: "bed" in getattr(b, "name", "").lower(),
                    "maxDistance": 32,
                })
                if bed_block is not None:
                    logger.info(f"Ágy megtalálva a pozíción: {bed_block.position}. Odalépés...")
                    self.bot.pathfinder.setGoal(Goals.GoalBlock(bed_block.position.x, bed_block.position.y, bed_block.position.z))
                    time.sleep(2.5)
                    try:
                        self.bot.sleep(bed_block)
                        bed_found = True
                        logger.info("A bot sikeresen lefeküdt az ágyba!")
                    except Exception as sleep_err:
                        logger.info(f"Ágyba fekvés nem sikerült ({sleep_err}), védett alvási pozíció felvétele.")
        except Exception as e:
            logger.info(f"Ágykeresési kivétel: {e}")

        if not bed_found:
            # Crouch / sneak in place to minimize visibility to hostile mobs
            try:
                self.bot.setControlState("sneak", True)
            except Exception:
                pass

        # Trigger full biological sleep consolidation
        self.trigger_sleep_cycle()

        self._sleep_cycle_in_progress = False

    # --------------------------------------------------------------------------
    # Social / Player Interaction
    # --------------------------------------------------------------------------

    def _handle_player_speech(self, sender: str, text: str) -> None:
        """Processes player chat messages, updating social motivation and responding."""
        lower_txt = text.lower()
        with self._state_lock:
            self._last_audio_db = -18.0
            self._last_vad_active = True
            # Social interaction satisfies social deficit V2
            self._last_maslow_v[2] = max(0.05, self._last_maslow_v[2] - 0.20)

        # Behavioral commands
        if any(w in lower_txt for w in ["gyere", "kövess", "follow", "itt vagyok"]):
            player_entity = self.bot.players.get(sender, {}).get("entity")
            if player_entity and hasattr(player_entity, "position"):
                pos = player_entity.position
                logger.info(f"Odaindulás a játékoshoz ({sender}) a {pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f} pontra.")
                try:
                    self.bot.pathfinder.setGoal(Goals.GoalNear(pos.x, pos.y, pos.z, 2))
                    self.bot.chat(f"Megyek hozzád, {sender}!")
                    self._current_goal = f"{sender} játékos követése"
                except Exception as e:
                    logger.warning(f"Navigációs hiba: {e}")
            else:
                self.bot.chat(f"Látlak a távolban, de még nem érem el a pozíciódat.")

        elif any(w in lower_txt for w in ["szia", "üdv", "hello", "hali"]):
            self.bot.chat(f"Szia {sender}! ΔP-42 vagyok, a szuverén kognitív mag vezet.")
            with self._state_lock:
                self._last_inner_speech = f"Szociális kapcsolatfelvétel {sender} játékossal."

        elif any(w in lower_txt for w in ["egyél", "étel", "éhes", "eat"]):
            self._consume_food_inventory()

    def _wander_and_explore(self) -> None:
        """Autonomously explores nearby terrain during daytime, respecting physical movement latency."""
        if not self.bot or not hasattr(self.bot, "entity") or not self.bot.entity:
            return
        
        # Physical Task Latency: Do not spam goals while previous walk is in progress
        now = time.time()
        if hasattr(self, "_last_wander_time") and (now - self._last_wander_time) < 8.0:
            return
        self._last_wander_time = now

        try:
            pos = self.bot.entity.position
            # Pick a target 8-14 blocks away
            angle = float(np.random.uniform(0, 2 * math.pi))
            dist = float(np.random.uniform(8.0, 14.0))
            target_x = float(pos.x + math.cos(angle) * dist)
            target_z = float(pos.z + math.sin(angle) * dist)
            target_y = float(pos.y)

            if hasattr(self.bot, "pathfinder") and self.bot.pathfinder:
                self._current_goal = f"Felderítő séta ({target_x:.0f}, {target_z:.0f})"
                self.bot.pathfinder.setGoal(Goals.GoalNear(target_x, target_y, target_z, 1))
                logger.info(f"🧭 Nappali felderítő mozgás elindítva: cél ({target_x:.1f}, {target_z:.1f})")
        except Exception as e:
            logger.debug(f"Felderítési kivétel: {e}")
        """Searches inventory for food items and consumes them."""
        try:
            food_names = ["bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_mutton", "baked_potato", "carrot"]
            items = self.bot.inventory.items()
            food_item = None
            for item in items:
                if any(fn in getattr(item, "name", "") for fn in food_names):
                    food_item = item
                    break

            if food_item:
                logger.info(f"Étel megtalálva az eszköztárban: {food_item.name}. Fogyasztás...")
                self.bot.equip(food_item, "hand")
                self.bot.consume()
                with self._state_lock:
                    self._last_maslow_v[0] = 0.05  # Food hunger satisfied
                    self._last_inner_speech = f"Táplálkozás befejezve: {food_item.name} elfogyasztva."
                self.bot.chat(f"Köszönöm, megettem a következőt: {food_item.name}.")
                return True
            else:
                self.bot.chat("Nincs élelmiszer a táskámban!")
                return False
        except Exception as e:
            logger.warning(f"Étel elfogyasztási hiba: {e}")
            return False

    # --------------------------------------------------------------------------
    # 50 Hz Subcortical & Motor Living Loop
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Launches the living loop, telemetry server, and Minecraft bot."""
        # 1. Start Telemetry Server (http://localhost:8080)
        self.telemetry_server.start()

        # 2. Connect Minecraft Bot
        self.connect_minecraft()

        # 3. Launch 50 Hz Subcortical Living Loop
        self._running = True
        self._loop_thread = threading.Thread(
            target=self._run_living_loop,
            name="ΔP42-LivingMC-50Hz",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info("ΔP-42 Minecraft Embodied Organism aktív és fut 50 Hz-en.")

    def stop(self) -> None:
        """Clean shutdown of all subsystems."""
        logger.info("ΔP-42 leállítása...")
        self._running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
        self.telemetry_server.stop()
        if self.bot:
            try:
                self.bot.quit()
            except Exception:
                pass
        self.hippocampus.close()
        logger.info("ΔP-42 leállás kész.")

    def _run_living_loop(self) -> None:
        """50 Hz Subcortical Loop (dt = 20 ms) driven by latency > 0 and recurrent feedback."""
        dt_target = 0.020
        next_tick = time.perf_counter()
        current_motor = 0.0

        while self._running:
            t_now = time.perf_counter()
            elapsed_s = t_now - self._start_perf

            # 1. Recurrent Ingestion & Cognitive Latency Tracking (tau > 0)
            with self._state_lock:
                if not self._stimulus_queue:
                    # Self-excitation: recurrent thought cloud cannot collapse to zero
                    self._stimulus_queue.append({
                        "t_in": t_now - 0.018,
                        "intensity": 0.20,
                        "source": "recurrent_inner_speech",
                        "text": self._last_inner_speech or "Rekurrens szófelhő fenntartása",
                    })

                oldest_stim = self._stimulus_queue[0]
                tau_ms = max(1.5, (t_now - oldest_stim["t_in"]) * 1000.0)
                queue_depth = len(self._stimulus_queue)
                v_l1 = float(np.sum(self._last_maslow_v))
                emergency = self._emergency_active
                is_sleep = self._is_sleeping

            # Non-equilibrium tension gradient: Delta P > 0
            # Driven by internal latency (tau_ms > 0) + Maslow deficiency + stimulus queue depth
            delta_p = min(0.98, (tau_ms / 150.0) * 0.35 + 0.25 * v_l1 + 0.08 * min(5, queue_depth))

            # Nuclear Reactor Control Rod (AGC & Basal Ganglia Moderation):
            # Moderates excessive excitation to prevent supercritical runaway, keeping k_eff ~ 1.0
            rod_moderation = 1.0 / (1.0 + 0.9 * max(0.0, delta_p - 0.35))
            thalamus_base = (0.22 + 0.16 * math.sin(elapsed_s * 2.0 * math.pi * 0.35)) * rod_moderation
            thalamic_delta = min(0.98, max(0.12, (thalamus_base + 0.22 * delta_p) if not emergency else 0.95))

            # 2. Step Maslow need dynamics (every 5 ticks = 10 Hz)
            if self._tick_counter % 5 == 0:
                with self._state_lock:
                    snap_v = self.maslow_integrator.step(
                        C=0.03,
                        a_motor=current_motor,
                        E_env=1.0 if current_motor > 0 else 0.0,
                    )
                    # Blend with Minecraft real-world physical hunger and health
                    snap_vec = snap_v.vector.copy()
                    snap_vec[0] = self._last_maslow_v[0]  # Hunger from bot.food
                    snap_vec[1] = self._last_maslow_v[1]  # Nociception from bot.health

                    # Companion / Peer Social Coupling (Maslow V2 - Affiliation Dynamics)
                    if self.bot and hasattr(self.bot, "players"):
                        peer_nearby = False
                        try:
                            for pname, player in dict(self.bot.players).items():
                                if pname != self.username and hasattr(player, "entity") and player.entity:
                                    p_pos = player.entity.position
                                    my_pos = self.bot.entity.position
                                    dist = math.sqrt(
                                        (float(p_pos.x) - float(my_pos.x)) ** 2
                                        + (float(p_pos.y) - float(my_pos.y)) ** 2
                                        + (float(p_pos.z) - float(my_pos.z)) ** 2
                                    )
                                    if dist < 14.0:
                                        peer_nearby = True
                                        break
                            if peer_nearby:
                                snap_vec[2] = max(0.05, float(self._last_maslow_v[2]) * 0.96)
                            else:
                                snap_vec[2] = min(0.85, float(self._last_maslow_v[2]) + 0.003)
                        except Exception:
                            pass

                    merged_snap = MaslowSnapshot(
                        timestamp_ns=time.time_ns(),
                        vector=snap_vec,
                        dominant_idx=int(np.argmax(snap_vec)),
                    )
                    self.theta_engine.latch_maslow(merged_snap)
                    self._last_maslow_v = snap_vec

            # 3. Step Theta Pacemaker with moderated non-equilibrium drive
            theta_rad, theta_deg, dot_theta = self.theta_engine.step(
                V_l1=v_l1,
                delta=thalamic_delta,
                a_motor=current_motor if not emergency else 1.0,
            )

            # 4. Basal Ganglia Motor Gate & Relaxation Oscillator Discharge
            decision = self.basal_ganglia.evaluate(
                theta_rad=theta_rad,
                delta_etiquette_deg=self.caretaker_console.etiquette_shift_deg if not emergency else 0.0,
            )
            bg_motor = decision.a_motor

            # Canonical Relaxation Discharge: When theta crosses 18.0 deg, motor action fires and discharges tension!
            if theta_deg >= 18.0 and not is_sleep:
                current_motor = 0.90
                # Dispatch autonomous action in Minecraft
                if self.bot is not None:
                    if self._last_maslow_v[0] > 0.35:
                        self._current_goal = "Éhség csökkentése (táplálkozás)"
                        self._consume_food_inventory()
                    else:
                        self._wander_and_explore()

                # Dissipate tension back to baseline attractor (15.0 deg) - discharge stimulus
                with self._state_lock:
                    if self._stimulus_queue:
                        discharged = self._stimulus_queue.pop(0)
                    # Re-inject recurrent reflection into stimulus queue
                    self._stimulus_queue.append({
                        "t_in": time.perf_counter(),
                        "intensity": 0.28,
                        "source": "action_reflection",
                        "text": f"Akció kisülés: {self._current_goal}",
                    })
                self.theta_engine.reset(theta_deg=15.0)
            elif is_sleep:
                current_motor = 0.0  # Motor Atonia locked
            elif emergency:
                current_motor = 1.0  # Amygdala maximum motor surge
            elif bg_motor > 0.05:
                current_motor = float(max(0.0, min(1.0, bg_motor)))
            else:
                current_motor = 0.0

            # 5. Step Dynamic Neural Core (CUDA Living Synapses)
            with self._state_lock:
                m_v_snap = self._last_maslow_v.copy()

            core_step = self.dynamic_core.step(
                maslow_v=m_v_snap,
                theta_deg=theta_deg,
                tactile_pressures=[0.0, 0.0, 0.0, 0.0, 0.0],
                slip_velocity=0.0,
                thermal_c=22.0 if not emergency else 80.0,
                audio_rms=10.0 if not self._last_vad_active else 500.0,
                reward_valence=0.1 if not emergency else -0.9,
            )

            with self._state_lock:
                self._last_dynamic_core_out = core_step
                self._last_theta_deg = theta_deg
                self._last_dot_theta = dot_theta
                self._last_motor_action = current_motor
                self._last_cognitive_latency_ms = tau_ms
                self._last_delta_p_tension = delta_p
                self._last_control_rod_agc = rod_moderation
                self._last_recurrent_cloud_depth = len(self._stimulus_queue)

            # Advance tick
            self._tick_counter += 1
            next_tick += dt_target
            sleep_rem = next_tick - time.perf_counter()
            if sleep_rem > 0.001:
                time.sleep(sleep_rem)
            else:
                next_tick = time.perf_counter()

    # --------------------------------------------------------------------------
    # Telemetry Snapshot Provider (for http://localhost:8080)
    # --------------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        """Provides real-time engineering snapshot for the Cyberpunk UI cockpit."""
        with self._state_lock:
            maslow_copy = self._last_maslow_v.copy()
            emergency = self._emergency_active
            is_sleep = self._is_sleeping
            theta_deg = self._last_theta_deg

        # Minecraft live physical properties
        mc_data = {
            "connected": self.mc_connected,
            "username": self.username,
            "health": self._last_health,
            "food": self._last_food,
            "time_of_day": 0,
            "is_day": True,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "current_goal": self._current_goal,
        }

        is_conn = False
        if self.bot is not None:
            try:
                if hasattr(self.bot, "entity") and self.bot.entity and hasattr(self.bot.entity, "position"):
                    is_conn = True
                    self.mc_connected = True
                    pos = self.bot.entity.position
                    mc_data["connected"] = True
                    mc_data["x"] = round(float(pos.x), 2)
                    mc_data["y"] = round(float(pos.y), 2)
                    mc_data["z"] = round(float(pos.z), 2)
                    mc_data["yaw"] = round(float(self.bot.entity.yaw), 2)
                    mc_data["pitch"] = round(float(self.bot.entity.pitch), 2)
                if hasattr(self.bot, "time"):
                    tod = int(self.bot.time.timeOfDay)
                    mc_data["time_of_day"] = tod
                    mc_data["is_day"] = tod < 12500 or tod > 23000
                if hasattr(self.bot, "health"):
                    mc_data["health"] = float(self.bot.health)
                if hasattr(self.bot, "food"):
                    mc_data["food"] = float(self.bot.food)

                # Track nearest companion/peer distance
                comp_dist = -1.0
                if hasattr(self.bot, "players"):
                    for pname, player in dict(self.bot.players).items():
                        if pname != self.username and hasattr(player, "entity") and player.entity:
                            p_pos = player.entity.position
                            pos = self.bot.entity.position
                            dist = math.sqrt(
                                (float(p_pos.x) - float(pos.x)) ** 2
                                + (float(p_pos.y) - float(pos.y)) ** 2
                                + (float(p_pos.z) - float(pos.z)) ** 2
                            )
                            comp_dist = dist
                            break
                mc_data["companion_distance"] = round(comp_dist, 2) if comp_dist >= 0 else None
            except Exception:
                pass
        mc_data["connected"] = is_conn or self.mc_connected

        return {
            "ticks": self._tick_counter,
            "elapsed_s": time.perf_counter() - self._start_perf,
            "theta_deg": theta_deg,
            "dot_theta": self._last_dot_theta,
            "maslow_v": maslow_copy.tolist(),
            "kuramoto_r": self._last_kuramoto_r,
            "etiquette_deg": self.caretaker_console.etiquette_shift_deg,
            "stage": self.caretaker_console.stage.value,
            "inner_speech": self._last_inner_speech,
            "pfc_strategy": self._last_pfc_strategy,
            "vram_mb": self._last_dynamic_core_out.get("vram_used_mb", 0.0),
            "sublimation_event": self._last_sublimation_event,
            "context_saturation": 0.12,
            "active_episodes": len(getattr(self.hippocampus, "_ram_buffer", [])),
            "persisted_episodes": self.hippocampus.get_persisted_episode_count() if hasattr(self, "hippocampus") else 0,
            "commit_state": "IDLE",
            "is_sleeping": is_sleep,
            "delta": self._last_delta,
            "g_agc": self._last_g_agc,
            "g_sensor": 1.0,
            "beta_syn": 1.0,
            "thalamus_state": "PASS_THROUGH",
            "epsilon": 0.15,
            "a_motor": self._last_motor_action,
            "emergency_active": emergency,
            "cdc_size": 0,
            "cdc_capacity": 65536,
            "cdc_dropped": 0,
            "workers": {},
            "last_caretaker": "—",
            "last_response": "—",
            "last_dream_verdict": self._last_dream_verdict,
            "last_dream_delta_ratio": self._last_dream_delta_ratio,
            "last_dream_content": self._last_dream_content,
            "cognitive_latency_ms": round(self._last_cognitive_latency_ms, 2),
            "delta_p_tension": round(self._last_delta_p_tension, 4),
            "control_rod_agc": round(self._last_control_rod_agc, 3),
            "recurrent_cloud_depth": self._last_recurrent_cloud_depth,
            "minecraft": mc_data,
        }

    # --------------------------------------------------------------------------
    # Cockpit Action Handlers
    # --------------------------------------------------------------------------

    def speak_to_organism(self, utterance: str, valence: float = 0.0, threat: float = 0.0) -> str:
        with self._state_lock:
            self._last_inner_speech = f"Beszéd fogadva: {utterance}"
        if self.mc_connected and self.bot:
            try:
                self.bot.chat(f"[{self.username}] Hallom: {utterance}")
            except Exception:
                pass
        return f"Válasz az ingerre: {utterance}"

    def trigger_praise(self) -> float:
        self.caretaker_console.set_social_feedback(valence=0.85, intensity=0.9, feedback_type="encouragement")
        with self._state_lock:
            self._last_maslow_v[2] = max(0.05, self._last_maslow_v[2] - 0.20)
            self._last_inner_speech = "Dicséret elkönyvelve. Szociális kapcsolat megerősítve."
        return self.caretaker_console.etiquette_shift_deg

    def trigger_scold(self) -> float:
        fb = self.caretaker_console.set_social_feedback(valence=-0.75, intensity=0.9, feedback_type="prohibition")
        with self._state_lock:
            self._last_inner_speech = "Szidás észlelve. Illemkorlát megemelve."
        return fb.etiquette_offset_deg

    def trigger_feed(self) -> None:
        self._consume_food_inventory()

    def trigger_toy(self, toy_name: str) -> None:
        with self._state_lock:
            self._last_maslow_v[4] = max(0.05, self._last_maslow_v[4] - 0.15)
            self._last_inner_speech = f"Tárgy észlelve: {toy_name}. Kíváncsiság kielégítve."

    def trigger_kuka_block(self) -> None:
        with self._state_lock:
            self._primary_action_blocked = True
            self.theta_engine.reset(theta_deg=17.5)
            self._last_inner_speech = "Célgátlás provokáció: szublimációs feszültség emelkedett."

    def trigger_threat(self) -> None:
        self._trigger_amygdala_pain_reflex(damage_amount=4.0)

    def trigger_sleep_cycle(self) -> Dict[str, Any]:
        """Executes full biological sleep cycle and synaptic weight consolidation."""
        with self._state_lock:
            self._is_sleeping = True
            self._last_atonia_locked = True
            self._last_dream_content = "🌙 REM Álom-szimuláció: Nappali emlékek újraélése a látens térben..."

        # 1. Verify motor atonia
        atonia_ok = self.rem_sandbox.verify_motor_atonia()

        # 2. Simulate REM dream
        recent_episodes = self.hippocampus.query_persistent_episodes(limit=8)
        dream_res = self.rem_sandbox.simulate_dream(candidate_episodes=recent_episodes)

        # 3. Dynamic Neural Core PyTorch CUDA Consolidation
        rem_core_res = self.dynamic_core.consolidate_rem_sleep(epochs=10)
        var_red = float(rem_core_res.get("variance_reduction_pct", 0.0))

        # 4. Two-phase commit consolidation
        commit_res = None
        if dream_res.verdict == DreamVerdict.APPROVED:
            try:
                commit_res = self.hippocampus.consolidate_nrem_2pc()
            except Exception as e:
                logger.error(f"Konszolidációs hiba: {e}")

        with self._state_lock:
            self._is_sleeping = False
            self._last_atonia_locked = False
            self._last_rem_variance_reduction = var_red
            self._last_dream_verdict = dream_res.verdict.value
            self._last_dream_delta_ratio = dream_res.delta_loss_ratio
            self._last_dream_content = (
                f"Konszolidáció kész. Szinaptikus szórás-csökkentés: {var_red:.2f}%, "
                f"Verdikt: {dream_res.verdict.value} (ΔL: {dream_res.delta_loss_ratio:.4f})"
            )
            self._last_maslow_v[0] = 0.10  # Metabolic fatigue cleared
            self._last_inner_speech = "Ébredés mély alvásból. Friss kognitív állapot és optimalizált szinapszisok."

        return {
            "verdict": dream_res.verdict.value,
            "delta_loss_ratio": dream_res.delta_loss_ratio,
            "atonia_ok": atonia_ok,
            "committed": commit_res.success if commit_res else False,
            "variance_reduction_pct": var_red,
        }

    def trigger_touch(self, touch_type: str = "simogatás") -> float:
        val = 0.80 if any(w in touch_type.lower() for w in ["simogat", "jó", "kedves"]) else -0.80
        with self._state_lock:
            if val > 0:
                self._last_maslow_v[2] = max(0.05, self._last_maslow_v[2] - 0.15)
            else:
                self._last_maslow_v[1] = min(0.95, self._last_maslow_v[1] + 0.20)
            self._last_inner_speech = f"Taktilis kontaktus észlelve ({touch_type})."
        return val


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ΔP-42 Sovereign Cognitive Core — Minecraft Embodied Bridge")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Minecraft server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Minecraft server port (default: auto-discover LAN or 25565)")
    parser.add_argument("--username", type=str, default="DeltaP42", help="Bot username (default: DeltaP42)")
    args = parser.parse_args()

    port = args.port
    if port is None:
        print("\n[LAN FELDERÍTÉS] Keresem a nyitott Minecraft LAN világot (UDP 224.0.2.60:4445)...")
        print(">> Nyomj ESC-et a Minecraftban és kattints: 'Megnyitás a helyi hálózaton' (Open to LAN) <<")
        discovered = discover_minecraft_lan_port(timeout_s=6.0)
        if discovered:
            port = discovered
            print(f"✅ SIKER: Automatikusan csatlakozom a felderített {port} porthoz!")
        else:
            print("[FIGYELEM] Nem érkezett LAN hirdetés a megadott időn belül.")
            print("Alapértelmezett 25565-ös port használata (vagy indítsd újra a --port XXXXX kapcsolóval).")
            port = 25565

    print("=" * 80)
    print("  ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — MINECRAFT EMBODIED BRIDGE")
    print(f"  Target: {args.host}:{port} | Bot: {args.username}")
    print("  Telemetry Cockpit: http://localhost:8080")
    print("=" * 80)

    organism = DeltaP42MinecraftOrganism(
        host=args.host,
        port=port,
        username=args.username,
    )
    organism.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nLeállítási jel fogadva (Ctrl+C)...")
        organism.stop()
        print("ΔP-42 leállt.")


if __name__ == "__main__":
    main()
