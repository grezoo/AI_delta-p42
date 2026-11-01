"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — DUAL-BOT COLONY SUPERVISOR
================================================================================
Supervises a collaborative multi-agent ecosystem in Minecraft Java Edition:
1. ΔP-42 (Sovereign Cognitive Core with biological Maslow, Theta pacemaker,
   Amygdala reflex, and PyTorch CUDA synaptic consolidation).
2. DeltaCraftsman (Experienced survival artisan equipped with procedural
   woodcutting, basic crafting, shelter construction, and defensive combat).

Telemetry & Multi-Agent Cockpit: http://localhost:8080
Colony Empirical Metrics:        scratch/telemetry_live/dual_colony_metrics.jsonl
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Node & Java paths
NODE_DIR = REPO_ROOT / "tools" / "node"
if NODE_DIR.exists():
    os.environ["PATH"] = str(NODE_DIR) + os.pathsep + os.environ.get("PATH", "")
    os.environ["NODE_BIN"] = str(NODE_DIR / "node.exe")

JAVA_CANDIDATES = [
    Path(os.environ.get("APPDATA", "")) / ".minecraft" / "runtime" / "java-runtime-delta" / "windows" / "java-runtime-delta" / "bin" / "java.exe",
    Path(os.environ.get("APPDATA", "")) / ".minecraft" / "runtime" / "java-runtime-gamma" / "windows" / "java-runtime-gamma" / "bin" / "java.exe",
    Path(os.environ.get("APPDATA", "")) / ".tlauncher" / "starter" / "jre_default" / "jre-21.0.11-windows-x64" / "bin" / "java.exe",
    Path("C:/Program Files/Java/jre1.8.0_511/bin/java.exe"),
]

def find_java_binary() -> str:
    for candidate in JAVA_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "java"

SERVER_DIR = REPO_ROOT / "server"
SERVER_JAR = SERVER_DIR / "server.jar"
LOG_DIR = REPO_ROOT / "scratch" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = REPO_ROOT / "scratch" / "telemetry_live"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
COLONY_METRICS_FILE = METRICS_DIR / "dual_colony_metrics.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "dual_colony_daemon.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ΔP42-DualColonyDaemon")

from nursery.minecraft_bridge import DeltaP42MinecraftOrganism
from nursery.expert_companion_bot import DeltaCraftsmanBot
from nursery.autonomous_minecraft_daemon import MinecraftServerProcess


class DualColonySupervisor:
    """Supervises the server and the two embodied bots in the Overworld."""

    def __init__(
        self,
        port: int = 25565,
        bot1_name: str = "DeltaP42",
        bot2_name: str = "DeltaCraftsman",
    ) -> None:
        self.port = port
        self.bot1_name = bot1_name
        self.bot2_name = bot2_name

        self.server = MinecraftServerProcess(port=port)
        self.organism: Optional[DeltaP42MinecraftOrganism] = None
        self.craftsman: Optional[DeltaCraftsmanBot] = None

        self._running = False
        self._colony_thread: Optional[threading.Thread] = None
        self._start_perf = time.perf_counter()

    def start(self) -> None:
        logger.info("=" * 80)
        logger.info("  ΔP-42 MULTI-AGENT EMBODIED COLONY SUPERVISOR INDUL")
        logger.info(f"  Headless Port: {self.port}")
        logger.info(f"  Bot 1 (Sovereign Core):   {self.bot1_name}")
        logger.info(f"  Bot 2 (Expert Craftsman): {self.bot2_name}")
        logger.info("  Neural Cockpit:           http://localhost:8080")
        logger.info(f"  Colony Telemetry Log:     {COLONY_METRICS_FILE}")
        logger.info("=" * 80)

        # 1. Start / Verify Headless Server
        if not self.server.start():
            logger.error("Minecraft szerver nem elérhető. Kilépés.")
            return

        time.sleep(2.0)

        # 2. Start Sovereign Cognitive Core (ΔP-42)
        logger.info(f"1/2: {self.bot1_name} indítása...")
        self.organism = DeltaP42MinecraftOrganism(
            host="127.0.0.1",
            port=self.port,
            username=self.bot1_name,
        )
        self.organism.start()

        time.sleep(3.0)

        # 3. Start Expert Craftsman Companion Bot
        logger.info(f"2/2: {self.bot2_name} indítása...")
        self.craftsman = DeltaCraftsmanBot(
            host="127.0.0.1",
            port=self.port,
            username=self.bot2_name,
            companion_username=self.bot1_name,
        )
        self.craftsman.start()

        # 4. Start Colony Data Logging Thread
        self._running = True
        self._colony_thread = threading.Thread(
            target=self._run_colony_monitor,
            name="ΔColony-Observer",
            daemon=True,
        )
        self._colony_thread.start()
        logger.info("Két-botos kolónia sikeresen felállt és együttműködik a szerveren.")

    def _run_colony_monitor(self) -> None:
        """Periodically records multi-agent collaborative dynamics."""
        last_log = time.time()

        while self._running:
            time.sleep(5.0)
            if not self.organism or not self.craftsman:
                continue

            # Log metrics every 30 seconds
            if time.time() - last_log >= 30.0:
                last_log = time.time()
                snap1 = self.organism.get_snapshot()
                snap2 = self.craftsman.get_snapshot()

                entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_s": round(time.perf_counter() - self._start_perf, 2),
                    "colony": {
                        "dp42": {
                            "health": snap1.get("minecraft", {}).get("health", 20.0),
                            "food": snap1.get("minecraft", {}).get("food", 20.0),
                            "coords": {"x": snap1.get("minecraft", {}).get("x", 0), "y": snap1.get("minecraft", {}).get("y", 0), "z": snap1.get("minecraft", {}).get("z", 0)},
                            "goal": snap1.get("minecraft", {}).get("current_goal", ""),
                            "theta_deg": snap1.get("theta_deg", 15.0),
                            "maslow_v": snap1.get("maslow_v", []),
                            "latency_ms": snap1.get("cognitive_latency_ms", 0.0),
                            "delta_p": snap1.get("delta_p_tension", 0.0),
                            "control_rod": snap1.get("control_rod_agc", 1.0),
                        },
                        "craftsman": snap2,
                        "distance_between_bots": snap2.get("distance_to_dp42", 0.0),
                    },
                }

                with open(COLONY_METRICS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def stop(self) -> None:
        logger.info("Két-botos kolónia leállítása...")
        self._running = False
        if self._colony_thread:
            self._colony_thread.join(timeout=2.0)
        if self.craftsman:
            self.craftsman.stop()
        if self.organism:
            self.organism.stop()
        self.server.stop()
        logger.info("Kolónia tisztán leállt.")


def main():
    parser = argparse.ArgumentParser(description="ΔP-42 Multi-Agent Colony Supervisor")
    parser.add_argument("--port", type=int, default=25565, help="Server port (default: 25565)")
    parser.add_argument("--bot1", type=str, default="DeltaP42", help="Sovereign Core bot username")
    parser.add_argument("--bot2", type=str, default="DeltaCraftsman", help="Craftsman companion username")
    args = parser.parse_args()

    colony = DualColonySupervisor(port=args.port, bot1_name=args.bot1, bot2_name=args.bot2)
    colony.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl+C fogadva...")
        colony.stop()


if __name__ == "__main__":
    main()
