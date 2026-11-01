"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — AUTONOMOUS MINECRAFT DAEMON
================================================================================
Runs a headless Minecraft 1.20.4 server and the embodied ΔP-42 cognitive bot
in the background 24/7 without requiring any GUI or manual user interaction.

Continuous empirical ethological data collection:
- Homeostatic dynamics (Maslow V0..V4)
- 50 Hz Phase angle oscillator (Theta Engine)
- Hard real-time Amygdala pain evasion (<= 1.2 ms)
- Circadian sleep consolidation (REM/NREM weight consolidation on CUDA tensors)
- Live cockpit dashboard at http://localhost:8080
================================================================================
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
import re
import socket
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
METRICS_FILE = METRICS_DIR / "minecraft_longrun_metrics.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "minecraft_daemon.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ΔP42-MinecraftDaemon")

from nursery.minecraft_bridge import DeltaP42MinecraftOrganism


# ==============================================================================
# Headless Minecraft Server Supervisor
# ==============================================================================

class MinecraftServerProcess:
    """Manages the background headless Minecraft Dedicated Server."""

    def __init__(self, port: int = 25565) -> None:
        self.port = port
        self.java_bin = find_java_binary()
        self.proc: Optional[subprocess.Popen] = None
        self._running = False

    def is_port_open(self) -> bool:
        """Checks if server is already accepting TCP connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                res = s.connect_ex(("127.0.0.1", self.port))
                return res == 0
        except Exception:
            return False

    def start(self) -> bool:
        if self.is_port_open():
            logger.info(f"Minecraft szerver már fut a 127.0.0.1:{self.port} porton.")
            return True

        if not SERVER_JAR.exists():
            logger.error(f"Szerver jar nem található: {SERVER_JAR}")
            return False

        logger.info(f"Headless Minecraft szerver indítása (Java: {self.java_bin})...")
        cmd = [
            self.java_bin,
            "-Xms1G",
            "-Xmx2G",
            "-jar",
            str(SERVER_JAR),
            "nogui",
        ]

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(SERVER_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._running = True

        # Wait for server to finish loading
        logger.info("Várakozás a világ betöltésére és a szerver indulására...")
        t_start = time.time()
        ready = False

        while self._running and time.time() - t_start < 90.0:
            if self.is_port_open():
                logger.info("✅ Minecraft szerver TCP port aktív és válaszol!")
                ready = True
                break
            line = self.proc.stdout.readline() if self.proc and self.proc.stdout else ""
            if line:
                clean_line = line.strip()
                if "Done (" in clean_line and "For help, type \"help\"" in clean_line:
                    logger.info(f"✅ Minecraft szerver sikeresen felállt: {clean_line}")
                    ready = True
                    break
            time.sleep(0.5)

        # Spawn background reader for server stdout
        threading.Thread(target=self._server_log_pump, daemon=True, name="MC-Server-Logs").start()
        return ready

    def _server_log_pump(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        server_log_file = LOG_DIR / "minecraft_server_runtime.log"
        with open(server_log_file, "a", encoding="utf-8") as f:
            for line in self.proc.stdout:
                if not self._running:
                    break
                f.write(line)
                f.flush()

    def stop(self) -> None:
        if not self.proc:
            return
        logger.info("Minecraft szerver leállítása ('stop')...")
        self._running = False
        try:
            if self.proc.stdin:
                self.proc.stdin.write("stop\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=15.0)
            logger.info("Minecraft szerver tisztán leállt.")
        except Exception as e:
            logger.warning(f"Kényszerített leállítás: {e}")
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


# ==============================================================================
# Autonomous 24/7 Supervisor Daemon
# ==============================================================================

class AutonomousMinecraftDaemon:
    """Supervises server, bot, telemetry, and empirical data recording."""

    def __init__(self, port: int = 25565, username: str = "DeltaP42") -> None:
        self.port = port
        self.username = username
        self.server = MinecraftServerProcess(port=port)
        self.organism: Optional[DeltaP42MinecraftOrganism] = None
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_perf = time.perf_counter()
        self._sleep_cycles_total = 0

    def start(self) -> None:
        logger.info("=" * 80)
        logger.info("  ΔP-42 SOVEREIGN COGNITIVE AUTONOMOUS DAEMON INDUL")
        logger.info(f"  Headless Port: {self.port} | Bot: {self.username}")
        logger.info("  Neural Cockpit: http://localhost:8080")
        logger.info(f"  Empirical Telemetry Log: {METRICS_FILE}")
        logger.info("=" * 80)

        # 1. Start Minecraft Headless Server
        server_ok = self.server.start()
        if not server_ok:
            logger.error("A Minecraft szerver nem tudott elindulni. Kilépés.")
            return

        # 2. Instantiate and start ΔP-42 Organism
        time.sleep(2.0)
        self.organism = DeltaP42MinecraftOrganism(
            host="127.0.0.1",
            port=self.port,
            username=self.username,
        )
        self.organism.start()

        # Wire Respawn on Death
        self._wire_bot_resiliency()

        # 3. Start Long-Run Empirical Data Logging Thread
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._run_empirical_collector,
            name="ΔP42-EmpiricalCollector",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Autonóm daemon sikeresen elindult. Folyamatos adatgyűjtés aktív.")

    def _wire_bot_resiliency(self) -> None:
        """Configures auto-respawn and auto-reconnect listeners."""
        if not self.organism or not hasattr(self.organism, "bot"):
            return

        from javascript import On

        @On(self.organism.bot, "death")
        def on_death(*args):
            logger.warning("💀 A bot életét vesztette! Önhelyreállítás (Auto-Respawn) 2 mp múlva...")
            time.sleep(2.0)
            try:
                self.organism.bot.respawn()
                logger.info("✅ Sikeres újraéledés! Folytatódik a kognitív életciklus.")
            except Exception as e:
                logger.error(f"Újraéledési hiba: {e}")

    def _run_empirical_collector(self) -> None:
        """Periodically logs empirical metrics for scientific analysis."""
        last_log_time = time.time()
        last_time_of_day = 0

        while self._running:
            time.sleep(5.0)
            if not self.organism:
                continue

            snap = self.organism.get_snapshot()
            mc = snap.get("minecraft", {})
            cur_tod = mc.get("time_of_day", 0)

            # Detect day-to-night transitions
            if cur_tod < last_time_of_day and cur_tod < 2000:
                self._sleep_cycles_total += 1
                logger.info(f"🌅 Új Minecraft nap virradt! Eddigi teljes alvási-konszolidációs ciklusok száma: {self._sleep_cycles_total}")
            last_time_of_day = cur_tod

            # Log to metrics file every 30 seconds
            if time.time() - last_log_time >= 30.0:
                last_log_time = time.time()
                entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "elapsed_s": round(time.perf_counter() - self._start_perf, 2),
                    "minecraft_time": cur_tod,
                    "is_day": mc.get("is_day", True),
                    "health": mc.get("health", 20.0),
                    "food": mc.get("food", 20.0),
                    "coords": {"x": mc.get("x", 0), "y": mc.get("y", 0), "z": mc.get("z", 0)},
                    "current_goal": mc.get("current_goal", ""),
                    "theta_deg": round(float(snap.get("theta_deg", 15.0)), 2),
                    "maslow_v": snap.get("maslow_v", []),
                    "kuramoto_r": round(float(snap.get("kuramoto_r", 0.85)), 3),
                    "a_motor": round(float(snap.get("a_motor", 0.0)), 3),
                    "is_sleeping": snap.get("is_sleeping", False),
                    "emergency_active": snap.get("emergency_active", False),
                    "dream_verdict": snap.get("last_dream_verdict", "—"),
                    "dream_delta_ratio": snap.get("last_dream_delta_ratio", 0.0),
                    "synaptic_variance_reduction_pct": snap.get("last_rem_variance_reduction", 0.0),
                    "total_cycles_completed": self._sleep_cycles_total,
                }
                with open(METRICS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def stop(self) -> None:
        logger.info("Autonóm daemon leállítása...")
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        if self.organism:
            self.organism.stop()
        self.server.stop()
        logger.info("Daemon leállás befejeződött.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="ΔP-42 Sovereign Autonomous Minecraft Daemon")
    parser.add_argument("--port", type=int, default=25565, help="Server port (default: 25565)")
    parser.add_argument("--username", type=str, default="DeltaP42", help="Bot username (default: DeltaP42)")
    args = parser.parse_args()

    daemon = AutonomousMinecraftDaemon(port=args.port, username=args.username)
    daemon.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Ctrl+C fogadva...")
        daemon.stop()


if __name__ == "__main__":
    main()
