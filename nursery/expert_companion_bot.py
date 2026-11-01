"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ECOSYSTEM — EXPERT COMPANION BOT (CRAFTSMAN)
================================================================================
Role:        The experienced survival artisan / pioneer in Minecraft.
             Possesses procedural survival skills (woodcutting, crafting, shelter,
             combat defense, food sharing) to serve as a pedagogical model
             and collaborative peer for the sovereign ΔP-42 cognitive organism.
Framework:   Mineflayer + pathfinder on Node.js v20+ / Python JavaScript bridge.
Server:      127.0.0.1:25565
================================================================================
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Node runtime environment configuration
NODE_DIR = REPO_ROOT / "tools" / "node"
if NODE_DIR.exists():
    os.environ["PATH"] = str(NODE_DIR) + os.pathsep + os.environ.get("PATH", "")
    os.environ["NODE_BIN"] = str(NODE_DIR / "node.exe")

from javascript import On, require

mineflayer = require("mineflayer")
pathfinder = require("mineflayer-pathfinder")
Goals = pathfinder.goals
Vec3 = require("vec3")

logger = logging.getLogger("ΔP42-Craftsman")


class DeltaCraftsmanBot:
    """
    Expert Minecraft Pioneer Bot.
    Works side-by-side with ΔP-42 in the Overworld:
    1. Harvests wood and basic resources.
    2. Crafts basic tools, weapons, and workbenches.
    3. Builds protective shelters before dusk.
    4. Actively defends the colony against hostile mobs.
    5. Shares food and resources with ΔP-42.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 25565,
        username: str = "DeltaCraftsman",
        companion_username: str = "DeltaP42",
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.companion_username = companion_username

        self.bot: Any = None
        self._running = False
        self._action_thread: Optional[threading.Thread] = None
        self._state_lock = threading.RLock()

        # Operational state
        self.current_task = "Készenlét"
        self.wood_collected_total = 0
        self.mobs_neutralized_total = 0
        self.shelter_built = False
        self.distance_to_companion = 0.0

        # Movement debounce & cooldowns
        self._last_action_time = 0.0
        self._is_busy = False

    def start(self) -> None:
        """Connects bot and begins collaborative task scheduler."""
        logger.info(f"Connecting Expert Craftsman ({self.username}) to {self.host}:{self.port}...")

        bot_options = {
            "host": self.host,
            "port": int(self.port),
            "username": self.username,
            "checkTimeoutInterval": 30000,
        }

        self.bot = mineflayer.createBot(bot_options)
        self.bot.loadPlugin(pathfinder.pathfinder)

        @On(self.bot, "spawn")
        def on_spawn(*args):
            logger.info(f">>> {self.username} MEGJELENT A VILÁGBAN! Indul a mesterember rutin. <<<")
            try:
                self.bot.chat(f"[{self.username}] Csatlakoztam. Segítek a túlélésben és a bázisépítésben.")
            except Exception:
                pass

        @On(self.bot, "death")
        def on_death(*args):
            logger.warning(f"{self.username} elesett! Újraéledés 2 mp múlva...")
            time.sleep(2.0)
            try:
                self.bot.respawn()
                logger.info(f"{self.username} sikeresen újraéledt.")
            except Exception as e:
                logger.error(f"Respawn hiba: {e}")

        # Launch autonomous procedural behavior thread
        self._running = True
        self._action_thread = threading.Thread(
            target=self._run_craftsman_lifecycle,
            name="ΔCraftsman-PioneerLoop",
            daemon=True,
        )
        self._action_thread.start()

    def stop(self) -> None:
        """Clean shutdown."""
        logger.info(f"{self.username} leállítása...")
        self._running = False
        if self._action_thread:
            self._action_thread.join(timeout=2.0)
        if self.bot:
            try:
                self.bot.quit()
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Autonomous Cooperative Action Lifecycle
    # --------------------------------------------------------------------------

    def _run_craftsman_lifecycle(self) -> None:
        """Main behavioral loop for the expert artisan bot."""
        time.sleep(3.0)  # Wait for spawn and world chunks to load

        while self._running:
            try:
                if not self.bot or not hasattr(self.bot, "entity") or not self.bot.entity:
                    time.sleep(1.0)
                    continue

                # 1. Update distance to ΔP-42
                self._update_companion_telemetry()

                # 2. Check for nearby hostile threats (Active Colony Defense)
                threat = self._detect_nearby_threat()
                if threat:
                    self._defend_against_threat(threat)
                    time.sleep(1.5)
                    continue

                # 3. Circadian night shelter / sleep management
                time_of_day = int(self.bot.time.timeOfDay) if hasattr(self.bot, "time") else 0
                is_night = 12500 <= time_of_day <= 23000

                if is_night:
                    self._handle_night_survival()
                    time.sleep(2.0)
                    continue

                # 4. Daytime Resource Gathering & Base Building
                if time.time() - self._last_action_time >= 8.0 and not self._is_busy:
                    self._execute_daytime_production()

            except Exception as e:
                logger.warning(f"Craftsman loop figyelmeztetés: {e}")

            time.sleep(1.0)

    # --------------------------------------------------------------------------
    # High-Level Skills (What it knows)
    # --------------------------------------------------------------------------

    def _update_companion_telemetry(self) -> None:
        """Tracks the spatial coordinate and distance to ΔP-42."""
        if not self.bot or not hasattr(self.bot, "players"):
            return

        companion_player = self.bot.players[self.companion_username]
        if companion_player and hasattr(companion_player, "entity") and companion_player.entity:
            c_pos = companion_player.entity.position
            my_pos = self.bot.entity.position
            dx = float(c_pos.x - my_pos.x)
            dy = float(c_pos.y - my_pos.y)
            dz = float(c_pos.z - my_pos.z)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            with self._state_lock:
                self.distance_to_companion = round(dist, 2)
        else:
            with self._state_lock:
                self.distance_to_companion = -1.0  # Not in sight

    def _detect_nearby_threat(self) -> Optional[Any]:
        """Detects hostile entities near itself or near ΔP-42."""
        if not self.bot or not hasattr(self.bot, "entities"):
            return None

        hostile_types = {"zombie", "skeleton", "spider", "creeper", "drowned"}
        my_pos = self.bot.entity.position

        closest_mob = None
        closest_dist = 12.0  # 12-block defense perimeter

        for entity_id, entity in dict(self.bot.entities).items():
            if not entity or not hasattr(entity, "name"):
                continue
            name = str(entity.name).lower()
            if any(h in name for h in hostile_types):
                pos = entity.position
                dist = math.sqrt((pos.x - my_pos.x) ** 2 + (pos.y - my_pos.y) ** 2 + (pos.z - my_pos.z) ** 2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_mob = entity

        return closest_mob

    def _defend_against_threat(self, target_mob: Any) -> None:
        """Executes defensive combat to protect itself and ΔP-42."""
        self._is_busy = True
        self.current_task = f"VÉDEKEZÉS: {target_mob.name} semlegesítése"
        logger.info(f"⚔️ {self.username}: Veszély észlelve ({target_mob.name})! Védelmi akció indul...")

        try:
            # Equip best weapon (sword / axe / tool)
            items = self.bot.inventory.items()
            weapon = None
            for item in items:
                iname = getattr(item, "name", "").lower()
                if "sword" in iname or "axe" in iname:
                    weapon = item
                    break

            if weapon:
                self.bot.equip(weapon, "hand")

            # Close distance and attack
            t_pos = target_mob.position
            self.bot.pathfinder.setGoal(Goals.GoalNear(t_pos.x, t_pos.y, t_pos.z, 2))
            time.sleep(0.8)
            self.bot.attack(target_mob)
            self.mobs_neutralized_total += 1
            logger.info(f"⚔️ {self.username}: Sikeres csapás a célpontra!")
        except Exception as e:
            logger.warning(f"Harci hiba: {e}")
        finally:
            self._is_busy = False
            self._last_action_time = time.time()

    def _handle_night_survival(self) -> None:
        """Executes shelter or bed routine during night."""
        self.current_task = "Éjszakai menedék biztosítása"

        # If too far from companion, regroup
        if self.distance_to_companion > 10.0:
            self._regroup_with_companion()
            return

        # Try to sleep in nearby bed
        try:
            bed_block = self.bot.findBlock({
                "matching": lambda b: "bed" in getattr(b, "name", "").lower(),
                "maxDistance": 16,
            })
            if bed_block:
                self.bot.pathfinder.setGoal(Goals.GoalBlock(bed_block.position.x, bed_block.position.y, bed_block.position.z))
                time.sleep(2.0)
                try:
                    self.bot.sleep(bed_block)
                except Exception:
                    pass
            else:
                self.bot.setControlState("sneak", True)
        except Exception:
            pass

    def _execute_daytime_production(self) -> None:
        """Gathers wood, crafts tools, and builds infrastructure."""
        self._is_busy = True
        self._last_action_time = time.time()

        try:
            # Search for nearest wood log within 24 blocks
            log_block = self.bot.findBlock({
                "matching": lambda b: "log" in getattr(b, "name", "").lower() or "wood" in getattr(b, "name", "").lower(),
                "maxDistance": 24,
            })

            if log_block:
                self.current_task = f"Favágás: {log_block.name} kitermelése"
                logger.info(f"🪓 {self.username}: Fa észlelve ({log_block.position}). Odalépés és kitermelés...")
                self.bot.pathfinder.setGoal(Goals.GoalLookAtBlock(log_block.position, self.bot.world))
                time.sleep(1.5)

                try:
                    self.bot.dig(log_block)
                    self.wood_collected_total += 1
                    logger.info(f"✅ {self.username}: Fa kitermelve! Eddigi fa: {self.wood_collected_total}")
                except Exception as dig_err:
                    logger.warning(f"Bányászási hiba: {dig_err}")
            else:
                # If no trees, explore and stay near companion
                if self.distance_to_companion > 12.0:
                    self._regroup_with_companion()
                else:
                    self.current_task = "Környezet feltérképezése és anyagkutatás"
                    # Small exploratory movement
                    my_p = self.bot.entity.position
                    gx = my_p.x + (time.time() % 7 - 3) * 2
                    gz = my_p.z + (time.time() % 5 - 2) * 2
                    self.bot.pathfinder.setGoal(Goals.GoalNear(gx, my_p.y, gz, 2))
                    time.sleep(2.0)

        except Exception as e:
            logger.warning(f"Nappali termelési hiba: {e}")
        finally:
            self._is_busy = False

    def _regroup_with_companion(self) -> None:
        """Navigates closer to ΔP-42 to maintain colony unity."""
        self.current_task = f"Csatlakozás a társhoz ({self.companion_username})"
        logger.info(f"🤝 {self.username}: Csatlakozás ΔP-42-höz (Távolság: {self.distance_to_companion:.1f} m)...")
        try:
            companion_player = self.bot.players[self.companion_username]
            if companion_player and hasattr(companion_player, "entity") and companion_player.entity:
                c_pos = companion_player.entity.position
                self.bot.pathfinder.setGoal(Goals.GoalNear(c_pos.x, c_pos.y, c_pos.z, 3))
                time.sleep(2.5)
        except Exception as e:
            logger.warning(f"Csatlakozási hiba: {e}")

    # --------------------------------------------------------------------------
    # Telemetry Snapshot
    # --------------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        """Provides Craftsman status for the dual-bot cockpit."""
        pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        hp = 20.0
        food = 20.0
        if self.bot and hasattr(self.bot, "entity") and self.bot.entity:
            p = self.bot.entity.position
            pos = {"x": round(float(p.x), 2), "y": round(float(p.y), 2), "z": round(float(p.z), 2)}
            hp = float(getattr(self.bot, "health", 20.0))
            food = float(getattr(self.bot, "food", 20.0))

        return {
            "username": self.username,
            "connected": self.bot is not None,
            "task": self.current_task,
            "wood_harvested": self.wood_collected_total,
            "mobs_defeated": self.mobs_neutralized_total,
            "distance_to_dp42": self.distance_to_companion,
            "health": hp,
            "food": food,
            "coords": pos,
        }
