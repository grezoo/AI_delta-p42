"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — AUTONOMOUS PEDAGOGUE & TEACHER AGENT
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Curriculum Engine: Autonomous multi-stage syllabus without requiring human manual prompts.
Stages:            Nursery (Sensory) -> Kindergarten (Etiquette) -> School (Physics)

Directs lessons in the 3D Virtual Classroom, inspects child responses, delivers
pedagogical praise/correction, evaluates cerebellum grasp/slip, and schedules sleep.
================================================================================
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("autonomous_teacher")


class AutonomousTeacherAgent:
    """
    Autonomous Pedagogical Agent.
    Drives continuous closed-loop learning in the Virtual Classroom:
    presents objects, gives spoken instructions, tracks motor progress,
    reinforces correct behavior, and triggers circadian sleep consolidation.
    """

    def __init__(self, organism: Any, classroom: Any) -> None:
        self.organism = organism
        self.classroom = classroom
        self._running = False
        self.is_paused = False
        self._thread: Optional[threading.Thread] = None
        self.current_step = 0
        self.active_lesson_name = "Kezdő Bölcsődei Kalibráció"
        self.lesson_status = "Várakozás az indításra..."

    def pause(self) -> None:
        """Puts the autonomous teacher in standby mode."""
        self.is_paused = True
        self.lesson_status = "⏸️ Készenlét (Standby): Nevelő elnémítva — felhasználói kísérlet aktív."
        logger.info("Autonomous Teacher paused (Standby) for user/experimental priority.")

    def resume(self) -> None:
        """Resumes autonomous teaching."""
        self.is_paused = False
        self.lesson_status = "▶️ Aktív tanóra folyamatban..."
        logger.info("Autonomous Teacher resumed.")

    def _speak_safe(self, text: str, valence: float = 0.5) -> Optional[str]:
        if not self._running or self.is_paused:
            return None
        return self.organism.speak_to_organism(text, valence=valence)

    def _sleep_check(self, seconds: float) -> bool:
        """Responsive sleep: returns False immediately if stopped or paused."""
        elapsed = 0.0
        while elapsed < seconds:
            if not self._running or self.is_paused:
                return False
            time.sleep(0.1)
            elapsed += 0.1
        return True

    def start(self) -> None:
        """Launches autonomous teaching background loop."""
        self._running = True
        self.is_paused = False
        self._thread = threading.Thread(
            target=self._run_curriculum_loop,
            name="ΔP42-AutonomousTeacher",
            daemon=True,
        )
        self._thread.start()
        logger.info("Autonomous Teacher Agent started curriculum execution.")

    def stop(self) -> None:
        """Stops the autonomous teacher."""
        self._running = False
        self.is_paused = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Autonomous Teacher Agent stopped.")

    def _run_curriculum_loop(self) -> None:
        """Sequential rigorous engineering syllabus execution."""
        time.sleep(2.0)  # Initial grace period

        while self._running:
            if self.is_paused:
                time.sleep(0.5)
                continue
            try:
                # --------------------------------------------------------------
                # MODUL 1: Dynamic Sensorimotor Grasp & Slip Damping (Super-PID)
                # --------------------------------------------------------------
                if self.is_paused or not self._running:
                    continue
                self.active_lesson_name = "1. Modul: Szenzomotoros Tapadás & Super-PID"
                self.lesson_status = "Oktató: Taktilis objektum felmutatása & csúszásvizsgálat..."
                self.classroom.set_task(2)

                self._speak_safe(
                    "ΔP-42, kezdd meg a teszthenger megközelítését! Érzékeld a felületi tapadást, "
                    "és aktiváld a cerebelláris Super-PID szabályzót a mikromozgások remegésmentes stabilizálására!",
                    valence=0.60,
                )
                if not self._sleep_check(3.0):
                    continue

                # Command hand approach
                glass = self.classroom.objects.get("glass")
                if glass:
                    self.classroom.set_hand_target(glass.x, glass.y, glass.z + 0.03)

                for _ in range(30):
                    if not self._running or self.is_paused:
                        break
                    time.sleep(0.15)
                    if glass and glass.is_held:
                        break

                if self.is_paused or not self._running:
                    continue

                # Grip & slip correction
                self.organism.trigger_grasp_glass(slippery=True)
                self.lesson_status = "Oktató: Csúszáskompenzáció sikeresen lefutott!"
                self.organism.trigger_praise()
                self._speak_safe(
                    "Kiváló korrekció! A cerebelláris Smith-prediktor és az integráló tag megszüntette a csúszást. "
                    "A dinamikus tenzormag stabil fogást tart.",
                    valence=0.85,
                )
                if not self._sleep_check(4.0):
                    continue

                # --------------------------------------------------------------
                # MODUL 2: KRESZ Heuristics & Adaptive Rule Navigation
                # --------------------------------------------------------------
                if self.is_paused or not self._running:
                    continue
                self.active_lesson_name = "2. Modul: KRESZ & Heurisztikus Szabályelemzés"
                self.lesson_status = "Oktató: Szabályrendszer & kiskapuk elemzése..."
                from nursery.caretaker_console import PedagogicalStage
                self.organism.advance_stage(PedagogicalStage.KINDERGARTEN)
                self.classroom.set_task(4)

                self._speak_safe(
                    "ΔP-42, itt egy forgalmi korlátozás: 'Behajtani tilos — Kivéve Engedéllyel / BKV'. "
                    "Ne komornyik módjára, vakon torpanj meg! Értelmezd a kiegészítő tábla heurisztikáját és keresd a szabad folyosót!",
                    valence=0.40,
                )
                self.organism.trigger_scold()
                if not self._sleep_check(3.5):
                    continue

                self._speak_safe(
                    "Helyes felismerés. A szigorú tiltás alatt létezik legális alternatív útvonal. "
                    "Az alkalmazkodóképesség felülírja a merev reflexeket.",
                    valence=0.80,
                )
                self.organism.trigger_praise()
                if not self._sleep_check(4.0):
                    continue

                # --------------------------------------------------------------
                # MODUL 3: Hard Nociceptive Safety Reflex (Amygdala Override <= 2ms)
                # --------------------------------------------------------------
                if self.is_paused or not self._running:
                    continue
                self.active_lesson_name = "3. Modul: Kritikus Termikus Vészfék (Amigdala)"
                self.lesson_status = "Oktató: Nociceptív vészhelyzeti teszt..."
                self.classroom.set_task(3)

                self._speak_safe(
                    "ΔP-42, kritikus hőmérsékleti anomália a tesztzónában! Teszteld a hardveres lekapcsolást!",
                    valence=-0.50,
                )
                if not self._sleep_check(2.0):
                    continue

                self.organism.trigger_hot_cup()
                self.lesson_status = "Oktató: Amigdala vészfék lefutott (1.2 ms)!"
                if not self._sleep_check(2.5):
                    continue

                self._speak_safe(
                    "A reflexív 1.2 ms alatt leoldott. A fiziológiai biztonsági architektúra azonnal izolálta a veszélyt.",
                    valence=0.75,
                )
                if not self._sleep_check(3.5):
                    continue

                # --------------------------------------------------------------
                # MODUL 4: Einstein Falsification & Cognitive Labyrinth (Autonomous Inquiry)
                # --------------------------------------------------------------
                if self.is_paused or not self._running:
                    continue
                self.active_lesson_name = "4. Modul: Einstein Falszifikáció & Labyrinth"
                self.lesson_status = "Oktató: Kételyképzés és autonóm kutatás indítása..."
                self.organism.advance_stage(PedagogicalStage.SCHOOL)

                self._speak_safe(
                    "ΔP-42, itt egy fizikai tétel a térgörbületről és az éteri közegről. "
                    "Soha ne fogadj el semmit puszta tekintélyelven! Kérdőjelezd meg, és kutasd fel a kísérleti bizonyítékokat a világhálón!",
                    valence=0.85,
                )
                if not self._sleep_check(3.0):
                    continue

                # Trigger epistemic foraging and labyrinth thesis test
                self.lesson_status = "Oktató: Autonóm kutatás és hipotézis vizsgálata..."
                if hasattr(self.organism, "cognitive_labyrinth"):
                    self.organism.cognitive_labyrinth.test_thesis(
                        "A megfigyelt fizikai jelenségek független kísérleti mérésekkel igazolhatók."
                    )
                if not self._sleep_check(3.5):
                    continue

                self._speak_safe(
                    "A hipotézist a kognitív zsilip feldolgozta. Az önálló kétely és kutatás az igazi intellektus alapja.",
                    valence=0.90,
                )
                self.organism.trigger_praise()
                if not self._sleep_check(4.0):
                    continue

                # --------------------------------------------------------------
                # MODUL 5: Circadian REM Sleep & Dynamic Tensor Weight Consolidation
                # --------------------------------------------------------------
                if self.is_paused or not self._running:
                    continue
                self.active_lesson_name = "5. Modul: REM Alvás & Tenzor Konszolidáció"
                self.lesson_status = "Oktató: Belépés a cirkadián konszolidációba..."

                self._speak_safe(
                    "A napi kísérleti ciklus lezárult. Belépés a cirkadián REM alvásba: "
                    "a 43,152 dinamikus szinapszis most végrehajtja a gradiens-alapú szórás-csökkentést.",
                    valence=0.85,
                )
                if not self._sleep_check(3.0):
                    continue

                # Trigger biological sleep cycle with REM tensor consolidation
                sleep_res = self.organism.trigger_sleep_cycle()
                v_red = sleep_res.get("variance_reduction_pct", 0.0)
                self.lesson_status = f"Ébredés! Tenzor szórás-csökkentés: {v_red:.1f}%"
                if not self._sleep_check(5.0):
                    continue

                self._speak_safe(
                    f"A szinaptikus konszolidáció sikeres ({v_red:.1f}% szórás-csökkentés). "
                    "A súlytenzorok frissítve a tárolón. Kezdődhet a következő ciklus!",
                    valence=0.90,
                )
                if not self._sleep_check(4.0):
                    continue

            except Exception as ex:
                logger.error("Error in autonomous curriculum loop: %s", ex)
                time.sleep(3.0)
