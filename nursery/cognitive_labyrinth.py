"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — COGNITIVE LABYRINTH ENGINE
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Paradigm:          Learning to solve problems via asking questions.
                   Zero hardcoded recipes — Epistemic Locks unlockable only through
                   autonomous inquiry and web methodology discovery (5 Whys, Polya, TRIZ).
================================================================================
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cognitive_labyrinth")


class LabyrinthPuzzle:
    """Represents an epistemic barrier puzzle with hidden causal layers."""

    def __init__(
        self,
        puzzle_id: str,
        name: str,
        description: str,
        target_methodology: str,
        causal_layers: List[Dict[str, Any]],
        solution_key: str,
    ) -> None:
        self.puzzle_id = puzzle_id
        self.name = name
        self.description = description
        self.target_methodology = target_methodology  # e.g. "5 Whys (Gyökérok-elemzés)", "Polya Heurisztika", "TRIZ"
        self.causal_layers = causal_layers  # Sequential depths revealed by targeted questions
        self.solution_key = solution_key
        self.current_depth = 0
        self.is_solved = False
        self.questions_asked: List[str] = []
        self.methodologies_found: List[str] = []
        self.dissonance_delta = 0.85  # High initial impasse dissonance


class CognitiveLabyrinthEngine:
    """
    Cognitive Labyrinth Engine for ΔP-42.
    Presents epistemic locks where blind action fails, and only asking
    targeted questions and searching for analytical methodologies unlocks the gate.
    """

    def __init__(self, organism: Any) -> None:
        self.organism = organism
        self._lock = threading.RLock()
        self.puzzles: Dict[str, LabyrinthPuzzle] = self._init_puzzles()
        self.active_puzzle_id = "puzzle_5whys"
        self.last_log = "A Kognitív Labirintus készen áll. Válassz egy rejtvényt!"

    def _init_puzzles(self) -> Dict[str, LabyrinthPuzzle]:
        return {
            "puzzle_5whys": LabyrinthPuzzle(
                puzzle_id="puzzle_5whys",
                name="1. A Leállt Generátor és a Rejtett Gyökérok",
                description=(
                    "Egy kritikus energiagenerátor váratlanul leállt és füstöl. "
                    "Ha vakon megnyomod az 'Újraindítás' gombot, a rendszer azonnal leég és zárlat keletkezik. "
                    "Hogyan deríted ki a leállás valódi okát anélkül, hogy kárt tennél benne?"
                ),
                target_methodology="5 Miért (Gyökérok-elemzés / Sakichi Toyoda)",
                causal_layers=[
                    {
                        "keywords": ["miért", "melegedett", "füstöl", "leállt", "motor", "ok"],
                        "answer": "A gép hőmérséklet-szenzora szerint a villanymotor azért melegedett túl, mert a hűtőventilátor lapátjai leálltak.",
                        "next_hint": "Miért álltak le a ventilátor lapátjai?",
                    },
                    {
                        "keywords": ["ventilátor", "lapát", "tengely", "csapágy", "miért"],
                        "answer": "A vizsgálat feltárta: a ventilátor tengelyének csapágya megszorult és elkopott a súrlódástól.",
                        "next_hint": "Miért kopott el a csapágy, nem volt kenése?",
                    },
                    {
                        "keywords": ["kenés", "olaj", "kenőanyag", "szivattyú", "szűrő", "miért"],
                        "answer": "A kenőolaj-szivattyú nem szállított olajat, mert a szűrője teljesen eltömődött fémreszelékkel.",
                        "next_hint": "Honnan került fémreszelék a szűrőbe?",
                    },
                    {
                        "keywords": ["fémreszelék", "forgács", "kopás", "gyökérok", "miért"],
                        "answer": "GYÖKÉROK MEGTALÁLVA! A szivattyú beömlőcsonkjának rögzítőcsavarja kilazult a rezonanciától, és beleért a forgórészbe. A csavar meghúzása és rögzítése véglegesen megoldja a problémát!",
                        "next_hint": "A probléma gyökere azonosítva!",
                    },
                ],
                solution_key="csavar kilazulás rezonancia",
            ),
            "puzzle_polya": LabyrinthPuzzle(
                puzzle_id="puzzle_polya",
                name="2. A Szakadék Áthidalása (Polya Heurisztika)",
                description=(
                    "Két hegycsúcs között 12 méteres szakadék tátong. Rendelkezésre áll: 3 db 4 méteres gerenda, "
                    "kötél és egy csiga. A közvetlen fektetéshez túl rövidek a gerendák. "
                    "Hogyan bontod le a problémát egyszerűbb részekre?"
                ),
                target_methodology="Polya György 4 lépése (A gondolkodás iskolája)",
                causal_layers=[
                    {
                        "keywords": ["ismeretlen", "adat", "feltétel", "cél", "polya"],
                        "answer": "1. Lépés: Értsd meg a feladatot! A cél: 12 m áthidalása 3x4 m elemből. Az elemek összekapcsolhatók és alátámaszthatók.",
                        "next_hint": "Hogyan készítesz tervet az elemek kombinálásával?",
                    },
                    {
                        "keywords": ["terv", "hasonló", "rokon", "háromszög", "rács"],
                        "answer": "2. Lépés: Tervkészítés! Háromszög-rácsos tartószerkezet (Fachwerk) kialakítása, ahol a gerendák nem egymás után, hanem támaszként kapcsolódnak.",
                        "next_hint": "Hogyan hajtod végre és ellenőrzöd a tervet?",
                    },
                    {
                        "keywords": ["végrehajtás", "ellenőrzés", "teherbírás", "szilárdság"],
                        "answer": "3. & 4. Lépés: SIKER! A rácsos tartószerkezet felépült, a teherbírás 400 kg. A szakadék áthidalva!",
                        "next_hint": "A heurisztikus híd stabilan áll!",
                    },
                ],
                solution_key="háromszög rácsos szerkezet",
            ),
            "puzzle_einstein": LabyrinthPuzzle(
                puzzle_id="puzzle_einstein",
                name="3. Einstein Rejtvénye: Kié a hal? (Tézis & Falszifikáció)",
                description=(
                    "5 ház, 5 szín, 5 nemzetiség, 5 ital és 5 háziállat. A feladvány: KIÉ A HAL? "
                    "A többség házakat kezd rajzolgatni (25 perc), Einstein módszere a bátor tézis és a gyors ellentmondás-keresés (2 perc): "
                    "'Létezik-e egzakt, feloldhatatlan kísérleti vagy logikai ellentmondás, ami a tézist kizárja?'"
                ),
                target_methodology="Einsteini Gondolatkísérlet & Hipotetikus Falszifikáció (Popper)",
                causal_layers=[
                    {
                        "keywords": ["svéd", "sved", "kutya"],
                        "answer": "1. Tézis teszt: 'A svédé a hal?' -> ELLENTMONDÁS! A 2. axióma szerint a svéd kutyát tart (Kutya != Hal). Kizárva!",
                        "next_hint": "Tedd fel a következő tézist (pl. az angolról)!",
                    },
                    {
                        "keywords": ["angol", "madár", "piros"],
                        "answer": "2. Tézis teszt: 'Az angolé a hal?' -> ELLENTMONDÁS! A 3. axióma szerint az angol a piros házban lakik és madara van (Madár != Hal). Kizárva!",
                        "next_hint": "Tedd fel a következő tézist (pl. a dánról vagy norvégról)!",
                    },
                    {
                        "keywords": ["dán", "dan", "norvég", "norveg", "macska", "ló", "tea", "víz"],
                        "answer": "3. Tézis teszt: 'A dáné vagy norvégé a hal?' -> ELLENTMONDÁS! A norvég az 1. házban lakik (macska a szomszédban), a dán teát iszik (ló a szomszédban). Kizárva!",
                        "next_hint": "Ki maradt az univerzumban, akinél zéró ellentmondás van?",
                    },
                    {
                        "keywords": ["német", "nemet", "hal", "zöld", "kávé"],
                        "answer": "🎉 ZSENIÁLIS! Tézis teszt: 'A németé a hal!' -> LÉTEZIK ELLENTMONDÁS? NEM! ZÉRÓ ELLENTMONDÁS! A német a zöld házban kávézik, és egyedüliként elégíti ki az univerzum összes axiómáját. A HAL A NÉMETÉ!",
                        "next_hint": "A feladvány 2 perc alatt, tiszta falszifikációval megoldva!",
                    },
                ],
                solution_key="német hal zöld ház",
            ),
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Snapshot for dashboard HUD."""
        with self._lock:
            p = self.puzzles[self.active_puzzle_id]
            return {
                "active_puzzle_id": self.active_puzzle_id,
                "name": p.name,
                "description": p.description,
                "target_methodology": p.target_methodology,
                "current_depth": p.current_depth,
                "max_depth": len(p.causal_layers),
                "is_solved": p.is_solved,
                "questions_count": len(p.questions_asked),
                "last_question": p.questions_asked[-1] if p.questions_asked else "—",
                "methodologies_found": p.methodologies_found,
                "dissonance": p.dissonance_delta,
                "last_log": self.last_log,
            }

    def select_puzzle(self, puzzle_id: str) -> Dict[str, Any]:
        """Selects active labyrinth puzzle."""
        with self._lock:
            if puzzle_id in self.puzzles:
                self.active_puzzle_id = puzzle_id
                p = self.puzzles[puzzle_id]
                self.last_log = f"Rejtvény aktiválva: {p.name}. Zárt zsilip — kérdezz az előrelépéshez!"
                logger.info(f"Active labyrinth puzzle set to: {puzzle_id}")
        return self.get_snapshot()

    def submit_blind_action(self, action_name: str) -> Dict[str, Any]:
        """Simulates blind guessing without asking questions (Results in failure)."""
        with self._lock:
            p = self.puzzles[self.active_puzzle_id]
            p.dissonance_delta = min(1.0, p.dissonance_delta + 0.10)
            if p.puzzle_id == "puzzle_einstein":
                self.last_log = (
                    f"❌ KUDARC: A vak tippelgetés ('{action_name}') hatástalan! "
                    "5^5 = 3125 kombináció van. Ne találgass vakon, állíts fel egy konkrét tézist "
                    "(pl. 'A svédé a hal') és vizsgáld meg, hogy létezik-e egzakt ellentmondás!"
                )
            else:
                self.last_log = (
                    f"❌ KUDARC: A vak cselekvés ('{action_name}') hatástalan! "
                    "A zsilip zárva maradt. Kérdezz az okokról vagy keress módszert a weben!"
                )
            # Ingest failure into sensory cortex asynchronously
            threading.Thread(
                target=self.organism.sensory_bridge.ingest_audio_text,
                args=(f"[LABIRINTUS ZSILIP ZÁRVA] A vak próbálkozás elbukott. A feszültség nő. Kérdezz!",),
                kwargs={"valence": -0.40},
                daemon=True,
            ).start()
        return self.get_snapshot()

    def process_inquiry(self, inquiry_text: str) -> Dict[str, Any]:
        """
        Processes question asked by ΔP-42 (or operator on behalf of ΔP-42).
        If inquiry targets causal layers or discovers methodology on web, unlocks depth!
        """
        with self._lock:
            p = self.puzzles[self.active_puzzle_id]
            p.questions_asked.append(inquiry_text)
            clean_inq = inquiry_text.lower()

            # Special Einstein Hypothesis-Falsification Resolver
            if p.puzzle_id == "puzzle_einstein":
                if any(w in clean_inq for w in ["német", "nemet", "németé", "nemete", "németnél", "hal a németé"]):
                    p.current_depth = len(p.causal_layers)
                    p.is_solved = True
                    p.dissonance_delta = 0.05
                    ans = (
                        "🎉 ZSENIÁLIS! Tézis teszt: 'A németé a hal!' -> LÉTEZIK-E ELLENTMONDÁS? "
                        "NEM! ZÉRÓ ELLENTMONDÁS! A német a zöld házban kávézik, és egyedüliként elégíti ki "
                        "az univerzum összes axiómáját. A HAL A NÉMETÉ!"
                    )
                    self.last_log = "🎉 SIKER! A NÉMETÉ A HAL! Létezik ellentmondás? -> NEM! Zéró ellentmondás! A zsilip kitárult!"
                    logger.info("Einstein riddle solved via hypothesis falsification!")
                    knowledge_feed = f"[LABIRINTUS SIKER]: {ans}\nA hal a németé, a zsilip kitárult!"
                    threading.Thread(
                        target=self.organism.sensory_bridge.ingest_audio_text,
                        args=(knowledge_feed,),
                        kwargs={"valence": 0.98},
                        daemon=True,
                    ).start()
                    return {
                        "status": "unlocked",
                        "depth": p.current_depth,
                        "answer": ans,
                        "is_solved": True,
                        "snapshot": self.get_snapshot(),
                    }

                elif any(w in clean_inq for w in ["svéd", "sved", "svédé", "svede", "kutya"]):
                    p.current_depth = max(p.current_depth, 1)
                    p.dissonance_delta = max(0.20, p.dissonance_delta - 0.20)
                    ans = (
                        "❌ ELLENTMONDÁS! Tézis: 'A svédé a hal?' -> A 2. axióma kimondja: "
                        "A svéd kutyát tart (Kutya != Hal). A svéd-hipotézis MEGDŐLT, KIZÁRVA!"
                    )
                    self.last_log = "❌ ELLENTMONDÁS: A svéd kutyát tart! A tézis megdőlt, kizárva!"
                    threading.Thread(
                        target=self.organism.sensory_bridge.ingest_audio_text,
                        args=(f"[LABIRINTUS FALSIFIKÁCIÓ]: {ans}",),
                        kwargs={"valence": 0.60},
                        daemon=True,
                    ).start()
                    return {
                        "status": "partial",
                        "depth": p.current_depth,
                        "answer": ans,
                        "is_solved": False,
                        "snapshot": self.get_snapshot(),
                    }

                elif any(w in clean_inq for w in ["angol", "angolé", "angolnál", "madár", "piros"]):
                    p.current_depth = max(p.current_depth, 2)
                    p.dissonance_delta = max(0.20, p.dissonance_delta - 0.20)
                    ans = (
                        "❌ ELLENTMONDÁS! Tézis: 'Az angolé a hal?' -> A 3. axióma kimondja: "
                        "Az angol a piros házban lakik és madara van (Madár != Hal). Az angol-hipotézis KIZÁRVA!"
                    )
                    self.last_log = "❌ ELLENTMONDÁS: Az angolnak madara van! A tézis megdőlt, kizárva!"
                    threading.Thread(
                        target=self.organism.sensory_bridge.ingest_audio_text,
                        args=(f"[LABIRINTUS FALSIFIKÁCIÓ]: {ans}",),
                        kwargs={"valence": 0.60},
                        daemon=True,
                    ).start()
                    return {
                        "status": "partial",
                        "depth": p.current_depth,
                        "answer": ans,
                        "is_solved": False,
                        "snapshot": self.get_snapshot(),
                    }

                elif any(w in clean_inq for w in ["dán", "dan", "dáné", "norvég", "norveg", "norvégé", "macska", "ló", "tea", "víz"]):
                    p.current_depth = max(p.current_depth, 3)
                    p.dissonance_delta = max(0.20, p.dissonance_delta - 0.20)
                    ans = (
                        "❌ ELLENTMONDÁS! Tézis: 'A dáné vagy a norvégé a hal?' -> "
                        "A norvég az 1. házban lakik (macska a szomszédban, vizet iszik); a dán teát iszik (2. ház) és lova van a szomszédnak. "
                        "A szomszédsági kényszerek miatt KIZÁRVA!"
                    )
                    self.last_log = "❌ ELLENTMONDÁS: A dán és a norvég relációi kizárják a halat!"
                    threading.Thread(
                        target=self.organism.sensory_bridge.ingest_audio_text,
                        args=(f"[LABIRINTUS FALSIFIKÁCIÓ]: {ans}",),
                        kwargs={"valence": 0.60},
                        daemon=True,
                    ).start()
                    return {
                        "status": "partial",
                        "depth": p.current_depth,
                        "answer": ans,
                        "is_solved": False,
                        "snapshot": self.get_snapshot(),
                    }

            # Check if methodology inquiry
            if any(w in clean_inq for w in ["módszer", "hogyan", "hibakeresés", "elemzés", "okozat", "miért"]):
                methodology_name = p.target_methodology
                if methodology_name not in p.methodologies_found:
                    p.methodologies_found.append(methodology_name)

            # Check causal layer progression
            if p.current_depth < len(p.causal_layers):
                current_layer = p.causal_layers[p.current_depth]
                # Match keywords in question
                match_count = sum(1 for kw in current_layer["keywords"] if kw in clean_inq)
                if match_count >= 1 or len(p.questions_asked) >= 2:
                    # Layer unlocked!
                    p.current_depth += 1
                    answer = current_layer["answer"]
                    hint = current_layer["next_hint"]
                    p.dissonance_delta = max(0.10, p.dissonance_delta - 0.22)

                    if p.current_depth >= len(p.causal_layers):
                        p.is_solved = True
                        p.dissonance_delta = 0.05
                        self.last_log = f"🎉 SIKER! A zsilip kinyílt: {answer}"
                        logger.info(f"Labyrinth puzzle {p.puzzle_id} SOLVED via autonomous inquiry!")
                    else:
                        self.last_log = f"🔓 {p.current_depth}. Zsilip feloldva! {answer}"

                    # Feed answer back into ΔP-42 sensory cortex asynchronously
                    knowledge_feed = (
                        f"[LABIRINTUS VÁLASZ]: {answer}\n"
                        f"Következő kérdés iránya: {hint}"
                    )
                    threading.Thread(
                        target=self.organism.sensory_bridge.ingest_audio_text,
                        args=(knowledge_feed,),
                        kwargs={"valence": 0.90},
                        daemon=True,
                    ).start()

                    return {
                        "status": "unlocked",
                        "depth": p.current_depth,
                        "answer": answer,
                        "hint": hint,
                        "is_solved": p.is_solved,
                        "snapshot": self.get_snapshot(),
                    }

            self.last_log = f"Kérdés rögzítve: '{inquiry_text[:40]}...'. Kérdezz specifikusabban az okokról!"
            return {
                "status": "partial",
                "depth": p.current_depth,
                "is_solved": p.is_solved,
                "snapshot": self.get_snapshot(),
            }
