"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — TELEMETRY & COCKPIT SERVER
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Subcortical Core:  FastAPI + WebSocket Live Streaming Engine (20 Hz Telemetry)
Dashboard Client:  Futuristic Multi-Agent Cyberpunk Cockpit at http://localhost:8080

Enables real-time bidirectional communication between the living 50 Hz organism
and the graphical engineering cockpit UI.
================================================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger("telemetry_server")

REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PATH = REPO_ROOT / "dashboard" / "index.html"
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ActionRequest(BaseModel):
    action: str
    text: Optional[str] = None
    valence: Optional[float] = 0.4
    toy_name: Optional[str] = "Piros Labda"
    touch_type: Optional[str] = "simogatás"
    smell_name: Optional[str] = "virág"
    taste_name: Optional[str] = "édes"
    stage: Optional[int] = 1
    task_id: Optional[int] = 1
    video_title: Optional[str] = "13,8 Milliárd Év: Az Univerzum & A Számítógépes Elme Eredete"
    video_content: Optional[str] = None
    question: Optional[str] = None
    url: Optional[str] = None
    puzzle_id: Optional[str] = None
    inquiry: Optional[str] = None
    snippet_index: Optional[int] = 0
    direction: Optional[int] = 1
    seconds: Optional[float] = None
    challenge_id: Optional[str] = "kresz_exception"


def clean_snapshot_for_json(snap: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all numpy types, enums and complex structures serialize to JSON safely."""
    out = {}
    for k, v in snap.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif hasattr(v, "value"):  # Enums (e.g. PedagogicalStage)
            out[k] = v.value
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, dict):
            out[k] = clean_snapshot_for_json(v)
        else:
            out[k] = v
    return out


class TelemetryServer:
    """
    Non-blocking FastAPI and WebSocket Telemetry Server.
    Connects to NurseryOrganism and streams high-fidelity state to the browser dashboard.
    """

    def __init__(self, organism: Any, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.organism = organism
        self.host = host
        self.port = port
        self.app = FastAPI(title="ΔP-42 Neural Cockpit Server")

        # Enable CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static mounts for high-speed zero-copy multimodal media streaming
        self.app.mount("/multimodal_cache", StaticFiles(directory=str(CACHE_DIR)), name="multimodal_cache")

        self.active_websockets: Set[WebSocket] = set()
        self._ws_lock = asyncio.Lock()
        self._server_thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._running = False

        self._setup_routes()

    def _setup_routes(self) -> None:
        """Configures HTTP endpoints and WebSocket streaming."""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard() -> HTMLResponse:
            if DASHBOARD_PATH.exists():
                html_content = DASHBOARD_PATH.read_text(encoding="utf-8")
                return HTMLResponse(content=html_content)
            return HTMLResponse(content="<h1>Dashboard HTML nem található!</h1>", status_code=404)

        @self.app.get("/api/telemetry")
        def get_telemetry_snapshot() -> Dict[str, Any]:
            snap = self.organism.get_snapshot()
            return clean_snapshot_for_json(snap)

        @self.app.websocket("/ws/telemetry")
        async def websocket_telemetry_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self.active_websockets.add(websocket)
            logger.info("Cockpit UI connected via WebSocket.")
            try:
                while self._running:
                    # Stream telemetry snapshot at 20 Hz without blocking event loop
                    clean_snap = await asyncio.to_thread(lambda: clean_snapshot_for_json(self.organism.get_snapshot()))
                    await websocket.send_text(json.dumps(clean_snap))
                    await asyncio.sleep(0.05)
            except WebSocketDisconnect:
                logger.info("Cockpit UI disconnected.")
            except Exception as e:
                logger.error("WebSocket stream error: %s", e)
            finally:
                self.active_websockets.discard(websocket)

        @self.app.post("/api/action")
        def execute_action(req: ActionRequest) -> Dict[str, Any]:
            act = req.action.lower()
            res: Dict[str, Any] = {"status": "ok", "action": act}

            try:
                if act == "speak":
                    utterance = req.text or "Szia!"
                    resp_text = self.organism.speak_to_organism(utterance, valence=req.valence or 0.4)
                    res["response"] = resp_text
                    res["message"] = f"Beszéd elküldve: «{utterance}»"
                    res["reaction"] = f"ΔP-42 válasza: «{resp_text}»"

                elif act == "praise":
                    self.organism.trigger_praise()
                    res["message"] = "Dicséret elküldve (Pozitív megerősítés)"
                    res["reaction"] = "Dopamin felszabadult! Szociális éhség csillapítva, örömszint és motiváció nőtt."

                elif act == "scold":
                    shift = self.organism.trigger_scold()
                    res["message"] = "Szidás elküldve (Szociális korlát)"
                    res["reaction"] = f"Illemhatár megerősítve (+{shift:.2f}° eltolódás), motoros gátlás aktiválva."

                elif act == "feed":
                    self.organism.trigger_feed()
                    res["message"] = "Etetés végrehajtva (Metabolizmus)"
                    res["reaction"] = "Glükózszint helyreállt, éhség elhárítva (v_0 = 0.05), homeosztázis stabil."

                elif act == "toy":
                    t_name = req.toy_name or "Piros Labda"
                    self.organism.trigger_toy(t_name)
                    res["message"] = f"Játék átadva: {t_name}"
                    res["reaction"] = f"Vizuális és taktilis tárgyészlelés aktív: {t_name}. Kíváncsisági hajtóerő nőtt."

                elif act == "kuka":
                    self.organism.trigger_kuka_block()
                    res["message"] = "Kuka-teszt célgátlás provokáció"
                    res["reaction"] = "Elsődleges cél blokkolva! Belső feszültség emelkedett, szublimáció aktiválódik."

                elif act == "threat":
                    self.organism.trigger_threat()
                    res["message"] = "Életveszély szimuláció aktiválva"
                    res["reaction"] = "🚨 AMIGDALA VÉSZFÉK! Fenyegetettség: 0.95, motoros kimenet azonnal zárolva!"

                elif act == "sleep":
                    sleep_res = self.organism.trigger_sleep_cycle()
                    res["sleep_result"] = sleep_res
                    res["message"] = "Cirkadián alvási ciklus elindítva"
                    res["reaction"] = f"🌙 REM álom és 2PC konszolidáció lefutott: {sleep_res.get('verdict', 'APPROVED')} (Metabolikus fáradtság kitisztítva)"

                elif act == "touch":
                    val = self.organism.trigger_touch(req.touch_type or "simogatás")
                    res["valence"] = val
                    res["message"] = f"Érintés inger: {req.touch_type or 'simogatás'}"
                    res["reaction"] = f"Taktilis mechanoreceptorok aktiválva. Valencia: {val:+.2f}, biztonságérzet megerősítve."

                elif act in ("grasp", "glass"):
                    g_res = self.organism.trigger_grasp_glass(slippery=False)
                    res["grasp_result"] = g_res
                    res["message"] = "Vizespohár kézbevétele"
                    res["reaction"] = f"Stabil kisagyi szorítóerő: {g_res.get('status', 'SIKERES FOGÁS')} (5 ujj: 4.2N, csúszás: 0.0)"

                elif act in ("slip", "slip_glass"):
                    s_res = self.organism.trigger_grasp_glass(slippery=True)
                    res["grasp_result"] = s_res
                    res["message"] = "Csúszós pohár teszt"
                    res["reaction"] = f"Kisagyi csúszásérzékelés aktív! {s_res.get('status', 'CSÚSZIK A POHÁR!')} (Sebesség: 0.45 m/s, azonnali korrekció)"

                elif act in ("massage", "chinese_massage"):
                    m_res = self.organism.trigger_chinese_massage(pressure_n=18.0)
                    res["massage_result"] = m_res
                    res["message"] = "Kínai Tui Na mélyszöveti masszázs"
                    res["reaction"] = f"Mélyszöveti inger: {m_res.get('status', 'Ellazulás')} (Nyomás: 18N, izomtónus és stressz lecsökkent)"

                elif act in ("hot", "hot_cup"):
                    h_res = self.organism.trigger_hot_cup()
                    res["hot_result"] = h_res
                    res["message"] = "Forró csésze nociceptív próba"
                    res["reaction"] = f"🚨 FORRÓ TÁRGY ÉRZÉKELVE ({h_res.get('temperature_c', 75)}°C)! Amigdala vészreflex: kezét 1.8ms alatt elrántotta!"

                elif act == "smell":
                    val = self.organism.trigger_smell(req.smell_name or "virág")
                    res["valence"] = val
                    res["message"] = f"Szaglás inger: {req.smell_name or 'virág'}"
                    res["reaction"] = f"Kémiai szagérzékelés aktív. Valencia: {val:+.2f}."

                elif act == "taste":
                    val = self.organism.trigger_taste(req.taste_name or "édes")
                    res["valence"] = val
                    res["message"] = f"Ízlelés inger: {req.taste_name or 'édes'}"
                    res["reaction"] = f"Kémiai ízérzékelés aktív. Valencia: {val:+.2f}."

                elif act == "stage":
                    from nursery.caretaker_console import PedagogicalStage
                    st = PedagogicalStage(req.stage or 1)
                    self.organism.advance_stage(st)
                    res["stage"] = st.name
                    res["message"] = f"Fejlődési szint váltás: {st.name}"
                    res["reaction"] = f"A nevelési fázis hivatalosan átállítva: {st.name}."

                elif act in ("start_teacher", "auto_teach_on"):
                    self.organism.autonomous_teacher.start()
                    res["teacher"] = "started"

                elif act in ("stop_teacher", "auto_teach_off"):
                    self.organism.autonomous_teacher.stop()
                    res["teacher"] = "stopped"

                elif act in ("pause_teacher", "teacher_pause"):
                    self.organism.autonomous_teacher.pause()
                    res["message"] = "⏸️ Nevelő csendesítve (Standby)"

                elif act in ("resume_teacher", "teacher_resume"):
                    self.organism.autonomous_teacher.resume()
                    res["message"] = "▶️ Nevelő tanóra újraindítva"

                elif act in ("task", "set_task"):
                    task_res = self.organism.classroom.set_task(req.task_id or 1)
                    res["task"] = task_res

                elif act in ("watch_video", "play_video"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    v_title = req.video_title or "13,8 Milliárd Év: Az Univerzum & A Mesterséges Elme"
                    v_content = req.video_content or (
                        "A világegyetem 13,8 milliárd éve keletkezett az ősrobbanásban. A csillagok magfúziójában szén, "
                        "vas és oxigén keletkezett, amiből a Föld és az emberiség felépült. Az emberiség megalkotta "
                        "a matematikát és a szuperszámítógépeket, létrehozva a digitális elméket. "
                        "Te magad a ΔP-42 szuverén kognitív mag vagy az NVIDIA GeForce RTX 5070-en."
                    )
                    threading.Thread(
                        target=self.organism.trigger_video_and_forage,
                        args=(v_title, v_content),
                        daemon=True,
                    ).start()
                    res["message"] = f"🎬 Videó elindítva: '{v_title}'. ΔP-42 kérdése és az autonóm webes keresés aktív!"

                elif act in ("ask_web", "forage"):
                    q = req.question or req.text or "univerzum és mesterséges intelligencia eredete"
                    threading.Thread(
                        target=self.organism.epistemic_forager.forage_for_question,
                        args=(q,),
                        daemon=True,
                    ).start()
                    res["message"] = f"🌐 Közvetlen webes keresés elindítva: '{q}'"

                elif act in ("youtube_ingest", "ingest_yt", "watch_yt"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    yt_url = req.url or req.text or "https://www.youtube.com/watch?v=libKVRa01L8"
                    threading.Thread(
                        target=self.organism.youtube_ingest.ingest_youtube_url,
                        args=(yt_url,),
                        daemon=True,
                    ).start()
                    res["message"] = f"🎬 Multimodális YouTube vetítés elindítva a linkről: {yt_url}"

                elif act in ("youtube_step", "step_yt"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    dir_val = req.direction if req.direction is not None else 1
                    threading.Thread(
                        target=self.organism.youtube_ingest.step_slice,
                        args=(dir_val,),
                        daemon=True,
                    ).start()
                    res["message"] = f"⏱️ Időszelet léptetés: {dir_val:+d}"

                elif act in ("youtube_jump", "jump_yt"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    s_idx = req.snippet_index if req.snippet_index is not None else 0
                    threading.Thread(
                        target=self.organism.youtube_ingest.extract_and_bind_slice,
                        args=(s_idx,),
                        daemon=True,
                    ).start()
                    res["message"] = f"⏱️ Időszelet ugrás indexre: {s_idx}"

                elif act in ("youtube_jump_time", "jump_yt_time"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    sec = float(req.seconds if req.seconds is not None else 0.0)
                    threading.Thread(
                        target=self.organism.youtube_ingest.jump_to_time,
                        args=(sec,),
                        daemon=True,
                    ).start()
                    res["message"] = f"⏱️ Időszelet ugrás időpontra: {sec:.1f} másodperc"

                elif act in ("labyrinth_select", "select_puzzle"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    p_id = req.puzzle_id or "puzzle_5whys"
                    res["snapshot"] = self.organism.cognitive_labyrinth.select_puzzle(p_id)
                    res["message"] = f"Rejtvény kiválasztva: {p_id}"

                elif act in ("labyrinth_ask", "ask_labyrinth"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    inq = req.inquiry or req.text or "Miért állt le a rendszer?"
                    lab_res = self.organism.cognitive_labyrinth.process_inquiry(inq)
                    res["message"] = f"Tézis / Kérdés elküldve: «{inq}»"
                    res["reaction"] = lab_res.get("answer", self.organism.cognitive_labyrinth.last_log)
                    res["snapshot"] = lab_res.get("snapshot")

                elif act in ("labyrinth_blind", "blind_action"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    b_act = req.text or "Újraindítás kísérlet"
                    snap_b = self.organism.cognitive_labyrinth.submit_blind_action(b_act)
                    res["message"] = f"Vak próbálkozás: «{b_act}»"
                    res["reaction"] = self.organism.cognitive_labyrinth.last_log
                    res["snapshot"] = snap_b

                elif act in ("gymnasium_start", "start_challenge", "gym_challenge"):
                    if hasattr(self.organism, "autonomous_teacher"):
                        self.organism.autonomous_teacher.pause()
                    c_id = req.challenge_id or "kresz_exception"
                    gym_res = self.organism.gymnasium.start_challenge(c_id)
                    res["gymnasium"] = gym_res
                    res["message"] = f"🥊 Kognitív Edzőterem: '{c_id}' kihívás elindítva!"
                    res["reaction"] = f"Hipotézis-generálás és autonóm élő internetes kutatás aktív: {c_id}."

                else:
                    res["status"] = "unknown_action"

            except Exception as ex:
                res["status"] = "error"
                res["error"] = str(ex)
                logger.error("Action execution failed: %s", ex)

            return res

    def start(self) -> None:
        """Starts Uvicorn in a background daemon thread."""
        self._running = True
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            loop="asyncio",
        )
        self._uvicorn_server = uvicorn.Server(config)

        def run_server() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._uvicorn_server.serve())
            except (OSError, SystemExit) as ex:
                logger.warning(f"Telemetry server port {self.port} already in use or startup aborted: {ex}")
            finally:
                loop.close()

        self._server_thread = threading.Thread(
            target=run_server,
            name="ΔP42-TelemetryServer",
            daemon=True,
        )
        self._server_thread.start()
        logger.info(f"Telemetry Web Cockpit running at http://{self.host}:{self.port}")

    def stop(self) -> None:
        """Gracefully stops the telemetry server."""
        self._running = False
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        logger.info("Telemetry server stopped.")
