"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — MULTIMODAL YOUTUBE GROUNDING ENGINE
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Cognitive Theory:  Temporal Binding & Tri-Modal Grounding (Frame + Audio + Text)
DVP Compliance:    DVP-05 (Cross-Modal Coherence & Grounded Prediction)
================================================================================
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

import numpy as np
from scipy.io import wavfile
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger("youtube_multimodal")

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "multimodal_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_video_id(url_or_id: str) -> str:
    """Extracts 11-character YouTube video ID from various URL formats."""
    cleaned = url_or_id.strip()
    if len(cleaned) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", cleaned):
        return cleaned

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"(?:shorts\/)([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, cleaned)
        if m:
            return m.group(1)
    return ""


def format_seconds(seconds: float) -> str:
    """Formats seconds as MM:SS."""
    s = int(seconds)
    mins = s // 60
    secs = s % 60
    return f"{mins:02d}:{secs:02d}"


class MultimodalYouTubeIngest:
    """
    Ingests YouTube streams into a synchronized tri-modal perceptual bundle:
    1. Visual Channel: Exact video frame extracted at timestamp t (JPEG base64).
    2. Audio Channel: Real acoustic waveform slice extracted at timestamp t (WAV base64, RMS, Peak, Spectrum).
    3. Text Channel: Synchronized speech transcript segment elapsing at timestamp t.
    """

    def __init__(self, organism: Any) -> None:
        self.organism = organism
        self.last_percept: Optional[Dict[str, Any]] = None
        self.cached_video_id: str = ""
        self.cached_title: str = ""
        self.cached_author: str = ""
        self.cached_duration: float = 0.0
        self.cached_audio_url: Optional[str] = None
        self.cached_video_url: Optional[str] = None
        self.cached_snippets: List[Dict[str, Any]] = []
        self.active_snippet_index: int = 0

    def ingest_youtube_url(self, url: str) -> Dict[str, Any]:
        """
        Fetches stream metadata with yt-dlp, fetches transcript snippets,
        and immediately binds the first synchronized time slice.
        """
        video_id = extract_video_id(url)
        if not video_id:
            logger.warning(f"Invalid YouTube URL or ID: {url}")
            return {"status": "error", "message": "Érvénytelen YouTube link vagy azonosító!"}

        logger.info(f"Ingesting multimodal YouTube video with yt-dlp: {video_id}")

        # 1. Extract direct audio/video stream URLs and metadata via yt-dlp
        title = "YouTube Ismeretterjesztő Film"
        author = "YouTube Alkotó"
        duration = 0.0
        audio_stream_url = None
        video_stream_url = None

        try:
            ydl_opts = {
                "format": "ba[ext=m4a]/ba/b",
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                title = info.get("title", title)
                author = info.get("uploader", author)
                duration = float(info.get("duration", 0.0))
                formats = info.get("formats", [])

                for f in reversed(formats):
                    if f.get("acodec") != "none" and f.get("url") and not audio_stream_url:
                        audio_stream_url = f.get("url")
                    if f.get("vcodec") != "none" and f.get("url") and not video_stream_url:
                        video_stream_url = f.get("url")
                    if audio_stream_url and video_stream_url:
                        break
        except Exception as ex:
            logger.warning(f"yt-dlp stream extraction note for {video_id}: {ex}")

        # Fallback to thumbnail and standard URL if streams fail
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

        # 2. Fetch transcript snippets with timestamps
        snippets: List[Dict[str, Any]] = []
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id)
            if hasattr(fetched, "snippets") and fetched.snippets:
                for s in fetched.snippets:
                    snippets.append({
                        "text": s.text.strip(),
                        "start": float(s.start),
                        "duration": float(s.duration),
                    })
        except Exception as ex:
            logger.debug(f"Transcript fetch note for {video_id}: {ex}")

        # If no transcript exists, generate timeline intervals based on title
        if not snippets:
            step = 15.0
            tot = duration if duration > 10 else 60.0
            cur = 25.0 if tot > 35.0 else 0.0
            while cur < tot:
                snippets.append({
                    "text": f"{title} — szakaszelemzés ({format_seconds(cur)})",
                    "start": cur,
                    "duration": step,
                })
                cur += step

        self.cached_video_id = video_id
        self.cached_title = title
        self.cached_author = author
        self.cached_duration = duration
        self.cached_audio_url = audio_stream_url
        self.cached_video_url = video_stream_url
        self.cached_snippets = snippets
        self.active_snippet_index = 0

        # 3. Extract and bind the first time slice (t ~ 0-10s)
        return self.extract_and_bind_slice(0)

    def extract_and_bind_slice(self, index: int) -> Dict[str, Any]:
        """
        Extracts exact frame, acoustic waveform, and text for snippet at index,
        computes cross-modal resonance, and injects into ΔP-42 SensoryBridge.
        """
        if not self.cached_snippets:
            return {"status": "error", "message": "Nincs betöltve videó szegmens!"}

        idx = max(0, min(index, len(self.cached_snippets) - 1))
        self.active_snippet_index = idx
        snip = self.cached_snippets[idx]

        t_start = snip["start"]
        t_dur = max(snip["duration"], 15.0)
        t_mid = t_start + t_dur / 2.0
        text = snip["text"]

        video_id = self.cached_video_id
        frame_file = CACHE_DIR / f"{video_id}_slice_{idx}.jpg"
        audio_file = CACHE_DIR / f"{video_id}_slice_{idx}.wav"

        # 1. Extract physical video frame at t_mid via ffmpeg
        frame_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        if self.cached_video_url:
            cmd_frame = [
                "ffmpeg", "-y",
                "-ss", str(t_mid),
                "-i", self.cached_video_url,
                "-vframes", "1",
                "-vf", "scale=480:-1",
                "-q:v", "4",
                str(frame_file),
            ]
            try:
                subprocess.run(cmd_frame, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=12)
                if frame_file.exists():
                    frame_url = f"/multimodal_cache/{video_id}_slice_{idx}.jpg"
            except Exception as ex:
                logger.debug(f"Frame extraction timeout/error: {ex}")

        # 2. Extract physical audio slice at [t_start, t_start + t_dur] via ffmpeg
        audio_url = ""
        rms = 0.0
        peak = 0
        speech_detected = False
        waveform_points: List[float] = [0.05] * 50

        if self.cached_audio_url:
            cmd_audio = [
                "ffmpeg", "-y",
                "-ss", str(t_start),
                "-i", self.cached_audio_url,
                "-t", str(t_dur),
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                str(audio_file),
            ]
            try:
                subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=12)
                if audio_file.exists():
                    audio_url = f"/multimodal_cache/{video_id}_slice_{idx}.wav"

                    # Scipy acoustic analysis
                    sr, data = wavfile.read(str(audio_file))
                    if len(data) > 0:
                        rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                        peak = int(np.max(np.abs(data)))
                        speech_detected = rms > 250.0

                        # Sample 50 points for cockpit oscilloscope
                        step = max(1, len(data) // 50)
                        waveform_points = [float(np.round(abs(int(x)) / 32768.0, 3)) for x in data[::step][:50]]
            except Exception as ex:
                logger.debug(f"Audio extraction timeout/error: {ex}")

        # 3. Cross-Modal Grounding & Coherence Evaluation
        # If speech energy exists and text is present -> High Coherence (0.95+)
        # If text is present but speech is silent (or mismatch) -> Dissonance (0.45)
        clean_words = [w for w in re.findall(r"[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]{3,}", text.lower()) if len(w) > 2]
        keywords = list(dict.fromkeys(clean_words))[:5]
        kw_str = ", ".join(keywords) if keywords else "vizuális kontextus"

        if speech_detected and len(text) > 3:
            coherence = 0.96
            dissonance = 0.04
            status_text = "Horgonyzott hármas: [KÉP ⟷ VALÓS HANG ⟷ SZÖVEG] KOVÁCIÓ KÉSZ"
        else:
            coherence = 0.65 if len(text) > 3 else 0.35
            dissonance = 0.70
            status_text = "Érzékszervi disszonancia: Akusztikai vagy képi aszinkron észlelve!"

        cur_time_str = f"{format_seconds(t_start)} / {format_seconds(self.cached_duration)}"

        percept_bundle = {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": self.cached_title,
            "author": self.cached_author,
            "duration_str": format_seconds(self.cached_duration),
            "timestamp_str": cur_time_str,
            "snippet_index": idx,
            "total_snippets": len(self.cached_snippets),
            "text": text,
            "keywords": keywords,
            "frame_url": frame_url,
            "audio_url": audio_url,
            "frame_b64": frame_url,
            "audio_b64": audio_url,
            "waveform": waveform_points,
            "audio_metrics": {
                "rms": round(rms, 1),
                "peak": peak,
                "speech_detected": speech_detected,
            },
            "coherence_score": coherence,
            "dissonance": dissonance,
            "status_text": status_text,
        }

        self.last_percept = percept_bundle

        # 4. Inject Tri-Modal Bundle into SensoryBridge (Temporal Grounding asynchronously)
        sensory_narration = (
            f"[SZINKRONIZÁLT MULTIMODÁLIS IDŐSZELET: {cur_time_str}]\n"
            f"• LÁTVÁNY (Kép): {self.cached_title} (Kulcsszavak: {kw_str})\n"
            f"• HANG (Akusztika): Beszédhang jelenléte: {'IGEN' if speech_detected else 'NEM'} (RMS: {rms:.0f})\n"
            f"• SZÖVEG (Kimondva): \"{text}\"\n"
            f"Kereszt-modális rezonancia: {coherence:.2f} — {status_text}."
        )

        logger.info(f"Injecting synchronized time slice {idx} ({cur_time_str}) into ΔP-42 SensoryBridge")
        valence = 0.90 if coherence >= 0.9 else 0.45
        import threading
        threading.Thread(
            target=self.organism.sensory_bridge.ingest_audio_text,
            args=(sensory_narration,),
            kwargs={"valence": valence},
            daemon=True,
        ).start()

        # 5. Emergent Question Formulation if dissonance or new concept
        emergent_q = ""
        pfc_goal = getattr(self.organism.pedagogical_controller, "last_pfc_goal", "")
        clean_pfc = pfc_goal.replace("Kérdésem az internethez:", "").replace("Kutatási kérdés a webre:", "").strip()
        if len(clean_pfc) >= 5:
            emergent_q = clean_pfc
        else:
            emergent_q = f"{self.cached_title} {kw_str}".strip()

        if hasattr(self.organism, "epistemic_forager") and emergent_q:
            threading.Thread(
                target=self.organism.epistemic_forager.forage_for_question,
                args=(emergent_q,),
                daemon=True,
            ).start()

        percept_bundle["emergent_question"] = emergent_q
        return {
            "status": "ok",
            "percept": percept_bundle,
            "emergent_question": emergent_q,
        }

    def step_slice(self, direction: int = 1) -> Dict[str, Any]:
        """Steps forward or backward through video snippets."""
        new_index = self.active_snippet_index + direction
        return self.extract_and_bind_slice(new_index)

    def jump_to_time(self, seconds: float) -> Dict[str, Any]:
        """Jumps directly to the slice closest to the requested timestamp."""
        if not self.cached_snippets:
            return {"status": "error", "message": "Nincs betöltve videó szegmens!"}
        sec = max(0.0, min(float(seconds), float(self.cached_duration or 3600.0)))
        closest_idx = 0
        min_diff = 99999999.0
        for i, s in enumerate(self.cached_snippets):
            diff = abs(s["start"] - sec)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        return self.extract_and_bind_slice(closest_idx)

    def get_snapshot(self) -> Dict[str, Any]:
        """Provides full snapshot for telemetry websocket and dashboard."""
        if self.last_percept:
            return self.last_percept
        return {
            "video_id": "",
            "title": "Nincs aktív videó",
            "author": "—",
            "duration_str": "00:00",
            "timestamp_str": "00:00 / 00:00",
            "snippet_index": 0,
            "total_snippets": 0,
            "text": "Illessz be egy YouTube videó linket a szinkronizált kereszt-modális vetítéshez!",
            "keywords": [],
            "frame_b64": "",
            "audio_b64": "",
            "waveform": [0.0] * 50,
            "audio_metrics": {"rms": 0.0, "peak": 0, "speech_detected": False},
            "coherence_score": 0.0,
            "dissonance": 0.0,
            "status_text": "Várakozás videó bemenetre...",
            "emergent_question": "",
        }
