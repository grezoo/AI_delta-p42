"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — EPISTEMIC WEB FORAGER
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Subcortical Drive:  Maslow Δ_2 (Epistemic Hunger) & Thalamic Dissonance Reduction
Function:           Direct, autonomous web exploration without external operator.
================================================================================
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("epistemic_forager")


class EpistemicWebForager:
    """
    Autonomous Epistemic Search Organ for ΔP-42.
    Allows the organism to translate emergent prefrontal curiosity questions
    into live web search queries, fetch real scientific/encyclopedic articles,
    and directly inject the findings into the SensoryBridge.
    """

    def __init__(self, organism: Any) -> None:
        self.organism = organism
        self._lock = threading.RLock()
        self.is_foraging = False
        self.last_question = ""
        self.last_query = ""
        self.last_results: List[Dict[str, str]] = []
        self.last_summary = ""
        self.total_searches = 0
        self.last_search_timestamp = 0.0

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot for Cockpit HUD."""
        with self._lock:
            return {
                "is_foraging": self.is_foraging,
                "last_question": self.last_question,
                "last_query": self.last_query,
                "last_results": self.last_results,
                "last_summary": self.last_summary,
                "total_searches": self.total_searches,
                "last_search_timestamp": self.last_search_timestamp,
            }

    def clean_text(self, text: str) -> str:
        """Removes HTML tags and normalizes whitespace."""
        clean = re.sub(r"<[^>]+>", " ", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def search_web_direct(self, query: str) -> List[Dict[str, str]]:
        """
        Executes real-time encyclopedic and web search using standard HTTP APIs.
        Returns top titles and informative snippets.
        """
        results: List[Dict[str, str]] = []
        clean_q = query.strip()
        if not clean_q:
            return results

        # 1. Primary: Hungarian Wikipedia API (Encyclopedic, fast, clean)
        try:
            wiki_url = (
                f"https://hu.wikipedia.org/w/api.php?action=query&list=search&srsearch="
                f"{urllib.parse.quote(clean_q)}&format=json&utf8=1&srlimit=4"
            )
            req = urllib.request.Request(
                wiki_url,
                headers={"User-Agent": "DP42-SovereignCognitiveOrganism/1.0 (NVIDIA RTX 5070; Research)"},
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_items = data.get("query", {}).get("search", [])
                for item in search_items:
                    snippet = self.clean_text(item.get("snippet", ""))
                    title = item.get("title", "")
                    if title and snippet:
                        results.append({
                            "source": "Wikipedia",
                            "title": title,
                            "snippet": snippet,
                        })
        except Exception as ex:
            logger.warning(f"Wikipedia search failed for '{clean_q}': {ex}")

        # 2. Fallback: DuckDuckGo Instant Answer API if results are few
        if len(results) < 2:
            try:
                ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
                req = urllib.request.Request(
                    ddg_url,
                    headers={"User-Agent": "DP42-SovereignCognitiveOrganism/1.0"},
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    ddg_data = json.loads(resp.read().decode("utf-8"))
                    abstract = ddg_data.get("AbstractText", "").strip()
                    heading = ddg_data.get("Heading", "Összefoglaló")
                    if abstract:
                        results.append({
                            "source": "DuckDuckGo Instant Answer",
                            "title": heading,
                            "snippet": abstract,
                        })
            except Exception as ex:
                logger.debug(f"DuckDuckGo fallback failed: {ex}")

        return results

    def forage_for_question(self, question: str) -> Dict[str, Any]:
        """
        Synchronous / Thread worker:
        1. Formulates search query from question.
        2. Queries web directly.
        3. Formulates structured knowledge payload.
        4. Injects directly into SensoryBridge.
        """
        with self._lock:
            self.is_foraging = True
            self.last_question = question
            self.last_query = self._extract_clean_search_terms(question)
            self.last_results = []
            self.last_summary = "Keresés folyamatban az interneten..."

        logger.info(f"Epistemic Web Forager initiated search for: '{self.last_query}'")
        t0 = time.perf_counter()

        # Execute web query
        found = self.search_web_direct(self.last_query)
        elapsed = time.perf_counter() - t0

        if not found:
            # Broader retry if too specific
            fallback_query = "ősrobbanás univerzum csillagok"
            found = self.search_web_direct(fallback_query)

        with self._lock:
            self.last_results = found
            self.total_searches += 1
            self.last_search_timestamp = time.time()
            self.is_foraging = False

            if found:
                summary_parts = []
                for item in found[:3]:
                    summary_parts.append(f"• {item['title']}: {item['snippet']}")
                self.last_summary = "\n".join(summary_parts)
            else:
                self.last_summary = "Nem érkezett közvetlen találat a lekérdezésre."

        # Ingest directly into the child's sensory cortex
        if found:
            top = found[0]
            sensory_payload = (
                f"[INTERNETES KUTATÁSI EREDMÉNY] A kérdésedre ezt találtam a weben: "
                f"'{top['title']}' — {top['snippet']}. "
                f"Ez segít megérteni a világ működését."
            )
            logger.info(f"Injecting web research knowledge into SensoryBridge: {top['title']}")
            self.organism.sensory_bridge.ingest_audio_text(sensory_payload, valence=0.90)

        return {
            "question": question,
            "query": self.last_query,
            "results_count": len(found),
            "results": found,
            "elapsed_s": round(elapsed, 2),
        }

    def _extract_clean_search_terms(self, text: str) -> str:
        """Strips conversational and meta filler words into core semantic search terms."""
        cleaned = re.sub(r'["\'\?\!\:\,\.]', " ", text)
        meta_fillers = {
            "hogy", "vajon", "miért", "mikor", "milyen", "lehet", "tudom", "kérlek",
            "szeretném", "megtudni", "keresni", "egy", "az", "a", "legfontosabb",
            "kérdésed", "kérdés", "kérdésem", "látottak", "alapján", "amire",
            "azonnal", "választ", "választak", "szeretnél", "mondd", "akarok",
            "akarsz", "tudni", "filmet", "film", "videó", "videót", "azonban",
            "esetben", "kapcsán", "vagy", "nem", "van", "meg", "fel", "itt", "most"
        }
        words = [w for w in cleaned.split() if w.lower() not in meta_fillers and len(w) > 2]
        if words:
            return " ".join(words[:5])
        return "univerzum csillagok intelligencia"
