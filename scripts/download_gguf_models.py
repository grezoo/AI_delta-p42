"""
ΔP-42 Sovereign Cognitive Architecture
Script: scripts/download_gguf_models.py
Author: Stage-Gate 5 Delivery

Direct, non-interactive downloader for the 5 physical GGUF models from Hugging Face Hub:
1. qwen2.5-1.5b-instruct-q4_k_m.gguf (~986 MB) -> PFC Executive Cortex
2. qwen2.5-0.5b-instruct-q4_k_m.gguf (~398 MB) -> Inner Monologue & Sensorimotor
3. Llama-3.2-1B-Instruct-Q4_K_M.gguf (~774 MB) -> Social Mirror Cortex
4. SmolLM2-360M-Instruct-Q4_K_M.gguf (~258 MB) -> Doubt & Critic Cortex
5. all-MiniLM-L6-v2-Q4_K_M.gguf (~65 MB)      -> Dream Embedding Cortex

Total download volume: ~2.48 GB.
Destination: C:\\GOOGLE PROJEKTEK\\ai_nursery\\models\\gguf\\
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEST_DIR = Path("C:/GOOGLE PROJEKTEK/ai_nursery/models/gguf")
DEST_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    {
        "name": "Dream Embedding (All-MiniLM-L6-v2)",
        "filename": "all-MiniLM-L6-v2-Q4_K_M.gguf",
        "url": "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf",
        "min_size_mb": 60.0,
    },
    {
        "name": "Critic / Falsification (SmolLM2-360M)",
        "filename": "SmolLM2-360M-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
        "min_size_mb": 240.0,
    },
    {
        "name": "Inner Speech & Sensorimotor (Qwen2.5-0.5B)",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "min_size_mb": 370.0,
    },
    {
        "name": "Social Mirror (Llama-3.2-1B)",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "min_size_mb": 730.0,
    },
    {
        "name": "PFC Strategy (Qwen2.5-1.5B)",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "min_size_mb": 940.0,
    },
]

def download_file(url: str, dest_path: Path, min_size_mb: float) -> bool:
    if dest_path.exists():
        actual_size_mb = dest_path.stat().st_size / (1024 * 1024)
        if actual_size_mb >= min_size_mb:
            print(f"[OK - ALREADY EXISTS] {dest_path.name} ({actual_size_mb:.1f} MB)")
            return True
        else:
            print(f"[INCOMPLETE] {dest_path.name} is only {actual_size_mb:.1f} MB. Re-downloading...")
            dest_path.unlink()

    temp_path = dest_path.with_suffix(".tmp")
    if temp_path.exists():
        temp_path.unlink()

    print(f"\n[DOWNLOADING] {dest_path.name}")
    print(f"  URL: {url}")
    print(f"  Target: {dest_path}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DP42-Nursery-Downloader",
    }
    req = urllib.request.Request(url, headers=headers)

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024 * 2  # 2 MB chunks

            with open(temp_path, "wb") as f:
                last_print = 0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print >= 2.0 or downloaded == total_size:
                        pct = (downloaded / total_size * 100.0) if total_size > 0 else 0.0
                        mb = downloaded / (1024 * 1024)
                        speed = mb / max(now - t0, 0.001)
                        print(f"  Progress: {mb:.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="\r")
                        last_print = now

            print(f"\n  Finished downloading {dest_path.name} in {time.time() - t0:.1f}s")
            temp_path.rename(dest_path)
            return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {dest_path.name}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

def main() -> None:
    print("=" * 70)
    print("ΔP-42 PHYSICAL GGUF MODEL DOWNLOADER")
    print(f"Target Directory: {DEST_DIR}")
    print("=" * 70)

    success_count = 0
    for idx, item in enumerate(MODELS, 1):
        print(f"\n--- Model {idx}/{len(MODELS)}: {item['name']} ---")
        dest = DEST_DIR / item["filename"]
        ok = download_file(item["url"], dest, item["min_size_mb"])
        if ok:
            success_count += 1
        else:
            print(f"Failed to download model {item['name']}. Stopping.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"ALL {success_count}/{len(MODELS)} PHYSICAL GGUF MODELS DOWNLOADED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
