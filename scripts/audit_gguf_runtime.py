"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 5
Module: scripts.audit_gguf_runtime
Author: Stage-Gate 5 AI Nursery Team

Executes an independent external code audit via Groq Cloud API (openai/gpt-oss-120b)
for nursery/gguf_runtime.py. Evaluates:
1. Canonical Architecture Budget: 6 models, <= 2.90 GB VRAM (2.88 GB), 4.17B parameters.
2. Zero Downgrade Mandate ("A fagyis lopás szigorú tilalma"): genuine threading, genuine Kuramoto phase metrics.
3. Lock-free CDC Ring Buffer integration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = REPO_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

def run_audit() -> None:
    target_file = REPO_ROOT / "nursery" / "gguf_runtime.py"
    output_file = SCRATCH_DIR / "groq_audit_gguf_runtime.md"

    if not target_file.exists():
        print(f"Error: target file {target_file} does not exist.")
        sys.exit(1)

    code_content = target_file.read_text(encoding="utf-8")

    spec_text = """
REFERENCIA SPECIFIKÁCIÓK:
1. DELTA_P42_MUKODESI_ELVEK.md (X. Fejezet) & megvalósitási terv.md (3.1..3.2):
   - 6 aszinkron kortikális modell: PFC (1.5B, 986 MB), Belső Beszéd (0.5B, 398 MB), Szociális Tükör (1B, 774 MB),
     Kétely/Kritikus (360M, 258 MB), Szenzomotoros Fordító (0.5B, 398 MB), Álom Beágyazó (All-MiniLM-L6-v2, 65 MB).
   - Teljes VRAM keret: <= 2.90 GB (2.88 GB weights).
   - Összesen: 4.17B paraméter.
2. Kuramoto Fázisszinkronizáció és Fizikai Stressz Dinamika:
   - A modellek inferencia-késleltetései (delta t_i) közvetlenül határozzák meg a fázisszögeket: theta_i = omega_i * delta t_i.
   - Valós idejű Kuramoto rendparaméter r(t) számítás a fázisszórás mérésére.
3. Determinisztikus C++20 CDCRingBuffer és CognitivePacket integráció (zero-mutex, thread-safe queue dispatch).
4. Zero Downgrade Mandate ("A fagyis lopás szigorú tilalma"): zéró csaló stub, zéró hardkódolt kamu érték.
"""

    prompt = f"""
Te vagy a ΔP-42 Szuverén Kognitív Architektúra független, szigorú és kíméletlen vezető rendszer-auditora.
A feladatod a Stage-Gate 5 'nursery/gguf_runtime.py' moduljának formai, matematikai és mérnöki auditja.

{spec_text}

VIZSGÁLANDÓ FORRÁSKÓD:
```python
{code_content}
```

KÖVETELMÉNYEK AZ AUDIT-JELENTÉSHEZ:
1. Tételes követelmény-összevetés (mind a 6 modell jelenléte, VRAM keret, aszinkron szálkezelés, Kuramoto r(t)).
2. Csalás- és lebutítás-vizsgálat ("A fagyis lopás szigorú tilalma").
3. Végleges minősítés: Kötelező megadni a záró sorban a következőt pontosan:
**VERDICT: PASS** vagy **VERDICT: FAIL**
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Te egy szigorú, kíméletlen vezető kiber-biológiai és beágyazott rendszer-auditor vagy."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    req = urllib.request.Request(GROQ_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    print(f"Submitting audit request for nursery/gguf_runtime.py to Groq Cloud ({MODEL})...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                output_file.write_text(content, encoding="utf-8")
                print(f"Audit completed successfully! Saved to {output_file}")
                
                # Check verdict
                if "**VERDICT: PASS**" in content:
                    print(">>> AUDIT RESULT: PASS")
                    sys.exit(0)
                else:
                    print(">>> AUDIT RESULT: FAIL or WARNING")
                    sys.exit(1)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                sys.exit(1)

if __name__ == "__main__":
    run_audit()
