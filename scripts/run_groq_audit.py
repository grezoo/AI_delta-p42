"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Groq 120B Independent Code Audit Runner
Features 27 & 28: Automated adversarial code review via Groq Cloud API (openai/gpt-oss-120b)
evaluating all 5 core modules against foundational specifications:
- DELTA_P42_MESTER_TEZIS.md
- DELTA_P42_MUKODESI_ELVEK.md
- Plan Review.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# UTF-8 stdout configuration
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = REPO_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

MODULES_TO_AUDIT = [
    {
        "name": "theta_engine",
        "file_path": REPO_ROOT / "core" / "theta_engine.py",
        "output_path": SCRATCH_DIR / "groq_audit_theta_engine.md",
        "max_lines": None,
        "spec_refs": """
REFERENCIA SPECIFIKÁCIÓK:
1. DELTA_P42_MESTER_TEZIS.md (III.1):
   - 50 Hz-es (dt=0.02 s) relaxációs fázisoszcillátor szigorúan SI radiánban.
   - Csatolt fázissebesség ODE: dot{theta}(t) = kv * ||V||_1 - K * sin(theta - theta_0) + alpha * delta - gamma * a_motor.
   - Paraméterek: kv=0.50 rad/s, K=0.80 rad/s (Kuramoto szinuszos visszahúzó rugó), alpha=0.50 rad/s (Thalamus disszonancia gyorsítás), gamma=0.40 rad/s (motoros kioltási fázisfék).
2. DELTA_P42_MUKODESI_ELVEK.md (I.1, X.2):
   - theta_0 = 15.0 deg (0.261799 rad, pi/12), theta_gate = 18.0 deg (0.314159 rad, pi/10).
   - Day 0 bootstrap kick: theta(0) = 15.5 deg (0.270526 rad).
   - delta_theta_sav = 0.35 deg (0.006109 rad).
3. Plan Review.md (REQ-MATH-01, REQ-FUNC-02, DVP-01):
   - Multi-rate szinkronizáció a 10 Hz MaslowSnapshot-tal (latch_maslow), thread-safe L1 átvétel.
   - Szigorú típusellenőrzés, numerikus stabilitás, zéró lebutítás, zéró hardkódolt állapot.
""",
    },
    {
        "name": "maslow_integrator",
        "file_path": REPO_ROOT / "core" / "maslow_integrator.py",
        "output_path": SCRATCH_DIR / "groq_audit_maslow_integrator.md",
        "max_lines": None,
        "spec_refs": """
REFERENCIA SPECIFIKÁCIÓK:
1. DELTA_P42_MESTER_TEZIS.md (I.3, III.2):
   - 10 Hz-es (dt=0.10 s) egzakt analitikus exponenciális integrátor:
     V_i(t+dt) = V_i(t) * e^(-lambda_i * dt) + V_i^infty * (1 - e^(-lambda_i * dt)).
   - Növekedési sebesség: G_i(C) = lambda_base + lambda_work * C(t).
   - Bazális entrópiavesztés / feszültségfelépülés: lambda_base = 0.1386 s^-1 (felezési idő t_1/2 = 5.0 s, tau = 7.215 s) üresjáratban (C=0) is működik.
   - Kognitív terhelés: lambda_work = 0.20 s^-1.
   - Környezeti kielégülés kioltási sebesség: R_i = eta_gain * a_motor * E_env (eta_gain = 2.0 s^-1).
   - lambda_i = G_i + R_i, V_i^infty = G_i / lambda_i.
   - Mesterséges clipping (np.clip) NÉLKÜLI matematikai stabilitási garancia: V_i in [0.0, 1.0] konvex kombináció révén.
2. DELTA_P42_MUKODESI_ELVEK.md (I.1, X.2):
   - 5D Maslow szükségletvektor: [metab, safety, social, comp, cog].
   - Day 0 kezdeti feltöltés: [0.15, 0.10, 0.25, 0.10, 0.30].
   - Megváltoztathatatlan atomi MaslowSnapshot dataclass (frozen=True, flags.writeable = False).
3. Plan Review.md (REQ-MATH-03, REQ-COMM-02): Zéró mock, zéró lebutítás.
""",
    },
    {
        "name": "thalamus_agc",
        "file_path": REPO_ROOT / "core" / "thalamus_agc.py",
        "output_path": SCRATCH_DIR / "groq_audit_thalamus_agc.md",
        "max_lines": None,
        "spec_refs": """
REFERENCIA SPECIFIKÁCIÓK:
1. DELTA_P42_MUKODESI_ELVEK.md (I.2, VI.1):
   - Normalizált dimenziófüggetlen RMS disszonancia metrika:
     delta(t) = (1 / sqrt(d_sens)) * ||x_ext - c_pred||_2 in [0.0, 1.0].
   - Dinamikus AGC adaptív erősítés:
     g_AGC = 1.0 / (1.0 + 0.5 * ||V||_1) in [0.10, 1.00] (kappa_agc = 0.50, beta_0 = 0.10).
   - Prediktív clipping és kérgi Compute Bypass állapotgép (FSM):
     * Ha delta < epsilon (0.15): COMPUTE_BYPASS (alacsony fogyasztás, 0 token kérgi kórusnak).
     * Ha delta >= epsilon (0.15): COMPUTE_ACTIVE, Ping! megszakítás PFC felé.
   - Szinesztéziás kevert szenzor-szükséglet tenzor stream (s_mixed).
2. DELTA_P42_MESTER_TEZIS.md (III.1, Megvalósítási Terv 3.1, 3.2):
   - 6 mikro-GGUF modell illesztési pontok és architektúra-leírók (PFC, Inner Speech, Amygdala, Hippocampus, Salience, Motor-Synthesizer).
   - 65536-os CDC lock-free ring buffer specifikáció és csomagstruktúra.
3. Plan Review.md (REQ-AGC-01..03, DVP-02): Zéró lebutítás, valódi tenzorvektorok.
""",
    },
    {
        "name": "basal_ganglia_gate",
        "file_path": REPO_ROOT / "core" / "basal_ganglia_gate.py",
        "output_path": SCRATCH_DIR / "groq_audit_basal_ganglia_gate.md",
        "max_lines": None,
        "spec_refs": """
REFERENCIA SPECIFIKÁCIÓK:
1. DELTA_P42_MESTER_TEZIS.md (III.3, V):
   - Bipoláris Striatum Go/No-Go kapuzás folytonos C0 rántásmentes átmenettel:
     a_gate(theta) = 2.0 * (sigma((theta - theta_gate) / Delta_theta_sav) - 0.5) * u
     ha theta >= theta_gate és Go nyert, különben 0.0.
   - theta_gate = 18.0 deg (0.314159 rad), Delta_theta_sav = 0.35 deg (0.006109 rad).
   - a_gate(theta_gate) = 0.0000 folytonosan indul, rántásmentes C0 onset.
2. DELTA_P42_MUKODESI_ELVEK.md (I.5, IV.1):
   - Biológiai biztonsági reteszek:
     * Sleep Lock (REM/NREM alvás): a_motor === 0.000 unconditionally.
     * Panic Lock (Amygdala riasztás): a_motor === 0.000 unconditionally.
   - Biológiai refrakter kioltási dinamika (limit ciklus visszatérés theta_0 = 15.0 deg-ig).
3. Plan Review.md (REQ-GATE-01..03, DVP-04, DVP-14):
   - Kisagyi Purkinje LTD plaszticitási interfész (20 Hz, anti-hebbian tanulás Delta_W = -eta * error * parallel_fiber).
   - Zéró lebutítás, valódi állapotkezelés.
""",
    },
    {
        "name": "gate1_verification",
        "file_path": REPO_ROOT / "core" / "gate1_verification.py",
        "output_path": SCRATCH_DIR / "groq_audit_gate1_verification.md",
        "max_lines": 460,  # Covers all logic classes and methods; omits trailing CLI printer
        "spec_refs": """
REFERENCIA SPECIFIKÁCIÓK:
1. Plan Review.md (DVP-01 Quantitative Protocol, Stage-Gate 1):
   - Standalone, automata és reprodukálható kvantitatív mérési motor.
   - Összekapcsolja a valós 4 kognitív almagot (ThetaEngine, MaslowIntegrator, ThalamusAGC, BasalGangliaGate).
   - Kritérium 1: Zárt határciklus stabilitás (10,000 lépés / 200 s szimuláció):
     * Ciklusidő T_cycle in [1.0, 5.0] s.
     * Csúcsamplitúdó relatív szórása sigma(A_peak) / mu(A_peak) <= 0.05 (5.0%).
   - Kritérium 2: Üresjárati motoros gyújtás t_ignition <= 10.0 s külső prompt nélkül (CO2 holtzóna cáfolata).
   - Kritérium 3: Autonóm iniciációs ráta Monte Carlo teszt (N=100 független futás sztochasztikus zajjal):
     * R_init = N_sikeres / 100 >= 0.85 (85.0%).
2. DELTA_P42_MESTER_TEZIS.md (V): Zéró csalás, valódi numerikus integráció, nem hardkódolt értékek.
3. Strukturált DVP01Results kimenet és terminál riport.
""",
    },
]


def read_text_file(path: Path, max_lines: Optional[int] = None) -> str:
    """Read a text file with UTF-8 encoding, optionally capped to max_lines."""
    with open(path, "r", encoding="utf-8") as f:
        if max_lines is not None:
            lines = [f.readline() for _ in range(max_lines)]
            return "".join(lines)
        return f.read()


def call_groq_api(
    prompt: str,
    max_tokens: int = 1200,
    system_instruction: str = "Te szigorú vezető kiber-biológiai és AI rendszermérnök, formális szoftverauditor vagy. Tömör, precíz, szabványos audit jelentést készítesz magyar nyelven.",
    max_retries: int = 8,
) -> Dict[str, str]:
    """
    Call Groq Cloud API with model openai/gpt-oss-120b and return reasoning + content.
    Uses reasoning_effort="low" to maximize token budget for complete content generation.
    Handles rate limits (TPM) by dynamically extracting wait times from 429 errors.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DeltaP42Auditor/1.0",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "reasoning_effort": "low",
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    data_bytes = json.dumps(payload).encode("utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(GROQ_URL, data=data_bytes, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choice = resp_data["choices"][0]["message"]
                content = choice.get("content", "")
                reasoning = choice.get("reasoning", "")
                usage = resp_data.get("usage", {})
                return {"content": content, "reasoning": reasoning, "usage": str(usage)}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"  [Attempt {attempt}/{max_retries}] HTTPError {e.code}: {err_body[:200]}")
            if e.code == 429:
                # Extract wait time if present: "please try again in 12.7875s"
                m = re.search(r"try again in ([0-9\.]+)s", err_body, re.IGNORECASE)
                if m:
                    wait_time = float(m.group(1)) + 2.0
                else:
                    wait_time = 20.0 * attempt
                print(f"  Rate limit hit. Sleeping {wait_time:.1f}s for quota replenishment...")
                time.sleep(wait_time)
            elif e.code == 413:
                raise RuntimeError(f"Payload too large for Groq TPM limit: {err_body}")
            elif e.code >= 500:
                time.sleep(5 * attempt)
            else:
                raise RuntimeError(f"Groq API Error {e.code}: {err_body}")
        except Exception as e:
            print(f"  [Attempt {attempt}/{max_retries}] Exception: {e}")
            time.sleep(5 * attempt)

    raise RuntimeError(f"Failed to query Groq API after {max_retries} attempts.")


def build_audit_prompt(
    module_name: str,
    module_code: str,
    module_spec_refs: str,
) -> str:
    """Construct an adversarial audit prompt for Groq 120B."""
    return f"""
# FÜGGETLEN KÓD- ÉS ARCHITEKTÚRA AUDIT: {module_name}
Projekt: ΔP-42 Szuverén Kognitív Architektúra (Gate 1 Core)
Auditor Modell: Groq 120B (openai/gpt-oss-120b)

## Referencia Specifikációk:
{module_spec_refs}

---

## Vizsgálandó Forráskód ({module_name}):
```python
{module_code}
```

---

## AUDITÓRIUMI FELADATOK ÉS KÖVETELMÉNYEK:
Hajts végre szigorú, mérnöki forráskód-auditot az 5 szempont szerint!

VÁLASZ FORMÁTUM (Légy tömör és precíz, kb. 350 szóban fedd le az összes pontot):
1. **Matematikai és Fizikai Hűség**: Pontosak-e az egyenletek, konstansok és SI mértékegységek?
2. **Zéró Lebutítás és "Fagyis Lopás" Mentesség**: Valódi fizikai állapot/szimuláció van-e? (Nincs mock, nincs hardkódolt csalás)
3. **Interfész és Típus-konzisztencia**: Helyesek-e a szignatúrák, immutability, hibakezelés?
4. **Kiber-Biológiai Stabilitás és Reteszek**: Megfelelően véd-e a rendszer? (Sleep/Panic/Interlocks, határciklus stabilitás)
5. **VÉGSŐ MINŐSÍTÉS (VERDICT)**: KÖTELEZŐEN írd ki nagybetűvel: **VERDICT: PASS** vagy **VERDICT: FAIL**, valamint egy 1-2 mondatos összegző indoklást.
"""


def run_all_audits() -> None:
    """Execute Groq 120B audit on all 5 modules and persist markdown reports."""
    print("=" * 80)
    print("DELTA P-42 GATE 1 - GROQ 120B INDEPENDENT CODE AUDIT PIPELINE")
    print(f"Model: {MODEL} | Endpoint: {GROQ_URL} | reasoning_effort: low")
    print("=" * 80)

    all_verdicts: Dict[str, str] = {}

    for idx, mod in enumerate(MODULES_TO_AUDIT, 1):
        name = mod["name"]
        file_path = mod["file_path"]
        out_path = mod["output_path"]
        spec_refs = mod["spec_refs"]
        max_lines = mod.get("max_lines")

        print(f"\n[{idx}/5] Auditing {name} ({file_path.name})...")
        if not file_path.exists():
            print(f"  ERROR: File {file_path} does not exist!")
            all_verdicts[name] = "FAIL (File not found)"
            continue

        code_text = read_text_file(file_path, max_lines=max_lines)
        prompt = build_audit_prompt(
            module_name=file_path.name,
            module_code=code_text,
            module_spec_refs=spec_refs,
        )

        t0 = time.time()
        result = call_groq_api(prompt, max_tokens=1200)
        elapsed = time.time() - t0

        reasoning = result.get("reasoning", "")
        content = result.get("content", "")
        usage = result.get("usage", "")

        # Format full markdown document
        md_content = f"""# Groq 120B Independent Code Audit: `{file_path.name}`

- **Audited Target**: `{file_path.relative_to(REPO_ROOT)}`
- **Auditor Model**: `Groq Cloud API ({MODEL})`
- **Execution Timestamp**: `{time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())}`
- **Audit Duration**: `{elapsed:.2f} s`
- **Token Usage**: `{usage}`

---

## 1. Auditor Reasoning Trace
{reasoning if reasoning else "*No explicit reasoning trace provided by model.*"}

---

## 2. Formal Audit Evaluation & Findings
{content}
"""

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Verdict check
        upper_content = content.upper()
        if "VERDICT: PASS" in upper_content or "**VERDICT: PASS**" in upper_content or "VERDICT: **PASS**" in upper_content:
            verdict = "PASS"
        elif "VERDICT: FAIL" in upper_content or "**VERDICT: FAIL**" in upper_content:
            verdict = "FAIL"
        elif "PASS" in upper_content and "FAIL" not in upper_content.split("5.")[-1]:
            verdict = "PASS"
        else:
            verdict = "CHECK"

        all_verdicts[name] = verdict
        print(f"  Audit completed in {elapsed:.2f} s -> Saved to {out_path.name} | Status: {verdict}")

    print("\n" + "=" * 80)
    print("GROQ 120B AUDIT PIPELINE SUMMARY:")
    for name, verdict in all_verdicts.items():
        print(f"  - {name:25s}: {verdict}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_audits()
