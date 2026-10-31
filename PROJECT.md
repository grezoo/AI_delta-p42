# Project: ΔP-42 Sovereign Cognitive Core (Gate 1)

## Architecture
The ΔP-42 Sovereign Cognitive Core (Gate 1) is a non-equilibrium thermodynamic engine operating as a coupled Hybrid Relaxation Oscillator (FitzHugh-Nagumo dynamical class). It overcomes the passive "prompt-dependent" failure mode ($CO_2$ dead-zone) of conventional AI by generating continuous endogenous tension through basal metabolic entropy accumulation, driving autonomous phase advancement, smooth motor gating, and closed limit cycle dynamics.

### Architectural Component Diagram
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SENSORY & COMPUTATIONAL BYPASS                        │
│    x_ext (Sensory Input) ──┐                                                │
│    c_pred (Prediction)   ──┴──► [ Thalamus AGC ] ──► delta(t) (RMS Dissonance)
│                                        │                                    │
│                                        ▼ g_AGC                              │
│                               Mixed Synesthetic Stream                      │
└────────────────────────────────────────┼────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────┐      ┌──────────────────────────────────────────┐
│     MASLOW INTEGRATOR     │      │               THETA ENGINE               │
│ Exact Analytical Integrator│      │          50 Hz Phase Oscillator          │
│ V_i(t+dt) = V_i*e^(-l*dt) │      │  dot(theta) = kv*||V||_1 - K*sin(th-th0) │
│    + V_inf*(1 - e^(-l*dt))│      │                 + alpha*delta - gamma*a  │
│  (lambda_base = 0.1386)   │      │         (Strict SI Radian Engine)        │
└─────────────┬─────────────┘      └────────────────────┬─────────────────────┘
              │                                         │
              │ MaslowSnapshot                          │ theta(t) [rad]
              │ (||V||_1)                               │
              ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BASAL GANGLIA MOTOR GATE                           │
│  Continuous C0 Jerk-Free Gating: a_gate = 2 * (sigma(...) - 0.5) * u        │
│  Threshold: theta_gate = 18.0 deg (0.3142 rad), Bandwidth: Delta_theta = 0.35 deg
│  Biological Interlocks: Sleep Lock (REM/NREM) & Panic Lock (Amygdala)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ a_motor(t)
                                       ▼
                       Quenching Feedback: R_i (Maslow) & -gamma (Theta)
```

## Feature Inventory
Every feature extracted from the foundational specifications is mapped to a designated milestone.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 50 Hz Timebase & Coupled Phase Velocity ODE | Evaluates $\dot{\theta} = k_v \|\mathbf{V}\|_1 - K \sin(\theta - \theta_0) + \alpha \delta - \gamma a_{\text{motor}}$ at $dt=20\text{ ms}$ strictly in SI radians. | M1 | Thesis III.1, Request R1 |
| 2 | SI Radian Angular Architecture & Physical Constants | Radians used exclusively for internal dynamics (`DEG2RAD = np.pi / 180.0`, $\theta_0=15.0^\circ$, $\theta_{gate}=18.0^\circ$, $\Delta\theta_{sav}=0.35^\circ$). | M1 | Thesis III.1, Plan Review |
| 3 | Kuramoto Homeostatic Spring Restoration | Stable resting attractor $-K \sin(\theta - \theta_0)$ with $K = 0.80\text{ rad/s}$ pulling back to $\theta_0$. | M1 | Thesis III.1 |
| 4 | Day 0 Pre-charge Bootstrap Kick | Initial cold-start phase $\theta(0) = 15.5^\circ$ ($0.270526\text{ rad}$) breaks resting symmetry ($+0.5^\circ$). | M1 | Principles X.2 |
| 5 | Thalamic Excitatory Forward Coupling | $+\alpha \delta(t)$ forward acceleration with $\alpha = 0.50\text{ rad/s}$ responding to predictive dissonance. | M1 | Thesis III.1 |
| 6 | Motor Quenching Phase Braking | $-\gamma a_{\text{motor}}(t)$ rapid phase cooling with $\gamma = 0.40\text{ rad/s}$ upon action trigger. | M1 | Thesis III.1 |
| 7 | Multi-Rate Sync Bridge (10 Hz -> 50 Hz) | Thread-safe, drop-free latching of 10 Hz `MaslowSnapshot` into 50 Hz Theta step loop. | M1 | Megvalósítási Terv Phase 1 |
| 8 | 10 Hz Exact Analytical Exponential Integrator | $V_i(t+dt) = V_i e^{-\lambda_i dt} + V_i^\infty (1 - e^{-\lambda_i dt})$ guaranteeing $V_i \in [0, 1]$ without artificial clipping. | M1 | Thesis III.2, Request R1 |
| 9 | Basal Entropy Production Rate ($\lambda_{base}$) | Autonomous basal drive $\lambda_{\text{base}} = 0.1386\text{ s}^{-1}$ ($\tau = 7.215\text{ s}$, half-life $5.0\text{ s}$) accumulating tension in idle. | M1 | Thesis I.3, III.2 |
| 10 | Cognitive Load Coupling ($\lambda_{work}$) | Need growth modulation $G_i(C) = \lambda_{base} + \lambda_{work} \cdot C(t)$ under cortical load $C(t) \in [0, 1]$. | M1 | Thesis III.2 |
| 11 | Environmental Quenching Dynamics ($R_i$) | Need discharge $R_i = \eta_{\text{gain}} \cdot a_{\text{motor}} \cdot E_{\text{env}}$ collapsing deficiency vacuum upon successful action. | M1 | Thesis III.2 |
| 12 | 5D Maslow Needs Vector Space | Vector $\mathbf{V} = [V_{metab}, V_{bizt}, V_{tarsas}, V_{komp}, V_{megism}]^T \in [0, 1]^5$. | M1 | Principles I.1 |
| 13 | Day 0 Pre-charge Need Vector Initialization | Non-zero baseline $\mathbf{V}(0) = [0.15, 0.10, 0.25, 0.10, 0.30]^T$ to seed active dynamics. | M1 | Principles X.2 |
| 14 | Immutable Atomic `MaslowSnapshot` Dataclass | Frozen, read-only dataclass (`flags.writeable = False`) carrying timestamps, L1 norm, dominant need. | M1 | Megvalósítási Terv Phase 1 |
| 15 | Normalized Dimension-Independent RMS Dissonance | $\delta(t) = \frac{1}{\sqrt{d_{sens}}} \|\mathbf{x}_{ext} - \mathbf{c}_{pred}\|_2 \in [0, 1]$ invariant to sensor dimension $d_{sens}$. | M2 | Principles II.1, Request R1 |
| 16 | Dynamic AGC Gain Scaling ($g_{AGC}$) | Automatic Gain Control $g_{AGC} = \frac{1}{1 + 0.5 \|\mathbf{V}\|_1} \in [0.10, 1.00]$ preventing excitotoxic runaway. | M2 | Principles II.1, Request R1 |
| 17 | Predictive Clipping & Cortical Compute Bypass FSM | Subcortical gating: if $\delta < 0.15$, enter low-power bypass; if $\delta \ge 0.15$, trigger `Ping!` interrupt. | M2 | Principles II.1 |
| 18 | Continuous $C^0$ Jerk-Free Striatum Motor Gate | $a_{gate}(\theta) = 2(\sigma((\theta - \theta_{gate})/\Delta\theta_{sav}) - 0.5) \cdot \mathbf{u}$ with $\Delta\theta_{sav} = 0.35^\circ$ ($0.006109\text{ rad}$). | M2 | Thesis III.3, Request R1 |
| 19 | Biological Motor Interlocks (Sleep & Panic Locks) | Absolute hardware overrides forcing $a_{\text{motor}} \equiv 0.000$ during REM/NREM sleep or Amygdala alarm. | M2 | Principles IV.1 |
| 20 | 6 Micro-GGUF Chorus Integration Interfaces | Typed architecture hooks, prompt envelopes, and tensor buffer structures for the 6-model chorus. | M2 | Megvalósítási Terv |
| 21 | 65536-Slot CDC Lock-Free Ring Buffer Specification | Memory-aligned 64-bit `CognitivePacket` ring buffer interfaces with atomic head/tail pointers and CRC-64. | M2 | Megvalósítási Terv |
| 22 | Cerebellar Spiking Reservoir & Purkinje LTD Plasticity | Interface hooks and learning rate dynamics for Purkinje LTD error-driven motor adaptation. | M2 | Megvalósítási Terv |
| 23 | DVP-01 Criterion 1: Closed Limit Cycle Stability | 10,000-step ($200\text{ s}$) test verifying $T_{\text{cycle}} \in [1.0, 5.0]\text{ s}$ and $CV(A_{peak}) \le 0.05$. | M3 | Plan Review DVP-01, Request R4 |
| 24 | DVP-01 Criterion 2: Idle Motor Ignition $\le 10.0\text{ s}$ | Automated verification proving phase tipping and motor firing occur in $\le 10.0\text{ s}$ with zero prompt. | M3 | Plan Review DVP-01, Request R4 |
| 25 | DVP-01 Criterion 3: Autonomous Initiation Rate $R_{init}$ | 100-run Monte Carlo verification requiring $R_{init} = N_{success}/100 \ge 0.85$ without external trigger. | M3 | Plan Review DVP-01, Request R4 |
| 26 | DVP-01 Unified Test Engine (`core/gate1_verification.py`) | Reproducible standalone CLI script executing all 3 criteria and emitting structured PASS/FAIL metrics. | M3 | Megvalósítási Terv, Request R4 |
| 27 | Groq 120B Independent Code Audit Pipeline | Automated adversarial code review via `openai/gpt-oss-120b` checking all 5 modules against thesis specs. | M4 | Request R2, Plan Review |
| 28 | Audit Log Persistence in `scratch/` | Mandatory generation and persistence of clean audit logs in `scratch/` with zero unaddressed divergence. | M4 | Request R2 |
| 29 | Comprehensive E2E Verification & Sentinel Gate Signoff | Verification that all unit tests, DVP-01 suite, and Groq audits pass with 100% compliance. | M4 | Request Acceptance Criteria |

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | Core Mathematical Engines | `core/theta_engine.py`, `core/maslow_integrator.py` (Features 1–14) | none | DONE |
| M2 | Sensory & Motor Gating + System Interfaces | `core/thalamus_agc.py`, `core/basal_ganglia_gate.py` (Features 15–22) | M1 contracts | DONE |
| M3 | DVP-01 Quantitative Verification Engine | `core/gate1_verification.py` (Features 23–26) | M1, M2 | DONE |
| M4 | Groq 120B Audit & Full Gate 1 Acceptance | Groq 120B audit pipeline, `scratch/` logs, full DVP-01 execution (Features 27–29) | M1, M2, M3 | DONE |

## Interface Contracts

### 1. `core/maslow_integrator.py` -> `core/theta_engine.py` & others
- **Snapshot Dataclass**:
  ```python
  @dataclass(frozen=True)
  class MaslowSnapshot:
      timestamp_ns: int
      vector: np.ndarray        # shape=(5,), dtype=np.float64, read-only (flags.writeable=False)
      l1_norm: float            # sum(vector) in [0.0, 5.0]
      l2_norm: float            # sqrt(sum(vector^2)) in [0.0, sqrt(5)]
      dominant_idx: int         # 0..4
      dominant_name: str        # 'metab', 'safety', 'social', 'comp', 'cog'
  ```
- **Integrator Class**:
  ```python
  class MaslowIntegrator:
      def __init__(self, dt: float = 0.1, lambda_base: float = 0.1386, lambda_work: float = 0.20, eta_gain: float = 2.0)
      def step(self, C: float = 0.0, a_motor: float = 0.0, E_env: Optional[np.ndarray] = None) -> MaslowSnapshot
      def get_snapshot(self) -> MaslowSnapshot
  ```

### 2. `core/thalamus_agc.py` -> `core/theta_engine.py` & downstream
- **Thalamus Output Dataclass**:
  ```python
  @dataclass(frozen=True)
  class ThalamusOutput:
      timestamp_ns: int
      delta: float              # RMS dissonance in [0.0, 1.0]
      g_agc: float              # Dynamic AGC gain in [0.10, 1.00]
      emit_ping: bool           # True if delta >= epsilon (0.15)
      bypass_active: bool       # True if delta < epsilon (0.15)
      s_mixed: np.ndarray       # Synesthetic mixed output stream
  ```
- **Thalamus Class**:
  ```python
  class ThalamusAGC:
      def __init__(self, epsilon_threshold: float = 0.15, kappa_agc: float = 0.50, beta_0: float = 0.10)
      def process(self, x_ext: np.ndarray, c_pred: np.ndarray, V: np.ndarray, theta: float, sim_time: float) -> ThalamusOutput
  ```

### 3. `core/basal_ganglia_gate.py` -> `core/theta_engine.py` & actuators
- **Gate Output Dataclass**:
  ```python
  @dataclass(frozen=True)
  class GateDecision:
      timestamp_ns: int
      a_motor: float            # Continuous motor activation in [0.0, 1.0]
      go_winner: bool           # True if Go pathway won striatal competition
      is_locked: bool           # True if Sleep or Panic lock is active
      theta_rad: float          # Current phase in SI radians
      theta_deg: float          # Current phase in degrees
  ```
- **Basal Ganglia Gate Class**:
  ```python
  class BasalGangliaGate:
      def __init__(self, theta_gate_deg: float = 18.0, delta_theta_deg: float = 0.35)
      def evaluate(self, theta_rad: float, u: float = 1.0, go_signal: float = 1.0, nogo_signal: float = 0.0, lock_sleep: bool = False, panic_amy: bool = False) -> GateDecision
  ```

### 4. `core/theta_engine.py` (Master Subcortical Loop)
- **Engine Class**:
  ```python
  class ThetaEngine:
      def __init__(self, dt: float = 0.02, kv: float = 0.50, K: float = 0.80, alpha: float = 0.50, gamma: float = 0.40, theta_0_deg: float = 15.0, theta_gate_deg: float = 18.0, theta_bootstrap_deg: float = 15.5)
      def step(self, V_l1: float, delta: float = 0.0, a_motor: float = 0.0) -> Tuple[float, float, float]  # returns (theta_rad, theta_deg, dot_theta)
  ```

## Code Layout
- `core/`: Primary implementation modules
  - `core/__init__.py`: Package export and version definition (`__version__ = "0.1.0-gate1"`)
  - `core/theta_engine.py`: 50 Hz SI radian relaxation phase oscillator (Owned by M1 Worker)
  - `core/maslow_integrator.py`: 10 Hz exact analytical exponential integrator (Owned by M1 Worker)
  - `core/thalamus_agc.py`: RMS dissonance, AGC gain & compute bypass (Owned by M2 Worker)
  - `core/basal_ganglia_gate.py`: Bipolar Striatum C0 jerk-free gate & interlocks (Owned by M2 Worker)
  - `core/gate1_verification.py`: Standalone quantitative DVP-01 test runner (Owned by M3 Worker)
- `tests/`: Unit and property tests
  - `tests/test_theta_engine.py`: Unit tests for phase velocity, Kuramoto spring, SI radians
  - `tests/test_maslow_integrator.py`: Unit tests for exponential solver, basal tension, stability
  - `tests/test_thalamus_agc.py`: Unit tests for RMS metric, AGC attenuation, compute bypass
  - `tests/test_basal_ganglia_gate.py`: Unit tests for C0 smoothness, sleep/panic interlocks
  - `tests/test_dvp01_integration.py`: Integration tests for closed-loop limit cycle
- `scripts/`: Operational tools
  - `scripts/run_groq_audit.py`: Independent Groq 120B audit automation script
- `scratch/`: Audit logs and transient outputs
  - `scratch/groq_audit_theta_engine.md`: Groq 120B audit log
  - `scratch/groq_audit_maslow_integrator.md`: Groq 120B audit log
  - `scratch/groq_audit_thalamus_agc.md`: Groq 120B audit log
  - `scratch/groq_audit_basal_ganglia_gate.md`: Groq 120B audit log
  - `scratch/groq_audit_gate1_verification.md`: Groq 120B audit log
