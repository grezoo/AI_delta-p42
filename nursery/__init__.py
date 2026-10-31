"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 4
Package: nursery
Author: Milestone 1 Worker (Nursery Subsystem Specialist)

AI Nursery Living Environment Subsystem.
Integrates sensory worldstream ingestion, Day 0 cold start bootstrap ramp-up,
Caretaker pedagogical control channel, closed-loop environmental consequence evaluation,
6-model cortical chorus orchestration, and the 3-arm comparative ontogenetic falsification experiment.

Canonical Version: 0.4.0-gate4
"""

from __future__ import annotations

__version__ = "0.4.0-gate4"

# 1. WorldStream Sensory Ingestion
from nursery.worldstream_server import (
    MAX_PAYLOAD_BYTES,
    VALENCE_MAX,
    VALENCE_MIN,
    WORLD_EVENT_FORMAT,
    WORLD_EVENT_SIZE,
    WORLDSTREAM_CADENCE_HZ,
    WORLDSTREAM_DT,
    EventModality,
    EventPriority,
    ModalityType,
    WorldEvent,
    WorldStreamServer,
)

# 2. Day 0 Cold Start Bootstrap Loader
from nursery.bootstrap_loader import (
    DEFAULT_V0,
    RAMP_UP_DURATION_S,
    RAMP_UP_SIGMOID_K,
    RAMP_UP_SIGMOID_T0,
    THETA_0_DEG,
    THETA_0_RAD,
    THETA_BOOTSTRAP_DEG,
    THETA_BOOTSTRAP_RAD,
    THETA_GATE_DEG,
    THETA_GATE_RAD,
    V0_CANONICAL,
    BootstrapLoader,
    BootstrapProfile,
    ConvergenceResult,
)

# 3. Caretaker Control-Channel Console & Pedagogical Model
from nursery.caretaker_console import (
    BASE_GATE_DEG,
    BASE_GATE_RAD,
    DELTA_THETA_ILLEM_MAX_DEG,
    DELTA_THETA_ILLEM_MAX_RAD,
    EMERGENCY_WCET_MAX_MS,
    CaretakerConsole,
    CaretakerTelemetry,
    PedagogicalStage,
    SocialFeedbackRecord,
)

# 4. Reaction Engine & Environmental Consequences
from nursery.reaction_engine import (
    DISCHARGE_TARGET_DEG,
    DISCHARGE_THRESHOLD_DEG,
    EMERGENCY_THREAT_THRESHOLD,
    ETA_GAIN_DEFAULT,
    NEED_QUENCH_TIME_MS,
    PRIMARY_ACTION_TARGET,
    SURROGATE_ACTION_TARGET,
    ConsequenceResult,
    DischargeEvent,
    ReactionEngine,
)

# 5. Cortical Chorus Ensemble
from nursery.cortex_ensemble import (
    CORTEX_SUBSYSTEM_NAMES,
    PFC_GOAL_TEMPLATES,
    RUNTIME_VRAM_GB,
    TOTAL_PARAMS,
    TOTAL_WEIGHTS_GB,
    CortexEnsemble,
    CorticalMessage,
    EnsembleState,
    GGUF_CHORUS_ROSTER,
)

# 6. GGUF Cortical Runtime (Stage-Gate 5)
from nursery.gguf_runtime import (
    CorticalInferenceRequest,
    CorticalInferenceResult,
    CorticalModelWorker,
    CorticalTelemetrySnapshot,
    GGUFCortexChorusRuntime,
)

# 7. Sensory Bridge & Pedagogical Loop Controller (Stage-Gate 5)
from nursery.sensory_bridge import (
    InteractionTurnRecord,
    PedagogicalLoopController,
    SensoryBridge,
)

# 8. Ethological Long-Horizon Telemetry (Stage-Gate 5)
from nursery.etology_telemetry import (
    EthologySample,
    EthologyStabilitySummary,
    EthologyTelemetryRecorder,
)

# 9. Long-Horizon 3-Arm Comparative Ontogenesis Experiment (DVP-12)
from nursery.three_arm_experiment import (
    DVP12_MAX_P_VALUE,
    DVP12_MIN_ADAPTATION_RATIO,
    MIN_TRIALS_FOR_SIGNIFICANCE,
    R_INIT_EXPECTED_ARM_C,
    R_INIT_THRESHOLD_ARM_A,
    W_EFF,
    W_HOMEO,
    W_TASK,
    ArmResult,
    ArmTelemetry,
    ArmType,
    ExperimentSummary,
    FalsificationReport,
    ThreeArmExperiment,
)

__all__ = [
    # Metadata
    "__version__",
    # WorldStream
    "EventModality",
    "ModalityType",
    "EventPriority",
    "WORLDSTREAM_CADENCE_HZ",
    "WORLDSTREAM_DT",
    "WORLD_EVENT_SIZE",
    "MAX_PAYLOAD_BYTES",
    "VALENCE_MIN",
    "VALENCE_MAX",
    "WORLD_EVENT_FORMAT",
    "WorldEvent",
    "WorldStreamServer",
    # Bootstrap
    "DEFAULT_V0",
    "V0_CANONICAL",
    "THETA_0_DEG",
    "THETA_0_RAD",
    "THETA_BOOTSTRAP_DEG",
    "THETA_BOOTSTRAP_RAD",
    "THETA_GATE_DEG",
    "THETA_GATE_RAD",
    "RAMP_UP_DURATION_S",
    "RAMP_UP_SIGMOID_K",
    "RAMP_UP_SIGMOID_T0",
    "BootstrapProfile",
    "ConvergenceResult",
    "BootstrapLoader",
    # Caretaker Console
    "PedagogicalStage",
    "DELTA_THETA_ILLEM_MAX_DEG",
    "DELTA_THETA_ILLEM_MAX_RAD",
    "EMERGENCY_WCET_MAX_MS",
    "BASE_GATE_DEG",
    "BASE_GATE_RAD",
    "SocialFeedbackRecord",
    "CaretakerTelemetry",
    "CaretakerConsole",
    # Reaction Engine
    "ETA_GAIN_DEFAULT",
    "DISCHARGE_THRESHOLD_DEG",
    "DISCHARGE_TARGET_DEG",
    "EMERGENCY_THREAT_THRESHOLD",
    "NEED_QUENCH_TIME_MS",
    "PRIMARY_ACTION_TARGET",
    "SURROGATE_ACTION_TARGET",
    "ConsequenceResult",
    "DischargeEvent",
    "ReactionEngine",
    # Cortex Ensemble
    "TOTAL_PARAMS",
    "TOTAL_WEIGHTS_GB",
    "RUNTIME_VRAM_GB",
    "CORTEX_SUBSYSTEM_NAMES",
    "PFC_GOAL_TEMPLATES",
    "CorticalMessage",
    "EnsembleState",
    "GGUF_CHORUS_ROSTER",
    "CortexEnsemble",
    # GGUF Runtime (Stage-Gate 5)
    "CorticalInferenceRequest",
    "CorticalInferenceResult",
    "CorticalTelemetrySnapshot",
    "CorticalModelWorker",
    "GGUFCortexChorusRuntime",
    # Sensory Bridge & Pedagogical Loop (Stage-Gate 5)
    "InteractionTurnRecord",
    "SensoryBridge",
    "PedagogicalLoopController",
    # Ethological Telemetry (Stage-Gate 5)
    "EthologySample",
    "EthologyStabilitySummary",
    "EthologyTelemetryRecorder",
    # 3-Arm Experiment
    "ArmType",
    "DVP12_MIN_ADAPTATION_RATIO",
    "DVP12_MAX_P_VALUE",
    "R_INIT_THRESHOLD_ARM_A",
    "R_INIT_EXPECTED_ARM_C",
    "MIN_TRIALS_FOR_SIGNIFICANCE",
    "W_TASK",
    "W_HOMEO",
    "W_EFF",
    "ArmResult",
    "ArmTelemetry",
    "ExperimentSummary",
    "FalsificationReport",
    "ThreeArmExperiment",
]
