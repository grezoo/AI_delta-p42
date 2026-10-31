"""
ΔP-42 Sovereign Cognitive Architecture - Stage-Gate 3 Memory Subsystem
Canonical memory hygiene, NREM Two-Phase Commit consolidation, 2.0s watchdog supervisor,
sleep hysteresis Schmitt trigger, and REM dream simulation sandbox.
"""

# Hippocampus NREM & 2PC Consolidation
from memory.hippocampus_nrem import (
    CommitResult,
    CommitState,
    ConsolidationResult,
    DEFAULT_MAX_EPISODES,
    DEFAULT_RAM_LIMIT_BYTES,
    DEFAULT_SLEEP_THRESHOLD,
    EpisodeRecord,
    EpisodicBuffer,
    EpisodicMemoryRecord,
    HippocampusMemoryException,
    HippocampusNREM,
    IntegrityProbeException,
    BufferOverflowException,
    PROBE_COSINE_THRESHOLD,
    TransactionAbortedException,
    TransactionStatus,
)

# Hard Real-Time 2.0s Watchdog Supervisor
from memory.nrem_watchdog import (
    DEFAULT_CHECK_INTERVAL_SEC,
    DEFAULT_WATCHDOG_TIMEOUT_SEC,
    FAULT_CODE_IO_TIMEOUT_STALL,
    MAX_REALTIME_LATENCY_MARGIN_SEC,
    NREMWatchdog,
    TimeoutFault,
    WatchdogAbortEvent,
    WatchdogState,
    WatchdogTimeoutError,
)

# Sleep Hysteresis Schmitt Trigger
from memory.sleep_hysteresis import (
    C_DEADBAND_WIDTH,
    C_SLEEP_THRESHOLD,
    C_WAKE_THRESHOLD,
    CognitiveState,
    DEFAULT_DEADBAND_WIDTH,
    DEFAULT_SLEEP_THRESHOLD as HYST_DEFAULT_SLEEP_THRESHOLD,
    DEFAULT_WAKE_THRESHOLD,
    HysteresisSnapshot,
    SleepHysteresis,
    SleepState,
)

# REM Dream Simulation Sandbox
from memory.rem_sandbox import (
    DEFAULT_MAX_LOSS_DEGRADATION,
    DreamResult,
    DreamVerdict,
    KnowledgeGraphSandbox,
    LOSS_EPSILON,
    MAX_DREAM_DEGRADATION_PCT,
    MOTOR_ATONIA_CLAMP,
    REMSandbox,
    REMSimulationResult,
)

# Canonical Subsystem Constants
WATCHDOG_TIMEOUT_SECONDS: float = 2.0

__all__ = [
    # Hippocampus NREM
    "EpisodeRecord",
    "EpisodicMemoryRecord",
    "EpisodicBuffer",
    "CommitState",
    "TransactionStatus",
    "CommitResult",
    "ConsolidationResult",
    "HippocampusNREM",
    "HippocampusMemoryException",
    "TransactionAbortedException",
    "IntegrityProbeException",
    "BufferOverflowException",
    "DEFAULT_RAM_LIMIT_BYTES",
    "DEFAULT_MAX_EPISODES",
    "PROBE_COSINE_THRESHOLD",
    # NREM Watchdog
    "NREMWatchdog",
    "TimeoutFault",
    "WatchdogAbortEvent",
    "WatchdogTimeoutError",
    "WatchdogState",
    "DEFAULT_WATCHDOG_TIMEOUT_SEC",
    "DEFAULT_CHECK_INTERVAL_SEC",
    "FAULT_CODE_IO_TIMEOUT_STALL",
    "MAX_REALTIME_LATENCY_MARGIN_SEC",
    "WATCHDOG_TIMEOUT_SECONDS",
    # Sleep Hysteresis
    "SleepState",
    "CognitiveState",
    "HysteresisSnapshot",
    "SleepHysteresis",
    "DEFAULT_WAKE_THRESHOLD",
    "DEFAULT_DEADBAND_WIDTH",
    "C_SLEEP_THRESHOLD",
    "C_WAKE_THRESHOLD",
    "C_DEADBAND_WIDTH",
    # REM Sandbox
    "DreamVerdict",
    "DreamResult",
    "REMSimulationResult",
    "KnowledgeGraphSandbox",
    "REMSandbox",
    "DEFAULT_MAX_LOSS_DEGRADATION",
    "MAX_DREAM_DEGRADATION_PCT",
    "MOTOR_ATONIA_CLAMP",
    "LOSS_EPSILON",
]
