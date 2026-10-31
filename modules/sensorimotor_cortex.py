"""
ΔP-42 Sovereign Cognitive Core (Gate 2) - Sensorimotor Cortex Module
50 Hz Virtual Proprioception, Joint Kinematics Integration, Analytical Forward
Kinematics (FK) & Jacobian, 64-Element CognitivePacket Payload Packaging,
and Real-Time Propagation Latency Monitoring (DVP-15 WCET <= 3.0 ms).

Canonical References:
- DELTA_P42_MESTER_TEZIS.md (Chapter III, V)
- DELTA_P42_MUKODESI_ELVEK.md (Chapter I.7-I.8 lines 225-247, Chapter V.3 lines 397-408)
- Plan Review.md (REQ-ANAT-04, DVP-15, RTM-18)
- ORIGINAL_REQUEST.md (Gate 2 Requirements R2, R4)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import numpy as np

# Canonical Module Identifiers
MODULE_ID: int = 2
MODULE_SENSORIMOTOR: int = 2
MODULE_THALAMUS: int = 2
MODULE_CEREBELLUM: int = 7

# Timing and Kinematics Canonical Constants
CADENCE_HZ: float = 50.0
DT_DEFAULT: float = 1.0 / CADENCE_HZ    # 0.020 s (20.0 ms)
N_JOINTS_DEFAULT: int = 6               # 6-DOF Articulated Manipulator Chain
LATENCY_WCET_THRESHOLD_MS: float = 3.0  # DVP-15 Pass/Fail threshold
PAYLOAD_FLOAT_COUNT: int = 64

# Default Link Lengths (meters): L1 (base height), L2 (shoulder), L3 (elbow), L4 (wrist 1), L5 (wrist 2), L6 (tool)
DEFAULT_LINK_LENGTHS: Tuple[float, ...] = (0.35, 0.30, 0.25, 0.15, 0.10, 0.05)

# Default Joint Angular Limits (radians)
DEFAULT_Q_MIN: Tuple[float, ...] = (-math.pi, -math.pi / 2.0, -2.0 * math.pi / 3.0, -math.pi / 2.0, -math.pi, -math.pi / 2.0)
DEFAULT_Q_MAX: Tuple[float, ...] = (math.pi, math.pi / 2.0, 2.0 * math.pi / 3.0, math.pi / 2.0, math.pi, math.pi / 2.0)
DEFAULT_QDOT_MAX: float = 5.0           # Max joint velocity (rad/s)
DEFAULT_TORQUE_MAX: float = 50.0        # Max joint torque (N*m)


@dataclass(frozen=True)
class ProprioceptiveState:
    """
    Immutable atomic spatial and proprioceptive state snapshot emitted by Sensorimotor Cortex.

    Attributes:
        timestamp_ns: System monotonic timestamp at snapshot generation.
        packet_uuid: Unique RFC-4122 v4 event identifier string.
        joint_positions: Joint angles q in SI radians (shape=(N_joints,)).
        joint_velocities: Joint velocities q_dot in rad/s (shape=(N_joints,)).
        joint_torques: Measured/simulated joint torques tau in N*m (shape=(N_joints,)).
        cartesian_pos: End-effector Cartesian coordinates [x, y, z] in meters (shape=(3,)).
        cartesian_vel: End-effector Cartesian velocity [vx, vy, vz] in m/s (shape=(3,)).
        state_vector: 64-element float32 aligned tensor formatted for CognitivePacket.
        latency_ms: Processing duration in milliseconds for DVP-15 verification.
    """
    timestamp_ns: int
    packet_uuid: str
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    cartesian_pos: np.ndarray
    cartesian_vel: np.ndarray
    state_vector: np.ndarray
    latency_ms: float

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(self.joint_positions)):
            raise ValueError("joint_positions contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.joint_velocities)):
            raise ValueError("joint_velocities contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.cartesian_pos)):
            raise ValueError("cartesian_pos contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.cartesian_vel)):
            raise ValueError("cartesian_vel contains non-finite values (NaN or Inf)")
        if not np.all(np.isfinite(self.state_vector)):
            raise ValueError("state_vector contains non-finite values (NaN or Inf)")
        if self.state_vector.shape != (PAYLOAD_FLOAT_COUNT,):
            raise ValueError(f"state_vector must have shape ({PAYLOAD_FLOAT_COUNT},)")


# Alias for spec_miner naming compatibility
ProprioceptiveSnapshot = ProprioceptiveState


class SensorimotorCortex:
    """
    50 Hz Virtual Proprioception & Spatial Presence Mapping Engine.
    Integrates 6-DOF joint kinematics, computes forward kinematics & analytical Jacobian,
    packages spatial state into 64-element float32 tensors, and monitors update latency.
    """

    def __init__(
        self,
        n_joints: int = N_JOINTS_DEFAULT,
        dt: float = DT_DEFAULT,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        link_lengths: Optional[Sequence[float]] = None,
        damping: float = 0.05,
        max_velocity: float = DEFAULT_QDOT_MAX,
        max_torque: float = DEFAULT_TORQUE_MAX,
    ) -> None:
        self.n_joints = int(n_joints)
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_velocity = float(max_velocity)
        self.max_torque = float(max_torque)

        # Setup Joint Limits [q_min, q_max]
        if joint_limits is not None:
            q_min = np.asarray(joint_limits[0], dtype=np.float64).flatten()
            q_max = np.asarray(joint_limits[1], dtype=np.float64).flatten()
            if q_min.shape[0] != self.n_joints or q_max.shape[0] != self.n_joints:
                raise ValueError(f"joint_limits dimension mismatch for {self.n_joints} joints")
            self._q_min = q_min.copy()
            self._q_max = q_max.copy()
        else:
            self._q_min = np.array(DEFAULT_Q_MIN[: self.n_joints], dtype=np.float64)
            self._q_max = np.array(DEFAULT_Q_MAX[: self.n_joints], dtype=np.float64)

        # Setup Link Lengths
        if link_lengths is not None:
            links = np.asarray(link_lengths, dtype=np.float64).flatten()
            if links.shape[0] != self.n_joints:
                raise ValueError(f"link_lengths must contain {self.n_joints} values")
            self._link_lengths = links.copy()
        else:
            self._link_lengths = np.array(DEFAULT_LINK_LENGTHS[: self.n_joints], dtype=np.float64)

        # Dynamic State Vectors
        self._q = np.zeros(self.n_joints, dtype=np.float64)
        self._q_dot = np.zeros(self.n_joints, dtype=np.float64)
        self._tau = np.zeros(self.n_joints, dtype=np.float64)

        # Latency & Telemetry Tracking
        self._total_steps: int = 0
        self._total_latency_ms: float = 0.0
        self._min_latency_ms: float = float("inf")
        self._max_latency_ms: float = 0.0
        self._violations_count: int = 0
        self._last_state: Optional[ProprioceptiveState] = None

    @property
    def q(self) -> np.ndarray:
        """Current joint positions in radians."""
        return self._q.copy()

    @property
    def q_dot(self) -> np.ndarray:
        """Current joint velocities in rad/s."""
        return self._q_dot.copy()

    @property
    def tau(self) -> np.ndarray:
        """Current joint torques in N*m."""
        return self._tau.copy()

    @property
    def last_state(self) -> Optional[ProprioceptiveState]:
        """Most recent proprioceptive state snapshot."""
        return self._last_state

    def compute_fk_and_jacobian(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes forward kinematics (FK) and analytical Jacobian J(q) in R^(3 x N_joints).

        Forward Kinematics formulation:
            q_1: Base rotation around Z axis.
            q_2..q_N: Articulated planar elevation angles in sagittal plane.
            theta_i = sum_(j=2)^i q_j
            r = sum_(i=2)^N l_i * cos(theta_i)
            x = cos(q_1) * r
            y = sin(q_1) * r
            z = l_1 + sum_(i=2)^N l_i * sin(theta_i)

        Analytical Jacobian J in R^(3 x N):
            col 1 (q_1): [-sin(q_1)*r, cos(q_1)*r, 0]^T
            col k (k >= 2): [cos(q_1)*dr/dq_k, sin(q_1)*dr/dq_k, dz/dq_k]^T
            where dr/dq_k = -sum_(i=k)^N l_i * sin(theta_i)
                  dz/dq_k =  sum_(i=k)^N l_i * cos(theta_i)

        Returns:
            Tuple[cartesian_pos, jacobian]:
                cartesian_pos: np.ndarray shape=(3,), dtype=float64
                jacobian: np.ndarray shape=(3, N_joints), dtype=float64
        """
        arr_q = np.asarray(q, dtype=np.float64).flatten()
        if arr_q.shape[0] != self.n_joints:
            raise ValueError(f"Joint vector shape {arr_q.shape[0]} != {self.n_joints}")
        if not np.all(np.isfinite(arr_q)):
            raise ValueError("Joint position vector contains NaN or Inf")

        n = self.n_joints
        l = self._link_lengths

        q1 = arr_q[0]
        cos_q1 = math.cos(q1)
        sin_q1 = math.sin(q1)

        # Planar cumulative angles theta_i = sum_{j=1}^{i-1} arr_q[j] for link i (using 0-based indexing: index 1..n-1)
        # Link 0 is base post of height l[0] along Z.
        # Links 1..n-1 are planar segments.
        cum_theta = np.zeros(n, dtype=np.float64)
        if n > 1:
            cum_theta[1:] = np.cumsum(arr_q[1:])

        # Compute radial distance r and height z
        r = 0.0
        z = l[0]
        cos_theta = np.zeros(n, dtype=np.float64)
        sin_theta = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            cos_theta[i] = math.cos(cum_theta[i])
            sin_theta[i] = math.sin(cum_theta[i])
            r += l[i] * cos_theta[i]
            z += l[i] * sin_theta[i]

        x = cos_q1 * r
        y = sin_q1 * r
        p_ee = np.array([x, y, z], dtype=np.float64)

        # Compute Analytical Jacobian J in R^(3 x n)
        J = np.zeros((3, n), dtype=np.float64)

        # Column 0: d/dq_1
        J[0, 0] = -sin_q1 * r
        J[1, 0] = cos_q1 * r
        J[2, 0] = 0.0

        # Columns k >= 1: d/dq_k
        for k in range(1, n):
            # sum from i=k to n-1
            dr_dqk = -np.sum(l[k:] * sin_theta[k:])
            dz_dqk = np.sum(l[k:] * cos_theta[k:])

            J[0, k] = cos_q1 * dr_dqk
            J[1, k] = sin_q1 * dr_dqk
            J[2, k] = dz_dqk

        return p_ee, J

    def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Alias for compute_fk_and_jacobian for API parity."""
        return self.compute_fk_and_jacobian(q)

    def pack_cognitive_payload(
        self,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        cartesian_pos: np.ndarray,
        cartesian_vel: np.ndarray,
        joint_torques: np.ndarray,
    ) -> np.ndarray:
        """
        Serializes kinematic and spatial state variables into standard 64-element
        float32 payload aligned with CognitivePacket.payload_tensor[64].

        Layout:
            [0:6]   : joint_positions (6 floats)
            [6:12]  : joint_velocities (6 floats)
            [12:15] : cartesian_pos [x, y, z] (3 floats)
            [15:18] : cartesian_vel [vx, vy, vz] (3 floats)
            [18:24] : joint_torques (6 floats)
            [24:64] : reserved zero-padding (40 floats)
        """
        payload = np.zeros(PAYLOAD_FLOAT_COUNT, dtype=np.float32)

        # Copy joint positions
        n_pos = min(len(joint_positions), 6)
        payload[0:n_pos] = np.asarray(joint_positions[:n_pos], dtype=np.float32)

        # Copy joint velocities
        n_vel = min(len(joint_velocities), 6)
        payload[6: 6 + n_vel] = np.asarray(joint_velocities[:n_vel], dtype=np.float32)

        # Copy Cartesian position
        n_cpos = min(len(cartesian_pos), 3)
        payload[12: 12 + n_cpos] = np.asarray(cartesian_pos[:n_cpos], dtype=np.float32)

        # Copy Cartesian velocity
        n_cvel = min(len(cartesian_vel), 3)
        payload[15: 15 + n_cvel] = np.asarray(cartesian_vel[:n_cvel], dtype=np.float32)

        # Copy joint torques
        n_tau = min(len(joint_torques), 6)
        payload[18: 18 + n_tau] = np.asarray(joint_torques[:n_tau], dtype=np.float32)

        return payload

    @classmethod
    def unpack_cognitive_payload(cls, payload: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Deserializes a 64-element float32 payload tensor back into constituent kinematic arrays.
        """
        arr = np.asarray(payload, dtype=np.float32).flatten()
        if arr.size != PAYLOAD_FLOAT_COUNT:
            raise ValueError(f"Payload tensor size must be {PAYLOAD_FLOAT_COUNT}, got {arr.size}")

        return {
            "joint_positions": arr[0:6].astype(np.float64),
            "joint_velocities": arr[6:12].astype(np.float64),
            "cartesian_pos": arr[12:15].astype(np.float64),
            "cartesian_vel": arr[15:18].astype(np.float64),
            "joint_torques": arr[18:24].astype(np.float64),
        }

    def step(
        self,
        motor_command: np.ndarray,
        dt: Optional[float] = None,
        external_feedback: Optional[np.ndarray] = None,
        packet_uuid: Optional[str] = None,
        arrival_timestamp_ns: Optional[int] = None,
    ) -> ProprioceptiveState:
        """
        Executes one 50 Hz kinematic simulation cycle.
        Integrates joint kinematics, updates positions/velocities, computes analytical
        FK and Jacobian, calculates Cartesian velocity v_ee = J * q_dot, and packages
        the output into an immutable ProprioceptiveState.

        Latency is strictly measured via time.perf_counter_ns() for DVP-15 compliance.
        """
        t_start_ns = time.perf_counter_ns()
        ref_start_ns = arrival_timestamp_ns if arrival_timestamp_ns is not None else t_start_ns

        step_dt = float(dt) if dt is not None else self.dt
        if step_dt <= 0.0:
            raise ValueError(f"dt must be strictly positive, got {step_dt}")

        # Input Validation: Zero Corruption Guard
        u = np.asarray(motor_command, dtype=np.float64).flatten()
        if u.size != self.n_joints:
            # If shape is smaller or larger, adapt safely or raise
            if u.size < self.n_joints:
                padded_u = np.zeros(self.n_joints, dtype=np.float64)
                padded_u[: u.size] = u
                u = padded_u
            else:
                u = u[: self.n_joints]

        if not np.all(np.isfinite(u)):
            raise ValueError("Motor command contains NaN or Inf (Zero Corruption Guard)")

        # Kinematic Integration with Damping:
        # q_dot(t+dt) = clip(q_dot(t)*(1 - damping) + u*dt, -v_max, v_max)
        # q(t+dt) = clip(q(t) + q_dot(t+dt)*dt, q_min, q_max)
        new_q_dot = self._q_dot * (1.0 - self.damping) + u * step_dt
        new_q_dot = np.clip(new_q_dot, -self.max_velocity, self.max_velocity)

        new_q = self._q + new_q_dot * step_dt

        # Physical Joint Limit Enforcement with Zero-Velocity Clamping at boundary
        for i in range(self.n_joints):
            if new_q[i] <= self._q_min[i]:
                new_q[i] = self._q_min[i]
                if new_q_dot[i] < 0.0:
                    new_q_dot[i] = 0.0
            elif new_q[i] >= self._q_max[i]:
                new_q[i] = self._q_max[i]
                if new_q_dot[i] > 0.0:
                    new_q_dot[i] = 0.0

        # Optional Sensor Fusion with External Feedback
        if external_feedback is not None:
            ext_arr = np.asarray(external_feedback, dtype=np.float64).flatten()
            if ext_arr.size >= self.n_joints:
                # Complementary filter fusion: 90% model prediction, 10% sensory feedback
                new_q = 0.90 * new_q + 0.10 * ext_arr[: self.n_joints]
                new_q = np.clip(new_q, self._q_min, self._q_max)

        # Simulated Torque Load (Inertia + Damping + Gravity compensation)
        new_tau = np.clip(u * 2.5 + self._q_dot * 0.8, -self.max_torque, self.max_torque)

        # Forward Kinematics and Analytical Jacobian
        p_ee, J = self.compute_fk_and_jacobian(new_q)

        # Cartesian Velocity: v_ee = J * q_dot
        v_ee = np.dot(J, new_q_dot)

        # Commit State
        self._q = new_q
        self._q_dot = new_q_dot
        self._tau = new_tau

        # Format 64-element payload tensor
        state_tensor = self.pack_cognitive_payload(
            joint_positions=new_q,
            joint_velocities=new_q_dot,
            cartesian_pos=p_ee,
            cartesian_vel=v_ee,
            joint_torques=new_tau,
        )

        # Measure propagation latency
        t_end_ns = time.perf_counter_ns()
        latency_ms = (t_end_ns - ref_start_ns) / 1_000_000.0

        # Update Telemetry
        self._total_steps += 1
        self._total_latency_ms += latency_ms
        if latency_ms < self._min_latency_ms:
            self._min_latency_ms = latency_ms
        if latency_ms > self._max_latency_ms:
            self._max_latency_ms = latency_ms
        if latency_ms > LATENCY_WCET_THRESHOLD_MS:
            self._violations_count += 1

        pkt_uuid = packet_uuid if packet_uuid is not None else str(uuid.uuid4())

        snapshot = ProprioceptiveState(
            timestamp_ns=t_end_ns,
            packet_uuid=pkt_uuid,
            joint_positions=new_q.copy(),
            joint_velocities=new_q_dot.copy(),
            joint_torques=new_tau.copy(),
            cartesian_pos=p_ee.copy(),
            cartesian_vel=v_ee.copy(),
            state_vector=state_tensor,
            latency_ms=latency_ms,
        )

        self._last_state = snapshot
        return snapshot

    def update_state(
        self,
        motor_command: np.ndarray,
        packet_uuid: Optional[str] = None,
        external_feedback: Optional[np.ndarray] = None,
        arrival_timestamp_ns: Optional[int] = None,
    ) -> ProprioceptiveState:
        """Alias for step() adhering to spec_miner specification."""
        return self.step(
            motor_command=motor_command,
            external_feedback=external_feedback,
            packet_uuid=packet_uuid,
            arrival_timestamp_ns=arrival_timestamp_ns,
        )

    def process_packet(self, packet: Any) -> ProprioceptiveState:
        """
        Processes an incoming CognitivePacket containing motor actuator commands.
        Extracts motor command tensor, processes 50 Hz kinematic update, and emits snapshot.
        """
        arr_cmd = np.zeros(self.n_joints, dtype=np.float64)
        pkt_uuid = None
        arr_ts = None

        if hasattr(packet, "payload_tensor"):
            payload = packet.payload_tensor
            arr_cmd = np.asarray(payload[: self.n_joints], dtype=np.float64)
            if hasattr(packet, "event_uuid"):
                pkt_uuid = str(packet.event_uuid)
            if hasattr(packet, "timestamp_ns") and packet.timestamp_ns > 0:
                arr_ts = packet.timestamp_ns
        elif isinstance(packet, (np.ndarray, list, tuple)):
            arr_cmd = np.asarray(packet[: self.n_joints], dtype=np.float64)

        return self.step(
            motor_command=arr_cmd,
            packet_uuid=pkt_uuid,
            arrival_timestamp_ns=arr_ts,
        )

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Returns high-resolution latency statistics for DVP-15 compliance verification.
        """
        mean_lat = (self._total_latency_ms / self._total_steps) if self._total_steps > 0 else 0.0
        min_lat = self._min_latency_ms if self._total_steps > 0 else 0.0

        return {
            "total_cycles": float(self._total_steps),
            "wcet_ms": float(self._max_latency_ms),
            "mean_ms": float(mean_lat),
            "min_ms": float(min_lat),
            "violations_count": float(self._violations_count),
            "wcet_threshold_ms": float(LATENCY_WCET_THRESHOLD_MS),
            "dvp15_passed": float(self._max_latency_ms <= LATENCY_WCET_THRESHOLD_MS and self._violations_count == 0),
        }

    def reset(self) -> None:
        """Resets kinematic state to rest (zero angles, zero velocities)."""
        self._q = np.zeros(self.n_joints, dtype=np.float64)
        self._q_dot = np.zeros(self.n_joints, dtype=np.float64)
        self._tau = np.zeros(self.n_joints, dtype=np.float64)
        self._last_state = None
        self._total_steps = 0
        self._total_latency_ms = 0.0
        self._min_latency_ms = float("inf")
        self._max_latency_ms = 0.0
        self._violations_count = 0
