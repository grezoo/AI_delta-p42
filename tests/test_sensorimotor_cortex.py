"""
ΔP-42 Sovereign Cognitive Core (Gate 2) - Sensorimotor Cortex Unit Tests
Verifies:
1. 50 Hz kinematic integration (dt = 20.0 ms) with velocity damping and physical joint limits.
2. DVP-15 real-time propagation latency: WCET <= 3.0 ms across 1,000 cycles.
3. Forward kinematics (FK) and analytical Jacobian consistency against numerical differentiation.
4. 64-element float32 payload packaging for CognitivePacket round-trip integrity.
5. Zero tensor corruption guard against NaNs and Infs (raises ValueError).
6. CognitivePacket integration via process_packet().
"""

import math
import numpy as np
import pytest

from comm.cognitive_packet import CognitivePacket
from modules.sensorimotor_cortex import (
    CADENCE_HZ,
    DT_DEFAULT,
    LATENCY_WCET_THRESHOLD_MS,
    N_JOINTS_DEFAULT,
    PAYLOAD_FLOAT_COUNT,
    ProprioceptiveSnapshot,
    ProprioceptiveState,
    SensorimotorCortex,
)


def test_sensorimotor_50hz_kinematic_integration():
    """Verifies that 50 Hz integration steps positions and velocities with damping."""
    dt = 0.020  # 50 Hz
    cortex = SensorimotorCortex(n_joints=6, dt=dt, damping=0.05)

    assert cortex.n_joints == 6
    assert math.isclose(cortex.dt, 0.020)

    # Initial state is at rest
    np.testing.assert_allclose(cortex.q, np.zeros(6))
    np.testing.assert_allclose(cortex.q_dot, np.zeros(6))

    # Apply constant motor command u = [1.0, 0.0, ...]
    u = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = cortex.step(u)

    # New velocity: q_dot = 0 * 0.95 + 1.0 * 0.020 = 0.020
    # New position: q = 0 + 0.020 * 0.020 = 0.0004
    assert math.isclose(state.joint_velocities[0], 0.020, abs_tol=1e-6)
    assert math.isclose(state.joint_positions[0], 0.0004, abs_tol=1e-6)

    # Step again
    state2 = cortex.step(u)
    # q_dot = 0.020 * 0.95 + 0.020 = 0.019 + 0.020 = 0.039
    assert math.isclose(state2.joint_velocities[0], 0.039, abs_tol=1e-6)


def test_sensorimotor_dvp15_latency_1000_cycles():
    """
    Formal DVP-15 Benchmark:
    Stream 1,000 cycles at 50 Hz; verify WCET <= 3.0 ms and zero latency violations.
    """
    cortex = SensorimotorCortex(n_joints=6, dt=DT_DEFAULT)
    rng = np.random.RandomState(42)

    for i in range(1000):
        # Generate varied motor commands
        u = rng.uniform(-2.0, 2.0, size=6)
        state = cortex.step(u)

        assert state is not None
        assert state.latency_ms <= LATENCY_WCET_THRESHOLD_MS
        assert np.all(np.isfinite(state.state_vector))

    stats = cortex.get_latency_stats()
    assert stats["total_cycles"] == 1000.0
    assert stats["wcet_ms"] <= LATENCY_WCET_THRESHOLD_MS
    assert stats["violations_count"] == 0.0
    assert stats["dvp15_passed"] == 1.0
    # Average latency in pure NumPy should be far below 1 ms
    assert stats["mean_ms"] < 1.0


def test_sensorimotor_fk_and_jacobian_consistency():
    """
    Verifies that analytical Jacobian J(q) matches numerical central finite differences:
        J_numerical[:, k] = (FK(q + eps*e_k) - FK(q - eps*e_k)) / (2 * eps)
    and verifies Cartesian velocity: v_ee = J * q_dot.
    """
    cortex = SensorimotorCortex(n_joints=6)
    q_test = np.array([0.4, 0.2, -0.3, 0.1, 0.5, -0.2], dtype=np.float64)
    q_dot_test = np.array([0.5, -0.2, 0.3, 0.4, -0.1, 0.2], dtype=np.float64)

    p_ee, J_analytical = cortex.compute_fk_and_jacobian(q_test)

    # Numerical Jacobian via finite difference
    eps = 1e-7
    J_numerical = np.zeros_like(J_analytical)
    for k in range(6):
        q_plus = q_test.copy()
        q_minus = q_test.copy()
        q_plus[k] += eps
        q_minus[k] -= eps

        p_plus, _ = cortex.compute_fk_and_jacobian(q_plus)
        p_minus, _ = cortex.compute_fk_and_jacobian(q_minus)

        J_numerical[:, k] = (p_plus - p_minus) / (2.0 * eps)

    # Analytical Jacobian must match numerical with high precision (< 1e-6)
    np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-6)

    # Step cortex with q_test and q_dot_test
    cortex._q = q_test.copy()
    cortex._q_dot = q_dot_test.copy()
    dt = 0.001
    p_before, _ = cortex.compute_fk_and_jacobian(cortex.q)
    state = cortex.step(np.zeros(6), dt=dt)

    # Verify Cartesian velocity matches J(new_q) * new_q_dot exactly
    _, J_after = cortex.compute_fk_and_jacobian(state.joint_positions)
    expected_v_ee = np.dot(J_after, state.joint_velocities)
    np.testing.assert_allclose(state.cartesian_vel, expected_v_ee, rtol=1e-5, atol=1e-6)

    # Also verify Cartesian velocity matches numerical finite displacement dp / dt
    dp_dt = (state.cartesian_pos - p_before) / dt
    np.testing.assert_allclose(state.cartesian_vel, dp_dt, rtol=1e-2, atol=1e-3)


def test_sensorimotor_64_element_payload_packaging():
    """Verifies 64-element float32 payload packaging and unpacking roundtrip."""
    cortex = SensorimotorCortex(n_joints=6)
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    q_dot = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6])
    p_ee = np.array([1.2, 0.5, 0.8])
    v_ee = np.array([0.05, -0.02, 0.01])
    tau = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

    payload = cortex.pack_cognitive_payload(q, q_dot, p_ee, v_ee, tau)

    assert payload.shape == (PAYLOAD_FLOAT_COUNT,)
    assert payload.dtype == np.float32

    # Unpack
    unpacked = SensorimotorCortex.unpack_cognitive_payload(payload)
    np.testing.assert_allclose(unpacked["joint_positions"], q, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(unpacked["joint_velocities"], q_dot, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(unpacked["cartesian_pos"], p_ee, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(unpacked["cartesian_vel"], v_ee, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(unpacked["joint_torques"], tau, rtol=1e-5, atol=1e-6)

    # Check zero padding
    np.testing.assert_allclose(payload[24:64], np.zeros(40, dtype=np.float32))


def test_sensorimotor_zero_tensor_corruption_guard():
    """Verifies that NaNs and Infs in motor commands are rejected with ValueError."""
    cortex = SensorimotorCortex(n_joints=6)

    # NaN in command
    with pytest.raises(ValueError, match="Zero Corruption Guard"):
        cortex.step(np.array([0.0, 1.0, np.nan, 0.0, 0.0, 0.0]))

    # Inf in command
    with pytest.raises(ValueError, match="Zero Corruption Guard"):
        cortex.step(np.array([0.0, 1.0, np.inf, 0.0, 0.0, 0.0]))

    # -Inf in command
    with pytest.raises(ValueError, match="Zero Corruption Guard"):
        cortex.step(np.array([0.0, 1.0, -np.inf, 0.0, 0.0, 0.0]))


def test_sensorimotor_joint_limits_and_velocity_clamping():
    """Verifies that physical joint limits strictly clamp positions and zero velocities at boundary."""
    # Joint limits: [-1.0, 1.0] for all joints
    limits = (np.full(6, -1.0), np.full(6, 1.0))
    cortex = SensorimotorCortex(n_joints=6, joint_limits=limits, max_velocity=3.0)

    # Large motor command pushing joint 0 past limit
    u_large = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        state = cortex.step(u_large)

    assert state.joint_positions[0] <= 1.0
    # Velocity directed outwards is clamped
    assert state.joint_positions[0] == pytest.approx(1.0, abs=1e-7)

    # Reverse direction
    u_reverse = np.array([-100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(100):
        state = cortex.step(u_reverse)

    assert state.joint_positions[0] >= -1.0
    assert state.joint_positions[0] == pytest.approx(-1.0, abs=1e-7)


def test_sensorimotor_cognitive_packet_integration():
    """Verifies processing incoming CognitivePacket and embedding proprioceptive payload into a packet."""
    cortex = SensorimotorCortex(n_joints=6)

    # Create an incoming CognitivePacket from Basal Ganglia (source=6) to Thalamus/Sensorimotor (target=2)
    motor_payload = np.zeros(64, dtype=np.float32)
    motor_payload[0:6] = [0.5, -0.3, 0.8, -0.2, 0.1, -0.4]

    pkt = CognitivePacket(
        timestamp_ns=123456789,
        source_module_id=6,
        target_module_id=2,
        payload_tensor=motor_payload,
    )

    state = cortex.process_packet(pkt)
    assert state is not None
    assert isinstance(state, ProprioceptiveState)
    assert state.packet_uuid == pkt.event_uuid
    assert np.all(np.isfinite(state.state_vector))

    # Test roundtrip packet emission
    out_pkt = CognitivePacket(
        timestamp_ns=state.timestamp_ns,
        event_uuid=state.packet_uuid,
        source_module_id=2,
        target_module_id=7,  # Feeds into Cerebellum
        payload_tensor=state.state_vector,
    )
    assert out_pkt.validate_crc() is True
    assert len(out_pkt.to_bytes()) == 304


def test_sensorimotor_reset():
    """Verifies that reset returns cortex to initial rest state and clears latency counters."""
    cortex = SensorimotorCortex(n_joints=6)
    cortex.step(np.ones(6))
    assert cortex.last_state is not None
    assert cortex.get_latency_stats()["total_cycles"] == 1.0

    cortex.reset()
    assert cortex.last_state is None
    np.testing.assert_allclose(cortex.q, np.zeros(6))
    np.testing.assert_allclose(cortex.q_dot, np.zeros(6))
    stats = cortex.get_latency_stats()
    assert stats["total_cycles"] == 0.0
