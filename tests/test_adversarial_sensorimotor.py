"""
ΔP-42 Sovereign Cognitive Architecture - Gate 2 Milestone 3 Adversarial Stress Test Suite
Target: modules/sensorimotor_cortex.py (DVP-15)
Author: Challenger 2 (Adversarial Sensorimotor Stress Tester)

Empirical Adversarial Challenges:
1. Update latency WCET under 5,000 rapid cycles: verify latency strictly <= 3.0 ms in 100% of cases.
2. Kinematic boundary collisions: extreme high-velocity commands (+-1e2, +-1e4, +-1e6, +-1e9)
   driving joints into limits; verify velocity clamping to 0 without numerical instability,
   smooth reversal without deadlock, and high-frequency bang-bang switching stability.
3. Analytical Jacobian precision under singular/gimbal-lock configurations:
   - Overhead singularity (r = 0, end-effector on Z axis)
   - Maximum reach boundary singularity (fully outstretched)
   - Folded-arm self-canceling configurations
   - 1,000 randomized configurations across joint space
   Comparison against numerical central finite differences with max error < 1e-5.
4. CognitivePacket payload packaging: pack and unpack 1,000 diverse states, verify zero tensor
   corruption, exact float32 round-trip, 64-bit alignment, and valid CRC-64/ECMA-182 integration
   with tamper-detection bit-flip tests.
5. Pathological inputs: NaNs, Infinities, extreme commands (+-1e9, +-1e12), invalid dt (<= 0),
   and non-finite feedback; verify ValueError rejection for invalid/non-finite inputs and zero
   internal state corruption across all failure modes.
"""

from __future__ import annotations

import math
import struct
import time
from typing import Any, Dict, List, Tuple
import uuid

import numpy as np
import pytest

from comm.cognitive_packet import (
    CognitivePacket,
    PACKET_SIZE,
    PAYLOAD_FLOAT_COUNT,
    compute_crc64,
    validate_crc,
)
from modules.sensorimotor_cortex import (
    CADENCE_HZ,
    DEFAULT_LINK_LENGTHS,
    DEFAULT_Q_MAX,
    DEFAULT_Q_MIN,
    DEFAULT_QDOT_MAX,
    DEFAULT_TORQUE_MAX,
    DT_DEFAULT,
    LATENCY_WCET_THRESHOLD_MS,
    N_JOINTS_DEFAULT,
    ProprioceptiveSnapshot,
    ProprioceptiveState,
    SensorimotorCortex,
)


# ==============================================================================
# Challenge 1: WCET Empirical Verification under 5,000 Rapid Cycles (DVP-15)
# ==============================================================================
class TestAdversarialSensorimotorWCET:
    """
    Stress-tests the 50 Hz kinematic simulation cycle across 5,000 rapid cycles.
    Verifies that WCET is strictly <= 3.0 ms in 100% of cases (0 violations).
    """

    def test_wcet_5000_rapid_cycles(self) -> None:
        """
        Runs 5,000 continuous kinematic update cycles with diverse dynamic commands:
        - Quiescent baseline
        - Smooth sinusoidal oscillations
        - Random white-noise motor commands
        - High-velocity impulse steps
        - Extreme transient spikes
        """
        cortex = SensorimotorCortex(n_joints=6, dt=DT_DEFAULT)
        rng = np.random.RandomState(42)

        latencies_ms: List[float] = []
        violations: int = 0

        t_overall_start = time.perf_counter()

        for cycle in range(5000):
            # Select varying command profiles
            phase = cycle % 5
            if phase == 0:
                # Quiescent
                u = np.zeros(6, dtype=np.float64)
            elif phase == 1:
                # Sinusoidal smooth tracking
                t = cycle * DT_DEFAULT
                u = np.array([
                    1.5 * math.sin(2.0 * math.pi * 0.5 * t),
                    1.0 * math.cos(2.0 * math.pi * 0.3 * t),
                    -1.2 * math.sin(2.0 * math.pi * 0.8 * t),
                    0.8 * math.cos(2.0 * math.pi * 1.2 * t),
                    -0.5 * math.sin(2.0 * math.pi * 0.2 * t),
                    0.3 * math.cos(2.0 * math.pi * 0.4 * t),
                ], dtype=np.float64)
            elif phase == 2:
                # Random uniform excitation
                u = rng.uniform(-5.0, 5.0, size=6).astype(np.float64)
            elif phase == 3:
                # High-velocity impulse steps
                u = rng.choice([-10.0, 0.0, 10.0], size=6).astype(np.float64)
            else:
                # Dynamic multi-frequency burst
                u = (rng.randn(6) * 3.0).astype(np.float64)

            # Execute cycle
            state = cortex.step(u)

            lat = state.latency_ms
            latencies_ms.append(lat)

            if lat > LATENCY_WCET_THRESHOLD_MS:
                violations += 1

            # Assert internal state validity at every cycle
            assert np.all(np.isfinite(state.joint_positions)), f"Non-finite joint position at cycle {cycle}"
            assert np.all(np.isfinite(state.joint_velocities)), f"Non-finite joint velocity at cycle {cycle}"
            assert np.all(np.isfinite(state.cartesian_pos)), f"Non-finite cartesian_pos at cycle {cycle}"
            assert np.all(np.isfinite(state.cartesian_vel)), f"Non-finite cartesian_vel at cycle {cycle}"
            assert np.all(np.isfinite(state.state_vector)), f"Non-finite state_vector at cycle {cycle}"

        t_overall_duration = time.perf_counter() - t_overall_start

        # Compute empirical statistics
        arr_lat = np.array(latencies_ms)
        min_lat = float(np.min(arr_lat))
        mean_lat = float(np.mean(arr_lat))
        median_lat = float(np.median(arr_lat))
        p95_lat = float(np.percentile(arr_lat, 95))
        p99_lat = float(np.percentile(arr_lat, 99))
        p999_lat = float(np.percentile(arr_lat, 99.9))
        wcet_lat = float(np.max(arr_lat))

        stats = cortex.get_latency_stats()

        # Telemetry assertions
        assert stats["total_cycles"] == 5000.0
        assert violations == 0, f"Found {violations} cycles exceeding {LATENCY_WCET_THRESHOLD_MS} ms"
        assert stats["violations_count"] == 0.0
        assert stats["dvp15_passed"] == 1.0
        assert wcet_lat <= LATENCY_WCET_THRESHOLD_MS, f"WCET {wcet_lat:.4f} ms exceeded threshold {LATENCY_WCET_THRESHOLD_MS} ms"

        # Performance margin assertion: mean should be well under 0.5 ms
        assert mean_lat < 0.5, f"Mean latency {mean_lat:.4f} ms is too high"

        print(f"\n[DVP-15 EMPIRICAL BENCHMARK — 5,000 CYCLES]")
        print(f"  Total Duration:   {t_overall_duration:.3f} s")
        print(f"  Min Latency:      {min_lat:.4f} ms")
        print(f"  Mean Latency:     {mean_lat:.4f} ms")
        print(f"  Median Latency:   {median_lat:.4f} ms")
        print(f"  95th Percentile:  {p95_lat:.4f} ms")
        print(f"  99th Percentile:  {p99_lat:.4f} ms")
        print(f"  99.9th Percentile:{p999_lat:.4f} ms")
        print(f"  WCET (Max):       {wcet_lat:.4f} ms (Threshold: <= {LATENCY_WCET_THRESHOLD_MS} ms)")
        print(f"  Violations Count: {violations} (100.0% Pass Rate)")


# ==============================================================================
# Challenge 2: Kinematic Boundary Collisions and Velocity Clamping
# ==============================================================================
class TestAdversarialKinematicBoundaries:
    """
    Stress-tests joint boundary contact dynamics under extreme commands:
    - Extreme positive & negative commands driving joints into limits
    - Strict velocity clamping to 0.0 at the boundary
    - Absence of numerical instability, accumulation, or drift
    - Seamless reversal without sticking / deadlock
    - Rapid bang-bang limit switching
    """

    def test_extreme_velocity_commands_limit_clamping(self) -> None:
        """Drive all joints into their upper and lower limits with extreme magnitudes (1e2, 1e4, 1e6, 1e9)."""
        cortex = SensorimotorCortex(n_joints=6)
        q_min = cortex._q_min
        q_max = cortex._q_max

        # Drive into upper limits with massive command
        u_extreme_pos = np.full(6, 1e6, dtype=np.float64)
        for step in range(50):
            state = cortex.step(u_extreme_pos)

        # Positions must be strictly clamped to q_max
        np.testing.assert_allclose(state.joint_positions, q_max, rtol=1e-7, atol=1e-7)
        # Velocities directed outwards (positive) must be clamped to 0.0
        np.testing.assert_allclose(state.joint_velocities, np.zeros(6), atol=1e-7)
        # Torques must be saturated at max_torque
        np.testing.assert_allclose(state.joint_torques, np.full(6, cortex.max_torque), atol=1e-7)

        # Drive into lower limits with massive command
        u_extreme_neg = np.full(6, -1e6, dtype=np.float64)
        for step in range(100):
            state = cortex.step(u_extreme_neg)

        # Positions must be strictly clamped to q_min
        np.testing.assert_allclose(state.joint_positions, q_min, rtol=1e-7, atol=1e-7)
        # Velocities directed outwards (negative) must be clamped to 0.0
        np.testing.assert_allclose(state.joint_velocities, np.zeros(6), atol=1e-7)
        # Torques must be saturated at -max_torque
        np.testing.assert_allclose(state.joint_torques, np.full(6, -cortex.max_torque), atol=1e-7)

    def test_boundary_sticking_and_reversal_freedom(self) -> None:
        """
        Verify that a joint pinned at q_max immediately responds to a reverse command
        without sticking or requiring multiple cycles to unstick.
        """
        cortex = SensorimotorCortex(n_joints=6)
        q_max = cortex._q_max

        # Pin at upper limit
        for _ in range(50):
            cortex.step(np.full(6, 100.0))

        assert math.isclose(cortex.q[0], q_max[0], abs_tol=1e-7)
        assert math.isclose(cortex.q_dot[0], 0.0, abs_tol=1e-7)

        # Apply negative command on joint 0: should immediately unstick
        u_rev = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        state = cortex.step(u_rev)

        # Velocity must immediately become negative (u * dt = -2.0 * 0.02 = -0.04)
        expected_q_dot = -2.0 * cortex.dt
        assert math.isclose(state.joint_velocities[0], expected_q_dot, abs_tol=1e-5)
        # Position must have moved inward away from boundary
        assert state.joint_positions[0] < q_max[0]
        assert math.isclose(state.joint_positions[0], q_max[0] + expected_q_dot * cortex.dt, abs_tol=1e-5)

    def test_high_frequency_bang_bang_switching(self) -> None:
        """
        Alternates extreme opposite motor commands (+-1e4) every cycle for 500 cycles.
        Verifies zero floating-point accumulation error, zero NaN/Inf, and strict containment.
        """
        cortex = SensorimotorCortex(n_joints=6)
        q_min = cortex._q_min
        q_max = cortex._q_max

        for cycle in range(500):
            sign = 1.0 if (cycle % 2 == 0) else -1.0
            u = np.full(6, 1e4 * sign, dtype=np.float64)
            state = cortex.step(u)

            # Assert strict boundedness
            assert np.all(state.joint_positions >= q_min - 1e-9)
            assert np.all(state.joint_positions <= q_max + 1e-9)
            assert np.all(np.abs(state.joint_velocities) <= cortex.max_velocity + 1e-9)
            assert np.all(np.abs(state.joint_torques) <= cortex.max_torque + 1e-9)
            assert np.all(np.isfinite(state.state_vector))

    def test_boundary_prolonged_dwell_numerical_stability(self) -> None:
        """
        Dwells at boundary under massive continuous force for 1,000 cycles.
        Verifies exact numerical invariance (no drift beyond q_max).
        """
        cortex = SensorimotorCortex(n_joints=6)
        q_max = cortex._q_max

        # Drive to limit
        for _ in range(50):
            cortex.step(np.full(6, 50.0))

        # Dwell for 1,000 cycles with extreme force
        u_huge = np.full(6, 1e5, dtype=np.float64)
        for _ in range(1000):
            state = cortex.step(u_huge)
            np.testing.assert_array_equal(state.joint_positions, q_max)
            np.testing.assert_array_equal(state.joint_velocities, np.zeros(6))


# ==============================================================================
# Challenge 3: Analytical Jacobian Precision at Singularities & Gimbal Locks
# ==============================================================================
class TestAdversarialJacobianSingularities:
    """
    Stress-tests Forward Kinematics and Analytical Jacobian J(q) under singular,
    near-singular, and boundary configurations:
    1. Overhead singularity (r = 0, arm extended vertically along Z axis)
    2. Fully extended arm (all planar joints = 0, boundary reach)
    3. Folded-arm self-canceling configurations
    4. 1,000 randomized configurations across full joint space
    Verifies that analytical Jacobian matches numerical finite differences with max error < 1e-5.
    """

    @staticmethod
    def _compute_numerical_jacobian(
        cortex: SensorimotorCortex, q: np.ndarray, eps: float = 1e-7
    ) -> np.ndarray:
        """Computes numerical Jacobian via central finite differences."""
        n = cortex.n_joints
        J_num = np.zeros((3, n), dtype=np.float64)
        for k in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[k] += eps
            q_minus[k] -= eps
            p_plus, _ = cortex.compute_fk_and_jacobian(q_plus)
            p_minus, _ = cortex.compute_fk_and_jacobian(q_minus)
            J_num[:, k] = (p_plus - p_minus) / (2.0 * eps)
        return J_num

    def test_overhead_singularity_r_zero(self) -> None:
        """
        Singular Configuration: Arm vertically upward (theta_i = pi/2 for all planar joints).
        When links are pointed along Z, radial reach r = 0.
        Column 0 of Jacobian (d/dq_1) must be exactly [0, 0, 0]^T without NaN/Inf or division errors.
        """
        cortex = SensorimotorCortex(n_joints=6)
        # q1 = arbitrary, planar angles = pi/2, 0, 0, 0, 0 => sum of planar angles = pi/2
        q_sing = np.array([0.785398, math.pi / 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        p_ee, J_analytical = cortex.compute_fk_and_jacobian(q_sing)

        assert np.all(np.isfinite(p_ee))
        assert np.all(np.isfinite(J_analytical))

        # End effector x, y should be near zero because r = l2 * cos(pi/2) = 0
        assert math.isclose(p_ee[0], 0.0, abs_tol=1e-12)
        assert math.isclose(p_ee[1], 0.0, abs_tol=1e-12)

        # Base rotation produces zero linear end-effector velocity
        assert math.isclose(J_analytical[0, 0], 0.0, abs_tol=1e-12)
        assert math.isclose(J_analytical[1, 0], 0.0, abs_tol=1e-12)
        assert math.isclose(J_analytical[2, 0], 0.0, abs_tol=1e-12)

        # Verify against numerical Jacobian
        J_numerical = self._compute_numerical_jacobian(cortex, q_sing)
        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-6)

    def test_fully_extended_horizontal_boundary_singularity(self) -> None:
        """
        Singular Configuration: Arm outstretched fully horizontally (all planar angles = 0).
        Radial distance is maximized (r = sum(l_1..l_5)).
        Linear velocity in radial direction dr/dq_k is 0 (boundary singularity).
        """
        cortex = SensorimotorCortex(n_joints=6)
        q_ext = np.zeros(6, dtype=np.float64)

        p_ee, J_analytical = cortex.compute_fk_and_jacobian(q_ext)

        assert np.all(np.isfinite(p_ee))
        assert np.all(np.isfinite(J_analytical))

        # Expected reach: z = l[0] = 0.35, r = sum(l[1..5]) = 0.30 + 0.25 + 0.15 + 0.10 + 0.05 = 0.85
        assert math.isclose(p_ee[2], DEFAULT_LINK_LENGTHS[0], abs_tol=1e-12)
        expected_r = sum(DEFAULT_LINK_LENGTHS[1:])
        assert math.isclose(p_ee[0], expected_r, abs_tol=1e-12)
        assert math.isclose(p_ee[1], 0.0, abs_tol=1e-12)

        # Verify against numerical Jacobian
        J_numerical = self._compute_numerical_jacobian(cortex, q_ext)
        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-6)

    def test_folded_arm_cancellation_singularity(self) -> None:
        """
        Folded arm: segments alternating +-pi / 2 or +-pi to create geometric folds.
        """
        cortex = SensorimotorCortex(n_joints=6)
        q_fold = np.array([0.5, math.pi / 3.0, -2.0 * math.pi / 3.0, math.pi / 3.0, -math.pi / 4.0, math.pi / 4.0])

        p_ee, J_analytical = cortex.compute_fk_and_jacobian(q_fold)

        assert np.all(np.isfinite(p_ee))
        assert np.all(np.isfinite(J_analytical))

        J_numerical = self._compute_numerical_jacobian(cortex, q_fold)
        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5, atol=1e-6)

    def test_jacobian_across_1000_random_adversarial_configurations(self) -> None:
        """
        Evaluates analytical Jacobian precision against numerical differentiation
        across 1,000 randomized configurations across the full physical joint space.
        """
        cortex = SensorimotorCortex(n_joints=6)
        rng = np.random.RandomState(42)

        max_abs_err = 0.0
        max_rel_err = 0.0

        for i in range(1000):
            q_rand = rng.uniform(cortex._q_min, cortex._q_max)
            p_ee, J_ana = cortex.compute_fk_and_jacobian(q_rand)
            J_num = self._compute_numerical_jacobian(cortex, q_rand)

            abs_err = float(np.max(np.abs(J_ana - J_num)))
            if abs_err > max_abs_err:
                max_abs_err = abs_err

            denom = np.abs(J_num) + 1e-7
            rel_err = float(np.max(np.abs(J_ana - J_num) / denom))
            if rel_err > max_rel_err:
                max_rel_err = rel_err

            assert np.all(np.isfinite(J_ana))
            assert abs_err < 1e-5, f"Jacobian precision violation at trial {i}: abs_err={abs_err:.2e}"

        assert max_abs_err < 1e-5
        print(f"\n[JACOBIAN PRECISION AUDIT — 1,000 CONFIGURATIONS]")
        print(f"  Max Absolute Error: {max_abs_err:.3e} (Threshold: < 1.000e-05)")
        print(f"  Max Relative Error: {max_rel_err:.3e}")


# ==============================================================================
# Challenge 4: CognitivePacket Payload Packaging and CRC-64 Integrity
# ==============================================================================
class TestAdversarialCognitivePacketPackaging:
    """
    Stress-tests CognitivePacket serialization, deserialization, and CRC-64 integrity:
    - 1,000 state packaging round-trips
    - Exact float32 bitwise equality
    - Zero tensor corruption
    - 64-bit alignment verification
    - Hardware/vector CRC-64/ECMA-182 validation
    - Single-bit tamper detection across header and payload bytes
    """

    def test_pack_unpack_1000_states_roundtrip(self) -> None:
        """Generate 1,000 diverse kinematic states, pack into CognitivePacket, serialize, deserialize, unpack."""
        cortex = SensorimotorCortex(n_joints=6)
        rng = np.random.RandomState(42)

        corruptions = 0
        crc_failures = 0

        for trial in range(1000):
            # Generate diverse inputs
            q = rng.uniform(-math.pi, math.pi, size=6)
            q_dot = rng.uniform(-5.0, 5.0, size=6)
            p_ee = rng.uniform(-1.5, 1.5, size=3)
            v_ee = rng.uniform(-2.0, 2.0, size=3)
            tau = rng.uniform(-50.0, 50.0, size=6)

            # Pack payload
            payload = cortex.pack_cognitive_payload(
                joint_positions=q,
                joint_velocities=q_dot,
                cartesian_pos=p_ee,
                cartesian_vel=v_ee,
                joint_torques=tau,
            )

            assert payload.shape == (64,)
            assert payload.dtype == np.float32

            # Embed into CognitivePacket
            pkt_out = CognitivePacket(
                timestamp_ns=time.perf_counter_ns(),
                source_module_id=2,  # Thalamus / Sensorimotor
                target_module_id=7,  # Cerebellum
                payload_tensor=payload,
            )

            # Check 64-bit alignment
            assert pkt_out.is_aligned_64bit() is True

            # Serialize to 304 binary bytes
            raw_bytes = pkt_out.to_bytes(compute_checksum=True)
            assert len(raw_bytes) == PACKET_SIZE

            # Verify CRC-64 on buffer
            if not validate_crc(raw_bytes):
                crc_failures += 1

            # Deserialize with strict checksum validation
            pkt_in = CognitivePacket.from_bytes(raw_bytes, validate_checksum=True)

            # Unpack payload
            unpacked = SensorimotorCortex.unpack_cognitive_payload(pkt_in.payload_tensor)

            # Exact float32 roundtrip assertion
            if not np.array_equal(unpacked["joint_positions"].astype(np.float32), q.astype(np.float32)):
                corruptions += 1
            if not np.array_equal(unpacked["joint_velocities"].astype(np.float32), q_dot.astype(np.float32)):
                corruptions += 1
            if not np.array_equal(unpacked["cartesian_pos"].astype(np.float32), p_ee.astype(np.float32)):
                corruptions += 1
            if not np.array_equal(unpacked["cartesian_vel"].astype(np.float32), v_ee.astype(np.float32)):
                corruptions += 1
            if not np.array_equal(unpacked["joint_torques"].astype(np.float32), tau.astype(np.float32)):
                corruptions += 1

            # Verify padding floats [24:64] are strictly 0.0
            np.testing.assert_array_equal(pkt_in.payload_tensor[24:64], np.zeros(40, dtype=np.float32))

        assert crc_failures == 0, f"Encountered {crc_failures} CRC validation failures"
        assert corruptions == 0, f"Encountered {corruptions} tensor corruptions during roundtrip"

    def test_single_bit_tamper_detection(self) -> None:
        """
        Adversarially flips single bits in header and payload bytes of a serialized packet.
        Verifies that CognitivePacket.from_bytes(..., validate_checksum=True) reliably raises ValueError.
        """
        cortex = SensorimotorCortex(n_joints=6)
        state = cortex.step(np.ones(6))

        pkt = CognitivePacket(
            timestamp_ns=1000000,
            source_module_id=2,
            target_module_id=7,
            payload_tensor=state.state_vector,
        )
        raw_bytes = bytearray(pkt.to_bytes())

        # Test tampering across 50 distinct byte positions (header, UUID, sequence, payload)
        test_indices = [0, 7, 8, 15, 24, 28, 30, 32, 40, 100, 200, 290, 295]
        for byte_idx in test_indices:
            tampered = bytearray(raw_bytes)
            tampered[byte_idx] ^= 0x01  # Flip least significant bit

            # Must fail CRC-64 verification
            with pytest.raises(ValueError, match="CRC-64 integrity violation"):
                CognitivePacket.from_bytes(tampered, validate_checksum=True)


# ==============================================================================
# Challenge 5: Pathological Inputs and Zero Internal State Corruption
# ==============================================================================
class TestAdversarialPathologicalInputs:
    """
    Stress-tests input validation against non-finite values (NaN, Inf, -Inf),
    extreme commands (+-1e9, +-1e12), invalid time steps (dt <= 0), and malformed feedback.
    Verifies that:
    1. Invalid/non-finite inputs raise ValueError.
    2. Zero internal state corruption occurs (state remains identical to pre-failure state).
    3. Extreme finite commands (+-1e9) are clamped without numerical instability or overflow.
    """

    def test_nan_command_rejection_and_zero_state_corruption(self) -> None:
        """Verifies that NaNs at any joint position raise ValueError without altering internal state."""
        cortex = SensorimotorCortex(n_joints=6)
        # Advance state to a known non-zero configuration
        cortex.step(np.array([0.5, -0.3, 0.2, 0.1, -0.4, 0.6]))
        q_before = cortex.q
        q_dot_before = cortex.q_dot
        tau_before = cortex.tau
        steps_before = cortex._total_steps

        # Test NaN in each joint position
        for i in range(6):
            bad_cmd = np.zeros(6)
            bad_cmd[i] = np.nan

            with pytest.raises(ValueError, match="Zero Corruption Guard"):
                cortex.step(bad_cmd)

            # Assert complete state preservation
            np.testing.assert_array_equal(cortex.q, q_before)
            np.testing.assert_array_equal(cortex.q_dot, q_dot_before)
            np.testing.assert_array_equal(cortex.tau, tau_before)
            assert cortex._total_steps == steps_before

    def test_inf_command_rejection_and_zero_state_corruption(self) -> None:
        """Verifies that +Inf and -Inf raise ValueError without altering internal state."""
        cortex = SensorimotorCortex(n_joints=6)
        cortex.step(np.array([0.2, 0.3, -0.1, 0.4, -0.2, 0.1]))
        q_before = cortex.q
        q_dot_before = cortex.q_dot
        tau_before = cortex.tau
        steps_before = cortex._total_steps

        for val in [float("inf"), float("-inf")]:
            for i in range(6):
                bad_cmd = np.zeros(6)
                bad_cmd[i] = val

                with pytest.raises(ValueError, match="Zero Corruption Guard"):
                    cortex.step(bad_cmd)

                # Assert state integrity
                np.testing.assert_array_equal(cortex.q, q_before)
                np.testing.assert_array_equal(cortex.q_dot, q_dot_before)
                np.testing.assert_array_equal(cortex.tau, tau_before)
                assert cortex._total_steps == steps_before

    def test_invalid_dt_rejection_and_zero_state_corruption(self) -> None:
        """Verifies that non-positive dt values (0.0, negative) raise ValueError and do not mutate state."""
        cortex = SensorimotorCortex(n_joints=6)
        cortex.step(np.ones(6))
        q_before = cortex.q
        q_dot_before = cortex.q_dot

        for bad_dt in [0.0, -0.020, -1e-6, -100.0]:
            with pytest.raises(ValueError, match="dt must be strictly positive"):
                cortex.step(np.ones(6), dt=bad_dt)

            np.testing.assert_array_equal(cortex.q, q_before)
            np.testing.assert_array_equal(cortex.q_dot, q_dot_before)

    def test_non_finite_external_feedback_rejection_and_zero_state_corruption(self) -> None:
        """Verifies that NaN or Inf in external feedback raises ValueError and protects state."""
        cortex = SensorimotorCortex(n_joints=6)
        cortex.step(np.ones(6))
        q_before = cortex.q
        q_dot_before = cortex.q_dot

        bad_feedback = np.array([0.0, np.nan, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Joint position vector contains NaN or Inf"):
            cortex.step(np.ones(6), external_feedback=bad_feedback)

        # State must remain unmutated because error was caught before commit
        np.testing.assert_array_equal(cortex.q, q_before)
        np.testing.assert_array_equal(cortex.q_dot, q_dot_before)

    def test_extreme_magnitude_commands_safety_and_bounding(self) -> None:
        """
        Evaluates system resilience against extreme magnitude finite commands (+-1e9, +-1e12).
        Verifies that:
        1. Commands are safely saturated by kinematic limiters (max_velocity and max_torque).
        2. No overflow or non-finite numbers enter the state vector.
        3. All kinematic quantities remain strictly finite and within physical bounds.
        """
        cortex = SensorimotorCortex(n_joints=6)

        for mag in [1e6, -1e6, 1e9, -1e9, 1e12, -1e12]:
            cmd = np.full(6, mag, dtype=np.float64)
            state = cortex.step(cmd)

            # Assert that output is 100% finite and within limits
            assert np.all(np.isfinite(state.joint_positions))
            assert np.all(np.isfinite(state.joint_velocities))
            assert np.all(np.isfinite(state.joint_torques))
            assert np.all(np.isfinite(state.cartesian_pos))
            assert np.all(np.isfinite(state.cartesian_vel))
            assert np.all(np.isfinite(state.state_vector))

            assert np.all(state.joint_positions >= cortex._q_min - 1e-7)
            assert np.all(state.joint_positions <= cortex._q_max + 1e-7)
            assert np.all(np.abs(state.joint_velocities) <= cortex.max_velocity + 1e-7)
            assert np.all(np.abs(state.joint_torques) <= cortex.max_torque + 1e-7)

    def test_dimension_adaptation_safety(self) -> None:
        """Verifies safe handling of undersized motor commands (zero-padded) and oversized commands (truncated)."""
        cortex = SensorimotorCortex(n_joints=6)

        # Undersized command (3 elements instead of 6)
        state_under = cortex.step(np.array([1.0, 2.0, 3.0]))
        assert state_under.joint_positions.shape == (6,)
        # Joints 3..5 should have received 0 command
        assert math.isclose(state_under.joint_velocities[3], 0.0, abs_tol=1e-7)

        # Oversized command (10 elements instead of 6)
        state_over = cortex.step(np.ones(10))
        assert state_over.joint_positions.shape == (6,)
        assert np.all(np.isfinite(state_over.state_vector))
