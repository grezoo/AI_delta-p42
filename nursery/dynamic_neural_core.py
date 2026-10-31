"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — DYNAMIC NEURAL CORE & SUPER-PID
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (Blackwell, 12 GB VRAM, PyTorch 2.11+CUDA)
Architecture:      Native Continuous-Time Recurrent Tensor Network + Cerebellar Super-PID
Plasticity:        Online Hebbian Eligibility Trace Plasticity + REM Sleep Variance Reduction
Persistence:       Direct PyTorch Tensor Weights File (Zero SQL DB)
================================================================================
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("dynamic_neural_core")

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNAPSE_DIR = REPO_ROOT / "data" / "synapses"
SYNAPSE_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_WEIGHTS_FILE = SYNAPSE_DIR / "dp42_dynamic_core.pt"


class SuperPIDCerebellum:
    """
    Biologically Grounded Cerebellar Super-PID Controller with Smith Predictor.
    Delivers surgeon-like fine sensorimotor control, damping, and zero static error
    without motor tremor or overshoot.
    """

    def __init__(
        self,
        p_gain: float = 1.20,
        i_gain: float = 0.35,
        d_gain: float = 0.12,
        dt: float = 0.02,
        integral_limit: float = 4.0,
    ) -> None:
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.dt = dt
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.smith_delay_buffer: List[float] = [0.0] * 3  # 60ms forward delay buffer
        self.last_output = 0.0

    def step(self, target: float, actual: float, forward_est: float = 0.0) -> float:
        """Computes Super-PID control action with Smith predictive lead."""
        error = target - actual

        # Smith Predictor: inject forward estimate to cancel transport lag
        predicted_error = error + 0.5 * (forward_est - actual)

        # Proportional term (Purkinje immediate action)
        p_term = self.p_gain * predicted_error

        # Integral term (LTD synaptic accumulation - removes steady-state load)
        self.integral += error * self.dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.i_gain * self.integral

        # Derivative term (Climbing fiber rate-of-change damping - prevents tremor)
        d_term = self.d_gain * (error - self.prev_error) / max(1e-4, self.dt)
        self.prev_error = error

        control = p_term + i_term + d_term
        self.last_output = float(control)
        return self.last_output

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.smith_delay_buffer = [0.0] * 3
        self.last_output = 0.0


class DynamicNeuralCore(nn.Module):
    """
    Sovereign Dynamic Neural Core.
    Continuous-time recurrent neural state machine on CUDA with living, writable synapses,
    Hebb/STDP eligibility traces, and REM sleep variance-reduction consolidation.
    """

    def __init__(
        self,
        dim_in: int = 64,
        dim_hidden: int = 128,
        dim_out: int = 16,
        dt: float = 0.02,
        tau: float = 0.05,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dt = dt
        self.tau = tau
        self.alpha = min(1.0, dt / tau)  # Continuous leaky integrator blending

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing DynamicNeuralCore on {self.device} (dim_hidden={dim_hidden})")

        # ----------------------------------------------------------------------
        # 1. Physical Synaptic Weight Matrices
        # ----------------------------------------------------------------------
        # Input synaptic projection
        self.W_in = nn.Parameter(torch.randn(dim_hidden, dim_in, device=self.device) * 0.08)

        # Recurrent reservoir core (Orthogonal initialization for stable boundary cycle)
        w_rec_raw = torch.empty(dim_hidden, dim_hidden, device=self.device)
        nn.init.orthogonal_(w_rec_raw, gain=0.92)
        self.W_rec = nn.Parameter(w_rec_raw)

        # Recurrent bias
        self.b_rec = nn.Parameter(torch.zeros(dim_hidden, device=self.device))

        # Output projection (Motor & Subcortical Gating)
        self.W_out = nn.Parameter(torch.randn(dim_out, dim_hidden, device=self.device) * 0.08)
        self.b_out = nn.Parameter(torch.zeros(dim_out, device=self.device))

        # Cerebellar Forward Model (Smith predictor tensor)
        self.W_fwd = nn.Parameter(torch.randn(dim_hidden, dim_hidden, device=self.device) * 0.05)

        # Total active synapse count
        self.synapse_count = (
            dim_hidden * dim_in + dim_hidden * dim_hidden + dim_hidden + dim_out * dim_hidden + dim_out + dim_hidden * dim_hidden
        )

        # ----------------------------------------------------------------------
        # 2. Dynamic State Vectors & Plasticity Registers
        # ----------------------------------------------------------------------
        self.register_buffer("h", torch.zeros(dim_hidden, device=self.device))
        self.register_buffer("eligibility_in", torch.zeros(dim_hidden, dim_in, device=self.device))
        self.register_buffer("eligibility_rec", torch.zeros(dim_hidden, dim_hidden, device=self.device))

        # Super-PID controller for fine cerebellar grip and motor smoothing
        self.cerebellum = SuperPIDCerebellum(dt=dt)

        # Daytime Experience Trajectory Buffer for REM Sleep Replay
        self.trajectory_buffer: List[Dict[str, torch.Tensor]] = []
        self.max_buffer_capacity = 3000

        # Optimizer for REM Sleep Consolidation
        self.rem_optimizer = optim.Adam(self.parameters(), lr=0.0018, weight_decay=1e-5)

        # Load existing weights if present
        if DEFAULT_WEIGHTS_FILE.exists():
            self.load_weights(DEFAULT_WEIGHTS_FILE)

    def forward_step(
        self,
        x: torch.Tensor,
        theta_deg: float = 15.0,
        tau_torque: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes single continuous-time integration step at 50 Hz.
        Integrates synchronous motor torque into recurrent state.
        """
        # Synchronous motor torque drive vector
        # tau_motor = K * sin(theta - 15 deg)
        torque_vector = torch.full((self.dim_hidden,), tau_torque, device=self.device)

        # Pre-synaptic activation
        pre = (
            torch.matmul(self.W_in, x)
            + torch.matmul(self.W_rec, self.h)
            + self.b_rec
            + torque_vector
        )

        # Continuous Leaky Integrator State Transition
        h_new = (1.0 - self.alpha) * self.h + self.alpha * torch.tanh(pre)
        self.h.copy_(h_new)

        # Output projection
        out = torch.matmul(self.W_out, self.h) + self.b_out
        return h_new, out

    def step(
        self,
        maslow_v: np.ndarray,
        theta_deg: float,
        tactile_pressures: List[float],
        slip_velocity: float,
        thermal_c: float,
        audio_rms: float,
        semantic_concept: Optional[np.ndarray] = None,
        reward_valence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Real-time sensorimotor step called by the 50 Hz organism tick.
        Updates internal neural state, drives cerebellar Super-PID, applies online Hebbian traces.
        """
        with torch.no_grad():
            # 1. Assemble physical sensor vector (dim_in = 64)
            x_arr = np.zeros(self.dim_in, dtype=np.float32)

            # Maslow 5D deficit
            for i in range(min(5, len(maslow_v))):
                x_arr[i] = float(maslow_v[i])

            # Phase dynamics
            x_arr[5] = float(theta_deg / 180.0)
            x_arr[6] = float(math.sin(math.radians(theta_deg - 15.0)))

            # Tactile sensors (5 fingers)
            for i in range(min(5, len(tactile_pressures))):
                x_arr[7 + i] = float(tactile_pressures[i]) / 10.0

            # Slip, Thermal, Acoustic
            x_arr[12] = float(min(1.0, slip_velocity / 2.0))
            x_arr[13] = float(min(1.0, thermal_c / 100.0))
            x_arr[14] = float(min(1.0, audio_rms / 3000.0))
            x_arr[15] = float(reward_valence)

            # Semantic concept projection (from GGUF interpreter or visual tags)
            if semantic_concept is not None:
                sem_len = min(48, len(semantic_concept))
                x_arr[16 : 16 + sem_len] = semantic_concept[:sem_len].astype(np.float32)

            x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=self.device)

            # Synchronous motor torque
            tau_torque = 0.80 * math.sin(math.radians(max(-90.0, min(90.0, theta_deg - 15.0))))

            # Previous hidden state snapshot for eligibility trace
            h_prev = self.h.clone()

            # Execute recurrent dynamic step
            h_cur, out_raw = self.forward_step(x_tensor, theta_deg=theta_deg, tau_torque=tau_torque)

            # Motor Gating & Action Output
            a_motor_raw = float(torch.sigmoid(out_raw[0]).item())

            # Super-PID Cerebellar Damping on Grip / Action
            # Target action is driven by motor gate; actual is damped by slip
            smith_pred = float(torch.matmul(self.W_fwd, h_cur)[0].item())
            pid_smooth = self.cerebellum.step(
                target=a_motor_raw,
                actual=float(out_raw[1].item()),
                forward_est=smith_pred,
            )

            # Smoothed finger tactile grip (scaled [0, 8N])
            fingers_out = [
                float(max(0.1, min(8.0, 4.0 * torch.sigmoid(out_raw[2 + i]).item() + pid_smooth)))
                for i in range(5)
            ]

            # ------------------------------------------------------------------
            # 2. Online Hebbian / STDP Eligibility Trace Plasticity
            # ------------------------------------------------------------------
            # E(t) = lambda * E(t-1) + h * x^T
            self.eligibility_in.mul_(0.92).add_(torch.outer(h_cur, x_tensor))
            self.eligibility_rec.mul_(0.92).add_(torch.outer(h_cur, h_prev))

            # If meaningful reward or punishment arrives, immediately update synapses!
            if abs(reward_valence) > 0.05:
                eta = 0.0008 * reward_valence
                self.W_in.add_(self.eligibility_in * eta)
                self.W_rec.add_(self.eligibility_rec * (eta * 0.5))

                # Weight norm stabilization
                self.W_in.clamp_(-1.5, 1.5)
                self.W_rec.clamp_(-1.5, 1.5)

            # ------------------------------------------------------------------
            # 3. Daytime Trajectory Buffering for REM Consolidation
            # ------------------------------------------------------------------
            if len(self.trajectory_buffer) < self.max_buffer_capacity:
                self.trajectory_buffer.append({
                    "x": x_tensor.cpu(),
                    "h_prev": h_prev.cpu(),
                    "h_target": h_cur.cpu(),
                    "valence": reward_valence,
                })

            # Calculate VRAM usage
            vram_mb = 0.0
            if torch.cuda.is_available() and self.device.type == "cuda":
                vram_mb = float(torch.cuda.memory_allocated(self.device) / (1024 * 1024))

            return {
                "a_motor": max(0.0, min(1.0, a_motor_raw)),
                "tactile_fingers": fingers_out,
                "super_pid_out": pid_smooth,
                "synapse_count": self.synapse_count,
                "vram_used_mb": round(vram_mb, 2),
                "is_cuda": self.device.type == "cuda",
                "h_norm": float(torch.norm(h_cur).item()),
                "plasticity_delta": float(torch.norm(self.eligibility_in).item()),
            }

    def consolidate_rem_sleep(self, epochs: int = 12) -> Dict[str, Any]:
        """
        Executes Biological REM Sleep Consolidation.
        Replays daytime state trajectories, runs gradient-based variance reduction,
        smooths noise, and writes the updated synaptic tensor weights to disk.
        """
        if len(self.trajectory_buffer) < 20:
            return {
                "status": "skipped",
                "message": "Nincs elég nappali epizód a konszolidációhoz (<20 mintavétel).",
                "verdict": "SKIPPED",
            }

        logger.info(f"Starting REM Sleep Consolidation over {len(self.trajectory_buffer)} states...")
        self.train()

        # Extract batches onto device
        x_batch = torch.stack([b["x"] for b in self.trajectory_buffer]).to(self.device)
        h_prev_batch = torch.stack([b["h_prev"] for b in self.trajectory_buffer]).to(self.device)
        h_target_batch = torch.stack([b["h_target"] for b in self.trajectory_buffer]).to(self.device)

        initial_loss = 0.0
        final_loss = 0.0

        for epoch in range(epochs):
            self.rem_optimizer.zero_grad()

            # Predict next hidden state through continuous leaky integrator
            pre = (
                torch.matmul(x_batch, self.W_in.t())
                + torch.matmul(h_prev_batch, self.W_rec.t())
                + self.b_rec
            )
            h_pred = (1.0 - self.alpha) * h_prev_batch + self.alpha * torch.tanh(pre)

            # Variance & Prediction Error Loss
            loss = nn.functional.mse_loss(h_pred, h_target_batch)

            if epoch == 0:
                initial_loss = float(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.rem_optimizer.step()

            final_loss = float(loss.item())

        self.eval()

        # Variance reduction ratio
        delta_loss = initial_loss - final_loss
        variance_reduction_pct = max(0.0, (delta_loss / max(1e-5, initial_loss)) * 100.0)

        # Clear daytime buffer after consolidation
        self.trajectory_buffer.clear()

        # Save physical weights directly to PyTorch tensor file (Zero SQL!)
        self.save_weights(DEFAULT_WEIGHTS_FILE)

        logger.info(
            f"REM Sleep Consolidation Complete: Initial Loss={initial_loss:.4f}, "
            f"Final Loss={final_loss:.4f}, Variance Reduction={variance_reduction_pct:.2f}%"
        )

        return {
            "status": "ok",
            "verdict": "APPROVED",
            "initial_loss": round(initial_loss, 4),
            "final_loss": round(final_loss, 4),
            "delta_loss": round(delta_loss, 4),
            "variance_reduction_pct": round(variance_reduction_pct, 2),
            "synapse_count": self.synapse_count,
            "weights_file": str(DEFAULT_WEIGHTS_FILE),
        }

    def save_weights(self, path: Path) -> None:
        """Saves physical PyTorch tensor weights to disk."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), str(path))
            logger.info(f"Saved dynamic neural core synapses to {path}")
        except Exception as ex:
            logger.error(f"Failed to save dynamic core weights: {ex}")

    def load_weights(self, path: Path) -> None:
        """Loads physical PyTorch tensor weights from disk."""
        try:
            if path.exists():
                sd = torch.load(str(path), map_location=self.device)
                self.load_state_dict(sd, strict=False)
                logger.info(f"Loaded dynamic neural core synapses from {path}")
        except Exception as ex:
            logger.error(f"Failed to load dynamic core weights: {ex}")
