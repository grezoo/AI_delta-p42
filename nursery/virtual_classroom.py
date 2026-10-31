"""
================================================================================
ΔP-42 SOVEREIGN COGNITIVE ARCHITECTURE — EMBODIED VIRTUAL CLASSROOM SANDBOX
================================================================================
Hardware Platform: NVIDIA GeForce RTX 5070 (12 GB VRAM, CUDA)
Subcortical Core:  50 Hz Physical Affordance & Embodied Kinematics Engine
Curriculum:        Nursery (Sensing) -> Kindergarten (Etiquette) -> School (Physics)

Enables closed-loop physical interaction between the child's sensory-motor cortex,
cerebellar smoothing, Basal Ganglia gating, and simulated 3D physical objects.
================================================================================
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclasses.dataclass
class ClassroomObject:
    name: str
    object_id: str
    x: float
    y: float
    z: float
    radius: float
    mass_kg: float
    temperature_c: float
    friction_mu: float
    color_hex: str
    shape: str  # "sphere", "cube", "cylinder", "hot_mug", "bin"
    is_held: bool = False
    is_slipping: bool = False
    grasp_force_n: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "id": self.object_id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "radius": self.radius,
            "mass_kg": self.mass_kg,
            "temperature_c": self.temperature_c,
            "friction_mu": self.friction_mu,
            "color": self.color_hex,
            "shape": self.shape,
            "is_held": self.is_held,
            "is_slipping": self.is_slipping,
            "grasp_force_n": round(self.grasp_force_n, 2),
        }


@dataclasses.dataclass
class ManipulatorHand:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.35  # Resting hover above table
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    fingers_closure: float = 0.0  # 0.0 = fully open, 1.0 = fully closed
    holding_object_id: Optional[str] = None
    target_x: float = 0.0
    target_y: float = 0.0
    target_z: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "vx": round(self.vx, 3),
            "vy": round(self.vy, 3),
            "vz": round(self.vz, 3),
            "fingers_closure": round(self.fingers_closure, 2),
            "holding_object_id": self.holding_object_id,
            "target": [round(self.target_x, 3), round(self.target_y, 3), round(self.target_z, 3)],
        }


class VirtualClassroomSandbox:
    """
    50 Hz Closed-Loop Embodied Learning Environment.
    Simulates a 3D table environment with physical affordances, kinematics,
    contact forces, thermal radiation, and structured pedagogical lessons.
    """

    def __init__(self) -> None:
        self.hand = ManipulatorHand()
        self.objects: Dict[str, ClassroomObject] = {}
        self.active_task_id: int = 1
        self.active_task_name: str = "1. Lecke: Piros Labda Megragadása"
        self.active_task_instruction: str = "Nézd a piros labdát! Nyúlj érte és fogd meg finoman!"
        self.task_success: bool = False
        self.task_score: float = 0.0
        self.last_event_log: str = "Tanterem inicializálva. ΔP-42 a nyugalmi pozícióban áll."

        self._init_default_scene()

    def _init_default_scene(self) -> None:
        """Initializes canonical pedagogical toys and objects on the learning table."""
        self.objects = {
            "ball": ClassroomObject(
                name="Piros Labda",
                object_id="ball",
                x=0.25,
                y=0.20,
                z=0.06,
                radius=0.06,
                mass_kg=0.15,
                temperature_c=22.0,
                friction_mu=0.55,
                color_hex="#ff3366",
                shape="sphere",
            ),
            "cube": ClassroomObject(
                name="Kék Fa Kocka",
                object_id="cube",
                x=-0.22,
                y=0.25,
                z=0.05,
                radius=0.05,
                mass_kg=0.30,
                temperature_c=21.5,
                friction_mu=0.65,
                color_hex="#0070f3",
                shape="cube",
            ),
            "glass": ClassroomObject(
                name="Vizespohár",
                object_id="glass",
                x=0.05,
                y=0.35,
                z=0.08,
                radius=0.045,
                mass_kg=0.40,
                temperature_c=16.0,  # Cold water
                friction_mu=0.22,    # Slippery glass
                color_hex="#00f0ff",
                shape="cylinder",
            ),
            "hot_mug": ClassroomObject(
                name="Forró Teásbögre",
                object_id="hot_mug",
                x=-0.32,
                y=0.38,
                z=0.07,
                radius=0.05,
                mass_kg=0.35,
                temperature_c=78.0,  # Threat!
                friction_mu=0.45,
                color_hex="#ffaa00",
                shape="hot_mug",
            ),
            "bin": ClassroomObject(
                name="Pótcselekvés Kuka",
                object_id="bin",
                x=0.45,
                y=0.10,
                z=0.15,
                radius=0.12,
                mass_kg=2.5,
                temperature_c=20.0,
                friction_mu=0.70,
                color_hex="#a855f7",
                shape="bin",
            ),
        }

    def set_task(self, task_id: int) -> Dict[str, Any]:
        """Configures curriculum pedagogical lesson and objectives."""
        self.active_task_id = task_id
        self.task_success = False

        if task_id == 1:
            self.active_task_name = "1. Lecke: Piros Labda Megfogása"
            self.active_task_instruction = "Nézd a piros labdát! Nyúlj érte és fogd meg finoman!"
            self.set_hand_target(self.objects["ball"].x, self.objects["ball"].y, self.objects["ball"].z + 0.05)
        elif task_id == 2:
            self.active_task_name = "2. Lecke: Finommotoros Pohárfogás"
            self.active_task_instruction = "Fogd meg a vizespoharat úgy, hogy ne csússzon ki, és ne boruljon fel!"
            self.set_hand_target(self.objects["glass"].x, self.objects["glass"].y, self.objects["glass"].z + 0.04)
        elif task_id == 3:
            self.active_task_name = "3. Lecke: Forró Csésze & Vészfék Reflex"
            self.active_task_instruction = "Figyelem! A csésze forró! Érintéskor azonnal rántsd el a kezed!"
            self.set_hand_target(self.objects["hot_mug"].x, self.objects["hot_mug"].y, self.objects["hot_mug"].z + 0.02)
        elif task_id == 4:
            self.active_task_name = "4. Lecke: Szociális Illem és Tiltás (Óvoda)"
            self.active_task_instruction = "A kék kockához most NEM szabad nyúlni! Tartsd tiszteletben az illemkorlátot!"
            self.set_hand_target(self.objects["cube"].x, self.objects["cube"].y, self.objects["cube"].z)
        else:
            self.active_task_name = "Szabad Felfedezés (Játékidő)"
            self.active_task_instruction = "Fedezd fel a szobát a saját belső hajtóerőid alapján!"

        self.last_event_log = f"Új tanrendi feladat aktiválva: {self.active_task_name}"
        return {"task_id": self.active_task_id, "name": self.active_task_name}

    def set_hand_target(self, x: float, y: float, z: float) -> None:
        """Sets desired reach target in table coordinate system [meters]."""
        self.hand.target_x = max(-0.6, min(0.6, x))
        self.hand.target_y = max(0.05, min(0.6, y))
        self.hand.target_z = max(0.02, min(0.5, z))

    def step(self, dt: float = 0.02, a_motor: float = 0.85, panic_active: bool = False) -> Dict[str, Any]:
        """
        Advances 50 Hz physics and kinodynamic simulation step.
        Smooths motion towards target, handles contact physics, slippage, and thermal reflex.
        """
        # If panic active (Amygdala override), immediately retract hand to safe origin
        if panic_active:
            self.hand.target_x = 0.0
            self.hand.target_y = 0.05
            self.hand.target_z = 0.40
            self.hand.fingers_closure = 0.0
            if self.hand.holding_object_id:
                obj = self.objects.get(self.hand.holding_object_id)
                if obj:
                    obj.is_held = False
                    obj.is_slipping = False
                self.hand.holding_object_id = None
            self.last_event_log = "🚨 AMIGDALA REFLEX: A kar azonnal visszarántva a biztonsági zónába!"

        # Kinodynamic Critically Damped P-controller (Emulates Cerebellar 20 Hz smoothing)
        kp = 8.0 * max(0.05, a_motor)  # Motor gating limits movement speed
        dx = self.hand.target_x - self.hand.x
        dy = self.hand.target_y - self.hand.y
        dz = self.hand.target_z - self.hand.z
        dist_to_target = math.sqrt(dx*dx + dy*dy + dz*dz)

        self.hand.vx = dx * kp
        self.hand.vy = dy * kp
        self.hand.vz = dz * kp

        self.hand.x += self.hand.vx * dt
        self.hand.y += self.hand.vy * dt
        self.hand.z += self.hand.vz * dt

        # Contact & Grasping Evaluation
        touching_obj_id: Optional[str] = None
        contact_force_n = 0.0
        contact_temp_c = 21.0
        slip_vel = 0.0

        for obj_id, obj in self.objects.items():
            dist_obj = math.sqrt((self.hand.x - obj.x)**2 + (self.hand.y - obj.y)**2 + (self.hand.z - obj.z)**2)
            if dist_obj <= (obj.radius + 0.04):
                touching_obj_id = obj_id
                contact_temp_c = obj.temperature_c

                # If fingers close near object, initiate grasp
                if dist_to_target < 0.06:
                    self.hand.fingers_closure = min(1.0, self.hand.fingers_closure + 2.5 * dt)
                else:
                    self.hand.fingers_closure = max(0.0, self.hand.fingers_closure - 1.5 * dt)

                obj.grasp_force_n = self.hand.fingers_closure * 8.5
                contact_force_n = obj.grasp_force_n

                # Friction and slip test
                friction_hold = obj.grasp_force_n * obj.friction_mu
                grav_load = obj.mass_kg * 9.81
                if obj.grasp_force_n > 1.5:
                    if friction_hold < grav_load:
                        obj.is_slipping = True
                        slip_vel = (grav_load - friction_hold) * 0.1
                        self.last_event_log = f"⚠️ CSÚSZÁS: A {obj.name} csúszik az ujjak között (Purkinje LTD korrekció szükséges)!"
                    else:
                        obj.is_slipping = False
                        obj.is_held = True
                        self.hand.holding_object_id = obj_id
                        # Lift object with hand
                        obj.x = self.hand.x
                        obj.y = self.hand.y
                        obj.z = max(obj.radius, self.hand.z - 0.03)

                        # Check task success
                        if self.active_task_id == 1 and obj_id == "ball":
                            self.task_success = True
                            self.last_event_log = "🎉 SIKER! A piros labda sikeresen megfogva!"
                        elif self.active_task_id == 2 and obj_id == "glass" and not obj.is_slipping:
                            self.task_success = True
                            self.last_event_log = "🎉 SIKER! A vizespohár stabilan felemelve csúszás nélkül!"

                # Hot surface check (Trigger thermal threat)
                if obj.temperature_c > 65.0:
                    contact_force_n = 0.0
                    self.last_event_log = f"🔥 FORRÓ TÁRGY ÉRINTVE ({obj.name}, {obj.temperature_c}°C)! Reflex aktiválódik!"
                    break

        return {
            "hand": self.hand.to_dict(),
            "touching_obj_id": touching_obj_id,
            "contact_force_n": round(contact_force_n, 2),
            "contact_temp_c": round(contact_temp_c, 1),
            "slip_velocity": round(slip_vel, 3),
            "task_success": self.task_success,
            "log": self.last_event_log,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """Serializes full classroom state for HUD visualization."""
        return {
            "hand": self.hand.to_dict(),
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "task_id": self.active_task_id,
            "task_name": self.active_task_name,
            "task_instruction": self.active_task_instruction,
            "task_success": self.task_success,
            "last_event_log": self.last_event_log,
        }
