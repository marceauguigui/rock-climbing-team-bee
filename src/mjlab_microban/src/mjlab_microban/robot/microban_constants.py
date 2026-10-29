# Copyright 2026 Marc Duclusaud
# Licensed under the Apache License, Version 2.0 (the "License")

import os
import numpy as np
from pathlib import Path
import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# ---------------------------------------------------------------------------
# Chemins des fichiers XML
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent / "microban"
ROBOT_XML = ASSETS_DIR / "robot.xml"
WALL_XML = ASSETS_DIR / "wall.xml"

assert ROBOT_XML.exists(), f"XML not found: {ROBOT_XML}"
assert WALL_XML.exists(), f"XML not found: {WALL_XML}"

def get_robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ROBOT_XML))

get_spec = get_robot_spec

def get_wall_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(WALL_XML))

# ---------------------------------------------------------------------------
# Configuration des Moteurs et Constantes
# ---------------------------------------------------------------------------
MOTOR_TO_ID = {
    "left_hip_yaw": 11,
    "left_hip_roll": 12,
    "left_hip_pitch": 13,
    "left_knee": 14,
    "left_ankle_pitch": 15,
    "left_ankle_roll": 16,
    "right_hip_yaw": 21,
    "right_hip_roll": 22,
    "right_hip_pitch": 23,
    "right_knee": 24,
    "right_ankle_pitch": 25,
    "right_ankle_roll": 26,
    "left_shoulder_pitch": 31,
    "left_shoulder_roll": 32,
    "left_elbow": 33,
    "right_shoulder_pitch": 41,
    "right_shoulder_roll": 42,
    "right_elbow": 43,
    "head": 51,
}

ID_TO_MOTOR = {v: k for k, v in MOTOR_TO_ID.items()}

NEUTRAL_POSE = {
    "left_hip_yaw": float(np.deg2rad(0.0)),
    "left_hip_roll": float(np.deg2rad(5.0)),
    "left_hip_pitch": float(np.deg2rad(-10.0)),
    "left_knee": float(np.deg2rad(0.0)),
    "left_ankle_pitch": float(np.deg2rad(0.0)),
    "left_ankle_roll": float(np.deg2rad(-5.0)),
    "right_hip_yaw": float(np.deg2rad(0.0)),
    "right_hip_roll": float(np.deg2rad(-5.0)),
    "right_hip_pitch": float(np.deg2rad(-10.0)),
    "right_knee": float(np.deg2rad(0.0)),
    "right_ankle_pitch": float(np.deg2rad(0.0)),
    "right_ankle_roll": float(np.deg2rad(5.0)),
    "left_shoulder_pitch": float(np.deg2rad(100.0)),
    "left_shoulder_roll": float(np.deg2rad(10.0)),
    "left_elbow": float(np.deg2rad(-20.0)),
    "right_shoulder_pitch": float(np.deg2rad(100.0)),
    "right_shoulder_roll": float(np.deg2rad(-10.0)),
    "right_elbow": float(np.deg2rad(-20.0)),
    "head": float(np.deg2rad(0.0)),
}

MOTOR_SIGN = {
    "left_hip_yaw": -1.0,
    "left_hip_roll": 1.0,
    "left_hip_pitch": -1.0,
    "left_knee": 1.0,
    "left_ankle_pitch": 1.0,
    "left_ankle_roll": -1.0,
    "right_hip_yaw": -1.0,
    "right_hip_roll": 1.0,
    "right_hip_pitch": 1.0,
    "right_knee": -1.0,
    "right_ankle_pitch": -1.0,
    "right_ankle_roll": -1.0,
    "left_shoulder_pitch": 1.0,
    "left_shoulder_roll": -1.0,
    "left_elbow": 1.0,
    "right_shoulder_pitch": -1.0,
    "right_shoulder_roll": -1.0,
    "right_elbow": -1.0,
    "head": 1.0,
}

KP_DEFAULT: int = 400
KP_RL: int = 125
KP_GAIN_PRM: float = 0.0022

BAM_VIN: float = 8.4
BAM_VIN_MIN: float = 6.0
BAM_VIN_DROP_RESISTANCE: float = 0.25
BAM_MAX_CURRENT: float = 1.75

PRESENT_CURRENT_UNIT_A: float = 0.001
OVERCURRENT_CUTOFF_A: float | None = None
OVERCURRENT_DEBOUNCE_TICKS: int = 2

PROXY_KT: float = 0.366
PROXY_R: float = 2.811
PROXY_VIN: float = BAM_VIN
PROXY_ERROR_GAIN: float = 0.0028773775
PROXY_MAX_PWM: float = 1.0
PROXY_KP: int = KP_RL
OVERCURRENT_PROXY_DELAY_TICKS: int = 3

VX_MAX: float = 0.7
VX_MAX_BACKWARD: float = 0.5
VY_MAX: float = 0.3
VTHETA_MAX_STATIONARY: float = 3.0
VTHETA_MAX_MOVING: float = 1.5

def _board_model() -> str:
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n")
    except OSError:
        return ""

BOARD_MODEL: str = _board_model()
IS_RADXA: bool = "Radxa" in BOARD_MODEL

MOTOR_SERIAL_PORT: str = "/dev/ttyS2" if IS_RADXA else "/dev/ttyAMA0"
IMU_I2C_BUS: int = 3 if IS_RADXA else 1
IMU_MOUNT_QUAT: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)

OBSERVATION_DOF_ORDER = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_elbow",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_elbow",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll"
]

MOVE_KEYS: dict[str, str] = {"h": "head", "s": "squat", "v": "walk"}
GAMEPAD_BUTTON_MOVES: dict[str, str] = {"A": "walk"}

# ---------------------------------------------------------------------------
# Configuration de l'État Initial et des Collisions du Robot
# ---------------------------------------------------------------------------
HOME_FRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.168),
    joint_pos=NEUTRAL_POSE,
    joint_vel={r".*": 0.0},
)

# Correction : contype et conaffinity ajoutés explicitement
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    contype=1,
    conaffinity=1,
    condim={r"^(left|right)_foot_collision$": 3, r".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# Gestion sécurisée du module bam avec repli sur XmlActuatorCfg
try:
    from bam.mjlab import BamActuatorCfg
    actuators = BamActuatorCfg(
        motor_name="xl330",
        model="m6",
        target_names_expr=(r".*",),
        kp_fw=125,
        vin_range=(7.0, 8.0),
        vin_drop_resistance_range=(0.0, 0.2),
        vin_min=6.0,
        delay_min_lag=3,
        delay_max_lag=6,
    )
except ImportError:
    from mjlab.actuator import XmlActuatorCfg
    actuators = XmlActuatorCfg(
        target_names_expr=(r".*",),
        delay_min_lag=0,
        delay_max_lag=3,
    )

MICROBAN_ROBOT_CFG = EntityCfg(
    spec_fn=get_robot_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# ---------------------------------------------------------------------------
# Configuration du Mur (Décor statique issu de wall.xml)
# ---------------------------------------------------------------------------
MICROBAN_WALL_CFG = EntityCfg(
    spec_fn=get_wall_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)

# ---------------------------------------------------------------------------
# Test de visualisation en local
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": MICROBAN_ROBOT_CFG,
            "wall": MICROBAN_WALL_CFG,
        },
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    model = scene.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    viewer.launch(model, data=data)