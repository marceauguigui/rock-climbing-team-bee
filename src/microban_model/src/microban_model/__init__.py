"""Microban v2 (Radxa) robot model: MuJoCo XML and URDF, in two versions that differ only by the forearms and the
feet (walking: the default; climbing: ball hands and climbing feet). Every joint's zero is its motor's zero (Dynamixel
Wizard 180 deg); joint names are microban's.

    import microban_model
    microban_model.mjcf()              # .../mjcf/robot.xml        (walking forearms and feet)
    microban_model.mjcf(climb=True)    # .../mjcf/robot_climb.xml  (ball hands, climbing feet)
    microban_model.urdf(), microban_model.urdf(climb=True)
"""
from pathlib import Path

ROOT = Path(__file__).parent
MJCF_DIR = ROOT / "mjcf"
URDF_DIR = ROOT / "urdf"
JOINTS = ("head", "right_shoulder_pitch", "right_shoulder_roll", "right_elbow", "right_hip_yaw", "right_hip_roll",
          "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll", "left_shoulder_pitch",
          "left_shoulder_roll", "left_elbow", "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
          "left_ankle_pitch", "left_ankle_roll")
"""The 19 actuated joints, in the models' order."""


def mjcf(climb: bool = False) -> Path:
    """The MuJoCo XML (its meshes: MJCF_DIR / "assets")."""
    return MJCF_DIR / ("robot_climb.xml" if climb else "robot.xml")


def urdf(climb: bool = False) -> Path:
    """The URDF (kinematics and inertias, no meshes): for PlaCo and the like."""
    return URDF_DIR / ("robot_climb.urdf" if climb else "robot.urdf")
