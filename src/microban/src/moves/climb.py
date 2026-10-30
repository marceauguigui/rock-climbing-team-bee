"""
climb.py: Climbing move using visual hold detection and Cartesian inverse kinematics.

This move integrates the full pipeline:
1. Detects holds on the wall image using `detection.detector.create_occupation_matrix`.
2. Plans the climbing steps and generates 3D waypoints using `src/moves/moves.py`.
3. Solves inverse kinematics at 50 Hz using Placo (CubicSpline3D) to command the 19 robot joints.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

# Resolve workspace root and load project modules via importlib to avoid namespace shadowing
_this_dir = Path(__file__).resolve().parent
_workspace_root = _this_dir.parents[3]
_workspace_src = _workspace_root / "src"
for p in [_workspace_src, _workspace_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load src/moves/moves.py
_moves_path = _workspace_src / "moves" / "moves.py"
_moves_spec = importlib.util.spec_from_file_location("workspace_moves", _moves_path)
_workspace_moves = importlib.util.module_from_spec(_moves_spec)
_moves_spec.loader.exec_module(_workspace_moves)

GRID_COLS = _workspace_moves.GRID_COLS
GRID_ROWS = _workspace_moves.GRID_ROWS
Robot = _workspace_moves.Robot
matrix_to_socket_coords = _workspace_moves.matrix_to_socket_coords
moves_sequence = _workspace_moves.moves_sequence
normalize_limb = _workspace_moves.normalize_limb

import numpy as np
import placo
from microban_model import JOINTS, MJCF_DIR


def wall_to_placo(pt: tuple[float, float, float] | np.ndarray, z_base: float = 0.026) -> np.ndarray:
    """
    Convert a point from Wall frame (meters) to Placo world frame.
    Wall frame:  X is right, Y is up, Z is away from wall (positive).
    Placo frame: X is normal to wall (< 0 away from wall), Y is right, Z is up.
    """
    xw, yw, zw = pt
    xp = -(zw - z_base)
    yp = xw
    zp = yw
    return np.array([xp, yp, zp], dtype=float)


def plan_from_occupation_matrix(occupation: np.ndarray, max_steps: int = 4) -> list[dict[str, Any]]:
    """
    Extract a sequence of starting holds and climb moves from an occupation matrix.
    occupation: 2D array (GRID_ROWS x GRID_COLS) with 1s at hold sockets.
    """
    # Find all detected holds: (row_y, col_x) in OpenCV matrix convention
    coords = np.argwhere(occupation == 1)
    if len(coords) < 4:
        raise ValueError(f"Found only {len(coords)} holds in matrix, need at least 4.")

    # Sort holds from bottom of wall (highest row_y) to top (lowest row_y)
    sorted_holds = [(int(c), int(r)) for r, c in coords]
    sorted_holds.sort(key=lambda pt: (-pt[1], pt[0]))

    # Midpoint of wall columns to distinguish left vs right holds
    mid_col = GRID_COLS / 2.0

    # Separate into left and right holds
    left_holds = [h for h in sorted_holds if h[0] < mid_col]
    right_holds = [h for h in sorted_holds if h[0] >= mid_col]

    if len(left_holds) < 2 or len(right_holds) < 2:
        # Fallback to simple split
        left_holds = sorted_holds[0::2]
        right_holds = sorted_holds[1::2]

    # Starting 4 limbs: lowest 2 left and lowest 2 right
    left_foot = left_holds[0]
    right_foot = right_holds[0]
    left_hand = left_holds[1]
    right_hand = right_holds[1]

    plan: list[dict[str, Any]] = [
        {"limb": "left_foot", "target_coords": left_foot},
        {"limb": "right_foot", "target_coords": right_foot},
        {"limb": "left_hand", "target_coords": left_hand},
        {"limb": "right_hand", "target_coords": right_hand},
    ]

    # Generate subsequent climb steps with higher holds
    l_idx = 2
    r_idx = 2
    steps = 0

    while steps < max_steps and (l_idx < len(left_holds) or r_idx < len(right_holds)):
        # Alternate moving right and left limbs
        if steps % 2 == 0 and r_idx < len(right_holds):
            limb = "right_hand" if steps % 4 < 2 else "right_foot"
            plan.append({"limb": limb, "target_coords": right_holds[r_idx]})
            r_idx += 1
            steps += 1
        elif l_idx < len(left_holds):
            limb = "left_hand" if steps % 4 < 2 else "left_foot"
            plan.append({"limb": limb, "target_coords": left_holds[l_idx]})
            l_idx += 1
            steps += 1
        elif r_idx < len(right_holds):
            limb = "right_hand"
            plan.append({"limb": limb, "target_coords": right_holds[r_idx]})
            r_idx += 1
            steps += 1
        else:
            break

    return plan


def plan_from_wall(wall_name: str = "N2", max_steps: int = 4) -> list[dict[str, Any]]:
    """
    Fallback plan generated from a bundled simulator wall (e.g. N2).
    """
    from climbing_sim import Wall

    wall = Wall.load(wall_name)
    plan: list[dict[str, Any]] = []

    # Helper to convert Wall Hold(col, row) to matrix (x, y)
    def hold_to_matrix(h):
        return (int(h.col), int((GRID_ROWS - 1) - h.row))

    # Add start holds
    for limb in ["left_foot", "right_foot", "left_hand", "right_hand"]:
        if limb in wall.start:
            plan.append({"limb": limb, "target_coords": hold_to_matrix(wall.start[limb])})

    # Pick climbing holds above start holds
    start_holds_set = set(wall.start.values())
    higher_holds = [h for h in wall.holds if h not in start_holds_set]
    higher_holds.sort(key=lambda h: (h.row, h.col))

    for i in range(min(max_steps, len(higher_holds))):
        h = higher_holds[i]
        limb = "right_hand" if h.col >= 24 else "left_hand"
        plan.append({"limb": limb, "target_coords": hold_to_matrix(h)})

    return plan


class ClimbMove:
    """Climbs the wall using hold detection and Cartesian inverse kinematics (Placo)."""

    description = "Visual climbing via detector.py and Placo IK"
    kp = 125
    options = {
        "image_path": "board.jpg",
        "wall_name": "N2",
        "step_duration": 1.5,
        "pause_duration": 0.3,
        "max_steps": 4,
        "auto_stop": False,
    }
    option_docs = {
        "image_path": "Path to the wall photo (uses detector.py). If empty or not found, uses wall_name.",
        "wall_name": "Bundled wall fallback when image is missing (e.g. N1, N2, N3).",
        "step_duration": "Duration of each limb movement in seconds.",
        "pause_duration": "Pause between successive limb movements in seconds.",
        "max_steps": "Maximum number of steps to climb.",
        "auto_stop": "Whether the move stops automatically after finishing all steps.",
    }
    sliders = {
        "speed": (0.2, 2.0, 1.0),
    }

    def __init__(self) -> None:
        self.slider = {k: v[2] for k, v in self.sliders.items()}
        self._robot = placo.RobotWrapper(
            str(MJCF_DIR / "robot_climb.xml"),
            placo.Flags.mjcf + placo.Flags.ignore_collisions,
        )
        self._solver = placo.KinematicsSolver(self._robot)

        # Configure joint limits
        self._robot.set_joint_limits("left_knee", 0.01, 2.3)
        self._robot.set_joint_limits("right_knee", 0.01, 2.3)

        # Configure ball tasks
        self._tasks: dict[str, Any] = {}
        for limb in ["right_hand", "left_hand", "right_foot", "left_foot"]:
            self._tasks[limb] = self._solver.add_position_task(f"{limb}_ball", np.zeros(3))

        # Body orientation task
        self._body_task = self._solver.add_orientation_task("body", np.eye(3))
        self._body_task.configure("body_orientation", "soft", 1e-3)

        # Joint regularization task
        self._joints_task = self._solver.add_joints_task()
        self._joints_task.configure("joints", "soft", 1e-6)
        self._joints_task.set_joints({j: 0 for j in self._robot.joint_names()})

        self._solver.add_regularization_task(1e-3)
        self._solver.enable_joint_limits(True)

    def _build_plan(self) -> list[dict[str, Any]]:
        """Extract plan from image if available, otherwise from wall fallback."""
        img_path = self.options.get("image_path", "")
        max_steps = int(self.options.get("max_steps", 4))

        if img_path and os.path.exists(img_path):
            try:
                import cv2
                from detection.detector import create_occupation_matrix

                image = cv2.imread(img_path)
                if image is not None:
                    print(f"[ClimbMove] Processing wall image from '{img_path}'...")
                    occupation, _, _ = create_occupation_matrix(image, debug=False)
                    plan = plan_from_occupation_matrix(occupation, max_steps=max_steps)
                    print(f"[ClimbMove] Extracted plan with {len(plan)} items from image.")
                    return plan
            except Exception as e:
                print(f"[ClimbMove] Error extracting plan from image '{img_path}': {e}")

        # Fallback to bundled wall
        wall_name = str(self.options.get("wall_name", "N2"))
        print(f"[ClimbMove] Using fallback wall '{wall_name}'...")
        return plan_from_wall(wall_name, max_steps=max_steps)

    def start(self, obs: dict[str, Any]) -> None:
        """Called when the move starts."""
        self.t0 = float(obs["time"])
        self._q0 = np.array(obs["q"], dtype=float).copy()

        # Update placo robot with initial joint angles if valid
        for name, val in zip(JOINTS, self._q0):
            if name in self._robot.joint_names():
                self._robot.set_joint(name, float(val))
        self._robot.update_kinematics()

        # 1. Generate plan
        plan = self._build_plan()

        # 2. Track robot limb positions
        self._climb_robot = Robot(plan[:4])

        # 3. Generate waypoints sequence
        self._moves_seq = moves_sequence(plan)
        print(f"[ClimbMove] Generated {len(self._moves_seq)} sequential limb moves.")

        # 4. Initialize task positions in Placo
        for limb, coords in self._climb_robot.limbs.items():
            self._tasks[limb].target_world = wall_to_placo(coords)

        # 5. Build cubic splines and schedules for each move
        step_duration = float(self.options.get("step_duration", 1.5))
        pause_duration = float(self.options.get("pause_duration", 0.3))

        self._moves_data: list[dict[str, Any]] = []
        current_time = 0.5  # initial 0.5s settle time

        for move in self._moves_seq:
            # Build Placo cubic spline for this move
            spline = placo.CubicSpline3D()
            waypoints = move["waypoints"]
            ratios = [0.15, 0.30, 0.60, 0.75, 0.90, 1.00]

            spline.add_point(0.0, np.array(move["from_coords"], dtype=float), np.zeros(3))
            for r, wp in zip(ratios, waypoints):
                t_point = round(r * step_duration, 4)
                spline.add_point(t_point, np.array(wp["coords"], dtype=float), np.zeros(3))

            self._moves_data.append({
                "limb": move["limb"],
                "from_coords": move["from_coords"],
                "to_coords": move["to_coords"],
                "spline": spline,
                "start_time": current_time,
                "end_time": current_time + step_duration,
                "duration": step_duration,
            })
            current_time += step_duration + pause_duration

        self._total_duration = current_time

    def step(self, obs: dict[str, Any]) -> dict[str, float] | None:
        """Called every 20 ms tick."""
        speed = float(getattr(self, "slider", {}).get("speed", 1.0))
        elapsed = (float(obs["time"]) - self.t0) * speed

        if elapsed >= self._total_duration:
            if self.options.get("auto_stop", False):
                return None  # Stop move automatically

        # Find active move
        active_move = None
        for m in self._moves_data:
            if m["start_time"] <= elapsed <= m["end_time"]:
                active_move = m
                break
            elif elapsed > m["end_time"]:
                # Keep limb at destination hold
                self._tasks[m["limb"]].target_world = wall_to_placo(m["to_coords"])

        if active_move is not None:
            # Evaluate spline along trajectory
            t_in_move = min(max(0.0, elapsed - active_move["start_time"]), active_move["duration"])
            pos_wall = active_move["spline"].pos(t_in_move)
            self._tasks[active_move["limb"]].target_world = wall_to_placo(pos_wall)

        # Solve kinematics
        self._solver.solve(True)
        self._robot.update_kinematics()

        # Return the 19 joint targets
        return {j: float(self._robot.get_joint(j)) for j in JOINTS}


if __name__ == "__main__":
    from climbing_sim import ClimbingSim, run_move

    print("Running ClimbMove in simulation...")
    sim = ClimbingSim("N2", cameras=[], kp_fw=125)
    sim.reset_hanging()

    move = ClimbMove()
    obs = run_move(move, sim, seconds=10.0)
    print("ClimbMove finished simulation successfully!")
