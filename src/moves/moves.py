"""
moves.py: Generation of spatial 3D trajectories and waypoints for climbing movements.

Takes the output of the planning module (plannification) and converts it into
a sequence of spatial Cartesian waypoints (Wall frame, in meters) for each limb.

Input plan shape:
plan = [
  {"limb": "right_hand", "target_coords": (x, y)},
  {"limb": "left_hand",  "target_coords": (x, y)},
  {"limb": "right_foot", "target_coords": (x, y)},
  {"limb": "left_foot",  "target_coords": (x, y)},
  # Subsequent movements:
  {"limb": "right_hand", "target_coords": (x, y)},
  ...
]

(x, y) are the coordinates of the hold's top-left in the OpenCV occupation matrix
(col x from 0 to 51, row y from 0 at top to 92 at bottom).
plan[:4] contains the four starting positions of the four limbs (in any order).
"""
from __future__ import annotations

from typing import Any

# Wall grid constants (meters), aligned with climbing_sim.wall
WALL_WIDTH = 0.800
WALL_HEIGHT = 1.400
HOLE_PITCH = 0.015
GRID_COLS = 52
GRID_ROWS = 93
GRID_X0 = 0.0175
GRID_Z0 = 0.010

SOCKET_OUT = 0.026       # Distance out of wall face (Z in wall frame)
SOCKET_BELOW = 0.014     # Distance below bolt axis (Y in wall frame)

# Default motion clearances (meters)
DEFAULT_EXIT_CLEARANCE_Y = 0.030   # ~30 mm upward motion to exit socket cup
DEFAULT_CLEARANCE_Z = 0.050        # ~50 mm away from wall for obstacle-free translation
DEFAULT_ABOVE_CLEARANCE_Y = 0.030  # ~30 mm above target hold socket before entry

# Canonical limb names mapping
LIMB_MAP = {
    "right_hand": "right_hand",
    "rh": "right_hand",
    "left_hand": "left_hand",
    "lh": "left_hand",
    "right_foot": "right_foot",
    "rf": "right_foot",
    "left_foot": "left_foot",
    "lf": "left_foot",
}


def normalize_limb(name: str) -> str:
    """Normalize a limb name to one of right_hand, left_hand, right_foot, left_foot."""
    key = str(name).strip().lower()
    if key not in LIMB_MAP:
        raise ValueError(f"Unknown limb '{name}'. Expected one of {list(LIMB_MAP.keys())}")
    return LIMB_MAP[key]


def matrix_to_socket_coords(
    coords: tuple[float, ...] | list[float],
    base_z: float | None = None,
) -> tuple[float, float, float]:
    """
    Convert 2D occupation matrix coordinates (x, y) into 3D Wall frame socket coordinates (meters).

    Matrix frame (OpenCV):
      - x: column index (0 to 51, left to right)
      - y: row index (0 at top of board, 92 at bottom of board)

    Wall frame (robot space):
      - x: horizontal along wall (meters, 0 at bottom-left corner plate origin)
      - y: vertical along wall (meters, 0 at bottom)
      - z: perpendicular out of wall towards robot (meters, positive)

    If a 3D coordinate (x, y, z) is provided directly, it is validated and returned.
    """
    if len(coords) == 3:
        z = coords[2] if base_z is None else base_z
        return (float(coords[0]), float(coords[1]), float(z))

    if len(coords) != 2:
        raise ValueError(f"Expected 2D (x, y) or 3D (x, y, z) coords, got {coords}")

    # Matrix column x corresponds to Hold col (left bolt)
    col = max(0, min(GRID_COLS - 3, int(round(coords[0]))))

    # Matrix row y (0 at top) inverted to Wall row (0 at bottom)
    row = max(0, min(GRID_ROWS - 1, (GRID_ROWS - 1) - int(round(coords[1]))))

    # Center of socket is 1 pitch to the right of the left bolt (col + 1)
    x_wall = GRID_X0 + (col + 1) * HOLE_PITCH

    # Socket is SOCKET_BELOW (14 mm) below the bolt axis
    y_wall = GRID_Z0 + row * HOLE_PITCH - SOCKET_BELOW

    # Socket center depth out from wall face
    z_wall = SOCKET_OUT if base_z is None else float(base_z)

    return (round(x_wall, 4), round(y_wall, 4), round(z_wall, 4))


class Robot:
    """Tracks the 3D Cartesian coordinates of the 4 limbs on the wall."""

    right_hand_coords: tuple[float, float, float]
    left_hand_coords: tuple[float, float, float]
    right_foot_coords: tuple[float, float, float]
    left_foot_coords: tuple[float, float, float]

    def __init__(self, base_coord: list[dict[str, Any]], base_z: float | None = None):
        """
        Initialize the 4 limbs from the first 4 elements of the plan.

        base_coord: list of 4 dicts, e.g. [{"limb": "right_hand", "target_coords": (x, y)}, ...]
        base_z: optional override for the socket depth Z (default: SOCKET_OUT = 0.026 m).
        """
        self.right_hand_coords = (0.0, 0.0, 0.0)
        self.left_hand_coords = (0.0, 0.0, 0.0)
        self.right_foot_coords = (0.0, 0.0, 0.0)
        self.left_foot_coords = (0.0, 0.0, 0.0)

        for item in base_coord:
            limb = normalize_limb(item["limb"])
            coords = matrix_to_socket_coords(item["target_coords"], base_z=base_z)
            self.set_limb_coords(limb, coords)

    def get_limb_coords(self, limb: str) -> tuple[float, float, float]:
        """Get 3D coordinates of a limb by name."""
        canonical = normalize_limb(limb)
        return getattr(self, f"{canonical}_coords")

    def set_limb_coords(self, limb: str, coords: tuple[float, float, float]) -> None:
        """Set 3D coordinates of a limb by name."""
        canonical = normalize_limb(limb)
        setattr(self, f"{canonical}_coords", coords)

    @property
    def limbs(self) -> dict[str, tuple[float, float, float]]:
        """Return a dictionary of all 4 limb coordinates."""
        return {
            "right_hand": self.right_hand_coords,
            "left_hand": self.left_hand_coords,
            "right_foot": self.right_foot_coords,
            "left_foot": self.left_foot_coords,
        }

    def __repr__(self) -> str:
        return (
            f"Robot(rh={self.right_hand_coords}, lh={self.left_hand_coords}, "
            f"rf={self.right_foot_coords}, lf={self.left_foot_coords})"
        )


def moves_sequence(
    objective_seq: list[dict[str, Any]],
    exit_clearance_y: float = DEFAULT_EXIT_CLEARANCE_Y,
    clearance_z: float = DEFAULT_CLEARANCE_Z,
    above_clearance_y: float = DEFAULT_ABOVE_CLEARANCE_Y,
    base_z: float | None = None,
) -> list[dict[str, Any]]:
    """
    Generate the sequence of movements with Cartesian waypoints from objective_seq.

    objective_seq[:4]: The 4 starting limb positions (any order).
    objective_seq[4:]: The sequence of target holds for limbs to move to.

    Motion phases per movement:
      1. exit:      Limb moves vertically upward out of the socket cup (+Y).
      2. clearance: Limb pulls back away from wall (+Z) to clearance distance.
      3. in_front:  Limb translates across in (X, Y) to align in front of target hold.
      4. above:     Limb rises above target hold socket (+Y).
      5. pre_enter: Limb pushes forward towards wall (to target Z) above the socket.
      6. enter:     Limb lowers vertically into the target socket cup.

    Returns:
      List of movement dicts (Option A):
      [
        {
          "limb": "right_hand",
          "from_coords": (x, y, z),
          "to_coords": (x, y, z),
          "waypoints": [
            {"phase": "exit", "coords": (x, y, z)},
            {"phase": "clearance", "coords": (x, y, z)},
            {"phase": "in_front", "coords": (x, y, z)},
            {"phase": "above", "coords": (x, y, z)},
            {"phase": "pre_enter", "coords": (x, y, z)},
            {"phase": "enter", "coords": (x, y, z)},
          ]
        },
        ...
      ]
    """
    if len(objective_seq) < 4:
        raise ValueError("objective_seq must contain at least 4 items for initial limb positions.")

    robot = Robot(objective_seq[:4], base_z=base_z)
    sequence: list[dict[str, Any]] = []

    for step in objective_seq[4:]:
        limb = normalize_limb(step["limb"])
        from_coords = robot.get_limb_coords(limb)
        to_coords = matrix_to_socket_coords(step["target_coords"], base_z=base_z)

        x_from, y_from, z_from = from_coords
        x_to, y_to, z_to = to_coords

        z_travel = round(max(z_from, z_to) + clearance_z, 4)

        # 1. Exit: lift upward out of the socket cup
        p_exit = (x_from, round(y_from + exit_clearance_y, 4), z_from)

        # 2. Clearance: pull away from the wall
        p_clearance = (x_from, round(y_from + exit_clearance_y, 4), z_travel)

        # 3. In front: move across in front of target hold
        p_in_front = (x_to, y_to, z_travel)

        # 4. Above: rise above target hold socket
        p_above = (x_to, round(y_to + above_clearance_y, 4), z_travel)

        # 5. Pre-enter: push forward towards the wall above socket
        p_pre_enter = (x_to, round(y_to + above_clearance_y, 4), z_to)

        # 6. Enter: descend into socket cup
        p_enter = to_coords

        waypoints = [
            {"phase": "exit", "coords": p_exit},
            {"phase": "clearance", "coords": p_clearance},
            {"phase": "in_front", "coords": p_in_front},
            {"phase": "above", "coords": p_above},
            {"phase": "pre_enter", "coords": p_pre_enter},
            {"phase": "enter", "coords": p_enter},
        ]

        # Update robot internal state with the new limb position
        robot.set_limb_coords(limb, to_coords)

        sequence.append({
            "limb": limb,
            "from_coords": from_coords,
            "to_coords": to_coords,
            "waypoints": waypoints,
        })

    return sequence


def move_to_placo_waypoints(
    move: dict[str, Any],
    duration: float = 1.5,
) -> list[tuple[float, tuple[float, float, float]]]:
    """
    Convert a movement dict into a list of timed waypoints (t, coords) suitable for Placo.

    duration: Total duration of the limb motion in seconds.
    Returns: [(t0, p0), (t1, p1), ..., (t_end, p_end)]
    """
    waypoints = move["waypoints"]
    time_ratios = [0.15, 0.30, 0.60, 0.75, 0.90, 1.00]

    timed: list[tuple[float, tuple[float, float, float]]] = [
        (0.0, move["from_coords"])
    ]

    for ratio, wp in zip(time_ratios, waypoints):
        timed.append((round(ratio * duration, 4), wp["coords"]))

    return timed


def create_placo_trajectory(
    move: dict[str, Any],
    duration: float = 1.5,
):
    """
    Create and return a placo.CubicSpline3D from a movement dict.
    Requires placo to be installed.
    """
    import numpy as np
    import placo

    spline = placo.CubicSpline3D()
    for t, pos in move_to_placo_waypoints(move, duration=duration):
        spline.add_point(t, np.array(pos, dtype=float), np.zeros(3))
    return spline
