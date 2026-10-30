import unittest
import numpy as np
from src.microban.src.moves.climb import (
    ClimbMove,
    plan_from_occupation_matrix,
    plan_from_wall,
    wall_to_placo,
)
from src.moves.moves import GRID_ROWS, GRID_COLS


class TestClimbMove(unittest.TestCase):
    def test_wall_to_placo(self):
        # Socket position (z = 0.026) -> x_placo should be 0.0
        pt = (0.3, 0.4, 0.026)
        placo_pt = wall_to_placo(pt, z_base=0.026)
        self.assertAlmostEqual(placo_pt[0], 0.0)
        self.assertAlmostEqual(placo_pt[1], 0.3)
        self.assertAlmostEqual(placo_pt[2], 0.4)

        # Clearance position (z = 0.076) -> x_placo should be -0.05
        pt_clear = (0.3, 0.4, 0.076)
        placo_clear = wall_to_placo(pt_clear, z_base=0.026)
        self.assertAlmostEqual(placo_clear[0], -0.05)

    def test_plan_from_occupation_matrix(self):
        occ = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
        # Create start holds and target holds
        occ[90, 20] = 1  # LF
        occ[90, 30] = 1  # RF
        occ[70, 18] = 1  # LH
        occ[70, 32] = 1  # RH
        occ[60, 18] = 1  # Target 1
        occ[60, 32] = 1  # Target 2

        plan = plan_from_occupation_matrix(occ, max_steps=2)
        self.assertGreaterEqual(len(plan), 5)
        # First 4 must be the 4 limbs
        start_limbs = {p["limb"] for p in plan[:4]}
        self.assertEqual(start_limbs, {"left_foot", "right_foot", "left_hand", "right_hand"})

    def test_plan_from_wall(self):
        plan = plan_from_wall("N2", max_steps=3)
        self.assertEqual(len(plan), 7)  # 4 start + 3 steps
        start_limbs = {p["limb"] for p in plan[:4]}
        self.assertEqual(start_limbs, {"left_foot", "right_foot", "left_hand", "right_hand"})

    def test_climb_move_step_execution(self):
        move = ClimbMove()
        move.options["wall_name"] = "N2"
        move.options["step_duration"] = 1.0
        move.options["pause_duration"] = 0.2
        move.options["max_steps"] = 2

        obs = {
            "time": 0.0,
            "q": [0.0] * 19,
        }
        move.start(obs)

        # Step at t = 0.0
        targets = move.step(obs)
        self.assertIsInstance(targets, dict)
        self.assertEqual(len(targets), 19)

        # Step into first move
        obs["time"] = 0.8
        targets = move.step(obs)
        self.assertIsInstance(targets, dict)
        self.assertEqual(len(targets), 19)


if __name__ == "__main__":
    unittest.main()
