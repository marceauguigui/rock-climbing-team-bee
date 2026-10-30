import unittest
from src.moves.moves import (
    Robot,
    moves_sequence,
    matrix_to_socket_coords,
    normalize_limb,
    move_to_placo_waypoints,
    create_placo_trajectory,
    SOCKET_OUT,
)


class TestMoves(unittest.TestCase):
    def test_normalize_limb(self):
        self.assertEqual(normalize_limb("rh"), "right_hand")
        self.assertEqual(normalize_limb("right_hand"), "right_hand")
        self.assertEqual(normalize_limb("lh"), "left_hand")
        self.assertEqual(normalize_limb("rf"), "right_foot")
        self.assertEqual(normalize_limb("lf"), "left_foot")
        with self.assertRaises(ValueError):
            normalize_limb("tail")

    def test_matrix_to_socket_coords(self):
        # Top-left corner (0, 0)
        p_top_left = matrix_to_socket_coords((0, 0))
        self.assertAlmostEqual(p_top_left[0], 0.0325, places=3)
        self.assertAlmostEqual(p_top_left[1], 1.376, places=3)
        self.assertAlmostEqual(p_top_left[2], SOCKET_OUT, places=3)

        # Bottom-left corner (0, 92)
        p_bottom_left = matrix_to_socket_coords((0, 92))
        self.assertAlmostEqual(p_bottom_left[0], 0.0325, places=3)
        self.assertAlmostEqual(p_bottom_left[1], -0.004, places=3)
        self.assertAlmostEqual(p_bottom_left[2], SOCKET_OUT, places=3)

        # 3D coordinate pass-through
        p_3d = matrix_to_socket_coords((0.1, 0.2, 0.05))
        self.assertEqual(p_3d, (0.1, 0.2, 0.05))

    def test_robot_init(self):
        start_plan = [
            {"limb": "right_hand", "target_coords": (20, 30)},
            {"limb": "left_hand", "target_coords": (10, 30)},
            {"limb": "right_foot", "target_coords": (18, 70)},
            {"limb": "left_foot", "target_coords": (12, 70)},
        ]
        robot = Robot(start_plan)
        rh = robot.get_limb_coords("right_hand")
        self.assertGreater(rh[1], 0.0)
        self.assertEqual(rh[2], SOCKET_OUT)

        all_limbs = robot.limbs
        self.assertIn("right_hand", all_limbs)
        self.assertIn("left_hand", all_limbs)
        self.assertIn("right_foot", all_limbs)
        self.assertIn("left_foot", all_limbs)

    def test_moves_sequence(self):
        plan = [
            # Initial 4 limbs
            {"limb": "right_hand", "target_coords": (20, 30)},
            {"limb": "left_hand", "target_coords": (10, 30)},
            {"limb": "right_foot", "target_coords": (18, 70)},
            {"limb": "left_foot", "target_coords": (12, 70)},
            # Move 1: right hand reaches higher
            {"limb": "right_hand", "target_coords": (22, 20)},
            # Move 2: left hand reaches higher
            {"limb": "left_hand", "target_coords": (12, 20)},
        ]

        seq = moves_sequence(plan)
        self.assertEqual(len(seq), 2)

        # Check move 1
        m1 = seq[0]
        self.assertEqual(m1["limb"], "right_hand")
        self.assertEqual(len(m1["waypoints"]), 6)

        phases = [wp["phase"] for wp in m1["waypoints"]]
        expected_phases = ["exit", "clearance", "in_front", "above", "pre_enter", "enter"]
        self.assertEqual(phases, expected_phases)

        wps = m1["waypoints"]
        from_coords = m1["from_coords"]
        to_coords = m1["to_coords"]

        # Phase 1: exit (Y rises above socket, X and Z unchanged)
        self.assertEqual(wps[0]["coords"][0], from_coords[0])
        self.assertGreater(wps[0]["coords"][1], from_coords[1])
        self.assertEqual(wps[0]["coords"][2], from_coords[2])

        # Phase 2: clearance (Z pulls back)
        self.assertEqual(wps[1]["coords"][0], from_coords[0])
        self.assertEqual(wps[1]["coords"][1], wps[0]["coords"][1])
        self.assertGreater(wps[1]["coords"][2], from_coords[2])

        # Phase 3: in_front (moves to target X, Y at cleared Z)
        self.assertEqual(wps[2]["coords"][0], to_coords[0])
        self.assertEqual(wps[2]["coords"][1], to_coords[1])
        self.assertEqual(wps[2]["coords"][2], wps[1]["coords"][2])

        # Phase 4: above (rises above target hold)
        self.assertEqual(wps[3]["coords"][0], to_coords[0])
        self.assertGreater(wps[3]["coords"][1], to_coords[1])
        self.assertEqual(wps[3]["coords"][2], wps[2]["coords"][2])

        # Phase 5: pre_enter (pushes towards wall to target Z while above socket)
        self.assertEqual(wps[4]["coords"][0], to_coords[0])
        self.assertEqual(wps[4]["coords"][1], wps[3]["coords"][1])
        self.assertEqual(wps[4]["coords"][2], to_coords[2])

        # Phase 6: enter (reaches target socket)
        self.assertEqual(wps[5]["coords"], to_coords)

    def test_placo_trajectory(self):
        plan = [
            {"limb": "right_hand", "target_coords": (20, 30)},
            {"limb": "left_hand", "target_coords": (10, 30)},
            {"limb": "right_foot", "target_coords": (18, 70)},
            {"limb": "left_foot", "target_coords": (12, 70)},
            {"limb": "right_hand", "target_coords": (20, 20)},
        ]
        seq = moves_sequence(plan)
        timed = move_to_placo_waypoints(seq[0], duration=2.0)
        self.assertEqual(len(timed), 7)  # initial + 6 waypoints
        self.assertEqual(timed[0][0], 0.0)
        self.assertEqual(timed[-1][0], 2.0)

        # Test Placo spline creation
        spline = create_placo_trajectory(seq[0], duration=2.0)
        pos_start = spline.pos(0.0)
        pos_end = spline.pos(2.0)
        self.assertAlmostEqual(pos_start[0], seq[0]["from_coords"][0], places=3)
        self.assertAlmostEqual(pos_end[0], seq[0]["to_coords"][0], places=3)


if __name__ == "__main__":
    unittest.main()
