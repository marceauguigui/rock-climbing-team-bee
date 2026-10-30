import time
import pinocchio
import placo
import numpy as np
from placo_utils.tf import tf
from placo_utils.visualization import robot_viz, points_viz

y_spacing = 0.15
z_spacing = 0.06

# "Holds" position (just for visualization)
holds = [[0, y, z] for y in [0, y_spacing] for z in np.arange(0, 1, z_spacing)]

robot = placo.RobotWrapper(
    "src/microban_model/src/microban_model/mjcf/robot_climb.xml", placo.Flags.mjcf + placo.Flags.ignore_collisions
)
solver = placo.KinematicsSolver(robot)

robot.set_joint_limits("left_knee", 0.01, 2.3)
robot.set_joint_limits("right_knee", 0.01, 2.3)

# Ball tasks
right_foot_task = solver.add_position_task("right_foot_ball", np.array([0, 0, 0]))
left_foot_task = solver.add_position_task("left_foot_ball", np.array([0, y_spacing, 0]))
right_hand_task = solver.add_position_task("right_hand_ball", np.array([0, 0, z_spacing * 5]))
left_hand_task = solver.add_position_task("left_hand_ball", np.array([0, y_spacing, z_spacing * 5]))

tasks = [right_foot_task, left_foot_task, right_hand_task, left_hand_task]

body_orientation_task = solver.add_orientation_task("body", np.eye(3))
body_orientation_task.configure("body_orientation", "soft", 1e-3)

# Joints
joints = solver.add_joints_task()
joints.configure("joints", "soft", 1e-6)
joints.set_joints({joint: 0 for joint in robot.joint_names()})

solver.add_regularization_task(1e-3)
solver.enable_joint_limits(True)

viz = robot_viz(robot)
dt = 0.02  # [s]
part_duration = 1.5  # [s]

while True:
    for task in tasks:
        # Building a cubic trajectory for the limb that rise with clearance
        waypoints = [
            [0, task.target_world],
            [part_duration * 0.25, task.target_world + np.array([-0.05, 0, 0])],
            [part_duration * 0.5, task.target_world + np.array([-0.05, 0, z_spacing])],
            [part_duration * 0.75,task.target_world + np.array([-0.025, 0, z_spacing + 0.02])],
            [part_duration, task.target_world + np.array([0, 0, z_spacing])],
            ]

        trajectory = placo.CubicSpline3D()
        for t, pos in waypoints:
            trajectory.add_point(t, pos, np.zeros(3))

        t = 0
        t0 = time.time()
        while time.time() - t0 < part_duration:
            task.target_world = trajectory.pos(time.time() - t0)

            solver.solve(True)
            robot.update_kinematics()

            points_viz("holds", holds)

            viz.display(robot.state.q)
            time.sleep(dt)
            t += dt