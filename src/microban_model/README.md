# microban_model

The model of Microban v2 (the Radxa version), shared by [microban](https://github.com/Rhoban/microban) (the robot's
code), [microban_sim](https://github.com/Rhoban/microban_sim) (Rhoban's robot simulator, which `make sim` runs) and the
simulators built on it (the hackathon's [climbing simulator](https://github.com/rock-climbing-robots/simulator)). MuJoCo XML and URDF, in two versions of the same robot that differ
only by the forearms and the feet:

| | MuJoCo | URDF (kinematics and inertias) | mass |
|---|---|---|---|
| walking forearms and feet (default) | `mjcf/robot.xml` | `urdf/robot.urdf` | 769.8 g |
| climbing: ball hands, climbing feet | `mjcf/robot_climb.xml` | `urdf/robot_climb.urdf` | 807.8 g |

```python
import microban_model
microban_model.mjcf()               # path of mjcf/robot.xml; climb=True: robot_climb.xml
microban_model.urdf(climb=True)     # path of urdf/robot_climb.urdf
microban_model.JOINTS               # the 19 actuated joints, in the models' order
```

Install: `uv add "microban-model @ git+https://github.com/Rhoban/microban_model"` (the meshes are stored with Git LFS:
uv fetches them with `[tool.uv.sources] microban-model = { git = "...", lfs = true }`).

## Conventions

* **Every joint's zero is its motor's zero** (Dynamixel Wizard: 180 deg). The shoulder pitch at 0 holds the arm
  forward, horizontal; along the body it is +pi/2.
* Joint names are microban's (`left_knee`, ...); the trunk frame: x forward, y left, z up. The IMU is the site `imu`
  (sensors `orientation`, `angular-velocity`, `imu_accel`...); the feet frames are the sites `left_foot`, `right_foot`;
  the climbing version adds the 25 mm balls' centres `left_hand_ball`, `right_hand_ball`, `left_foot_ball`,
  `right_foot_ball`.

## What is in it

* Geometry and masses from the Onshape CAD (onshape-to-robot 1.8.3); the walking forearms and feet are those of
  Microban v1.
* Actuators: XL330 as torque motors (class `xl330`: armature, damping, friction loss), for a servo model such as
  [BAM](https://github.com/Rhoban/bam) that computes the torque.
* Collisions (group 3): boxes for the servos that stick out, the trunk as four convex hulls, the other parts as convex
  pieces, primitives for the walking forearms and feet, the hand / foot necks and spheres for the balls of the
  climbing version. Hands and feet collide with the rest of the robot (contype 3 / conaffinity 2), the other parts
  only with the world.
* No scene: each user adds its floor, lights or wall (MuJoCo's `MjSpec`, or an XML that includes the robot).
