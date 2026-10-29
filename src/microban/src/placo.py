import numpy as np
import placo

# 1. Chargement de votre robot
urdf_path = "/home/noar/rock-climbing-team-bee/src/microban_model/src/microban_model/urdf/robot_climb.urdf"
robot = placo.RobotWrapper(urdf_path)

print(f"Robot chargé ! Masse totale : {robot.total_mass():.2f} kg")
print("Joints configurés :", list(robot.joint_names()))

# 2. Mise à jour de la cinématique initiale
robot.update_kinematics()

# Obtenir la position d'une main ou d'un pied (frame de l'URDF)
T_world_rhand = robot.get_T_world_frame("right_hand")
print("Position actuelle main droite (X, Y, Z) :", T_world_rhand.translation)