from pathlib import Path
import numpy as np
import placo
import microban_model

# 1. Chargement de l'URDF
package_dir = Path(microban_model.__file__).parent
urdf_path = package_dir / "urdf" / "robot_climb.urdf"

if not urdf_path.exists():
    root_dir = Path(__file__).resolve().parents[2]
    urdf_path = root_dir / "src" / "microban_model" / "src" / "microban_model" / "urdf" / "robot_climb.urdf"

robot = placo.RobotWrapper(str(urdf_path))
robot.update_kinematics()

# 2. Lecture de la position initiale
target_frame = "right_hand_ball"
T_world_rhand = robot.get_T_world_frame(target_frame)

initial_pos = T_world_rhand[:3, 3]
print(f"Position initiale de '{target_frame}' : {initial_pos}")

# ==========================================
# 3. CINÉMATIQUE INVERSE (IK)
# ==========================================
print("\n--- Lancement de l'IK ---")

# Création du solveur
solver = placo.KinematicsSolver(robot)

# Ajout d'une tâche de régularisation pour minimiser les grands mouvements
# Dans les versions récentes, cela peut se faire via add_regularization_task ou en paramétrant les joints
try:
    solver.add_regularization_task(1e-4)
except AttributeError:
    # Alternative si la méthode a aussi changé de nom
    print("Note : add_regularization_task non disponible, on utilise le comportement par défaut.")

# Cible : Monter de 10 cm, avancer de 5 cm
target_pos = initial_pos.copy()
target_pos[0] += 0.05
target_pos[2] += 0.10
print(f"Cible visée pour '{target_frame}' : {target_pos}")

# Tâche pour la main
hand_task = solver.add_position_task(target_frame, target_pos)
hand_task.configure("hand_target", "hard", 1.0)

# Masquage des joints si besoin (ex: on ne veut pas bouger la tête ou les jambes)
solver.mask_dof("head") 
# Note : on laisse le reste libre pour que le corps compense si besoin

# Résolution
for i in range(100):
    solver.solve(True)
    robot.update_kinematics()

# Vérification
final_T = robot.get_T_world_frame(target_frame)
final_pos = final_T[:3, 3]
error = np.linalg.norm(final_pos - target_pos)

print(f"\nPosition finale atteinte : {final_pos}")
print(f"Erreur (distance restante) : {error*1000:.2f} mm")

print("\nAngles moteurs résultants pour le bras droit :")
for name in robot.joint_names():
    if "right_shoulder" in name or "right_elbow" in name:
        print(f"  {name}: {robot.get_joint(name):.3f} rad")