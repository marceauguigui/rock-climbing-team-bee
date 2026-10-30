
import numpy as np
from climbing_sim import ClimbingSim, JOINTS


LIMBS = ("right_hand", "left_hand", "right_foot", "left_foot")
RIGHT_LEG_INDICES = [JOINTS.index(name) for name in (
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
)]


def is_reachable(
    sim, limb, hold, q_start, trunk_pos, trunk_quat, tolerance=0.001
):
    """Teste l'atteignabilité géométrique IK, sans valider le chemin ni sa dynamique."""
    q, error = sim.joint_angles_for(
        {limb: hold.socket},
        trunk_pos,
        trunk_quat,
        q_start=q_start,
    )
    return error <= tolerance, q, error


def other_limbs_stable(
    init_limbs_pos, candidate_limbs_pos, actual_limb, tolerance=0.001
):
    """Vérifie géométriquement si les autres extrémités restent en place."""
    for limb in LIMBS:
        if limb == actual_limb:
            continue

        distance = np.linalg.norm(
            candidate_limbs_pos[limb] - init_limbs_pos[limb]
        )
        if distance > tolerance:
            return False

    return True


def test_joint_tracking(sim, joint_index=4, delta=0.1, steps=50):
    """Mesure le suivi d'une petite consigne articulaire."""
    print("\n=== Test de suivi articulaire ===")
    state_before = sim.get_state()

    try:
        q_start = sim.joint_positions()
        q_target = q_start.copy()
        q_target[joint_index] += delta

        print(f"Articulation : {JOINTS[joint_index]} (index {joint_index})")
        print(f"Angle initial : {q_start[joint_index]:.4f} rad")
        print(f"Angle cible   : {q_target[joint_index]:.4f} rad")

        for step in range(steps):
            sim.step(q_target)

            if (step + 1) % 10 == 0:
                measured = sim.joint_positions()[joint_index]
                error = q_target[joint_index] - measured

                print(
                    f"Pas {step + 1:3d} | "
                    f"mesuré={measured:.4f} rad | "
                    f"écart={error:.4f} rad"
                )
    finally:
        sim.set_state(state_before)


def test_foot_lift(sim, limb="right_foot", height=0.02, steps=50):
    """
    Demande au pied de monter verticalement depuis sa position actuelle.
    Ne désactive pas le contact physique avec une prise.
    """
    print("\n=== Test de soulèvement du pied ===")
    state_before = sim.get_state()

    try:
        q_start = sim.joint_positions()
        trunk_pos, trunk_quat = sim.trunk_pose()
        positions_before = sim.ball_positions()
        foot_start = positions_before[limb].copy()

        target_position = foot_start.copy()
        target_position[2] += height

        q_target, ik_error = sim.joint_angles_for(
            {limb: target_position},
            trunk_pos,
            trunk_quat,
            q_start=q_start,
        )

        theoretical_positions = sim.ball_positions_at(
            q_target, trunk_pos, trunk_quat
        )
        theoretical_foot = theoretical_positions[limb]
        theoretical_lift = (
            theoretical_foot[2] - foot_start[2]
        ) * 1000

        print("Accrochages initiaux :", sim.hooked())
        print("Pied :", limb)
        print(f"Hauteur demandée : {height * 1000:.1f} mm")
        print(f"Erreur IK : {ik_error * 1000:.2f} mm")
        print(f"Hauteur calculée par IK : {theoretical_lift:.2f} mm")

        if ik_error > 0.005:
            print(
                "Attention : erreur IK supérieure à 5 mm. "
                "La cible n'est pas suffisamment précise."
            )

        print("\nSuivi de la consigne :")

        for step in range(steps):
            sim.step(q_target)

            if (step + 1) % 10 == 0:
                q_measured = sim.joint_positions()
                positions = sim.ball_positions()
                foot_now = positions[limb]

                actual_lift = (
                    foot_now[2] - foot_start[2]
                ) * 1000

                if limb == "right_foot":
                    joint_error = np.linalg.norm(
                        q_measured[RIGHT_LEG_INDICES]
                        - q_target[RIGHT_LEG_INDICES]
                    )
                else:
                    joint_error = np.linalg.norm(
                        q_measured - q_target
                    )

                distance_to_target = (
                    np.linalg.norm(foot_now - target_position) * 1000
                )

                print(
                    f"Pas {step + 1:3d} | "
                    f"soulèvement={actual_lift:6.2f} mm | "
                    f"distance cible={distance_to_target:6.2f} mm | "
                    f"erreur articulaire={joint_error:.4f} rad"
                )

        print("\nRésultat final :")
        print("Position initiale :", foot_start)
        print("Position finale :", sim.ball_positions()[limb])
        print("Accrochages finaux :", sim.hooked())
        print("Pose finale du tronc :", sim.trunk_pose())

    finally:
        sim.set_state(state_before)
        print("État restauré :", sim.hooked())


def diagnose_foot_motor(sim, hold_index=4, steps=100):
    """
    Mesure les courants moteurs et les erreurs articulaires
    pendant le déplacement du pied droit vers une prise.
    """
    print("\n=== Diagnostic moteur : pied droit ===")
    state_before = sim.get_state()

    try:
        q_start = sim.observation()["q"].copy()
        trunk_pos, trunk_quat = sim.trunk_pose()
        hold = sim.wall.holds[hold_index]

        q_target, ik_error = sim.joint_angles_for(
            {"right_foot": hold.socket},
            trunk_pos,
            trunk_quat,
            q_start=q_start,
        )

        print(f"Prise visée : {hold_index}")
        print(f"Erreur IK : {ik_error * 1000:.3f} mm")
        print("Indices de la jambe droite :", RIGHT_LEG_INDICES)

        for step in range(steps):
            sim.step(q_target)

            if (step + 1) % 10 == 0:
                obs = sim.observation()
                q = obs["q"]
                currents = np.asarray(obs["current"])

                joint_errors = (
                    q_target[RIGHT_LEG_INDICES]
                    - q[RIGHT_LEG_INDICES]
                )
                leg_currents = currents[RIGHT_LEG_INDICES]

                print(
                    f"Pas {step + 1:3d} | "
                    f"erreur articulaire="
                    f"{np.linalg.norm(joint_errors):.4f} rad | "
                    f"courant max={np.max(leg_currents):.3f} A | "
                    f"courants={np.round(leg_currents, 3)}"
                )

        print("Accrochages finaux :", sim.hooked())

    finally:
        sim.set_state(state_before)
        print("État restauré :", sim.hooked())


def test_foot_reach(sim, hold_index=4, steps=100):
    """Teste une cible IK du pied droit vers une prise donnée."""
    print("\n=== Test de déplacement du pied droit vers une prise ===")
    state_before = sim.get_state()

    try:
        q_start = sim.joint_positions()
        trunk_pos, trunk_quat = sim.trunk_pose()
        hold = sim.wall.holds[hold_index]

        reachable, q_target, error = is_reachable(
            sim,
            "right_foot",
            hold,
            q_start,
            trunk_pos,
            trunk_quat,
        )

        print("Accrochages initiaux :", sim.hooked())
        print(f"Prise visée : {hold_index}")
        print(f"Erreur IK : {error * 1000:.3f} mm")
        print("Atteignable géométriquement selon l'IK :", reachable)
        print("Angles initiaux :", q_start[RIGHT_LEG_INDICES])
        print("Angles cibles   :", q_target[RIGHT_LEG_INDICES])

        positions_target = sim.ball_positions_at(
            q_target, trunk_pos, trunk_quat
        )
        theoretical_distance = np.linalg.norm(
            positions_target["right_foot"] - hold.socket
        )

        print(
            f"Distance théorique à la prise : "
            f"{theoretical_distance * 1000:.3f} mm"
        )

        if not reachable:
            print("Test dynamique ignoré : erreur IK trop élevée.")
            return

        for step in range(steps):
            sim.step(q_target)

            if (step + 1) % 10 == 0:
                q_measured = sim.joint_positions()
                angle_error = np.linalg.norm(
                    q_measured[RIGHT_LEG_INDICES]
                    - q_target[RIGHT_LEG_INDICES]
                )

                foot_distance = np.linalg.norm(
                    sim.ball_positions()["right_foot"] - hold.socket
                )

                print(
                    f"Pas {step + 1:3d} | "
                    f"erreur angulaire={angle_error:.4f} rad | "
                    f"distance à la prise={foot_distance * 1000:.1f} mm"
                )

        print("Accrochages finaux :", sim.hooked())
        print("Pose finale du tronc :", sim.trunk_pose())

    finally:
        sim.set_state(state_before)
        print("État restauré :", sim.hooked())


def test_candidate_holds(sim, hold_count=5):
    """Évalue les prises candidates par IK, occupation et stabilité géométrique."""
    print("\n=== Analyse des prises candidates ===")

    q_start = sim.joint_positions()
    trunk_pos, trunk_quat = sim.trunk_pose()
    holds = sim.wall.holds
    hooked = sim.hooked()

    balls_initial = sim.ball_positions_at(
        q_start, trunk_pos, trunk_quat
    )

    for limb in LIMBS:
        print(f"\nMembre : {limb}")

        for index, hold in enumerate(holds[:hold_count]):
            reachable, q_candidate, error = is_reachable(
                sim,
                limb,
                hold,
                q_start,
                trunk_pos,
                trunk_quat,
            )

            occupants = [
                other_limb
                for other_limb, occupied_index in hooked.items()
                if occupied_index == index
            ]

            print(
                f"Prise {index} | erreur IK={error * 1000:.1f} mm | "
                f"occupants={occupants or 'aucun'}"
            )

            if limb in occupants:
                print("  Déjà accrochée par ce membre.")
                continue

            if occupants:
                print("  Prise occupée.")
                continue

            if not reachable:
                print("  Hors de portée selon le seuil IK.")
                continue

            balls_candidate = sim.ball_positions_at(
                q_candidate, trunk_pos, trunk_quat
            )

            stable = other_limbs_stable(
                balls_initial, balls_candidate, limb
            )

            print(
                "  Géométrie des autres membres : "
                + ("conservée" if stable else "modifiée")
            )
            print(
                f"  Distance articulaire totale : "
                f"{np.linalg.norm(q_candidate - q_start):.4f} rad"
            )


if __name__ == "__main__":
    sim = ClimbingSim("N2", cameras=[])

    try:
        sim.reset_hanging()

        print("Accrochages initiaux :", sim.hooked())
        print("Positions des membres :", sim.ball_positions())
        print("Pose du tronc :", sim.trunk_pose())

        test_joint_tracking(sim)
        test_foot_lift(
            sim,
            limb="right_foot",
            height=0.02,
            steps=50,
        )
        diagnose_foot_motor(
            sim,
            hold_index=4,
            steps=100,
        )
        test_foot_reach(
            sim,
            hold_index=4,
            steps=100,
        )
        test_candidate_holds(sim)

    finally:
        sim.close()
