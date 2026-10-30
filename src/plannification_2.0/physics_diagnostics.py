"""MuJoCo contact measurements for physical-motion diagnostics."""

import re

import mujoco
import numpy as np


_HOLD_NAME = re.compile(r"^hold(\d+)_")


def contact_snapshot(sim, *, expected_contacts=(), initial_pairs=(), tick=None, phase=None):
    """Return contact geometry and the solver's six-axis wrench in contact coordinates.

    Force components are N and torque components are N*m. The first three
    components use the contact frame (normal, then two friction directions).
    """
    expected_contacts = set(expected_contacts)
    initial_pairs = set(initial_pairs)
    robot_root = sim.model.body_rootid[sim.model.body("trunk").id]
    contacts = []
    for index, contact in enumerate(sim.data.contact):
        geom_ids = (int(contact.geom1), int(contact.geom2))
        geom_names = [sim.model.geom(geom).name for geom in geom_ids]
        body_ids = [int(sim.model.geom_bodyid[geom]) for geom in geom_ids]
        body_names = [sim.model.body(body).name for body in body_ids]
        roots = [sim.model.body_rootid[body] for body in body_ids]
        hold_indices = []
        for body_name in body_names:
            match = _HOLD_NAME.match(body_name)
            hold_indices.append(int(match.group(1)) if match else None)

        expected = any(
            hold_index is not None
            and (geom_ids[1 - side], hold_index) in expected_contacts
            for side, hold_index in enumerate(hold_indices)
        )
        pair = tuple(sorted(geom_ids))
        preexisting = pair in initial_pairs
        touching = bool(contact.dist <= 0.0)
        if any(hold_index is not None for hold_index in hold_indices):
            category = "robot_hold" if robot_root in roots else "hold"
        elif "wall" in geom_names:
            category = "robot_wall" if robot_root in roots else "wall"
        elif "floor" in geom_names:
            category = "robot_floor" if robot_root in roots else "floor"
        elif roots[0] == robot_root and roots[1] == robot_root:
            category = "robot_robot"
        elif robot_root in roots:
            category = "robot_other"
        else:
            category = "environment"

        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        classification = (
            "nearby_noncontact"
            if not touching
            else "expected_hold_contact"
            if expected
            else "preexisting_unexpected"
            if preexisting and robot_root in roots
            else "new_unexpected"
            if robot_root in roots
            else "environment_contact"
        )
        contacts.append({
            "tick": tick,
            "phase": phase,
            "geom_ids": list(geom_ids),
            "geom_names": geom_names,
            "body_ids": body_ids,
            "body_names": body_names,
            "category": category,
            "classification": classification,
            "preexisting": preexisting,
            "expected": expected,
            "position_m": np.asarray(contact.pos, dtype=float).tolist(),
            "frame": frame.tolist(),
            "distance_m": float(contact.dist),
            "touching": touching,
            "wrench_contact_frame": wrench.tolist(),
            "normal_force_N": float(wrench[0]),
            "tangential_force_N": wrench[1:3].tolist(),
            "torque_contact_frame_Nm": wrench[3:].tolist(),
            "force_world_N": (frame.T @ wrench[:3]).tolist(),
            "torque_world_Nm": (frame.T @ wrench[3:]).tolist(),
        })
    return contacts
