"""Contact-aware path planning for ordered limb-to-hold moves."""

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict

import numpy as np

from climbing_sim import LIMBS

if __package__:
    from .validate_plan import (
        _contacting_holds,
        DEFAULT_TRANSFER_TRAJECTORY,
        _new_collisions,
        _hold_socket,
        _ik_command_error,
        _theoretical_transfer_path,
        _validate_trajectory_config,
        _source_hold,
        _support_snapshot,
        validate_plan,
    )
else:
    from validate_plan import (
        _contacting_holds,
        DEFAULT_TRANSFER_TRAJECTORY,
        _new_collisions,
        _hold_socket,
        _ik_command_error,
        _theoretical_transfer_path,
        _validate_trajectory_config,
        _source_hold,
        _support_snapshot,
        validate_plan,
    )


def _contact_state(sim):
    return tuple(
        (limb, tuple(sorted(_contacting_holds(sim, limb))))
        for limb in LIMBS
    )


def _search_state(sim):
    trunk_pos, trunk_quat = sim.trunk_pose()
    quat = np.asarray(trunk_quat, dtype=float)
    if quat[0] < 0:
        quat = -quat
    return (
        _contact_state(sim),
        tuple(np.rint(sim.joint_positions() / 0.05).astype(int)),
        tuple(np.rint(np.asarray(trunk_pos) / 0.005).astype(int)),
        tuple(np.rint(quat / 0.02).astype(int)),
    )


def _candidate_moves(sim, occupied):
    """Return IK-valid transfer candidates with geometric and joint costs."""
    holds = sim.wall.holds
    candidates = []
    for limb in LIMBS:
        deadline = getattr(sim, "_planner_deadline", float("inf"))
        if time.monotonic() >= deadline:
            _record_timing(sim, "candidate_generation_timed_out", 1)
            break
        source = _source_hold(sim, limb)
        if source is None:
            continue
        supports, reason = _support_snapshot(sim, limb)
        if reason:
            continue
        source_hold = holds[source]
        trunk_pos, trunk_quat = sim.trunk_pose()
        q_start = sim.joint_positions()
        limits = np.asarray(sim.joint_range, dtype=float)
        targets = {name: data["position"] for name, data in supports.items()}
        for hold_index, hold in enumerate(holds):
            if time.monotonic() >= deadline:
                _record_timing(sim, "candidate_generation_timed_out", 1)
                break
            if hold_index in occupied or hold.row < source_hold.row:
                continue
            targets[limb] = _hold_socket(sim, hold_index)
            ik_started = time.perf_counter()
            q, error = sim.joint_angles_for(
                targets, trunk_pos, trunk_quat, q_start=q_start
            )
            _record_timing(sim, "ik_s", time.perf_counter() - ik_started)
            q = np.asarray(q, dtype=float)
            invalid = _ik_command_error(sim, q, error, "planner")
            if invalid:
                continue
            distance = float(np.linalg.norm(
                np.asarray(sim.ball_positions()[limb]) - hold.socket
            ))
            margin = float(np.min(np.minimum(
                q - limits[:, 0], limits[:, 1] - q
            )))
            effort = float(np.linalg.norm(q - q_start))
            candidates.append({
                "gain": hold.row - source_hold.row,
                "distance_m": distance,
                "limb": limb,
                "source_hold": source,
                "hold_index": hold_index,
                "ik_error_m": float(error),
                "joint_effort_rad": effort,
                "joint_limit_margin_rad": margin,
                "support_count": len(supports),
                "target_q": q.tolist(),
                "trunk_pos": np.asarray(trunk_pos, dtype=float).tolist(),
                "trunk_quat": np.asarray(trunk_quat, dtype=float).tolist(),
            })
    return candidates


def _record_timing(sim, key, elapsed):
    profile = getattr(sim, "_planner_profile", None)
    if profile is not None:
        profile[key] += elapsed


def _candidate_rank(candidate, strategy):
    gain = candidate["gain"]
    distance = candidate["distance_m"]
    error = candidate["ik_error_m"]
    effort = candidate["joint_effort_rad"]
    margin = candidate["joint_limit_margin_rad"]
    if strategy in {"progress", "combined"}:
        return (-gain, distance, error, effort, -margin)
    if strategy in {"balanced", "transfers_first", "useful_repositions"}:
        score = (
            (gain + 1) / (distance + 0.03)
            - 2.0 * error
            - 0.03 * effort
            + 0.05 * margin
        )
        return (-score, distance, error, -gain)
    return (gain, distance, error, effort)


def _normalize_candidate(candidate):
    if isinstance(candidate, dict):
        return candidate
    gain, distance, limb, hold_index = candidate
    return {
        "gain": gain,
        "distance_m": distance,
        "limb": limb,
        "source_hold": None,
        "hold_index": hold_index,
        "ik_error_m": 0.0,
        "joint_effort_rad": 0.0,
        "joint_limit_margin_rad": 0.0,
        "support_count": 3,
    }


def _frontier_utility(candidates, strategy):
    if not candidates:
        return (0, float("-inf"), 0)
    best = min(candidates, key=lambda item: _candidate_rank(item, strategy))
    rank = _candidate_rank(best, strategy)
    return (
        max(item["gain"] for item in candidates),
        -rank[0],
        min(len(candidates), 10),
    )


def _quat_multiply(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ))


def _support_quality(sim):
    count = 0
    distances = []
    positions = sim.ball_positions()
    for limb in LIMBS:
        contacts = _contacting_holds(sim, limb)
        if not contacts:
            continue
        count += 1
        distances.append(min(
            float(np.linalg.norm(positions[limb] - _hold_socket(sim, hold)))
            for hold in contacts
        ))
    return count, -max(distances, default=float("inf"))


def _candidate_repositions(sim):
    """Build small trunk-translation commands with all current balls anchored."""
    trunk_pos, trunk_quat = sim.trunk_pose()
    balls = sim.ball_positions()
    q_start = sim.joint_positions()
    candidates = []
    for axis in range(3):
        for direction in (1.0, -1.0):
            if time.monotonic() >= getattr(
                sim, "_planner_deadline", float("inf")
            ):
                _record_timing(sim, "candidate_generation_timed_out", 1)
                return candidates
            offset = np.zeros(3)
            offset[axis] = 0.01 * direction
            ik_started = time.perf_counter()
            q, error = sim.joint_angles_for(
                balls,
                np.asarray(trunk_pos, dtype=float) + offset,
                trunk_quat,
                q_start=q_start,
            )
            _record_timing(sim, "ik_s", time.perf_counter() - ik_started)
            q = np.asarray(q, dtype=float)
            if _ik_command_error(sim, q, error, "trunk_reposition"):
                continue
            candidates.append({
                "target_q": q.tolist(),
                "ticks": 50,
                "requested_trunk_offset_m": offset.tolist(),
                "target_trunk_pos": (
                    np.asarray(trunk_pos, dtype=float) + offset
                ).tolist(),
                "target_trunk_quat": np.asarray(trunk_quat, dtype=float).tolist(),
                "ik_error_m": float(error),
            })
    return candidates


def _run_reposition(sim, candidate):
    """Move the trunk through joint commands while preserving its four supports."""
    support_holds = {limb: _source_hold(sim, limb) for limb in LIMBS}
    if any(hold is None for hold in support_holds.values()):
        return {"success": False, "reason": "support_lost_before_reposition"}

    positions = sim.ball_positions()
    support_pairs = {
        (sim.model.geom(f"{limb}_ball_coll").id, hold)
        for limb, hold in support_holds.items()
    }
    prior_collisions = set(_new_collisions(sim, support_pairs))
    start_trunk = np.asarray(sim.trunk_pose()[0], dtype=float)
    target_q = np.asarray(candidate["target_q"], dtype=float)
    if _ik_command_error(sim, target_q, 0.0, "trunk_reposition"):
        return {"success": False, "reason": "invalid_reposition_command"}

    max_current = 0.0
    support_checks = 0
    unexpected_contacts = 0
    for tick in range(1, candidate["ticks"] + 1):
        observation = sim.step(target_q)
        max_current = max(
            max_current,
            float(np.max(np.abs(observation["current"]))),
        )
        actual = sim.ball_positions()
        for limb, hold in support_holds.items():
            support_checks += 1
            drift = float(np.linalg.norm(actual[limb] - positions[limb]))
            if hold not in _contacting_holds(sim, limb) or drift >= 0.006:
                return {
                    "success": False,
                    "reason": f"support_lost_during_reposition:{limb}",
                    "tick": tick,
                    "support_drift_m": drift,
                    "support_checks": support_checks,
                    "supports_retained": support_checks - 1,
                    "max_current_A": max_current,
                }
        new_collisions = set(_new_collisions(sim, support_pairs)) - prior_collisions
        unexpected_contacts += len(new_collisions)
        if new_collisions:
            return {
                "success": False,
                "reason": "new_collision_during_reposition",
                "tick": tick,
                "new_collision_pairs": sorted(new_collisions),
                "support_checks": support_checks,
                "supports_retained": support_checks,
                "unexpected_contacts_observed": unexpected_contacts,
                "max_current_A": max_current,
            }

    final_trunk = np.asarray(sim.trunk_pose()[0], dtype=float)
    displacement = float(np.linalg.norm(final_trunk - start_trunk))
    if displacement < 0.002:
        return {
            "success": False,
            "reason": "trunk_reposition_below_measurement_threshold",
            "trunk_displacement_m": displacement,
            "support_checks": support_checks,
            "supports_retained": support_checks,
            "unexpected_contacts_observed": unexpected_contacts,
            "max_current_A": max_current,
        }
    return {
        "success": True,
        "reason": "trunk_repositioned",
        "support_holds": support_holds,
        "trunk_start_m": start_trunk.tolist(),
        "trunk_end_m": final_trunk.tolist(),
        "trunk_displacement_m": displacement,
        "requested_trunk_offset_m": candidate["requested_trunk_offset_m"],
        "target_q": target_q.tolist(),
        "ticks": candidate["ticks"],
        "support_checks": support_checks,
        "supports_retained": support_checks,
        "unexpected_contacts_observed": unexpected_contacts,
        "max_current_A": max_current,
    }


def plan_theoretical_to_top(
    sim, max_moves=64, max_branches=128, time_limit_s=60.0,
    trajectory_config=DEFAULT_TRANSFER_TRAJECTORY,
):
    """Search IK-feasible transfer/reposition sequences without physical stepping."""
    started = time.monotonic()
    _validate_trajectory_config(trajectory_config)
    initial_state = sim.get_state()
    holds = sim.wall.holds
    if not holds:
        return {
            "success": False, "status": "THEORETICAL_FAILURE",
            "reason": "wall_has_no_holds", "plan": [], "execution_actions": [],
            "planning_mode": "theoretical",
            "physical_validation_status": "NOT_TESTED",
        }
    if (
        isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 0
        or isinstance(max_branches, bool) or not isinstance(max_branches, int)
        or max_branches < 1
        or isinstance(time_limit_s, bool)
        or not isinstance(time_limit_s, (int, float))
        or not np.isfinite(time_limit_s) or time_limit_s <= 0
    ):
        return {
            "success": False, "status": "THEORETICAL_FAILURE",
            "reason": "invalid_search_limits", "plan": [], "execution_actions": [],
            "planning_mode": "theoretical",
            "physical_validation_status": "NOT_TESTED",
        }

    top_row = max(int(hold.row) for hold in holds)
    top_indices = {i for i, hold in enumerate(holds) if hold.row == top_row}
    contacts = {limb: _source_hold(sim, limb) for limb in LIMBS}
    missing = [
        limb for limb, index in contacts.items()
        if index is None or np.linalg.norm(
            np.asarray(sim.ball_positions()[limb]) - _hold_socket(sim, index)
        ) >= 0.006
    ]
    if missing:
        return {
            "success": False, "status": "INVALID_INITIAL_STATE",
            "reason": "initial_support_missing", "missing_supports": missing,
            "top_row": top_row, "max_progress_row": max(
                (int(holds[index].row) for index in contacts.values() if index is not None),
                default=None,
            ),
            "plan": [], "execution_actions": [], "steps": [],
            "branches_explored": 0, "nodes_explored": 0,
            "planning_mode": "theoretical",
            "physical_validation_status": "NOT_TESTED",
        }

    initial_pose = sim.trunk_pose()
    initial = {
        "contacts": contacts,
        "q": np.asarray(sim.joint_positions(), dtype=float).copy(),
        "trunk_pos": np.asarray(initial_pose[0], dtype=float).copy(),
        "trunk_quat": np.asarray(initial_pose[1], dtype=float).copy(),
    }
    limits = np.asarray(sim.joint_range, dtype=float)
    limb_reach = {}
    model = getattr(sim, "model", None)
    if model is not None and all(hasattr(model, name) for name in (
        "body_parentid", "body_pos", "site_bodyid"
    )):
        trunk_id = model.body("trunk").id
        for limb in LIMBS:
            site = model.site(f"{limb}_ball")
            body_id = int(model.site_bodyid[site.id])
            reach = float(np.linalg.norm(site.pos))
            while body_id != trunk_id:
                reach += float(np.linalg.norm(model.body_pos[body_id]))
                body_id = int(model.body_parentid[body_id])
            limb_reach[limb] = reach
    deadline = started + float(time_limit_s)
    branches = 0
    nodes = 0
    visited = set()
    limit_reason = None
    best = None
    failure_reasons = Counter()

    def row_progress(state):
        return max(int(holds[index].row) for index in state["contacts"].values())

    def state_key(state):
        quat = state["trunk_quat"]
        if quat[0] < 0:
            quat = -quat
        return (
            tuple(state["contacts"][limb] for limb in LIMBS),
            tuple(np.rint(state["q"] / 0.05).astype(int)),
            tuple(np.rint(state["trunk_pos"] / 0.005).astype(int)),
            tuple(np.rint(quat / 0.02).astype(int)),
        )

    def serialize_state(state):
        return {
            "contacts": {
                limb: {
                    "hold_index": index,
                    "coords": (int(holds[index].col), int(holds[index].row)),
                }
                for limb, index in state["contacts"].items()
            },
            "joint_angles_rad": state["q"].tolist(),
            "trunk_pose": {
                "position_m": state["trunk_pos"].tolist(),
                "quaternion_wxyz": state["trunk_quat"].tolist(),
            },
            "max_contact_row": row_progress(state),
        }

    def ranking(state, sequence, min_margin, candidates=()):
        future_rows = [
            int(item["target_coords"][1])
            for item in candidates if item["kind"] == "transfer"
        ]
        progressing_limbs = {
            item["limb"]
            for item in candidates
            if item["kind"] == "transfer" and item["gain"] > 0
        }
        return (
            row_progress(state),
            max(future_rows, default=row_progress(state)),
            len(progressing_limbs),
            len(state["contacts"]),
            min_margin,
            -len(sequence),
        )

    def solve(targets, state, trunk_pos=None, trunk_quat=None):
        q, error = sim.joint_angles_for(
            targets,
            state["trunk_pos"] if trunk_pos is None else trunk_pos,
            state["trunk_quat"] if trunk_quat is None else trunk_quat,
            q_start=state["q"],
        )
        q = np.asarray(q, dtype=float)
        invalid = _ik_command_error(sim, q, error, "theoretical")
        if invalid:
            failure_reasons[invalid["reason"]] += 1
            return None
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        return q, float(error), margin

    def transfer_candidates(state):
        occupied = set(state["contacts"].values())
        results = []
        current_balls = sim.ball_positions_at(
            state["q"], state["trunk_pos"], state["trunk_quat"]
        )
        for limb in LIMBS:
            source = state["contacts"][limb]
            supports = {
                member: current_balls[member]
                for member, index in state["contacts"].items()
                if member != limb
            }
            for target_index, hold in enumerate(holds):
                if time.monotonic() >= deadline:
                    return results, True
                if target_index in occupied or int(hold.row) < int(
                    holds[source].row
                ):
                    continue
                target = _hold_socket(sim, target_index)
                if (
                    limb in limb_reach
                    and np.linalg.norm(target - state["trunk_pos"])
                    > limb_reach[limb] + 0.001
                ):
                    failure_reasons["outside_limb_reach_bound"] += 1
                    continue
                solved = solve({limb: target}, state)
                if solved is None:
                    continue
                q, error, margin = solved
                predicted_balls = sim.ball_positions_at(
                    q, state["trunk_pos"], state["trunk_quat"]
                )
                support_drift = max(
                    float(np.linalg.norm(predicted_balls[member] - position))
                    for member, position in supports.items()
                )
                if support_drift > 0.001:
                    failure_reasons["theoretical_support_geometry_violation"] += 1
                    continue
                trajectory, trajectory_error = _theoretical_transfer_path(
                    sim,
                    limb,
                    current_balls[limb],
                    target,
                    source,
                    target_index,
                    supports,
                    state["trunk_pos"],
                    state["trunk_quat"],
                    state["q"],
                    deadline=deadline,
                    config=trajectory_config,
                )
                if (
                    trajectory_error
                    and trajectory_error["reason"] == "time_limit_reached"
                ):
                    return results, True
                if trajectory_error:
                    failure_reasons[trajectory_error["reason"]] += 1
                    continue
                q = np.asarray(
                    trajectory["waypoints"][-1]["joint_angles_rad"],
                    dtype=float,
                )
                error = trajectory["max_ik_error_m"]
                margin = trajectory["min_joint_limit_margin_rad"]
                results.append({
                    "kind": "transfer",
                    "limb": limb,
                    "source_hold": source,
                    "target_hold": target_index,
                    "target_coords": (int(hold.col), int(hold.row)),
                    "target_q": q,
                    "ik_error_m": error,
                    "joint_limit_margin_rad": margin,
                    "gain": int(hold.row) - int(holds[source].row),
                    "distance_m": float(np.linalg.norm(
                        current_balls[limb] - _hold_socket(sim, target_index)
                    )),
                    "trunk_pos": state["trunk_pos"].copy(),
                    "trunk_quat": state["trunk_quat"].copy(),
                    "targets": {limb: target, **supports},
                    "waypoints": trajectory["waypoints"],
                    "max_joint_step_rad": trajectory["max_joint_step_rad"],
                })
        results.sort(key=lambda item: _candidate_rank({
            "gain": item["gain"],
            "distance_m": item["distance_m"],
            "ik_error_m": item["ik_error_m"],
            "joint_effort_rad": float(np.linalg.norm(item["target_q"] - state["q"])),
            "joint_limit_margin_rad": item["joint_limit_margin_rad"],
        }, "combined"))
        return results, False

    def reposition_candidates(state):
        targets = {
            limb: _hold_socket(sim, index)
            for limb, index in state["contacts"].items()
        }
        results = []
        requests = []
        for axis in range(3):
            for direction in (1.0, -1.0):
                offset = np.zeros(3)
                distances = (0.01, 0.025) if axis == 2 else (0.01,)
                requests.extend(
                    (
                        np.eye(3)[axis] * distance * direction,
                        np.zeros(3),
                    )
                    for distance in distances
                )
        for axis in range(3):
            for direction in (1.0, -1.0):
                rotation = np.zeros(3)
                rotation[axis] = np.deg2rad(5.0) * direction
                requests.append((np.zeros(3), rotation))
        for offset, rotation in requests:
            if time.monotonic() >= deadline:
                return results, True
            target_pos = state["trunk_pos"] + offset
            half_angle = float(np.linalg.norm(rotation)) / 2.0
            if half_angle:
                axis = rotation / np.linalg.norm(rotation)
                delta_quat = np.concatenate((
                    [np.cos(half_angle)],
                    axis * np.sin(half_angle),
                ))
                target_quat = _quat_multiply(delta_quat, state["trunk_quat"])
            else:
                target_quat = state["trunk_quat"]
            solved = solve(targets, state, target_pos, target_quat)
            if solved is None:
                continue
            q, error, margin = solved
            results.append({
                "kind": "reposition",
                "target_q": q,
                "ik_error_m": error,
                "joint_limit_margin_rad": margin,
                "requested_trunk_offset_m": offset,
                "requested_trunk_rotation_rad": rotation,
                "trunk_pos": target_pos,
                "trunk_quat": target_quat.copy(),
                "targets": targets,
            })
        return results, False

    def visit(state, plan, sequence, steps, min_margin, consecutive_repositions=0):
        nonlocal branches, nodes, limit_reason, best
        nodes += 1
        if any(index in top_indices for index in state["contacts"].values()):
            entry = {
                "state": state,
                "plan": plan,
                "sequence": sequence,
                "steps": steps,
                "ranking": ranking(state, sequence, min_margin),
                "progress": row_progress(state),
            }
            return {**entry, "success": True, "reason": "theoretical_top_reached"}
        if time.monotonic() >= deadline:
            limit_reason = "time_limit_reached"
            return None
        if len(sequence) >= max_moves:
            limit_reason = "max_moves_reached"
            return None
        key = state_key(state)
        if key in visited:
            failure_reasons["repeated_state"] += 1
            return None
        visited.add(key)

        candidates, timed_out = transfer_candidates(state)
        if timed_out:
            limit_reason = "time_limit_reached"
            candidates = candidates or []
        lateral_candidates = [
            item for item in candidates if item["gain"] <= 0
        ]
        candidates = [
            item for item in candidates if item["gain"] > 0
        ]
        score = ranking(state, sequence, min_margin, candidates)
        entry = {
            "state": state,
            "plan": plan,
            "sequence": sequence,
            "steps": steps,
            "ranking": score,
            "progress": row_progress(state),
        }
        if best is None or score > best["ranking"]:
            best = entry
        if timed_out:
            return None
        if not candidates and consecutive_repositions >= 2:
            failure_reasons["no_ik_feasible_action"] += 1
            return None
        candidate_index = 0
        repositions_generated = False
        while True:
            if candidate_index >= len(candidates):
                if repositions_generated or consecutive_repositions >= 2:
                    break
                if branches >= max_branches:
                    limit_reason = "branch_limit_reached"
                    return None
                if time.monotonic() >= deadline:
                    limit_reason = "time_limit_reached"
                    return None
                repositions, timed_out = reposition_candidates(state)
                repositions_generated = True
                if timed_out:
                    limit_reason = "time_limit_reached"
                    return None
                repositions.sort(key=lambda item: (
                    -float(item["requested_trunk_offset_m"][2]),
                    -float(np.linalg.norm(item["requested_trunk_rotation_rad"])),
                    float(np.linalg.norm(item["requested_trunk_offset_m"])),
                ))
                candidates.extend(repositions)
                candidates.extend(lateral_candidates)
                if candidate_index >= len(candidates):
                    failure_reasons["no_ik_feasible_action"] += 1
                    break
            if branches >= max_branches:
                limit_reason = "branch_limit_reached"
                return None
            if time.monotonic() >= deadline:
                limit_reason = "time_limit_reached"
                return None
            candidate = candidates[candidate_index]
            candidate_index += 1
            branches += 1
            target_q = candidate["target_q"]
            target_pos = candidate["trunk_pos"]
            target_quat = candidate["trunk_quat"]
            predicted_balls = sim.ball_positions_at(target_q, target_pos, target_quat)
            if candidate["kind"] == "transfer":
                limb = candidate["limb"]
                source = candidate["source_hold"]
                target = candidate["target_hold"]
                action = {
                    "kind": "transfer",
                    "status": "THEORETICAL",
                    "physical_validation_status": "NOT_TESTED",
                    "limb": limb,
                    "source_hold": source,
                    "source_coords": (int(holds[source].col), int(holds[source].row)),
                    "target_hold": target,
                    "target_coords": candidate["target_coords"],
                    "move": {
                        "limb": limb,
                        "target_coords": candidate["target_coords"],
                        "waypoints": candidate["waypoints"],
                    },
                    "trajectory_validation": "THEORETICAL_IK_VALIDATED",
                    "max_joint_waypoint_step_rad": candidate[
                        "max_joint_step_rad"
                    ],
                    "target_joint_angles_rad": target_q.tolist(),
                    "trunk_pose": {
                        "position_m": target_pos.tolist(),
                        "quaternion_wxyz": target_quat.tolist(),
                    },
                    "ik_result": {
                        "valid": True,
                        "error_m": candidate["ik_error_m"],
                        "joint_limit_margin_rad": candidate["joint_limit_margin_rad"],
                    },
                    "supports_during_transfer": {
                        member: hold_index
                        for member, hold_index in state["contacts"].items()
                        if member != limb
                    },
                    "predicted_ball_positions_m": {
                        member: np.asarray(position).tolist()
                        for member, position in predicted_balls.items()
                    },
                    "physical_failure_reason": None,
                }
                next_contacts = dict(state["contacts"])
                next_contacts[limb] = target
                next_plan = plan + [{
                    "limb": limb,
                    "target_coords": candidate["target_coords"],
                }]
                next_repositions = 0
            else:
                current_balls = sim.ball_positions_at(
                    state["q"], state["trunk_pos"], state["trunk_quat"]
                )
                action = {
                    "kind": "reposition",
                    "status": "THEORETICAL",
                    "physical_validation_status": "NOT_TESTED",
                    "target_q": target_q.tolist(),
                    "ticks": 50,
                    "requested_trunk_offset_m": candidate["requested_trunk_offset_m"].tolist(),
                    "requested_trunk_rotation_rad": candidate[
                        "requested_trunk_rotation_rad"
                    ].tolist(),
                    "trunk_pose": {
                        "start_position_m": state["trunk_pos"].tolist(),
                        "position_m": target_pos.tolist(),
                        "quaternion_wxyz": target_quat.tolist(),
                    },
                    "target_joint_angles_rad": target_q.tolist(),
                    "ik_result": {
                        "valid": True,
                        "error_m": candidate["ik_error_m"],
                        "joint_limit_margin_rad": candidate["joint_limit_margin_rad"],
                    },
                    "support_holds": dict(state["contacts"]),
                    "trunk_displacement_m": float(np.linalg.norm(target_pos - state["trunk_pos"])),
                    "trunk_rotation_rad": float(np.linalg.norm(
                        candidate["requested_trunk_rotation_rad"]
                    )),
                    "predicted_ball_displacements_m": {
                        member: float(np.linalg.norm(
                            predicted_balls[member] - current_balls[member]
                        ))
                        for member in LIMBS
                    },
                    "physical_failure_reason": None,
                }
                next_contacts = dict(state["contacts"])
                next_plan = plan
                next_repositions = consecutive_repositions + 1
            next_state = {
                "contacts": next_contacts,
                "q": target_q.copy(),
                "trunk_pos": target_pos.copy(),
                "trunk_quat": target_quat.copy(),
            }
            action["resulting_state"] = serialize_state(next_state)
            result = visit(
                next_state,
                next_plan,
                sequence + [action],
                steps + [action],
                min(min_margin, candidate["joint_limit_margin_rad"]),
                next_repositions,
            )
            if result is not None and result["success"]:
                return result
            if limit_reason:
                return None
        return None

    try:
        initial_margin = float(np.min(np.minimum(
            initial["q"] - limits[:, 0], limits[:, 1] - initial["q"]
        )))
        found = visit(initial, [], [], [], initial_margin)
        selected = found if found is not None else best
        if selected is None:
            selected = {
                "plan": [], "sequence": [], "steps": [], "progress": row_progress(initial),
                "ranking": ranking(initial, [], initial_margin),
            }
        success = bool(found and found["success"])
        status = (
            "COMPLETE_THEORETICAL_PATH" if success
            else "THEORETICAL_SEARCH_LIMIT_REACHED" if limit_reason
            else "THEORETICAL_PARTIAL" if selected["sequence"]
            else "THEORETICAL_FAILURE"
        )
        return {
            "success": success,
            "status": status,
            "reason": "theoretical_top_reached" if success else (
                limit_reason or "no_ik_feasible_route"
            ),
            "planning_mode": "theoretical",
            "planner_version": "0.2.0",
            "physical_validation_status": "NOT_TESTED",
            "search_parameters": {
                "max_moves": max_moves,
                "max_branches": max_branches,
                "time_limit_s": float(time_limit_s),
            },
            "trajectory_config": asdict(trajectory_config),
            "initial_state": serialize_state(initial),
            "state_path": [
                serialize_state(initial),
                *[
                    action["resulting_state"]
                    for action in selected["sequence"]
                ],
            ],
            "top_row": top_row,
            "initial_max_progress_row": max(
                int(holds[index].row) for index in contacts.values()
            ),
            "max_progress_row": selected["progress"],
            "plan": selected["plan"],
            "execution_actions": selected["sequence"],
            "steps": selected["steps"],
            "transfer_count": sum(
                action["kind"] == "transfer" for action in selected["sequence"]
            ),
            "reposition_count": sum(
                action["kind"] == "reposition" for action in selected["sequence"]
            ),
            "branches_explored": branches,
            "nodes_explored": nodes,
            "failure_reasons": dict(failure_reasons),
            "elapsed_s": time.monotonic() - started,
            "theoretical_model": "IK endpoint geometry and joint limits; no collision or dynamics validation",
        }
    finally:
        sim.set_state(initial_state)


def plan_to_top(
    sim, max_moves=64, max_branches=128, time_limit_s=60.0,
    search_strategy="combined",
):
    """Search physical transfers and joint-commanded trunk repositionings."""
    initial_state = sim.get_state()
    started = time.monotonic()
    profile = Counter({
        "transfer_generation_s": 0.0,
        "reposition_generation_s": 0.0,
        "post_reposition_generation_s": 0.0,
        "ik_s": 0.0,
        "physical_validation_s": 0.0,
        "state_snapshot_s": 0.0,
        "state_restore_s": 0.0,
        "candidate_generation_timed_out": 0,
    })
    sim._planner_profile = profile
    try:
        if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 0:
            return {
                "success": False, "status": "FAILURE",
                "reason": "invalid_max_moves", "plan": [],
                "steps": [], "branches_explored": 0,
            }
        if isinstance(max_branches, bool) or not isinstance(max_branches, int) or max_branches < 1:
            return {
                "success": False, "status": "FAILURE",
                "reason": "invalid_max_branches", "plan": [],
                "steps": [], "branches_explored": 0,
            }
        if (
            isinstance(time_limit_s, bool)
            or not isinstance(time_limit_s, (int, float))
            or not np.isfinite(time_limit_s)
            or time_limit_s <= 0
        ):
            return {
                "success": False, "status": "FAILURE",
                "reason": "invalid_time_limit", "plan": [],
                "steps": [], "branches_explored": 0,
            }
        strategies = {
            "legacy", "transfers_first", "progress", "balanced",
            "useful_repositions", "combined",
        }
        if search_strategy not in strategies:
            return {
                "success": False, "status": "FAILURE",
                "reason": "invalid_search_strategy", "plan": [],
                "steps": [], "branches_explored": 0,
            }
        holds = sim.wall.holds
        if not holds:
            return {
                "success": False, "status": "FAILURE",
                "reason": "wall_has_no_holds", "plan": [],
                "steps": [], "branches_explored": 0,
            }

        top_row = max(hold.row for hold in holds)
        top_indices = {
            index for index, hold in enumerate(holds) if hold.row == top_row
        }
        missing_supports = []
        for limb in LIMBS:
            source = _source_hold(sim, limb)
            if source is None or np.linalg.norm(
                np.asarray(sim.ball_positions()[limb]) - _hold_socket(sim, source)
            ) >= 0.006:
                missing_supports.append(limb)
        if missing_supports:
            initial_rows = [
                holds[index].row
                for limb in LIMBS
                for index in _contacting_holds(sim, limb)
            ]
            return {
                "success": False,
                "status": "INVALID_INITIAL_STATE",
                "reason": "initial_support_missing",
                "missing_supports": missing_supports,
                "top_row": int(top_row),
                "top_hold_indices": sorted(top_indices),
                "plan": [],
                "execution_actions": [],
                "steps": [],
                "step_reached": 0,
                "branches_explored": 0,
                "failure_reasons": {},
                "nodes_explored": 0,
                "transfer_candidates_tested": 0,
                "reposition_candidates_tested": 0,
                "repositions_validated": 0,
                "transfers_validated": 0,
                "support_checks": 0,
                "supports_retained": 0,
                "unexpected_contacts_observed": 0,
                "max_progress_row": max(initial_rows) if initial_rows else None,
                "elapsed_s": time.monotonic() - started,
                "profile": profile,
            }

        branches = 0
        visited = set()
        failures = Counter()
        counters = Counter()
        candidate_diagnostics = []
        deadline = started + float(time_limit_s)
        sim._planner_deadline = deadline
        limit_reason = None

        def progress():
            return max(
                holds[index].row
                for limb in LIMBS
                for index in _contacting_holds(sim, limb)
            )

        def reached_top():
            return any(
                _contacting_holds(sim, limb) & top_indices for limb in LIMBS
            )

        def partial_ranking(plan, sequence, potential=(0, 0.0, 0)):
            if search_strategy == "legacy":
                return progress(), len(plan), len(sequence)
            return (
                progress(),
                len(plan),
                _support_quality(sim),
                potential,
                -len(sequence),
            )

        def search(plan, steps, sequence, consecutive_repositions=0):
            nonlocal branches, limit_reason
            counters["nodes_explored"] += 1
            if reached_top():
                return {
                    "success": True, "reason": "top_hold_contacted",
                    "plan": plan, "steps": steps, "sequence": sequence,
                    "progress": progress(),
                    "ranking": partial_ranking(plan, sequence),
                }
            if time.monotonic() >= deadline:
                limit_reason = "time_limit_reached"
                failures[limit_reason] += 1
                return {
                    "success": False, "reason": limit_reason,
                    "plan": plan, "steps": steps, "sequence": sequence,
                    "progress": progress(),
                    "ranking": partial_ranking(plan, sequence),
                }
            if len(sequence) >= max_moves:
                limit_reason = "max_moves_reached"
                failures[limit_reason] += 1
                return {
                    "success": False, "reason": limit_reason,
                    "plan": plan, "steps": steps, "sequence": sequence,
                    "progress": progress(),
                    "ranking": partial_ranking(plan, sequence),
                }

            state_key = _search_state(sim)
            if state_key in visited:
                failures["repeated_state"] += 1
                return {
                    "success": False, "reason": "repeated_state",
                    "plan": plan, "steps": steps, "sequence": sequence,
                    "progress": progress(),
                    "ranking": partial_ranking(plan, sequence),
                }
            visited.add(state_key)

            occupied = {
                hold_index
                for limb in LIMBS
                for hold_index in _contacting_holds(sim, limb)
            }
            if time.monotonic() >= deadline:
                limit_reason = "time_limit_reached"
                failures[limit_reason] += 1
                return {
                    "success": False, "reason": limit_reason,
                    "plan": plan, "steps": steps, "sequence": sequence,
                    "progress": progress(),
                    "ranking": partial_ranking(plan, sequence),
                }
            generation_start = time.perf_counter()
            transfers = _candidate_moves(sim, occupied)
            profile["transfer_generation_s"] += time.perf_counter() - generation_start
            if (
                time.monotonic() >= deadline
                or profile["candidate_generation_timed_out"]
            ):
                limit_reason = "time_limit_reached"
                failures[limit_reason] += 1
                return {
                    "success": False, "reason": limit_reason,
                    "plan": plan, "steps": steps, "sequence": sequence,
                    "progress": progress(),
                    "ranking": partial_ranking(plan, sequence),
                }
            transfers = [_normalize_candidate(item) for item in transfers]
            transfers.sort(key=lambda item: _candidate_rank(item, search_strategy))
            candidates = [("transfer", item) for item in transfers]
            repositions_generated = False
            if search_strategy == "legacy":
                generation_start = time.perf_counter()
                repositions = _candidate_repositions(sim)
                profile["reposition_generation_s"] += (
                    time.perf_counter() - generation_start
                )
                if (
                    time.monotonic() >= deadline
                    or profile["candidate_generation_timed_out"]
                ):
                    limit_reason = "time_limit_reached"
                    failures[limit_reason] += 1
                    return {
                        "success": False, "reason": limit_reason,
                        "plan": plan, "steps": steps, "sequence": sequence,
                        "progress": progress(),
                        "ranking": partial_ranking(plan, sequence),
                    }
                candidates = [
                    ("reposition", item) for item in repositions
                ] + candidates
                repositions_generated = True
            if search_strategy != "legacy" and consecutive_repositions >= 2:
                candidates = [item for item in candidates if item[0] != "reposition"]
            direct_potential = _frontier_utility(transfers, search_strategy)
            node_ranking = partial_ranking(
                plan, sequence, direct_potential
            )
            best = None
            snapshot_started = time.perf_counter()
            state = sim.get_state()
            profile["state_snapshot_s"] += time.perf_counter() - snapshot_started
            candidate_index = 0
            branches_at_node = branches
            while True:
                if candidate_index >= len(candidates):
                    if search_strategy == "legacy" or repositions_generated:
                        break
                    if branches >= max_branches:
                        limit_reason = "branch_limit_reached"
                        failures[limit_reason] += 1
                        break
                    if time.monotonic() >= deadline:
                        limit_reason = "time_limit_reached"
                        failures[limit_reason] += 1
                        break
                    generation_start = time.perf_counter()
                    repositions = _candidate_repositions(sim)
                    profile["reposition_generation_s"] += (
                        time.perf_counter() - generation_start
                    )
                    repositions_generated = True
                    if (
                        time.monotonic() >= deadline
                        or profile["candidate_generation_timed_out"]
                    ):
                        limit_reason = "time_limit_reached"
                        failures[limit_reason] += 1
                        break
                    if consecutive_repositions < 2:
                        candidates.extend(
                            ("reposition", item) for item in repositions
                        )
                    if candidate_index >= len(candidates):
                        break
                if time.monotonic() >= deadline:
                    limit_reason = "time_limit_reached"
                    failures[limit_reason] += 1
                    break
                if branches >= max_branches:
                    limit_reason = "branch_limit_reached"
                    failures[limit_reason] += 1
                    break
                kind, candidate = candidates[candidate_index]
                candidate_index += 1
                branches += 1
                restore_started = time.perf_counter()
                sim.set_state(state)
                profile["state_restore_s"] += time.perf_counter() - restore_started
                if kind == "transfer":
                    limb = candidate["limb"]
                    hold_index = candidate["hold_index"]
                    counters["transfer_candidates_tested"] += 1
                    action = {
                        "limb": limb,
                        "target_coords": (
                            int(sim.wall.holds[hold_index].col),
                            int(sim.wall.holds[hold_index].row),
                        ),
                    }
                    validation_started = time.perf_counter()
                    outcome = validate_plan(sim, [action], restore=False)
                    profile["physical_validation_s"] += time.perf_counter() - validation_started
                    diagnostic = {
                        "kind": "transfer",
                        "limb": limb,
                        "source_hold": candidate.get("source_hold"),
                        "source_coords": (
                            None if candidate.get("source_hold") is None
                            else (
                                int(sim.wall.holds[candidate["source_hold"]].col),
                                int(sim.wall.holds[candidate["source_hold"]].row),
                            )
                        ),
                        "target_hold": hold_index,
                        "target_coords": action["target_coords"],
                        "target_joint_angles_rad": candidate.get("target_q"),
                        "trunk_pose": {
                            "position_m": candidate.get("trunk_pos"),
                            "quaternion_wxyz": candidate.get("trunk_quat"),
                        },
                        "ik_result": {
                            "valid": True,
                            "error_m": candidate.get("ik_error_m"),
                        },
                        "physical_validation_status": (
                            "PHYSICALLY_VALIDATED"
                            if outcome["success"] else "PHYSICALLY_FAILED"
                        ),
                        "physical_failure_reason": (
                            None if outcome["success"] else outcome["reason"]
                        ),
                    }
                    candidate_diagnostics.append(diagnostic)
                    if not outcome["success"]:
                        failures[outcome["reason"]] += 1
                        continue
                    counters["transfers_validated"] += 1
                    for measurement in outcome["steps"][0].get("waypoints", []):
                        support_states = measurement.get("supports", {}).values()
                        counters["support_checks"] += len(support_states)
                        counters["supports_retained"] += sum(
                            state.get("valid", False) for state in support_states
                        )
                    support_states = outcome["steps"][0].get("supports", {}).values()
                    counters["support_checks"] += len(support_states)
                    counters["supports_retained"] += sum(
                        state.get("valid", False) for state in support_states
                    )
                    counters["unexpected_contacts_observed"] += len(
                        outcome["steps"][0].get("collision_contacts", [])
                    )
                    next_plan = plan + [action]
                    next_steps = steps + [outcome["steps"][0]]
                    next_sequence = sequence + [{
                        "kind": "transfer",
                        "move": outcome.get("plan", [action])[0],
                        **diagnostic,
                        "status": "PHYSICALLY_VALIDATED",
                    }]
                    next_consecutive_repositions = 0
                else:
                    counters["reposition_candidates_tested"] += 1
                    validation_started = time.perf_counter()
                    outcome = _run_reposition(sim, candidate)
                    profile["physical_validation_s"] += time.perf_counter() - validation_started
                    if not outcome["success"]:
                        failures[outcome["reason"]] += 1
                        counters["support_checks"] += outcome.get("support_checks", 0)
                        counters["supports_retained"] += outcome.get("supports_retained", 0)
                        counters["unexpected_contacts_observed"] += outcome.get(
                            "unexpected_contacts_observed", 0
                        )
                        continue
                    counters["repositions_validated"] += 1
                    counters["support_checks"] += outcome["support_checks"]
                    counters["supports_retained"] += outcome["supports_retained"]
                    counters["unexpected_contacts_observed"] += outcome[
                        "unexpected_contacts_observed"
                    ]
                    if search_strategy in {"useful_repositions", "combined"}:
                        generation_start = time.perf_counter()
                        sim._planner_deadline = deadline
                        after_transfers = [
                            _normalize_candidate(item)
                            for item in _candidate_moves(sim, occupied)
                        ]
                        profile["post_reposition_generation_s"] += (
                            time.perf_counter() - generation_start
                        )
                        if (
                            time.monotonic() >= deadline
                            or profile["candidate_generation_timed_out"]
                        ):
                            limit_reason = "time_limit_reached"
                            failures[limit_reason] += 1
                            break
                        after_potential = _frontier_utility(
                            after_transfers, search_strategy
                        )
                        if after_potential <= direct_potential:
                            failures["reposition_no_transfer_benefit"] += 1
                            continue
                    next_plan = plan
                    next_steps = steps
                    next_sequence = sequence + [{
                        "kind": "reposition", **outcome,
                        "requested_trunk_rotation_rad": candidate.get(
                            "requested_trunk_rotation_rad", [0.0, 0.0, 0.0]
                        ),
                        "target_trunk_pos": candidate.get("target_trunk_pos"),
                        "target_trunk_quat": candidate.get("target_trunk_quat"),
                        "ik_result": {
                            "valid": True,
                            "error_m": candidate.get("ik_error_m"),
                        },
                        "status": "PHYSICALLY_VALIDATED",
                        "physical_validation_status": "PHYSICALLY_VALIDATED",
                    }]
                    next_consecutive_repositions = consecutive_repositions + 1

                result = search(
                    next_plan, next_steps, next_sequence,
                    next_consecutive_repositions,
                )
                if result["success"]:
                    return result
                if best is None or result["ranking"] > best["ranking"]:
                    best = result
                if limit_reason:
                    break

            restore_started = time.perf_counter()
            sim.set_state(state)
            profile["state_restore_s"] += time.perf_counter() - restore_started
            if best is not None:
                return best
            if branches == branches_at_node and not limit_reason:
                failures["no_reachable_action"] += 1
                reason = "no_reachable_action"
            else:
                reason = limit_reason or "no_feasible_progression"
            return {
                "success": False,
                "reason": reason,
                "plan": plan,
                "steps": steps,
                "sequence": sequence,
                "progress": progress(),
                "ranking": node_ranking,
            }

        result = search([], [], [])
        status = "SUCCESS" if result["success"] else (
            "SEARCH_LIMIT_REACHED" if limit_reason else
            "PARTIAL" if result["sequence"] else "FAILURE"
        )
        return {
            "success": result["success"],
            "status": status,
            "reason": result["reason"],
            "top_row": int(top_row),
            "top_hold_indices": sorted(top_indices),
            "plan": result["plan"],
            "execution_actions": result["sequence"],
            "steps": result["steps"],
            "step_reached": len(result["plan"]),
            "branches_explored": branches,
            "nodes_explored": counters["nodes_explored"],
            "transfer_candidates_tested": counters["transfer_candidates_tested"],
            "reposition_candidates_tested": counters["reposition_candidates_tested"],
            "repositions_validated": counters["repositions_validated"],
            "transfers_validated": counters["transfers_validated"],
            "support_checks": counters["support_checks"],
            "supports_retained": counters["supports_retained"],
            "unexpected_contacts_observed": counters["unexpected_contacts_observed"],
            "max_progress_row": int(result["progress"]),
            "elapsed_s": time.monotonic() - started,
            "failure_reasons": dict(failures),
            "candidate_diagnostics": candidate_diagnostics,
            "planning_mode": "physical",
            "physical_validation_status": (
                "PHYSICALLY_VALIDATED" if result["success"] else "PARTIAL"
            ),
            "profile": profile,
            "search_strategy": search_strategy,
        }
    finally:
        sim.__dict__.pop("_planner_profile", None)
        sim.__dict__.pop("_planner_deadline", None)
        restore_started = time.perf_counter()
        sim.set_state(initial_state)
        profile["state_restore_s"] += time.perf_counter() - restore_started
        profile["total_s"] = time.monotonic() - started


def plan_path(sim, holds, max_branches=32):
    """Find the first feasible sequence, backtracking over limb choices.

    A target can be ``{"coords": (col, row)}``, ``{"hold_index": i}``, or
    an explicit move with ``limb``. This bounded depth-first search is not a
    global path-length or stability optimizer.
    """
    initial_state = sim.get_state()
    try:
        if not isinstance(holds, Sequence) or isinstance(holds, (str, bytes, dict)):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "invalid_targets:expected_sequence",
                "plan": [],
                "steps": [],
            }
        if isinstance(max_branches, bool) or not isinstance(max_branches, int) or max_branches < 1:
            return {
                "success": False,
                "step_reached": 0,
                "reason": "invalid_max_branches",
                "plan": [],
                "steps": [],
            }

        branches = 0

        def search(index):
            nonlocal branches
            if index == len(holds):
                return {"success": True, "plan": [], "steps": []}

            target = holds[index]
            template = dict(target) if isinstance(target, dict) else {"target_coords": target}
            if "target_coords" not in template and "coords" in template:
                template["target_coords"] = template["coords"]
            candidates = [template["limb"]] if "limb" in template else LIMBS
            reference_state = sim.get_state()
            failures = []
            best_failure = None

            for limb in candidates:
                if branches >= max_branches:
                    return {
                        "success": False,
                        "failed_step": index + 1,
                        "reason": "branch_limit_reached",
                        "candidate_failures": failures,
                    }
                branches += 1
                sim.set_state(reference_state)
                action = {**template, "limb": limb}
                outcome = validate_plan(sim, [action], restore=False)
                if not outcome["success"]:
                    failures.append({
                        "limb": limb,
                        "reason": outcome["reason"],
                        "validation": outcome["steps"][0] if outcome["steps"] else {},
                    })
                    continue

                remaining = search(index + 1)
                if remaining["success"]:
                    step = {
                        "success": True,
                        "target": target,
                        "limb": limb,
                        "validation": outcome["steps"][0],
                        "candidate_failures": failures,
                    }
                    return {
                        "success": True,
                        "plan": outcome["plan"] + remaining["plan"],
                        "steps": [step] + remaining["steps"],
                    }
                failures.append({
                    "limb": limb,
                    "reason": "sequence_cannot_continue",
                    "diagnostic": remaining,
                })
                branch_failure = {
                    **remaining,
                    "candidate_failures": failures.copy(),
                    "partial_plan": outcome["plan"] + remaining.get("partial_plan", []),
                    "partial_steps": [{
                        "success": True,
                        "target": target,
                        "limb": limb,
                        "validation": outcome["steps"][0],
                        "candidate_failures": failures[:-1],
                    }] + remaining.get("partial_steps", []),
                }
                if (
                    best_failure is None
                    or branch_failure.get("failed_step", 0)
                    > best_failure.get("failed_step", 0)
                ):
                    best_failure = branch_failure
                if remaining["reason"] == "branch_limit_reached":
                    sim.set_state(reference_state)
                    return branch_failure

            sim.set_state(reference_state)
            return best_failure or {
                "success": False,
                "failed_step": index + 1,
                "reason": "no_feasible_limb",
                "candidate_failures": failures,
                "partial_plan": [],
                "partial_steps": [],
            }

        searched = search(0)
        if not searched["success"]:
            return {
                "success": False,
                "step_reached": searched.get("failed_step", 0) - 1,
                "failed_step": searched.get("failed_step"),
                "reason": searched["reason"],
                "plan": searched.get("partial_plan", []),
                "steps": searched.get("partial_steps", []),
                "candidate_failures": searched.get("candidate_failures", []),
                "branches_explored": branches,
                "max_branches": max_branches,
            }
        reason = (
            "success_with_contacts"
            if any(not step["validation"].get("collision_free", True) for step in searched["steps"])
            else "success"
        )
        return {
            "success": True,
            "step_reached": len(searched["steps"]),
            "reason": reason,
            "plan": searched["plan"],
            "steps": searched["steps"],
            "branches_explored": branches,
            "max_branches": max_branches,
        }
    finally:
        sim.set_state(initial_state)
