"""Execute a climbing plan against measured simulator state."""

import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from climbing_sim import LIMBS
from climbing_sim.scene import HOLD_SPHERES


_HOLD_NAME = re.compile(r"^hold(\d+)_")
_BALL_RADIUS = 0.0125
_HOOK_TOLERANCE = 0.006
_MAX_IK_ERROR = 0.001
_MIN_WAYPOINT_PROGRESS_M = 0.0001
_MAX_WAYPOINT_ERROR_M = 0.006
_STALLED_WAYPOINT_LIMIT = 2


@dataclass(frozen=True)
class TransferTrajectoryConfig:
    exit_clearance_m: float = 0.045
    wall_standoff_margin_m: float = 0.005
    approach_height_m: float = 0.045
    max_waypoint_spacing_m: float = 0.005
    max_insertion_step_m: float = 0.005
    max_joint_step_rad: float = 0.15
    socket_centering_tolerance_m: float = _MAX_IK_ERROR
    exit_ticks_per_waypoint: int = 30
    transit_ticks_per_waypoint: int = 80
    approach_ticks_per_waypoint: int = 80
    insertion_ticks_per_waypoint: int = 100


DEFAULT_TRANSFER_TRAJECTORY = TransferTrajectoryConfig()
_PHASE_ORDER = {
    "exit": 0,
    "transit": 1,
    "transfer": 1,
    "approach": 2,
    "insertion": 3,
    "target": 3,
}
_PHASE_STATES = {
    "exit": "EXITING",
    "transit": "TRANSITING",
    "transfer": "TRANSITING",
    "approach": "APPROACHING",
    "insertion": "INSERTING",
    "target": "INSERTING",
}
_REQUIRED_TRANSFER_PHASES = {"exit", "transit", "approach", "insertion"}


class WaypointPhaseError(ValueError):
    """A waypoint path has an unknown or inconsistent phase."""


def _hold_index(sim, geom_id):
    body_id = sim.model.geom_bodyid[geom_id]
    match = _HOLD_NAME.match(sim.model.body(body_id).name)
    return int(match.group(1)) if match else None


def _contacting_holds(sim, limb):
    ball_geom = sim.model.geom(f"{limb}_ball_coll").id
    result = set()
    for contact in sim.data.contact:
        if contact.dist > 0.0:
            continue
        if contact.geom1 == ball_geom:
            hold = _hold_index(sim, contact.geom2)
        elif contact.geom2 == ball_geom:
            hold = _hold_index(sim, contact.geom1)
        else:
            continue
        if hold is not None:
            result.add(hold)
    return result


def has_hold_contact(sim, limb, hold_index):
    """Return whether the limb ball is in MuJoCo contact with a hold."""
    return hold_index in _contacting_holds(sim, limb)


def _resolve_hold(sim, target, holds):
    if isinstance(target, dict):
        coords = target.get("target_coords", target.get("coords"))
        index = target.get("hold_index", target.get("target_index"))
        if coords is not None:
            coords = tuple(coords)
            for i, hold in enumerate(holds):
                if (hold.col, hold.row) == coords:
                    return i
            raise ValueError(f"No hold at grid coordinates {coords}")
        if index is None:
            index = target.get("id")
        if index is None:
            raise ValueError("Move requires target_coords or hold_index")
        return int(index)
    if isinstance(target, (int, np.integer)):
        return int(target)
    for i, hold in enumerate(holds):
        if target is hold:
            return i
    raise ValueError(f"Cannot resolve hold target {target!r}")


def _hold_socket(sim, hold_index):
    hold = sim.wall.holds[hold_index]
    return np.asarray(hold.socket, dtype=float)


def _source_hold(sim, limb):
    contacts = _contacting_holds(sim, limb)
    if not contacts:
        return None
    ball = np.asarray(sim.ball_positions()[limb], dtype=float)
    return min(contacts, key=lambda i: np.linalg.norm(ball - _hold_socket(sim, i)))


def _support_snapshot(sim, moving_limb):
    positions = sim.ball_positions()
    supports = {}
    for limb in LIMBS:
        if limb == moving_limb:
            continue
        source = _source_hold(sim, limb)
        if source is None:
            return None, f"support_lost_before_move:{limb}"
        supports[limb] = {
            "hold_index": source,
            "position": np.asarray(positions[limb], dtype=float).copy(),
        }
    return supports, None


def _contact_pairs(sim):
    return {
        tuple(sorted((c.geom1, c.geom2)))
        for c in sim.data.contact
        if c.dist <= 0.0
    }


def _collision_geometry(sim):
    root_by_geom = np.asarray(sim.model.body_rootid)[sim.model.geom_bodyid]
    holds_by_geom = []
    details_by_geom = []
    for geom_id, body_id in enumerate(sim.model.geom_bodyid):
        body_name = sim.model.body(body_id).name
        match = _HOLD_NAME.match(body_name)
        holds_by_geom.append(int(match.group(1)) if match else None)
        details_by_geom.append({
            "body_name": body_name,
            "geom_name": sim.model.geom(geom_id).name,
        })
    return {
        "robot_root": sim.model.body_rootid[sim.model.body("trunk").id],
        "root_by_geom": root_by_geom,
        "holds_by_geom": holds_by_geom,
        "details_by_geom": details_by_geom,
    }


def _new_collisions(sim, expected_contacts, phase=None, geometry=None, cache=None):
    geometry = _collision_geometry(sim) if geometry is None else geometry
    cache = {} if cache is None else cache
    expected_contacts = frozenset(expected_contacts)
    robot_root = geometry["robot_root"]
    collisions = []
    for contact in sim.data.contact:
        if contact.dist > 0.0:
            continue
        pair = tuple(sorted((contact.geom1, contact.geom2)))
        cache_key = (phase, pair)
        is_collision = cache.get(cache_key)
        if is_collision is None:
            geom1, geom2 = pair
            hold1 = geometry["holds_by_geom"][geom1]
            hold2 = geometry["holds_by_geom"][geom2]
            expected = (
                hold1 is not None and (geom2, hold1) in expected_contacts
            ) or (
                hold2 is not None and (geom1, hold2) in expected_contacts
            )
            is_collision = (
                not expected
                and (
                    geometry["root_by_geom"][geom1] == robot_root
                    or geometry["root_by_geom"][geom2] == robot_root
                )
            )
            cache[cache_key] = is_collision
        if is_collision:
            collisions.append(pair)
    return collisions


def _describe_contacts(sim, pairs, initial_pairs=(), phases=None, geometry=None):
    phases = phases or {}
    geometry = _collision_geometry(sim) if geometry is None else geometry
    distances = {}
    for contact in sim.data.contact:
        if contact.dist <= 0.0:
            pair = tuple(sorted((contact.geom1, contact.geom2)))
            distances[pair] = min(distances.get(pair, float("inf")), float(contact.dist))
    return [
        {
            "geom_ids": list(pair),
            "body_names": [geometry["details_by_geom"][geom]["body_name"] for geom in pair],
            "geom_names": [geometry["details_by_geom"][geom]["geom_name"] for geom in pair],
            "preexisting": pair in initial_pairs,
            "phases": sorted(phases.get(pair, ())),
            "distance_m": distances.get(pair),
            "currently_contacting": pair in distances,
        }
        for pair in sorted(pairs)
    ]


def _support_status(sim, supports):
    actual = sim.ball_positions()
    status = {}
    for limb, support in supports.items():
        hold_index = support["hold_index"]
        distance = float(np.linalg.norm(
            np.asarray(actual[limb], dtype=float) - support["position"]
        ))
        retained = (
            has_hold_contact(sim, limb, hold_index)
            and distance < _HOOK_TOLERANCE
        )
        status[limb] = {
            "hold_index": hold_index,
            "contact": has_hold_contact(sim, limb, hold_index),
            "drift_m": distance,
            "valid": retained,
        }
    return status


def _explicit_path(action):
    points = action.get("waypoints")
    if points is None:
        return None
    result = []
    previous_phase = -1
    for point_index, item in enumerate(points):
        if isinstance(item, dict):
            position = item.get("position")
            phase = item.get("phase", "transfer")
            ticks = item.get("ticks")
            if ticks is None:
                ticks = {
                    "exit": DEFAULT_TRANSFER_TRAJECTORY.exit_ticks_per_waypoint,
                    "transit": DEFAULT_TRANSFER_TRAJECTORY.transit_ticks_per_waypoint,
                    "transfer": DEFAULT_TRANSFER_TRAJECTORY.transit_ticks_per_waypoint,
                    "approach": DEFAULT_TRANSFER_TRAJECTORY.approach_ticks_per_waypoint,
                    "insertion": DEFAULT_TRANSFER_TRAJECTORY.insertion_ticks_per_waypoint,
                    "target": DEFAULT_TRANSFER_TRAJECTORY.insertion_ticks_per_waypoint,
                }.get(phase)
        else:
            position, phase = item, "transfer"
            ticks = DEFAULT_TRANSFER_TRAJECTORY.transit_ticks_per_waypoint
        if phase not in _PHASE_ORDER:
            raise WaypointPhaseError(f"Unknown waypoint phase {phase!r}")
        if _PHASE_ORDER[phase] < previous_phase:
            raise WaypointPhaseError(
                "Waypoint phases must be ordered exit, transit, approach, insertion"
            )
        previous_phase = _PHASE_ORDER[phase]
        if phase == "target" and point_index != len(points) - 1:
            raise WaypointPhaseError("A target-phase waypoint must be last")
        if isinstance(ticks, bool) or not isinstance(ticks, (int, np.integer)) or ticks <= 0:
            raise ValueError(f"Waypoint ticks must be a positive integer, got {ticks!r}")
        point = np.asarray(position, dtype=float)
        if point.shape != (3,):
            raise ValueError("Each waypoint must contain 3 world-space metres")
        if not np.all(np.isfinite(point)):
            raise ValueError("Waypoint coordinates must be finite")
        result.append((point, phase, int(ticks)))
    if not result:
        raise ValueError("An explicit waypoint path cannot be empty")
    return result


def _validate_trajectory_config(config):
    for name in (
        "exit_clearance_m",
        "wall_standoff_margin_m",
        "approach_height_m",
        "max_waypoint_spacing_m",
        "max_insertion_step_m",
        "max_joint_step_rad",
        "socket_centering_tolerance_m",
    ):
        value = getattr(config, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    for name in (
        "exit_ticks_per_waypoint",
        "transit_ticks_per_waypoint",
        "approach_ticks_per_waypoint",
        "insertion_ticks_per_waypoint",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _wall_frame(sim):
    """Return wall-local lateral, up, outward axes and the front-face point."""
    wall_geom = sim.model.geom("wall").id
    rotation = np.asarray(sim.data.geom_xmat[wall_geom], dtype=float).reshape(3, 3)
    origin = np.asarray(sim.data.geom_xpos[wall_geom], dtype=float)
    size = np.asarray(sim.model.geom_size[wall_geom][:3], dtype=float)
    normal_axis = int(np.argmin(size))
    normal = rotation[:, normal_axis].copy()
    trunk = np.asarray(sim.trunk_pose()[0], dtype=float)
    if np.dot(trunk - origin, normal) < 0:
        normal *= -1
    tangent_axes = [index for index in range(3) if index != normal_axis]
    up_axis = max(tangent_axes, key=lambda axis: abs(rotation[2, axis]))
    up = rotation[:, up_axis].copy()
    if up[2] < 0:
        up *= -1
    lateral = rotation[:, next(axis for axis in tangent_axes if axis != up_axis)].copy()
    if np.dot(np.cross(lateral, up), normal) < 0:
        lateral *= -1
    face = origin + normal * size[normal_axis]
    return lateral, up, normal, face


def _hold_lip_projection(sim, hold_index, outward):
    if hold_index is None:
        return max(
            float(np.dot(outward, sphere["pos"])) + float(sphere["r"])
            for sphere in HOLD_SPHERES
        )
    body_id = next(
        body for body in range(sim.model.nbody)
        if _HOLD_NAME.match(sim.model.body(body).name)
        and int(_HOLD_NAME.match(sim.model.body(body).name).group(1)) == hold_index
    )
    rotation = np.asarray(sim.data.xmat[body_id], dtype=float).reshape(3, 3)
    return max(
        float(np.dot(outward, rotation @ np.asarray(sphere["pos"], dtype=float)))
        + float(sphere["r"])
        for sphere in HOLD_SPHERES
    )


def _append_segment(path, start, end, phase, ticks, max_spacing):
    distance = float(np.linalg.norm(end - start))
    count = max(1, math.ceil(distance / max_spacing))
    for index in range(1, count + 1):
        point = start + (index / count) * (end - start)
        path.append((point, phase, ticks))


def _candidate_paths(
    sim, start, target, source_hold, target_hold=None,
    config=DEFAULT_TRANSFER_TRAJECTORY,
):
    """Build four wall-frame phases; no direct start-to-target interpolation."""
    _validate_trajectory_config(config)
    start = np.asarray(start, dtype=float)
    target = np.asarray(target, dtype=float)
    if start.shape != (3,) or target.shape != (3,):
        raise ValueError("Transfer start and target must be 3D world positions")
    lateral, up, outward, face = _wall_frame(sim)
    lip_projection = max(
        _hold_lip_projection(sim, hold, outward)
        for hold in (source_hold, target_hold)
        if hold is not None
    )
    standoff = max(
        lip_projection + _BALL_RADIUS + config.wall_standoff_margin_m,
        config.wall_standoff_margin_m,
    )
    exit_distance = max(config.exit_clearance_m, standoff)
    source_socket = (
        _hold_socket(sim, source_hold) if source_hold is not None else start
    )
    source_offset = start - source_socket
    source_tangent = source_offset - np.dot(source_offset, outward) * outward

    exit_point = start + outward * exit_distance
    transit_start = source_socket + source_tangent + outward * standoff
    transit_end = (
        target + outward * standoff - up * config.approach_height_m
    )
    approach_point = (
        target + outward * standoff + up * config.approach_height_m
    )
    target_standoff = target + outward * standoff
    path = []
    _append_segment(
        path, start, exit_point, "exit",
        config.exit_ticks_per_waypoint, config.max_waypoint_spacing_m,
    )
    _append_segment(
        path, exit_point, transit_start, "transit",
        config.transit_ticks_per_waypoint, config.max_waypoint_spacing_m,
    )
    _append_segment(
        path, transit_start, transit_end, "transit",
        config.transit_ticks_per_waypoint, config.max_waypoint_spacing_m,
    )
    _append_segment(
        path, transit_end, approach_point, "approach",
        config.approach_ticks_per_waypoint, config.max_waypoint_spacing_m,
    )
    _append_segment(
        path, target_standoff, target, "insertion",
        config.insertion_ticks_per_waypoint,
        config.max_insertion_step_m,
    )
    for point, _, _ in path:
        wall_distance = float(np.dot(point - face, outward))
        if wall_distance < -1e-9:
            raise ValueError("Generated transfer waypoint crosses the wall plane")
    return [path]


def _theoretical_transfer_path(
    sim, limb, start, target, source_hold, target_hold, supports,
    trunk_pos, trunk_quat, q_start, deadline=None,
    config=DEFAULT_TRANSFER_TRAJECTORY,
):
    path = _candidate_paths(
        sim, start, target, source_hold, target_hold, config
    )[0]
    limits = np.asarray(sim.joint_range, dtype=float)
    seed = np.asarray(q_start, dtype=float)
    planned = []
    max_joint_step = 0.0
    min_joint_margin = float("inf")
    max_ik_error = 0.0
    for waypoint_index, (point, phase, ticks) in enumerate(path, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            return None, {
                "reason": "time_limit_reached",
                "phase": phase,
                "waypoint_index": waypoint_index,
            }
        targets = {name: position for name, position in supports.items()}
        targets[limb] = point
        q, error = sim.joint_angles_for(
            targets, trunk_pos, trunk_quat, q_start=seed
        )
        q = np.asarray(q, dtype=float)
        invalid = _ik_command_error(
            sim, q, error, "theoretical",
            max_error_m=config.socket_centering_tolerance_m,
        )
        if invalid:
            return None, {
                **invalid,
                "phase": phase,
                "waypoint_index": waypoint_index,
            }
        joint_step = float(np.max(np.abs(q - seed)))
        if joint_step > config.max_joint_step_rad:
            return None, {
                "reason": "joint_path_discontinuity",
                "phase": phase,
                "waypoint_index": waypoint_index,
                "joint_waypoint_step_rad": joint_step,
                "joint_waypoint_step_limit_rad": config.max_joint_step_rad,
            }
        predicted = sim.ball_positions_at(q, trunk_pos, trunk_quat)
        support_drift = max(
            float(np.linalg.norm(
                np.asarray(predicted[member], dtype=float) - position
            ))
            for member, position in supports.items()
        )
        if support_drift > 0.001:
            return None, {
                "reason": "theoretical_support_geometry_violation",
                "phase": phase,
                "waypoint_index": waypoint_index,
                "support_drift_m": support_drift,
            }
        margin = float(np.min(np.minimum(
            q - limits[:, 0], limits[:, 1] - q
        )))
        planned.append({
            "position": point.tolist(),
            "phase": phase,
            "ticks": ticks,
            "joint_angles_rad": q.tolist(),
            "ik_error_m": float(error),
            "joint_step_rad": joint_step,
            "joint_limit_margin_rad": margin,
        })
        seed = q
        max_joint_step = max(max_joint_step, joint_step)
        min_joint_margin = min(min_joint_margin, margin)
        max_ik_error = max(max_ik_error, float(error))
    return {
        "waypoints": planned,
        "max_joint_step_rad": max_joint_step,
        "min_joint_limit_margin_rad": min_joint_margin,
        "max_ik_error_m": max_ik_error,
    }, None


def _ik(sim, limb, point, supports, q_start=None):
    targets = {name: data["position"] for name, data in supports.items()}
    targets[limb] = point
    trunk_pos, trunk_quat = sim.trunk_pose()
    q, error = sim.joint_angles_for(
        targets, trunk_pos, trunk_quat,
        q_start=sim.joint_positions() if q_start is None else q_start,
    )
    return np.asarray(q, dtype=float), error


def _ik_command_error(sim, q, error, stage, max_error_m=_MAX_IK_ERROR):
    if q.shape != (19,):
        return {
            "reason": f"{stage}_invalid_command_shape",
            "command_shape": list(q.shape),
        }
    if not np.all(np.isfinite(q)):
        return {"reason": f"{stage}_non_finite_command"}
    limits = np.asarray(sim.joint_range, dtype=float)
    if limits.shape != (19, 2) or not np.all(np.isfinite(limits)):
        return {"reason": "invalid_simulator_joint_limits"}
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        return {
            "reason": f"{stage}_joint_limit_violation",
            "joint_limits": limits.tolist(),
        }
    try:
        ik_error = np.asarray(error, dtype=float)
    except (TypeError, ValueError):
        return {"reason": f"{stage}_invalid_ik_error"}
    if ik_error.ndim != 0 or not np.isfinite(ik_error):
        return {"reason": f"{stage}_invalid_ik_error"}
    ik_error = float(ik_error)
    if ik_error > max_error_m:
        return {
            "reason": f"{stage}_ik_error_exceeded",
            "ik_error_m": ik_error,
            "ik_error_limit_m": max_error_m,
        }
    return None


def _expected_contacts(sim, limb, target_index, supports, source_hold, phase):
    expected = {
        (sim.model.geom(f"{name}_ball_coll").id, support["hold_index"])
        for name, support in supports.items()
    }
    moving_ball = sim.model.geom(f"{limb}_ball_coll").id
    if phase == "exit" and source_hold is not None:
        expected.add((moving_ball, source_hold))
    if phase in {"target", "insertion"}:
        expected.add((moving_ball, target_index))
    return expected


def _phase_state(phase):
    return _PHASE_STATES.get(phase, "TRANSITING")


def _add_unexpected_contacts(sim, contacts, phases, expected, phase, geometry, cache):
    found = _new_collisions(sim, expected, phase, geometry, cache)
    for pair in found:
        contacts.add(pair)
        phases.setdefault(pair, set()).add(phase)


def _record_contact_sample(
    sim, contacts, phases, expected, phase, tick, geometry, cache,
    initial_pairs, active, history,
):
    _add_unexpected_contacts(
        sim, contacts, phases, expected, phase, geometry, cache
    )
    current = {}
    for contact in sim.data.contact:
        if contact.dist > 0.0:
            continue
        pair = tuple(sorted((contact.geom1, contact.geom2)))
        geom1, geom2 = pair
        if not (
            geometry["root_by_geom"][geom1] == geometry["robot_root"]
            or geometry["root_by_geom"][geom2] == geometry["robot_root"]
        ):
            continue
        hold1 = geometry["holds_by_geom"][geom1]
        hold2 = geometry["holds_by_geom"][geom2]
        is_expected = (
            hold1 is not None and (geom2, hold1) in expected
        ) or (
            hold2 is not None and (geom1, hold2) in expected
        )
        key = (pair, is_expected)
        current[key] = min(current.get(key, float("inf")), float(contact.dist))

    for key in active.keys() - current.keys():
        active[key]["last_tick"] = tick - 1
        active[key]["ended_tick"] = tick
        active[key]["currently_contacting"] = False
        del active[key]

    for (pair, is_expected), distance in current.items():
        event = active.get((pair, is_expected))
        if event is None:
            event = {
                "geom_ids": list(pair),
                "body_names": [
                    geometry["details_by_geom"][geom]["body_name"] for geom in pair
                ],
                "geom_names": [
                    geometry["details_by_geom"][geom]["geom_name"] for geom in pair
                ],
                "classification": (
                    "expected_hold_contact"
                    if is_expected
                    else "preexisting_unexpected"
                    if pair in initial_pairs
                    else "new_unexpected"
                ),
                "preexisting": pair in initial_pairs,
                "expected": is_expected,
                "first_tick": tick,
                "last_tick": tick,
                "ended_tick": None,
                "phases": [phase],
                "distance_m": distance,
                "currently_contacting": True,
            }
            history.append(event)
            active[(pair, is_expected)] = event
        else:
            event["last_tick"] = tick
            event["distance_m"] = min(event["distance_m"], distance)
            event["currently_contacting"] = True
            if phase not in event["phases"]:
                event["phases"].append(phase)


def _measure(sim, limb, target_index, supports, source_hold):
    actual = sim.ball_positions()
    target_distance = float(np.linalg.norm(
        np.asarray(actual[limb], dtype=float) - _hold_socket(sim, target_index)
    ))
    current = np.asarray(sim.observation()["current"], dtype=float)
    return {
        "target_distance_m": target_distance,
        "target_contact": has_hold_contact(sim, limb, target_index),
        "source_contact": (
            has_hold_contact(sim, limb, source_hold)
            if source_hold is not None else False
        ),
        "supports": _support_status(sim, supports),
        "current_max_A": float(np.max(np.abs(current))),
    }


def _target_acquired(measurement):
    return (
        measurement["target_contact"]
        and measurement["target_distance_m"] < _HOOK_TOLERANCE
        and not measurement["source_contact"]
    )


def _run_path(
    sim, limb, target_index, supports, source_hold, path, initial_pairs,
    trace_callback=None, path_index=0,
    config=DEFAULT_TRANSFER_TRAJECTORY,
):
    start = np.asarray(sim.ball_positions()[limb], dtype=float)
    target_distance_start = float(np.linalg.norm(start - _hold_socket(sim, target_index)))
    measurements = []
    collision_pairs = set()
    collision_phases = {}
    collision_geometry = _collision_geometry(sim)
    collision_cache = {}
    contact_history = []
    active_contacts = {}
    tick = 0
    max_joint_error = 0.0
    max_joint_step = 0.0
    max_current = 0.0
    ik_seed = np.asarray(sim.joint_positions(), dtype=float)
    initial_expected = _expected_contacts(
        sim, limb, target_index, supports, source_hold, "initial"
    )
    if source_hold is not None:
        initial_expected.add((sim.model.geom(f"{limb}_ball_coll").id, source_hold))
    _record_contact_sample(
        sim, collision_pairs, collision_phases, initial_expected, "initial", tick,
        collision_geometry, collision_cache, initial_pairs, active_contacts,
        contact_history,
    )
    previous_endpoint = start
    stalled_waypoints = 0
    reverse_waypoints = 0
    for point_index, (endpoint, phase, ticks) in enumerate(path):
        distance = float(np.linalg.norm(endpoint - previous_endpoint))
        subdivisions = max(
            1, math.ceil(distance / config.max_waypoint_spacing_m)
        )
        for fraction_index in range(1, subdivisions + 1):
            fraction = fraction_index / subdivisions
            waypoint = previous_endpoint + fraction * (endpoint - previous_endpoint)
            measured_before = np.asarray(sim.ball_positions()[limb], dtype=float)
            error_before = float(np.linalg.norm(measured_before - waypoint))
            expected_contacts = _expected_contacts(
                sim, limb, target_index, supports, source_hold, phase
            )
            for _ in range(ticks):
                q_start = np.asarray(sim.joint_positions(), dtype=float)
                q, ik_error = _ik(
                    sim, limb, waypoint, supports, q_start=q_start
                )
                command_error = _ik_command_error(
                    sim, q, ik_error, "intermediate",
                    max_error_m=config.socket_centering_tolerance_m,
                )
                if command_error:
                    return False, {
                        **command_error,
                        "state": "FAILED",
                        "failed_phase": phase,
                        "ik_error_m": (
                            float(ik_error) if np.isscalar(ik_error) else None
                        ),
                        "waypoints": measurements,
                        "max_joint_error_rad": max_joint_error,
                        "max_current_A": max_current,
                        "contact_history": contact_history,
                        "ticks_executed": tick,
                    }
                joint_step = float(np.max(np.abs(q - ik_seed)))
                max_joint_step = max(max_joint_step, joint_step)
                if joint_step > config.max_joint_step_rad:
                    return False, {
                        "reason": "joint_path_discontinuity",
                        "state": "FAILED",
                        "failed_phase": phase,
                        "joint_waypoint_step_rad": joint_step,
                        "joint_waypoint_step_limit_rad": config.max_joint_step_rad,
                        "max_joint_waypoint_step_rad": max_joint_step,
                        "waypoints": measurements,
                        "collision_contacts": _describe_contacts(
                            sim, collision_pairs, initial_pairs,
                            collision_phases, collision_geometry,
                        ),
                        "max_joint_error_rad": max_joint_error,
                        "max_current_A": max_current,
                        "contact_history": contact_history,
                        "ticks_executed": tick,
                    }
                position_before = np.asarray(
                    sim.ball_positions()[limb], dtype=float
                )
                joints_before = np.asarray(sim.joint_positions(), dtype=float)
                sim.step(q)
                tick += 1
                observation = sim.observation()
                if trace_callback is not None:
                    trace_callback({
                        "sim": sim,
                        "path_index": path_index,
                        "tick": tick,
                        "phase": phase,
                        "phase_state": _phase_state(phase),
                        "waypoint_m": waypoint.copy(),
                        "target_q": q.copy(),
                        "position_before_m": position_before,
                        "joints_before_rad": joints_before,
                        "ik_error_m": float(ik_error),
                        "joint_waypoint_step_rad": joint_step,
                        "limb": limb,
                        "target_index": target_index,
                        "source_hold": source_hold,
                        "supports": supports,
                    })
                ik_seed = q.copy()
                measured_q = np.asarray(observation["q"], dtype=float)
                max_current = max(
                    max_current,
                    float(np.max(np.abs(observation["current"]))),
                )
                max_joint_error = max(
                    max_joint_error,
                    float(np.max(np.abs(q - measured_q))),
                )
                _record_contact_sample(
                    sim, collision_pairs, collision_phases, expected_contacts,
                    phase, tick, collision_geometry, collision_cache,
                    initial_pairs, active_contacts, contact_history,
                )
                support_state = _support_status(sim, supports)
                if any(not state["valid"] for state in support_state.values()):
                    return False, {
                        "reason": "support_lost",
                        "state": "FAILED",
                        "failed_phase": phase,
                        "supports": support_state,
                        "waypoints": measurements,
                        "collision_contacts": _describe_contacts(
                            sim, collision_pairs, initial_pairs, collision_phases,
                            collision_geometry,
                        ),
                        "max_joint_error_rad": max_joint_error,
                        "max_current_A": max_current,
                        "contact_history": contact_history,
                        "ticks_executed": tick,
                    }
                new_collisions = set(collision_pairs) - initial_pairs
                if new_collisions:
                    return False, {
                        "reason": "unexpected_collision",
                        "state": "FAILED",
                        "failed_phase": phase,
                        "new_collision_pairs": [
                            list(pair) for pair in sorted(new_collisions)
                        ],
                        "supports": support_state,
                        "waypoints": measurements,
                        "collision_contacts": _describe_contacts(
                            sim, collision_pairs, initial_pairs, collision_phases,
                            collision_geometry,
                        ),
                        "max_joint_error_rad": max_joint_error,
                        "max_current_A": max_current,
                        "contact_history": contact_history,
                        "ticks_executed": tick,
                    }
            support_state = _support_status(sim, supports)
            if any(not state["valid"] for state in support_state.values()):
                return False, {
                    "reason": "support_lost",
                    "state": "FAILED",
                    "failed_phase": phase,
                    "supports": support_state,
                    "waypoints": measurements,
                    "collision_contacts": _describe_contacts(
                        sim, collision_pairs, initial_pairs, collision_phases,
                        collision_geometry,
                    ),
                    "max_joint_error_rad": max_joint_error,
                    "max_current_A": max_current,
                    "contact_history": contact_history,
                    "ticks_executed": tick,
                }
            measure = _measure(sim, limb, target_index, supports, source_hold)
            tracking_error = float(np.linalg.norm(
                np.asarray(sim.ball_positions()[limb], dtype=float) - waypoint
            ))
            measure.update({
                "waypoint_m": waypoint.tolist(),
                "phase": phase,
                "phase_state": _phase_state(phase),
                "ticks": ticks,
                "ik_error_m": float(ik_error),
                "joint_waypoint_step_rad": joint_step,
                "trajectory_error_m": tracking_error,
                "waypoint_progress_m": error_before - tracking_error,
                "collision_contacts": _describe_contacts(
                    sim, collision_pairs, initial_pairs, collision_phases,
                    collision_geometry,
                ),
                "max_current_A": max_current,
            })
            measurements.append(measure)
            progress = error_before - tracking_error
            at_endpoint = fraction_index == subdivisions
            phase_complete = (
                at_endpoint
                and (
                    point_index + 1 == len(path)
                    or path[point_index + 1][1] != phase
                )
            )
            if phase_complete and tracking_error > _MAX_WAYPOINT_ERROR_M:
                return False, {
                    "reason": "phase_tracking_error",
                    "state": "FAILED",
                    "failed_phase": phase,
                    "trajectory_error_m": tracking_error,
                    "waypoint_progress_m": progress,
                    "waypoints": measurements,
                    "collision_contacts": measure["collision_contacts"],
                    "max_joint_error_rad": max_joint_error,
                    "max_current_A": max_current,
                    "contact_history": contact_history,
                    "ticks_executed": tick,
                }
            if (
                source_hold is not None
                and measure["source_contact"]
                and (phase != "exit" or phase_complete)
            ):
                return False, {
                    "reason": (
                        "exit_not_cleared" if phase == "exit"
                        else "starting_contact_persists"
                    ),
                    "state": "FAILED",
                    "failed_phase": phase,
                    "source_hold": source_hold,
                    "waypoints": measurements,
                    "collision_contacts": measure["collision_contacts"],
                    "max_joint_error_rad": max_joint_error,
                    "max_current_A": max_current,
                    "contact_history": contact_history,
                    "ticks_executed": tick,
                }
            if tracking_error > _MAX_WAYPOINT_ERROR_M:
                if progress < -_MIN_WAYPOINT_PROGRESS_M:
                    reverse_waypoints += 1
                    stalled_waypoints = 0
                    if reverse_waypoints >= _STALLED_WAYPOINT_LIMIT:
                        return False, {
                            "reason": "trajectory_tracking_error",
                            "state": "FAILED",
                            "failed_phase": phase,
                            "trajectory_error_m": tracking_error,
                            "waypoint_progress_m": progress,
                            "waypoints": measurements,
                            "collision_contacts": measure["collision_contacts"],
                            "max_joint_error_rad": max_joint_error,
                            "max_current_A": max_current,
                            "contact_history": contact_history,
                            "ticks_executed": tick,
                        }
                elif abs(progress) <= _MIN_WAYPOINT_PROGRESS_M:
                    stalled_waypoints += 1
                    reverse_waypoints = 0
                    if stalled_waypoints >= _STALLED_WAYPOINT_LIMIT:
                        return False, {
                            "reason": "movement_blocked",
                            "state": "FAILED",
                            "failed_phase": phase,
                            "trajectory_error_m": tracking_error,
                            "waypoint_progress_m": progress,
                            "waypoints": measurements,
                            "collision_contacts": measure["collision_contacts"],
                            "max_joint_error_rad": max_joint_error,
                            "max_current_A": max_current,
                            "contact_history": contact_history,
                            "ticks_executed": tick,
                        }
                else:
                    stalled_waypoints = 0
                    reverse_waypoints = 0
            else:
                stalled_waypoints = 0
                reverse_waypoints = 0
        previous_endpoint = endpoint

    q, ik_error = _ik(
        sim, limb, _hold_socket(sim, target_index), supports, q_start=ik_seed
    )
    command_error = _ik_command_error(
        sim, q, ik_error, "final",
        max_error_m=config.socket_centering_tolerance_m,
    )
    if command_error:
        return False, {
            **command_error,
            "state": "FAILED",
            "failed_phase": "insertion",
            "waypoints": measurements,
            "collision_contacts": _describe_contacts(
                sim, collision_pairs, initial_pairs, collision_phases,
                collision_geometry,
            ),
            "max_joint_error_rad": max_joint_error,
            "max_current_A": max_current,
            "contact_history": contact_history,
            "ticks_executed": tick,
        }
    final_expected_contacts = _expected_contacts(
        sim, limb, target_index, supports, source_hold, "insertion"
    )
    for _ in range(100):
        position_before = np.asarray(sim.ball_positions()[limb], dtype=float)
        joints_before = np.asarray(sim.joint_positions(), dtype=float)
        sim.step(q)
        tick += 1
        observation = sim.observation()
        if trace_callback is not None:
            trace_callback({
                "sim": sim,
                "path_index": path_index,
                "tick": tick,
                "phase": "insertion",
                "waypoint_m": _hold_socket(sim, target_index),
                "target_q": q.copy(),
                "position_before_m": position_before,
                "joints_before_rad": joints_before,
                "ik_error_m": float(ik_error),
                "limb": limb,
                "target_index": target_index,
                "source_hold": source_hold,
                "supports": supports,
            })
        max_current = max(
            max_current,
            float(np.max(np.abs(observation["current"]))),
        )
        max_joint_error = max(
            max_joint_error,
            float(np.max(np.abs(q - observation["q"]))),
        )
        _record_contact_sample(
            sim, collision_pairs, collision_phases, final_expected_contacts,
            "target", tick, collision_geometry, collision_cache,
            initial_pairs, active_contacts, contact_history,
        )
        support_state = _support_status(sim, supports)
        if any(not state["valid"] for state in support_state.values()):
            return False, {
                "reason": "support_lost",
                "state": "FAILED",
                "failed_phase": "insertion",
                "supports": support_state,
                "waypoints": measurements,
                "collision_contacts": _describe_contacts(
                    sim, collision_pairs, initial_pairs, collision_phases,
                    collision_geometry,
                ),
                "max_joint_error_rad": max_joint_error,
                "max_current_A": max_current,
                "contact_history": contact_history,
                "ticks_executed": tick,
            }
        new_collisions = set(collision_pairs) - initial_pairs
        if new_collisions:
            return False, {
                "reason": "unexpected_collision",
                "state": "FAILED",
                "failed_phase": "insertion",
                "new_collision_pairs": [
                    list(pair) for pair in sorted(new_collisions)
                ],
                "supports": support_state,
                "waypoints": measurements,
                "collision_contacts": _describe_contacts(
                    sim, collision_pairs, initial_pairs, collision_phases,
                    collision_geometry,
                ),
                "max_joint_error_rad": max_joint_error,
                "max_current_A": max_current,
                "contact_history": contact_history,
                "ticks_executed": tick,
            }
    final = _measure(sim, limb, target_index, supports, source_hold)
    final["ik_error_m"] = ik_error
    final["max_joint_error_rad"] = max_joint_error
    final["max_joint_waypoint_step_rad"] = max_joint_step
    final["max_current_A"] = max_current
    final["waypoints"] = measurements
    final["collision_contacts"] = _describe_contacts(
        sim, collision_pairs, initial_pairs, collision_phases,
        collision_geometry,
    )
    final["target_progress_m"] = target_distance_start - final["target_distance_m"]
    final["collision_free"] = not collision_pairs
    final["contact_history"] = contact_history
    final["ticks_executed"] = tick
    acquired = (
        _target_acquired(final)
        and final["target_progress_m"] > _MIN_WAYPOINT_PROGRESS_M
    )
    if acquired:
        reason = "success_with_contacts" if collision_pairs else "success"
    elif _target_acquired(final):
        reason = "target_progress_insufficient"
    else:
        reason = "target_not_physically_acquired"
    return acquired, {
        "reason": reason,
        "state": "ANCHORED" if acquired else "FAILED",
        "failed_phase": None if acquired else "insertion",
        **final,
    }


def _validate_move(
    sim, action, holds, trace_callback=None,
    config=DEFAULT_TRANSFER_TRAJECTORY,
):
    try:
        limb = action["limb"]
        if limb not in LIMBS:
            raise ValueError(f"Unknown limb {limb!r}")
        target_index = _resolve_hold(sim, action, holds)
        if not 0 <= target_index < len(holds):
            raise ValueError(f"Hold index {target_index} is out of range")
        explicit = _explicit_path(action)
        if explicit:
            normalized_phases = {
                {"transfer": "transit", "target": "insertion"}.get(phase, phase)
                for _, phase, _ in explicit
            }
            if not _REQUIRED_TRANSFER_PHASES.issubset(normalized_phases):
                raise WaypointPhaseError(
                    "Explicit transfer paths must include exit, transit, "
                    "approach, and insertion phases"
                )
    except WaypointPhaseError as error:
        return {
            "success": False,
            "reason": "invalid_waypoint_phase",
            "detail": str(error),
        }, None
    except (KeyError, TypeError, ValueError) as error:
        return {"success": False, "reason": f"invalid_move:{error}"}, None

    target = _hold_socket(sim, target_index)
    positions = sim.ball_positions()
    supports, reason = _support_snapshot(sim, limb)
    if reason:
        return {"success": False, "reason": reason, "limb": limb}, None
    source_hold = _source_hold(sim, limb)
    occupied_by = [
        other for other in LIMBS
        if other != limb and has_hold_contact(sim, other, target_index)
    ]
    if occupied_by:
        return {
            "success": False,
            "reason": "target_hold_occupied",
            "limb": limb,
            "target_index": target_index,
            "occupied_by": occupied_by,
        }, None
    if source_hold == target_index:
        initial_pairs = _contact_pairs(sim)
        expected = _expected_contacts(
            sim, limb, target_index, supports, source_hold, "target"
        )
        contacts = set(_new_collisions(sim, expected))
        phases = {pair: {"initial"} for pair in contacts}
        descriptions = _describe_contacts(sim, contacts, initial_pairs, phases)
        result = {
            "success": True,
            "reason": "success_with_contacts" if contacts else "already_in_target_contact",
            "limb": limb,
            "target_index": target_index,
            "geometrically_reachable": True,
            "path_feasible": True,
            "target_contact": True,
            "source_contact": has_hold_contact(sim, limb, source_hold),
            "target_distance_m": float(np.linalg.norm(
                np.asarray(positions[limb], dtype=float) - target
            )),
            "supports": _support_status(sim, supports),
            "collision_free": not contacts,
            "collision_contacts": descriptions,
            "max_current_A": float(np.max(np.abs(sim.observation()["current"]))),
            "ik_error_m": 0.0,
        }
        return result, dict(action)
    q, ik_error = _ik(sim, limb, target, supports)
    command_error = _ik_command_error(
        sim, q, ik_error, "endpoint",
        max_error_m=config.socket_centering_tolerance_m,
    )
    if command_error:
        return {
            "success": False,
            **command_error,
            "limb": limb,
            "target_index": target_index,
        }, None

    baseline_pairs = _contact_pairs(sim)
    paths = [explicit] if explicit is not None else _candidate_paths(
        sim,
        np.asarray(positions[limb], dtype=float),
        target,
        source_hold,
        target_hold=target_index,
        config=config,
    )
    failures = []
    failed_state = None
    for path_index, path in enumerate(paths):
        initial_state = sim.get_state()
        if trace_callback is None:
            feasible, result = _run_path(
                sim, limb, target_index, supports, source_hold, path,
                baseline_pairs, config=config,
            )
        else:
            feasible, result = _run_path(
                sim, limb, target_index, supports, source_hold, path,
                baseline_pairs, trace_callback=trace_callback,
                path_index=path_index,
                config=config,
            )
        result["waypoint_path"] = [
            {"position": point.tolist(), "phase": phase, "ticks": ticks}
            for point, phase, ticks in path
        ]
        result.update({
            "limb": limb,
            "target_index": target_index,
            "geometrically_reachable": True,
            "ik_error_m": ik_error,
            "path_feasible": feasible,
        })
        if feasible:
            result["success"] = True
            committed = dict(action)
            committed["waypoints"] = result["waypoint_path"]
            return result, committed
        failures.append(result)
        failed_state = sim.get_state()
        sim.set_state(initial_state)
    failure = dict(failures[-1]) if failures else {
        "reason": "no_candidate_path",
        "limb": limb,
        "target_index": target_index,
    }
    failure["success"] = False
    failure["candidate_failures"] = failures
    if failed_state is not None:
        sim.set_state(failed_state)
    return failure, None


def validate_plan(
    sim, plan, restore=True, trace_callback=None,
    config=DEFAULT_TRANSFER_TRAJECTORY,
):
    """Validate and execute each move, stopping at the first physical failure.

    ``restore`` defaults to preserving the caller's simulator state. Set it to
    false to leave the simulator at the resulting or failed measured state.
    """
    initial_state = sim.get_state() if restore else None
    steps = []
    committed_plan = []
    try:
        if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes, dict)):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "invalid_plan:expected_sequence",
                "steps": steps,
                "plan": committed_plan,
            }
        holds = sim.wall.holds
        for index, action in enumerate(plan, start=1):
            if not isinstance(action, dict):
                result = {"success": False, "reason": "invalid_move:expected_mapping"}
                steps.append(result)
                return {
                    "success": False,
                    "step_reached": index - 1,
                    "failed_step": index,
                    "reason": result["reason"],
                    "steps": steps,
                    "plan": committed_plan,
                }
            result, committed = _validate_move(
                sim, action, holds, trace_callback=trace_callback,
                config=config,
            )
            steps.append(result)
            if not result["success"]:
                return {
                    "success": False,
                    "step_reached": index - 1,
                    "failed_step": index,
                    "reason": result["reason"],
                    "steps": steps,
                    "plan": committed_plan,
                }
            committed_plan.append(committed)
        reason = (
            "success_with_contacts"
            if any(not step.get("collision_free", True) for step in steps)
            else "success"
        )
        return {
            "success": True,
            "step_reached": len(steps),
            "reason": reason,
            "steps": steps,
            "plan": committed_plan,
        }
    finally:
        if restore:
            sim.set_state(initial_state)
