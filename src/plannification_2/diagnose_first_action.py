"""Reproduce and record every physical attempt for the first saved transfer."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from climbing_sim import ClimbingSim, JOINTS, LIMBS
from planner.main import load_plan
from planner.physics_diagnostics import contact_snapshot
from planner.validate_plan import (
    DEFAULT_TRANSFER_TRAJECTORY,
    _contact_pairs,
    _contacting_holds,
    _source_hold,
    _support_snapshot,
    validate_plan,
)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _initial_state_check(sim, planned):
    q = np.asarray(sim.joint_positions(), dtype=float)
    planned_q = np.asarray(planned["joint_angles_rad"], dtype=float)
    trunk_pos, trunk_quat = sim.trunk_pose()
    expected_pose = planned["trunk_pose"]
    expected_pos = np.asarray(expected_pose["position_m"], dtype=float)
    expected_quat = np.asarray(expected_pose["quaternion_wxyz"], dtype=float)
    actual_contacts = {
        limb: _source_hold(sim, limb) for limb in LIMBS
    }
    position_error = float(np.linalg.norm(trunk_pos - expected_pos))
    quaternion_error = min(
        float(np.linalg.norm(trunk_quat - expected_quat)),
        float(np.linalg.norm(trunk_quat + expected_quat)),
    )
    q_error = float(np.max(np.abs(q - planned_q)))
    return {
        "matches_plan": (
            q_error <= 1e-8
            and position_error <= 1e-8
            and quaternion_error <= 1e-8
            and actual_contacts == {
                limb: planned["contacts"][limb]["hold_index"]
                for limb in LIMBS
            }
        ),
        "joint_position_max_error_rad": q_error,
        "trunk_position_error_m": position_error,
        "trunk_quaternion_error": quaternion_error,
        "planned_contacts": {
            limb: planned["contacts"][limb]["hold_index"] for limb in LIMBS
        },
        "measured_contacts": actual_contacts,
    }


def _model_summary(sim):
    actuator = sim.servos.ctrl.model.actuator
    dof_addresses = _joint_dof_addresses(sim)
    return {
        "control_dt_s": float(sim.control_dt),
        "physics_dt_s": float(sim.model.opt.timestep),
        "physics_steps_per_control_step": int(sim.n_sub),
        "command_latency_ms": float(
            (sim.servos.latency.maxlen - 1) * sim.model.opt.timestep * 1000
        ),
        "firmware_position_kp": float(sim.servos.kp),
        "motor_torque_constant_Nm_per_A": float(sim.servos.kt),
        "motor_current_limit_A": _json_ready(actuator.max_current),
        "supply_voltage_V": _json_ready(getattr(actuator, "vin", None)),
        "controller_error_gain": _json_ready(
            getattr(actuator, "error_gain", None)
        ),
        "controller_max_pwm": _json_ready(getattr(actuator, "max_pwm", None)),
        "supply_drop_resistance_ohm": _json_ready(
            sim.servos.ctrl.vin_drop_resistance
        ),
        "minimum_supply_voltage_V": _json_ready(sim.servos.ctrl.vin_min),
        "joint_frictionloss_Nm": np.asarray(
            sim.model.dof_frictionloss, dtype=float
        )[dof_addresses].tolist(),
        "joint_damping_Nm_s_per_rad": np.asarray(
            sim.model.dof_damping, dtype=float
        )[dof_addresses].tolist(),
        "torque_enabled": sim.servos.torque.tolist(),
        "joint_order": list(JOINTS),
        "joint_limits_rad": np.asarray(sim.joint_range, dtype=float).tolist(),
    }


def _joint_dof_addresses(sim):
    return [
        int(np.asarray(sim.model.joint(name).dofadr).reshape(-1)[0])
        for name in JOINTS
    ]


def _sample_callback(samples, initial_pairs):
    def record(event):
        sim = event["sim"]
        limb = event["limb"]
        waypoint = np.asarray(event["waypoint_m"], dtype=float)
        observation = sim.observation()
        q_actual = np.asarray(sim.joint_positions(), dtype=float)
        q_measured = np.asarray(observation["q"], dtype=float)
        q_target = np.asarray(event["target_q"], dtype=float)
        q_applied = np.asarray(sim.servos.ctrl.q_target, dtype=float)
        dof_addresses = _joint_dof_addresses(sim)
        qvel = np.asarray(sim.data.qvel, dtype=float)[dof_addresses]
        ball = np.asarray(sim.ball_positions()[limb], dtype=float)
        error = float(np.linalg.norm(ball - waypoint))
        error_before = float(np.linalg.norm(
            np.asarray(event["position_before_m"], dtype=float) - waypoint
        ))

        currents = np.asarray(sim.servos.currents(), dtype=float)
        current_limit = np.asarray(
            sim.servos.ctrl.model.actuator.max_current, dtype=float
        )
        if current_limit.ndim == 0:
            current_limit = np.full(len(JOINTS), float(current_limit))
        ctrl = np.asarray(sim.data.ctrl[:len(JOINTS)], dtype=float)
        actuator_force = np.asarray(
            sim.data.actuator_force[:len(JOINTS)], dtype=float
        )
        joint_ranges = np.asarray(sim.joint_range, dtype=float)
        limit_margin = np.minimum(
            q_target - joint_ranges[:, 0],
            joint_ranges[:, 1] - q_target,
        )
        actuator = sim.servos.ctrl.model.actuator
        duty_cycle = np.asarray(actuator.duty_cycle, dtype=float)
        battery_current = max(
            float(np.sum(duty_cycle * actuator_force / sim.servos.kt)), 0.0
        )
        effective_voltage = max(
            float(actuator.vin)
            - float(sim.servos.ctrl.vin_drop_resistance) * battery_current,
            float(sim.servos.ctrl.vin_min),
        )
        expected = {
            (sim.model.geom(name + "_ball_coll").id, support["hold_index"])
            for name, support in event["supports"].items()
        }
        moving_geom = sim.model.geom(limb + "_ball_coll").id
        if event["phase"] == "exit" and event["source_hold"] is not None:
            expected.add((moving_geom, event["source_hold"]))
        if event["phase"] in {"target", "insertion"}:
            expected.add((moving_geom, event["target_index"]))

        samples.append({
            "path_attempt": int(event["path_index"]),
            "tick_in_attempt": int(event["tick"]),
            "simulation_time_s": float(sim.data.time),
            "phase": event["phase"],
            "phase_state": event["phase_state"],
            "limb": limb,
            "source_hold": event["source_hold"],
            "target_hold": int(event["target_index"]),
            "contacting_holds_by_limb": {
                name: sorted(_contacting_holds(sim, name)) for name in LIMBS
            },
            "waypoint_target_m": waypoint.tolist(),
            "ball_position_before_m": np.asarray(
                event["position_before_m"], dtype=float
            ).tolist(),
            "ball_position_m": ball.tolist(),
            "cartesian_error_to_waypoint_m": error,
            "cartesian_progress_to_waypoint_m": error_before - error,
            "trunk_position_m": np.asarray(
                sim.trunk_pose()[0], dtype=float
            ).tolist(),
            "trunk_quaternion_wxyz": np.asarray(
                sim.trunk_pose()[1], dtype=float
            ).tolist(),
            "joint_target_rad": q_target.tolist(),
            "joint_waypoint_step_rad": float(event["joint_waypoint_step_rad"]),
            "joint_controller_target_rad": q_applied.tolist(),
            "joint_position_rad": q_actual.tolist(),
            "joint_position_sensor_rad": q_measured.tolist(),
            "joint_velocity_rad_s": qvel.tolist(),
            "joint_error_target_minus_actual_rad": (q_target - q_actual).tolist(),
            "joint_error_target_minus_sensor_rad": (q_target - q_measured).tolist(),
            "joint_target_limit_margin_rad": limit_margin.tolist(),
            "ik_error_m": float(event["ik_error_m"]),
            "motor_current_A": currents.tolist(),
            "motor_current_sensor_A": np.asarray(
                observation["current"], dtype=float
            ).tolist(),
            "motor_current_limit_A": current_limit.tolist(),
            "motor_current_saturated": (
                currents >= current_limit * (1.0 - 1e-3)
            ).tolist(),
            "motor_command_torque_Nm": ctrl.tolist(),
            "motor_applied_torque_Nm": actuator_force.tolist(),
            "joint_frictionloss_Nm": np.asarray(
                sim.model.dof_frictionloss, dtype=float
            )[dof_addresses].tolist(),
            "joint_damping_Nm_s_per_rad": np.asarray(
                sim.model.dof_damping, dtype=float
            )[dof_addresses].tolist(),
            "firmware_control_output": np.asarray(
                sim.servos.ctrl.control, dtype=float
            ).tolist(),
            "firmware_duty_cycle": duty_cycle.tolist(),
            "estimated_effective_supply_voltage_V": effective_voltage,
            "estimated_battery_current_A": battery_current,
            "servo_torque_enabled": sim.servos.torque.tolist(),
            "latency_queue_depth": len(sim.servos.latency),
            "controller_target_minus_command_rad": (
                q_applied - q_target
            ).tolist(),
            "contacts": contact_snapshot(
                sim,
                expected_contacts=expected,
                initial_pairs=initial_pairs,
                tick=event["tick"],
                phase=event["phase"],
            ),
        })

    return record


def _outcome_signature(result):
    step = result.get("steps", [{}])[0]
    failures = step.get("candidate_failures", [])
    attempts = failures + ([step] if not failures else [])
    return {
        "success": result.get("success"),
        "reason": result.get("reason"),
        "attempts": [
            {
                "reason": attempt.get("reason"),
                "ticks_executed": attempt.get("ticks_executed"),
                "max_current_A": attempt.get("max_current_A"),
                "max_joint_error_rad": attempt.get("max_joint_error_rad"),
                "waypoints": [
                    {
                        "phase": waypoint.get("phase"),
                        "waypoint_m": waypoint.get("waypoint_m"),
                        "trajectory_error_m": waypoint.get("trajectory_error_m"),
                        "waypoint_progress_m": waypoint.get("waypoint_progress_m"),
                        "max_current_A": waypoint.get("max_current_A"),
                    }
                    for waypoint in attempt.get("waypoints", [])
                ],
            }
            for attempt in attempts
        ],
    }


def _attempt_summaries(result, samples, action, start_ball):
    step = result.get("steps", [{}])[0]
    attempts = step.get("candidate_failures", [])
    if not attempts:
        attempts = [step]
    summaries = []
    for index, attempt in enumerate(attempts):
        path_samples = [
            sample for sample in samples
            if sample["path_attempt"] == index
        ]
        exit_samples = [
            sample for sample in path_samples if sample["phase"] == "exit"
        ]
        exit_target = (
            np.asarray(exit_samples[-1]["waypoint_target_m"], dtype=float)
            if exit_samples else None
        )
        exit_vector = (
            exit_target - start_ball if exit_target is not None else None
        )
        direction = (
            (exit_vector / np.linalg.norm(exit_vector)).tolist()
            if exit_vector is not None and np.linalg.norm(exit_vector) > 0
            else None
        )
        new_contacts = Counter(
            (
                tuple(contact["body_names"]),
                tuple(contact["geom_names"]),
            )
            for sample in path_samples
            for contact in sample["contacts"]
            if contact["touching"]
            and contact["classification"] == "new_unexpected"
        )
        floor_contacts = [
            contact
            for sample in exit_samples
            for contact in sample["contacts"]
            if contact["touching"] and contact["category"] == "robot_floor"
        ]
        summaries.append({
            "path_attempt": index,
            "reason": attempt.get("reason"),
            "ticks_executed": attempt.get("ticks_executed"),
            "max_current_A": attempt.get("max_current_A"),
            "max_joint_error_rad": attempt.get("max_joint_error_rad"),
            "exit_direction_unit": direction,
            "exit_endpoint_m": (
                exit_target.tolist() if exit_target is not None else None
            ),
            "exit_source_contact_ticks": sum(
                action["source_hold"]
                in sample["contacting_holds_by_limb"][action["limb"]]
                for sample in exit_samples
            ),
            "exit_sample_count": len(exit_samples),
            "floor_contact_max_normal_force_N": max(
                (contact["normal_force_N"] for contact in floor_contacts),
                default=0.0,
            ),
            "floor_contact_min_distance_m": min(
                (contact["distance_m"] for contact in floor_contacts),
                default=None,
            ),
            "new_unexpected_contacts": [
                {
                    "body_names": list(key[0]),
                    "geom_names": list(key[1]),
                    "active_sample_count": count,
                }
                for key, count in new_contacts.most_common(10)
            ],
            "planned_waypoint_path": attempt.get("waypoint_path", []),
            "waypoints": [
                {
                    "phase": item.get("phase"),
                    "target_m": item.get("waypoint_m"),
                    "tracking_error_m": item.get("trajectory_error_m"),
                    "progress_m": item.get("waypoint_progress_m"),
                    "current_max_A": item.get("max_current_A"),
                }
                for item in attempt.get("waypoints", [])
            ],
        })
    return summaries


def run_diagnostic(
    plan_path, output_path, seed=0, settle_s=1.0,
    limb=None, target_hold=None, config=DEFAULT_TRANSFER_TRAJECTORY,
):
    plan = load_plan(plan_path)
    if not plan.get("execution_actions"):
        raise ValueError("The plan has no execution actions")
    if (limb is None) != (target_hold is None):
        raise ValueError("Specify both limb and target hold, or neither")
    if limb is None:
        action = plan["execution_actions"][0]
        if action.get("kind") != "transfer":
            raise ValueError("The first theoretical action is not a transfer")
    else:
        if limb not in LIMBS:
            raise ValueError(f"Unknown limb {limb!r}")
        source = plan["state_path"][0]["contacts"][limb]
        action = {
            "kind": "transfer",
            "limb": limb,
            "source_hold": int(source["hold_index"]),
            "source_coords": source["coords"],
            "target_hold": int(target_hold),
            "move": {"limb": limb, "hold_index": int(target_hold)},
        }

    sim = ClimbingSim(plan["wall"], cameras=[], seed=seed)
    initial_state = None
    try:
        sim.reset_hanging(settle_s=settle_s)
        initial_state = sim.get_state()
        initial_pairs = _contact_pairs(sim)
        initial_model_summary = _model_summary(sim)
        initial_state_check = _initial_state_check(
            sim, plan["state_path"][0]
        )
        if not initial_state_check["matches_plan"]:
            raise RuntimeError(
                "Reset state does not match the saved theoretical initial state: "
                f"{initial_state_check}"
            )
        initial_contacts = contact_snapshot(
            sim, initial_pairs=initial_pairs, tick=0, phase="initial"
        )
        initial_q = np.asarray(sim.joint_positions(), dtype=float)
        initial_observation = sim.observation()
        dof_addresses = _joint_dof_addresses(sim)
        initial_qvel = np.asarray(sim.data.qvel, dtype=float)[dof_addresses]
        initial_currents = np.asarray(sim.servos.currents(), dtype=float)
        initial_applied_target = np.asarray(
            sim.servos.ctrl.q_target, dtype=float
        )
        initial_trunk = sim.trunk_pose()
        supports, support_error = _support_snapshot(sim, action["limb"])
        if support_error:
            raise RuntimeError(support_error)
        target_index = int(action["target_hold"])
        start_ball = np.asarray(sim.ball_positions()[action["limb"]], dtype=float)
        target_ball = np.asarray(
            sim.wall.holds[target_index].socket, dtype=float
        )
        samples = []
        callback = _sample_callback(samples, initial_pairs)

        result = validate_plan(
            sim,
            [action["move"]],
            restore=False,
            trace_callback=callback,
            config=config,
        )
        first_signature = _outcome_signature(result)
        first_attempts = _attempt_summaries(
            result, samples, action, start_ball
        )
        trajectory_config = asdict(config)
        default_config = asdict(DEFAULT_TRANSFER_TRAJECTORY)
        modified_parameters = {
            key: value for key, value in trajectory_config.items()
            if value != default_config[key]
        }
        final_failure_state = {
            "simulation_time_s": float(sim.data.time),
            "joint_positions_rad": np.asarray(
                sim.joint_positions(), dtype=float
            ).tolist(),
            "moving_ball_position_m": np.asarray(
                sim.ball_positions()[action["limb"]], dtype=float
            ).tolist(),
            "trunk_position_m": np.asarray(
                sim.trunk_pose()[0], dtype=float
            ).tolist(),
            "trunk_quaternion_wxyz": np.asarray(
                sim.trunk_pose()[1], dtype=float
            ).tolist(),
            "contacts": contact_snapshot(
                sim, initial_pairs=initial_pairs, phase="after_failure"
            ),
        }

        sim.set_state(initial_state)
        repeat = validate_plan(
            sim,
            [action["move"]],
            restore=False,
            config=config,
        )
        repeat_signature = _outcome_signature(repeat)
        report = {
            "experiment": {
                "name": "first_theoretical_transfer_physical_reproduction",
                "wall": plan["wall"],
                "plan_file": str(plan_path),
                "action_index": 1,
                "action": action,
                "seed": seed,
                "settle_s": settle_s,
                "command": (
                    "uv run python -m planner.diagnose_first_action "
                    f"--plan-file {plan_path} --output {output_path} "
                    f"--seed {seed} --settle-s {settle_s}"
                    + (
                        f" --limb {limb} --target-hold {target_hold}"
                        if limb is not None else ""
                    )
                    + "".join(
                        f" --{phase}-ticks-per-waypoint "
                        f"{trajectory_config[f'{phase}_ticks_per_waypoint']}"
                        for phase in ("exit", "transit", "approach", "insertion")
                        if trajectory_config[f"{phase}_ticks_per_waypoint"]
                        != default_config[f"{phase}_ticks_per_waypoint"]
                    )
                    + (
                        f" --max-waypoint-spacing-m "
                        f"{trajectory_config['max_waypoint_spacing_m']}"
                        if trajectory_config["max_waypoint_spacing_m"]
                        != default_config["max_waypoint_spacing_m"]
                        else ""
                    )
                ),
                "modified_parameters": modified_parameters,
                "transfer_trajectory_config": trajectory_config,
                "initial_model_and_servo_parameters": initial_model_summary,
                "theoretical_initial_state_match": initial_state_check,
                "state_restore_replay_deterministic": first_signature == repeat_signature,
                "repeat_outcome_signature": repeat_signature,
                "initial_simulation_time_s": float(initial_state["data"].time),
                "initial_joint_positions_rad": initial_q.tolist(),
                "initial_joint_positions_sensor_rad": np.asarray(
                    initial_observation["q"], dtype=float
                ).tolist(),
                "initial_joint_velocities_rad_s": initial_qvel.tolist(),
                "initial_motor_currents_A": initial_currents.tolist(),
                "initial_controller_target_rad": initial_applied_target.tolist(),
                "initial_trunk_pose": {
                    "position_m": np.asarray(initial_trunk[0], dtype=float).tolist(),
                    "quaternion_wxyz": np.asarray(initial_trunk[1], dtype=float).tolist(),
                },
                "source_ball_position_m": start_ball.tolist(),
                "target_hold_position_m": target_ball.tolist(),
                "initial_contacts": initial_contacts,
            },
            "physical_result": result,
            "attempt_summaries": first_attempts,
            "final_state_after_failed_attempt": final_failure_state,
            "samples": samples,
            "conclusion": (
                "Exact cause not inferred from IK convergence alone; use per-tick "
                "motor, tracking, and contact samples to identify constraints."
            ),
            "limitations": [
                "The logged state is MuJoCo simulator state, not a physical robot measurement.",
                "Contact forces are solver outputs and do not by themselves prove causation.",
                "Only the first transfer is tested; later actions are not executed.",
            ],
        }
    finally:
        if initial_state is not None:
            sim.set_state(initial_state)
        sim.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_ready(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-file", type=Path,
        default=Path("planner/n2-theoretical-plan.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("planner/diagnostics/n2-first-action.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--limb", choices=LIMBS)
    parser.add_argument("--target-hold", type=int)
    parser.add_argument("--max-waypoint-spacing-m", type=float, default=None)
    for phase in ("exit", "transit", "approach", "insertion"):
        parser.add_argument(
            f"--{phase}-ticks-per-waypoint",
            type=int,
            default=None,
        )
    args = parser.parse_args(argv)
    config_values = {
        f"{phase}_ticks_per_waypoint": value
        for phase in ("exit", "transit", "approach", "insertion")
        if (value := getattr(args, f"{phase}_ticks_per_waypoint")) is not None
    }
    if args.max_waypoint_spacing_m is not None:
        config_values["max_waypoint_spacing_m"] = args.max_waypoint_spacing_m
    config = replace(DEFAULT_TRANSFER_TRAJECTORY, **config_values)
    report = run_diagnostic(
        args.plan_file, args.output, seed=args.seed, settle_s=args.settle_s,
        limb=args.limb, target_hold=args.target_hold, config=config,
    )
    signature = report["physical_result"]
    step = signature.get("steps", [{}])[0]
    print(
        f"Initial state matches plan: "
        f"{report['experiment']['theoretical_initial_state_match']['matches_plan']}"
    )
    print(
        f"Physical result: {step.get('reason', signature.get('reason'))}; "
        f"sampled control ticks: {len(report['samples'])}; "
        f"replay deterministic: "
        f"{report['experiment']['state_restore_replay_deterministic']}"
    )
    print(f"Diagnostic JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
