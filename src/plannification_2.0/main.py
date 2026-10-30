"""Command-line entry point for planning and executing a climb to the top hold."""

import argparse
import copy
import json
from pathlib import Path

from climbing_sim import ClimbingSim, LIMBS
from climbing_sim.wall import bundled_walls

if __package__:
    from .contact_planner import (
        _configured_start_contacts,
        _run_reposition,
        plan_theoretical_to_top,
        plan_to_top,
    )
    from .validate_plan import (
        TransferTrajectoryConfig,
        _source_hold,
        validate_plan,
    )
else:
    from contact_planner import (
        _configured_start_contacts,
        _run_reposition,
        plan_theoretical_to_top,
        plan_to_top,
    )
    from validate_plan import (
        TransferTrajectoryConfig,
        _source_hold,
        validate_plan,
    )


def hold_coordinates(sim, hold_index):
    """Convert a wall's internal hold index to its left-hole grid coordinates."""
    hold = sim.wall.holds[hold_index]
    return int(hold.col), int(hold.row)


def extract_wall_holds(sim):
    """Describe the configured holds using their source geometry and relationships."""
    holds = []
    for index, hold in enumerate(sim.wall.holds):
        holds.append({
            "id": index,
            "coords": hold_coordinates(sim, index),
            "socket": tuple(float(value) for value in hold.socket),
            "row": int(hold.row),
            "clashes": [
                hold_coordinates(sim, other)
                for other in sim.wall.clashes(hold)
            ],
            "exit_blocks": [
                hold_coordinates(sim, other)
                for other in sim.wall.exit_blocks(hold)
            ],
        })
    return holds


def execute_plan(sim, plan):
    """Execute a search result or legacy transfer list and stop on first failure."""
    if isinstance(plan, dict):
        if (
            plan.get("planning_mode") == "theoretical"
            and plan.get("physical_validation_status") != "PHYSICALLY_VALIDATED"
        ):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "theoretical_plan_requires_physical_validation",
                "steps": [],
            }
        if (
            plan.get("planning_mode") == "realistic"
            and plan.get("physical_validation_status") != "PHYSICALLY_VALIDATED"
        ):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "realistic_plan_not_fully_validated",
                "steps": [],
            }
        if any(
            action.get("status") == "THEORETICAL"
            and action.get("physical_validation_status") != "PHYSICALLY_VALIDATED"
            for action in plan.get("execution_actions", [])
            if isinstance(action, dict)
        ):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "theoretical_plan_requires_physical_validation",
                "steps": [],
            }
        if not plan.get("success"):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "plan_not_complete",
                "steps": [],
            }
        actions = plan.get("execution_actions")
        if not isinstance(actions, list):
            return {
                "success": False,
                "step_reached": 0,
                "reason": "missing_execution_actions",
                "steps": [],
            }
    else:
        actions = [
            {"kind": "transfer", "move": action}
            for action in plan
        ]
    trajectory_config = TransferTrajectoryConfig(
        **plan.get("trajectory_config", {})
    ) if isinstance(plan, dict) else TransferTrajectoryConfig()

    completed = []
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            return {
                "success": False,
                "step_reached": index - 1,
                "reason": f"invalid_action:{index}",
                "steps": completed,
            }
        if action.get("kind") == "reposition":
            result = _run_reposition(sim, action)
            if not result["success"]:
                return {
                    "success": False,
                    "step_reached": index - 1,
                    "reason": result["reason"],
                    "steps": completed + [result],
                }
            completed.append(result)
            continue
        if action.get("kind") != "transfer":
            return {
                "success": False,
                "step_reached": index - 1,
                "reason": f"invalid_action:{index}",
                "steps": completed,
            }
        action = action.get("move")
        if (
            not isinstance(action, dict)
            or not {"limb", "target_coords"}.issubset(action)
            or set(action) - {"limb", "target_coords", "waypoints"}
            or action["limb"] not in LIMBS
            or not isinstance(action["target_coords"], tuple)
            or len(action["target_coords"]) != 2
            or any(type(value) is not int for value in action["target_coords"])
        ):
            return {
                "success": False,
                "step_reached": index - 1,
                "reason": f"invalid_move:{index}",
                "steps": completed,
            }
        result = validate_plan(
            sim, [action], restore=False, config=trajectory_config
        )
        step = result["steps"][0] if result["steps"] else {}
        if not result["success"] or not step.get("target_contact", False):
            return {
                "success": False,
                "step_reached": index - 1,
                "reason": step.get("reason", result["reason"]),
                "steps": completed + [step],
            }
        if any(
            not support.get("valid", False)
            for support in step.get("supports", {}).values()
        ):
            return {
                "success": False,
                "step_reached": index - 1,
                "reason": "support_lost_after_move",
                "steps": completed + [step],
            }
        completed.append(step)
    return {
        "success": True,
        "step_reached": len(completed),
        "reason": "success",
        "steps": completed,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    return value


def save_plan(path, plan):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(plan), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _restore_plan_tuples(value):
    if isinstance(value, dict):
        return {
            key: tuple(item) if key in {"target_coords", "source_coords"}
            and isinstance(item, list) and len(item) == 2
            else _restore_plan_tuples(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_restore_plan_tuples(item) for item in value]
    return value


def load_plan(path):
    with Path(path).open(encoding="utf-8") as file:
        return _restore_plan_tuples(json.load(file))


def validate_plan_result(sim, plan):
    """Validate a saved route physically and restore the simulator afterward."""
    if not isinstance(plan, dict) or not isinstance(plan.get("execution_actions"), list):
        return {
            "success": False, "reason": "invalid_plan_result",
            "physical_validation_status": "NOT_TESTED", "actions": [],
        }
    result = copy.deepcopy(plan)
    theoretical_status = result.get("status")
    theoretical_steps = copy.deepcopy(result.get("steps", []))
    actions = result["execution_actions"]
    result["theoretical_status"] = theoretical_status
    result["theoretical_steps"] = theoretical_steps
    if (
        not result.get("success")
        or theoretical_status != "COMPLETE_THEORETICAL_PATH"
        or result.get("planning_mode") != "theoretical"
    ):
        result.update({
            "success": False,
            "physical_validation_status": "NOT_TESTED",
            "physical_validation_reason": "complete_theoretical_plan_required",
            "steps": [
                {
                    "action_index": index,
                    "kind": action.get("kind") if isinstance(action, dict) else None,
                    "status": "NOT_TESTED",
                    "reason": "complete_theoretical_plan_required",
                }
                for index, action in enumerate(actions, 1)
            ],
            "physical_validation": {
                "status": "NOT_TESTED",
                "reason": "complete_theoretical_plan_required",
                "total_actions": len(actions),
                "validated_actions": 0,
                "failed_actions": 0,
                "not_tested_actions": len(actions),
                "first_failure": None,
            },
        })
        return result
    state = sim.get_state()
    physical_steps = []
    all_validated = True
    failure_reason = None
    first_failure = None
    trajectory_config = TransferTrajectoryConfig(
        **plan.get("trajectory_config", {})
    )
    try:
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                all_validated = False
                failure_reason = f"invalid_action:{index + 1}"
                first_failure = {
                    "action_index": index + 1,
                    "kind": None,
                    "reason": failure_reason,
                }
                physical_steps.append({
                    **first_failure, "status": "PHYSICALLY_FAILED"
                })
                physical_steps.extend(
                    {
                        "action_index": remaining_index,
                        "kind": (
                            remaining.get("kind")
                            if isinstance(remaining, dict) else None
                        ),
                        "status": "NOT_TESTED",
                        "reason": "stopped_after_invalid_action",
                    }
                    for remaining_index, remaining in enumerate(
                        actions[index + 1:], index + 2
                    )
                )
                break
            previous = action.get("physical_validation")
            if previous is not None:
                action.setdefault("physical_validation_history", []).append({
                    "status": action.get("physical_validation_status"),
                    "reason": action.get("physical_failure_reason"),
                    "result": previous,
                })
            if action.get("kind") == "reposition":
                outcome = _run_reposition(sim, action)
                valid = outcome["success"]
            elif action.get("kind") == "transfer":
                move = action.get("move")
                if not isinstance(move, dict):
                    outcome = {"success": False, "reason": "invalid_transfer_action"}
                    valid = False
                else:
                    move = dict(move)
                    if isinstance(move.get("target_coords"), list):
                        move["target_coords"] = tuple(move["target_coords"])
                    outcome = validate_plan(
                        sim, [move], restore=False, config=trajectory_config
                    )
                    step = outcome["steps"][0] if outcome.get("steps") else {}
                    valid = (
                        outcome.get("success", False)
                        and step.get("target_contact", False)
                        and len(step.get("supports", {})) == len(LIMBS) - 1
                        and all(
                            support.get("valid", False)
                            for support in step.get("supports", {}).values()
                        )
                    )
                    if not valid and outcome.get("success"):
                        outcome["reason"] = "support_lost_after_move"
            else:
                outcome = {"success": False, "reason": "invalid_action_kind"}
                valid = False
            action["physical_validation_status"] = (
                "PHYSICALLY_VALIDATED" if valid else "PHYSICALLY_FAILED"
            )
            action["physical_failure_reason"] = (
                None if valid else outcome.get("reason", "physical_validation_failed")
            )
            action["physical_validation"] = outcome
            physical_steps.append({
                "action_index": index + 1,
                "kind": action.get("kind"),
                "status": action["physical_validation_status"],
                "reason": action["physical_failure_reason"],
            })
            if not valid:
                all_validated = False
                failure_reason = action["physical_failure_reason"]
                first_failure = {
                    "action_index": index + 1,
                    "kind": action.get("kind"),
                    "reason": failure_reason,
                    "limb": action.get("limb"),
                    "source_hold": action.get("source_hold"),
                    "target_hold": action.get("target_hold"),
                }
                for remaining_index, remaining in enumerate(
                    actions[index + 1:], index + 2
                ):
                    if not isinstance(remaining, dict):
                        physical_steps.append({
                            "action_index": remaining_index,
                            "kind": None,
                            "status": "NOT_TESTED",
                            "reason": "stopped_after_first_failure",
                        })
                        continue
                    prior = remaining.get("physical_validation")
                    if prior is not None:
                        remaining.setdefault(
                            "physical_validation_history", []
                        ).append({
                            "status": remaining.get(
                                "physical_validation_status"
                            ),
                            "reason": remaining.get("physical_failure_reason"),
                            "result": prior,
                        })
                    remaining["physical_validation_status"] = "NOT_TESTED"
                    remaining["physical_failure_reason"] = None
                    remaining.pop("physical_validation", None)
                    physical_steps.append({
                        "action_index": remaining_index,
                        "kind": remaining.get("kind"),
                        "status": "NOT_TESTED",
                        "reason": "stopped_after_first_failure",
                    })
                break
        validated_count = sum(
            step["status"] == "PHYSICALLY_VALIDATED"
            for step in physical_steps
        )
        failed_count = sum(
            step["status"] == "PHYSICALLY_FAILED"
            for step in physical_steps
        )
        not_tested_count = len(actions) - validated_count - failed_count
        result["success"] = all_validated
        physical_status = (
            "COMPLETE_PHYSICALLY_VALIDATED"
            if all_validated
            else "PARTIALLY_PHYSICALLY_VALIDATED"
            if validated_count
            else "PHYSICAL_VALIDATION_FAILED"
        )
        result["status"] = physical_status
        result["physical_validation_status"] = (
            "PHYSICALLY_VALIDATED" if all_validated else "PHYSICALLY_FAILED"
        )
        result["physical_validation_reason"] = failure_reason
        result["steps"] = physical_steps
        result["physical_validation"] = {
            "status": physical_status,
            "reason": failure_reason,
            "total_actions": len(actions),
            "validated_actions": validated_count,
            "failed_actions": failed_count,
            "not_tested_actions": not_tested_count,
            "first_failure": first_failure,
        }
        return result
    finally:
        sim.set_state(state)


def build_plan(
    sim, max_moves=64, max_branches=128, time_limit_s=60.0,
    planning_mode="physical",
):
    """Plan a climb from the simulator's current hanging state without moving it."""
    wall_holds = extract_wall_holds(sim)
    initial_supports = (
        _configured_start_contacts(sim)
        if planning_mode == "theoretical"
        else {limb: _source_hold(sim, limb) for limb in LIMBS}
    )
    if planning_mode == "theoretical":
        search = plan_theoretical_to_top(
            sim, max_moves=max_moves, max_branches=max_branches,
            time_limit_s=time_limit_s,
        )
    elif planning_mode == "physical":
        search = plan_to_top(
            sim, max_moves=max_moves, max_branches=max_branches,
            time_limit_s=time_limit_s,
        )
    else:
        raise ValueError(f"Unknown planning mode: {planning_mode}")
    search.update({
        "wall": sim.wall.name,
        "initial_supports": initial_supports,
        "initial_max_progress_row": max(
            (
                int(sim.wall.holds[index].row)
                for index in initial_supports.values()
                if index is not None
            ),
            default=None,
        ),
        "wall_hold_count": len(wall_holds),
        "holds": wall_holds,
    })
    return search


def plan_all_walls(max_moves, max_branches, time_limit_s):
    """Search and checkpoint a theoretical route for every bundled wall."""
    planner_dir = Path(__file__).parent
    results = []
    summary = {
        "planning_mode": "theoretical",
        "physical_validation_status": "NOT_TESTED",
        "walls": results,
    }
    summary_path = planner_dir / "theoretical-plans-summary.json"
    for wall_name in bundled_walls():
        sim = ClimbingSim(wall_name, cameras=[])
        try:
            sim.reset_hanging(settle_s=0)
            result = build_plan(
                sim,
                max_moves=max_moves,
                max_branches=max_branches,
                time_limit_s=time_limit_s,
                planning_mode="theoretical",
            )
            plan_path = planner_dir / f"{wall_name.lower()}-theoretical-plan.json"
            save_plan(plan_path, result)
            print_result(result)
            results.append({
                "wall": wall_name,
                "status": result.get("status"),
                "theoretical_validation_status": result.get(
                    "theoretical_validation_status", "NOT_COMPLETED"
                ),
                "success": result.get("success", False),
                "reason": result.get("reason"),
                "initial_row": result.get("initial_max_progress_row"),
                "final_row": result.get("max_progress_row"),
                "transfer_count": result.get("transfer_count", 0),
                "reposition_count": result.get("reposition_count", 0),
                "branches_explored": result.get("branches_explored", 0),
                "nodes_explored": result.get("nodes_explored", 0),
                "elapsed_s": result.get("elapsed_s", 0.0),
                "plan_file": plan_path.name,
            })
            save_plan(summary_path, summary)
        finally:
            sim.close()
    return summary


def print_result(result):
    print("\n=== PLANNING RESULT ===\n")
    if result.get("planning_mode") == "realistic":
        actions = result.get("execution_actions", [])
        physical = result.get("physical_validation", {})
        validated = physical.get("validated_actions", sum(
            action.get("physical_validation_status") == "PHYSICALLY_VALIDATED"
            for action in actions
        ))
        failed = physical.get("failed_actions", sum(
            action.get("physical_validation_status") == "PHYSICALLY_FAILED"
            for action in actions
        ))
        untested = physical.get("not_tested_actions", sum(
            action.get("physical_validation_status") == "NOT_TESTED"
            for action in actions
        ))
        first_failure = physical.get("first_failure") or next((
            (index, action)
            for index, action in enumerate(actions, 1)
            if action.get("physical_validation_status") == "PHYSICALLY_FAILED"
        ), None)
        print(f"MODE : REALISTIC\nMUR : {result.get('wall', '?')}")
        print(
            "Résultat théorique initial : "
            f"{result.get('theoretical_status', 'unknown')}"
        )
        print(
            f"Mouvements : validés physiquement {validated}, "
            f"échoués {failed}, non testés {untested}"
        )
        if first_failure:
            if isinstance(first_failure, dict):
                index = first_failure.get("action_index")
                reason = first_failure.get("reason")
            else:
                index, action = first_failure
                reason = action.get("physical_failure_reason")
            print(
                f"Premier échec : action {index}, {reason}"
            )
        print(
            f"STATUT PHYSIQUE : "
            f"{physical.get('status', result.get('status'))}"
        )
        return
    if result.get("planning_mode") == "theoretical":
        print(f"MODE : THEORETICAL\nMUR : {result['wall']}")
        print(
            f"Progression : rangée {result.get('initial_max_progress_row', '?')} "
            f"→ rangée {result.get('max_progress_row', '?')}"
        )
        print(
            f"Statut : {result['status']} — "
            f"validation physique : {result.get('physical_validation_status', 'NOT_TESTED')}"
        )
        transfers = 0
        repositions = 0
        for index, action in enumerate(result.get("execution_actions", []), start=1):
            if action.get("kind") == "transfer":
                transfers += 1
                print(
                    f"{index}. Déplacer {action['limb']} : prise "
                    f"{action['source_coords']} → {action['target_coords']}"
                )
            else:
                repositions += 1
                print(
                    f"{index}. Repositionner le tronc : "
                    f"translation {action.get('requested_trunk_offset_m')} m, "
                    f"rotation {action.get('requested_trunk_rotation_rad')} rad"
                )
            print(
                f"   IK : {'valide' if action.get('ik_result', {}).get('valid') else 'invalide'}"
                f" | Validation physique : "
                f"{action.get('physical_validation_status', 'NOT_TESTED')}"
            )
            if action.get("physical_failure_reason"):
                print(f"   Motif : {action['physical_failure_reason']}")
        print(f"Transferts : {transfers} | Repositionnements : {repositions}")
        print(
            "Vérifications : IK par waypoint, limites articulaires et appuis "
            "géométriques; collisions, stabilité statique et dynamique non validées."
        )
        print(f"Search time: {result.get('elapsed_s', 0.0):.2f}s")
        return
    print(f"Wall: {result['wall']}")
    print(f"Initial supports: {result['initial_supports']}")
    top_holds = [
        item["coords"]
        for item in result["holds"]
        if item["id"] in result.get("top_hold_indices", [])
    ]
    print(f"Top holds: {top_holds}")
    print("\nPlan:")
    for index, action in enumerate(result["plan"], start=1):
        col, row = action["target_coords"]
        print(f"{index}. {action['limb']} -> ({col}, {row})")
    print(f"\nNumber of movements: {len(result['plan'])}")
    reposition_count = sum(
        action.get("kind") == "reposition"
        for action in result.get("execution_actions", [])
    )
    print(
        "Trunk repositionings in execution sequence: "
        f"{reposition_count}"
    )
    print(f"Planning status: {result['status']}")
    if not result["success"]:
        print(f"Reason: {result['reason']}")
    print(f"Branches explored: {result['branches_explored']}")
    print(f"Nodes explored: {result.get('nodes_explored', 0)}")
    print(
        "Candidates: "
        f"{result.get('transfer_candidates_tested', 0)} transfers, "
        f"{result.get('reposition_candidates_tested', 0)} trunk repositions "
        f"({result.get('repositions_validated', 0)} validated)"
    )
    print(
        f"Maximum progress row: {result.get('max_progress_row', 'unknown')}"
        f" | Search time: {result.get('elapsed_s', 0.0):.2f}s"
    )
    print(
        "Support checks: "
        f"{result.get('supports_retained', 0)}/"
        f"{result.get('support_checks', 0)} retained"
        f" | Unexpected contacts observed: "
        f"{result.get('unexpected_contacts_observed', 0)}"
    )
    if result.get("failure_reasons"):
        print(f"Candidate failures: {result['failure_reasons']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", default="N2", help="competition wall: N1, N2, or N3")
    parser.add_argument(
        "--mode", choices=("plan", "execute", "validate"), default="plan",
        help="plan, execute a physically validated plan, or validate a saved route",
    )
    parser.add_argument(
        "--planning-mode",
        choices=("physical", "theoretical", "realistic"),
        default="physical",
        help="legacy physical search, theoretical route search, or route validation",
    )
    parser.add_argument("--load-plan-file", type=Path)
    parser.add_argument("--plan-file", type=Path, help="save the plan/result as JSON")
    parser.add_argument("--max-moves", type=int, default=64)
    parser.add_argument("--max-branches", type=int, default=128)
    parser.add_argument(
        "--time-limit-s", type=float, default=600.0,
        help="soft search budget; checked between physical candidate validations",
    )
    parser.add_argument(
        "--settle-s", type=float, default=1.0,
        help="settling time after reset_hanging (seconds)",
    )
    args = parser.parse_args(argv)
    if args.wall.lower() == "all":
        if (
            args.mode != "plan"
            or args.planning_mode != "theoretical"
            or args.load_plan_file is not None
            or args.plan_file is not None
        ):
            parser.error(
                "--wall all requires theoretical planning without a loaded plan "
                "or --plan-file"
            )
        return plan_all_walls(
            args.max_moves, args.max_branches, args.time_limit_s
        )
    if args.mode == "validate" and args.load_plan_file is None:
        parser.error("--mode validate requires --load-plan-file")
    if args.planning_mode == "realistic" and args.load_plan_file is None:
        parser.error("--planning-mode realistic requires --load-plan-file")
    if (
        args.planning_mode == "realistic"
        and args.plan_file is not None
        and args.plan_file.resolve() == args.load_plan_file.resolve()
    ):
        parser.error("realistic validation output must not overwrite the theoretical plan")

    sim = ClimbingSim(args.wall, cameras=[])
    try:
        settle_s = (
            0.0
            if args.mode == "plan" and args.planning_mode == "theoretical"
            else args.settle_s
        )
        sim.reset_hanging(settle_s=settle_s)
        print(f"Initial supports: {sim.hooked()}")
        if args.load_plan_file:
            result = load_plan(args.load_plan_file)
            if result.get("wall") != sim.wall.name:
                raise ValueError(
                    f"Plan wall {result.get('wall')!r} does not match {sim.wall.name!r}"
                )
        else:
            result = build_plan(
                sim,
                max_moves=args.max_moves,
                max_branches=args.max_branches,
                time_limit_s=args.time_limit_s,
                planning_mode=args.planning_mode,
            )
        if args.mode == "validate" or args.planning_mode == "realistic":
            result = validate_plan_result(sim, result)
            result["planning_mode"] = "realistic"
        print_result(result)
        if args.plan_file:
            save_plan(args.plan_file, result)
        if args.mode == "execute":
            if result.get("planning_mode") == "theoretical" and (
                result.get("physical_validation_status") != "PHYSICALLY_VALIDATED"
            ):
                print(
                    "Execution skipped: theoretical route has not passed physical validation."
                )
            elif not result["success"]:
                print("Execution skipped: no complete plan was found.")
            else:
                execution = execute_plan(sim, result)
                print(
                    f"Execution: {'SUCCESS' if execution['success'] else 'FAILURE'}"
                    f" ({execution['reason']}; "
                    f"{execution['step_reached']} movements completed)"
                )
        return result["plan"]
    finally:
        sim.close()


if __name__ == "__main__":
    main()