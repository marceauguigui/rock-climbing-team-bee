"""Reproducible, bounded physical benchmark for climbing-plan validation."""

import argparse
import csv
import json
import platform
from collections import Counter, deque
from pathlib import Path

import numpy as np
import mujoco

from climbing_sim import ClimbingSim, LIMBS

if __package__:
    from .contact_planner import plan_path, plan_to_top
    from .validate_plan import (
        _hold_socket,
        _ik,
        _ik_command_error,
        _contacting_holds,
        _source_hold,
        _support_snapshot,
        has_hold_contact,
        validate_plan,
    )
else:
    from contact_planner import plan_path, plan_to_top
    from validate_plan import (
        _hold_socket,
        _ik,
        _ik_command_error,
        _contacting_holds,
        _source_hold,
        _support_snapshot,
        has_hold_contact,
        validate_plan,
    )


def _same_state(sim, left, right):
    state_signature = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = mujoco.mj_stateSize(sim.model, state_signature)
    left_data = np.empty(state_size)
    right_data = np.empty(state_size)
    mujoco.mj_getState(sim.model, left["data"], left_data, state_signature)
    mujoco.mj_getState(sim.model, right["data"], right_data, state_signature)
    return (
        np.array_equal(left_data, right_data, equal_nan=True)
        and _same_value(left["servos"], right["servos"])
        and _same_value(left["sensors"], right["sensors"])
    )


def _same_value(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return np.array_equal(left, right, equal_nan=True)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple, deque)) and isinstance(right, (list, tuple, deque)):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _accessible_candidates(sim, limb, occupied):
    supports, reason = _support_snapshot(sim, limb)
    if reason:
        return [], reason
    candidates = []
    for hold_index in range(len(sim.wall.holds)):
        if hold_index in occupied:
            continue
        q, error = _ik(sim, limb, _hold_socket(sim, hold_index), supports)
        command_error = _ik_command_error(sim, q, error, "benchmark")
        if command_error is None:
            candidates.append({
                "hold_index": hold_index,
                "ik_error_m": float(error),
                "distance_m": float(np.linalg.norm(
                    sim.ball_positions()[limb] - _hold_socket(sim, hold_index)
                )),
            })
    candidates.sort(key=lambda item: (item["ik_error_m"], item["distance_m"], item["hold_index"]))
    return candidates, None


def _move_record(
    sim, limb, target_index, initial_state, result, candidate_count,
    initial_target_distance,
):
    step = result.get("steps", [{}])[0]
    contacts = step.get("collision_contacts", [])
    history = step.get("contact_history", [])
    support_samples = [
        waypoint.get("supports", {})
        for waypoint in step.get("waypoints", [])
    ]
    if step.get("supports"):
        support_samples.append(step["supports"])
    supports_retained = (
        None
        if not support_samples
        else all(
            len(supports) == len(LIMBS) - 1
            and all(support.get("valid", False) for support in supports.values())
            for supports in support_samples
        )
    )
    unexpected_history = [event for event in history if not event["expected"]]
    unexpected_pairs = {
        tuple(event["geom_ids"]) for event in unexpected_history
    }
    source_hold = step.get("source_hold")
    success = result.get("success", False)
    reason = result.get("reason", "missing_reason")
    contact_assessment = (
        "none_detected"
        if not unexpected_history
        else "present_during_success"
        if success
        else "present_during_physical_failure"
        if reason in {
            "support_lost",
            "movement_blocked",
            "trajectory_tracking_error",
            "starting_contact_persists",
        }
        else "present_during_other_failure"
    )
    return {
        "test_type": "single_move",
        "sequence_length": 1,
        "limb": limb,
        "members": [limb],
        "source_hold": source_hold,
        "target_hold": target_index,
        "targets": [target_index],
        "accessible_candidate_count": candidate_count,
        "initial_target_distance_m": initial_target_distance,
        "final_target_distance_m": step.get(
            "target_distance_m",
            float(np.linalg.norm(
                sim.ball_positions()[limb] - _hold_socket(sim, target_index)
            )),
        ),
        "ik_error_m": step.get("ik_error_m"),
        "max_joint_error_rad": step.get("max_joint_error_rad"),
        "max_current_A": step.get("max_current_A"),
        "ticks": step.get("ticks_executed", 0),
        "supports_retained": supports_retained,
        "unexpected_contact_count": len(unexpected_pairs),
        "unexpected_contact_pairs": [list(pair) for pair in sorted(unexpected_pairs)],
        "unexpected_contact_episodes": len(unexpected_history),
        "preexisting_unexpected_contact_episodes": sum(
            event["preexisting"] for event in unexpected_history
        ),
        "new_contact_episodes": sum(
            event["classification"] == "new_unexpected"
            for event in unexpected_history
        ),
        "contact_history": history,
        "contacts": contacts,
        "contact_assessment": contact_assessment,
        "success": success,
        "status": reason,
        "failure_class": reason if not success else None,
        "candidate_failures": None,
        "state_restored": None,
    }


def _test_single_moves(sim, baseline, max_targets):
    records = []
    discovery = {}
    for limb in LIMBS:
        sim.set_state(baseline)
        occupied = {
            hold_index
            for member in LIMBS
            for hold_index in _contacting_holds(sim, member)
        }
        candidates, support_error = _accessible_candidates(sim, limb, occupied)
        discovery[limb] = {
            "accessible_targets": [item["hold_index"] for item in candidates],
            "occupied_holds": sorted(occupied),
            "support_error": support_error,
        }
        for candidate in candidates[:max_targets]:
            sim.set_state(baseline)
            initial_state = sim.get_state()
            source = _source_hold(sim, limb)
            target_index = candidate["hold_index"]
            initial_target_distance = float(np.linalg.norm(
                sim.ball_positions()[limb] - _hold_socket(sim, target_index)
            ))
            result = validate_plan(
                sim,
                [{"limb": limb, "hold_index": target_index}],
                restore=False,
            )
            record = _move_record(
                sim, limb, target_index, initial_state, result, len(candidates),
                initial_target_distance,
            )
            record["source_hold"] = source
            record["ik_error_m"] = candidate["ik_error_m"]
            sim.set_state(baseline)
            record["state_restored"] = _same_state(sim, initial_state, sim.get_state())
            if not record["state_restored"]:
                raise RuntimeError(f"State restoration failed for {limb} -> hold{target_index}")
            records.append(record)
    return records, discovery


def _best_initial_candidates(sim):
    occupied = {
        hold_index
        for limb in LIMBS
        for hold_index in _contacting_holds(sim, limb)
    }
    candidates = {}
    for limb in LIMBS:
        reachable, _ = _accessible_candidates(sim, limb, occupied)
        for item in reachable:
            previous = candidates.get(item["hold_index"])
            score = (item["ik_error_m"], item["distance_m"], limb)
            if previous is None or score < previous[0]:
                candidates[item["hold_index"]] = (score, item)
    return [
        index for index, _ in sorted(
            candidates.items(), key=lambda entry: (entry[1][0], entry[0])
        )
    ]


def _manual_sequence(sim, baseline, known_moves, length):
    sim.set_state(baseline)
    moves = []
    reason = "success"
    final_holds = None
    try:
        for move_index in range(length):
            if move_index < len(known_moves):
                action = dict(known_moves[move_index])
            else:
                occupied = {
                    index for limb in LIMBS
                    for index in _contacting_holds(sim, limb)
                }
                options = []
                for limb_index, limb in enumerate(LIMBS):
                    candidates, support_error = _accessible_candidates(
                        sim, limb, occupied
                    )
                    if support_error:
                        continue
                    options.extend(
                        (
                            item["ik_error_m"],
                            item["distance_m"],
                            limb_index,
                            limb,
                            item["hold_index"],
                        )
                        for item in candidates
                    )
                if not options:
                    reason = "no_geometrically_reachable_unoccupied_target"
                    break
                _, _, _, limb, hold_index = min(options)
                action = {"limb": limb, "hold_index": hold_index}

            initial_state = sim.get_state()
            limb = action["limb"]
            target_index = action["hold_index"]
            source_hold = _source_hold(sim, limb)
            initial_distance = float(np.linalg.norm(
                sim.ball_positions()[limb] - _hold_socket(sim, target_index)
            ))
            result = validate_plan(sim, [action], restore=False)
            metrics = _move_record(
                sim, limb, target_index, initial_state, result, 0,
                initial_distance,
            )
            metrics["source_hold"] = source_hold
            moves.append({
                "action": action,
                "result": result,
                "metrics": metrics,
            })
            if not result["success"]:
                reason = result["reason"]
                break
        final_holds = {limb: _source_hold(sim, limb) for limb in LIMBS}
    finally:
        sim.set_state(baseline)
    restored = _same_state(sim, baseline, sim.get_state())
    for move in moves:
        move["metrics"]["state_restored"] = restored
    if not restored:
        raise RuntimeError(f"Manual sequence of length {length} did not restore state")
    return moves, reason, final_holds


def _test_sequences(sim, baseline, max_branches, single_records):
    records = []
    known_moves = [
        {"limb": item["limb"], "hold_index": item["target_hold"]}
        for item in single_records if item["success"]
    ]
    initial_target_order = _best_initial_candidates(sim)
    seed_targets = list(dict.fromkeys(
        [move["hold_index"] for move in known_moves] + initial_target_order
    ))
    for length in (2, 3):
        manual_moves, manual_reason, final_holds = _manual_sequence(
            sim, baseline, known_moves, length
        )
        manual_success = (
            len(manual_moves) == length
            and all(move["result"]["success"] for move in manual_moves)
        )
        manual_targets = [
            move["action"]["hold_index"] for move in manual_moves
        ]
        targets = (
            manual_targets
            if len(manual_targets) == length
            else seed_targets[:length]
        )
        if len(targets) < length:
            planner_result = {
                "success": False,
                "reason": "insufficient_distinct_geometrically_reachable_targets",
                "steps": [],
                "branches_explored": 0,
            }
        else:
            planner_result = plan_path(
                sim,
                [{"hold_index": index} for index in targets],
                max_branches=max_branches,
            )
        plan_restored = _same_state(sim, baseline, sim.get_state())
        if not plan_restored:
            raise RuntimeError("plan_path did not restore the simulator state")

        manual_metrics = [move["metrics"] for move in manual_moves]
        manual_history = [
            event for step in manual_metrics
            for event in step.get("contact_history", [])
        ]
        manual_success_with_contacts = any(
            step["unexpected_contact_count"] for step in manual_metrics
        )
        success = bool(planner_result["success"] and manual_success)
        contact_assessment = (
            "none_detected"
            if not manual_history or not any(not event["expected"] for event in manual_history)
            else "present_during_success"
            if success
            else "present_during_physical_failure"
            if manual_reason in {
                "support_lost",
                "movement_blocked",
                "trajectory_tracking_error",
                "starting_contact_persists",
            }
            else "present_during_other_failure"
        )
        status = (
            "success_with_contacts"
            if success and manual_success_with_contacts
            else "success"
            if success
            else planner_result["reason"]
            if not planner_result["success"]
            else manual_reason
        )
        unexpected_pairs = {
            tuple(event["geom_ids"])
            for event in manual_history if not event["expected"]
        }
        records.append({
            "test_type": "sequence",
            "sequence_length": length,
            "limb": None,
            "members": [move["action"]["limb"] for move in manual_moves],
            "source_hold": None,
            "target_hold": None,
            "targets": targets,
            "success": success,
            "status": status,
            "contact_assessment": contact_assessment,
            "failure_class": None if success else status,
            "planner_status": planner_result["reason"],
            "planner_branches_explored": planner_result.get("branches_explored", 0),
            "manual_status": manual_reason,
            "manual_physical_verified": manual_success,
            "manual_step_count": len(manual_moves),
            "manual_supports_retained": all(
                step.get("supports_retained", False) for step in manual_metrics
            ),
            "manual_moves": manual_metrics,
            "manual_contact_history": manual_history,
            "max_joint_error_rad": max(
                (step.get("max_joint_error_rad") or 0.0 for step in manual_metrics),
                default=0.0,
            ),
            "max_current_A": max(
                (step.get("max_current_A") or 0.0 for step in manual_metrics),
                default=0.0,
            ),
            "ticks": sum(step.get("ticks", 0) for step in manual_metrics),
            "unexpected_contact_count": len(unexpected_pairs),
            "unexpected_contact_pairs": [list(pair) for pair in sorted(unexpected_pairs)],
            "final_physical_holds": final_holds,
            "state_restored": plan_restored,
        })
    return records, known_moves


def _summary(records, discovery):
    total = len(records)
    successful = [record for record in records if record["success"]]
    successful_clean = [
        record for record in successful
        if record.get("status") == "success"
    ]
    successful_contacts = [
        record for record in successful
        if record.get("status") == "success_with_contacts"
    ]
    successful_without_new_contacts = [
        record for record in successful
        if record.get("new_contact_episodes", 0) == 0
        and not any(
            event["classification"] == "new_unexpected"
            for event in record.get("manual_contact_history", [])
        )
    ]
    failures = Counter(
        record.get("failure_class") for record in records if not record["success"]
    )
    by_member = {}
    for limb in LIMBS:
        member_records = [
            record for record in records
            if record["test_type"] == "single_move" and record["limb"] == limb
        ]
        by_member[limb] = {
            "attempts": len(member_records),
            "successes": sum(record["success"] for record in member_records),
            "accessible_targets": discovery[limb]["accessible_targets"],
            "failures": dict(Counter(
                record["failure_class"] for record in member_records
                if not record["success"]
            )),
        }
    by_sequence = {
        str(length): [
            record for record in records
            if record["test_type"] == "sequence"
            and record["sequence_length"] == length
        ]
        for length in (2, 3)
    }
    return {
        "total_trials": total,
        "successes": len(successful),
        "success_rate": len(successful) / total if total else 0.0,
        "success_without_unexpected_contacts": len(successful_clean),
        "success_without_unexpected_contacts_rate": len(successful_clean) / total if total else 0.0,
        "success_with_unexpected_contacts": len(successful_contacts),
        "success_with_unexpected_contacts_rate": len(successful_contacts) / total if total else 0.0,
        "success_without_new_unexpected_contacts": len(successful_without_new_contacts),
        "success_without_new_unexpected_contacts_rate": (
            len(successful_without_new_contacts) / total if total else 0.0
        ),
        "failure_reasons": dict(failures),
        "ik_failures": sum(
            "ik" in (reason or "").lower() or "reachable" in (reason or "").lower()
            for reason in failures
            for _ in range(failures[reason])
        ),
        "tracking_failures": sum(
            failures.get(reason, 0)
            for reason in ("trajectory_tracking_error", "movement_blocked")
        ),
        "support_losses": failures.get("support_lost", 0),
        "blocked_moves": failures.get("movement_blocked", 0),
        "unexpected_contact_episodes": sum(
            record.get("unexpected_contact_episodes", 0)
            + sum(
                not event["expected"]
                for event in record.get("manual_contact_history", [])
            )
            for record in records
        ),
        "trials_with_unexpected_contacts": sum(
            record.get("unexpected_contact_count", 0) > 0
            for record in records
        ),
        "unexpected_contact_pairs_observed": sum(
            record.get("unexpected_contact_count", 0) for record in records
        ),
        "new_unexpected_contact_episodes": sum(
            record.get("new_contact_episodes", 0)
            + sum(
                event["classification"] == "new_unexpected"
                for event in record.get("manual_contact_history", [])
            )
            for record in records
        ),
        "physical_failures_with_contacts": sum(
            (
                bool(record.get("unexpected_contact_episodes"))
                or any(
                    not event["expected"]
                    for event in record.get("manual_contact_history", [])
                )
            )
            and record.get("failure_class") in {
                "support_lost",
                "movement_blocked",
                "trajectory_tracking_error",
                "starting_contact_persists",
            }
            for record in records
        ),
        "by_member": by_member,
        "by_sequence_length": {
            length: {
                "trials": len(entries),
                "successes": sum(item["success"] for item in entries),
                "reasons": dict(Counter(
                    item.get("failure_class") for item in entries if not item["success"]
                )),
            }
            for length, entries in by_sequence.items()
        },
    }


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_benchmark(wall="N2", seed=0, max_targets=2, max_branches=8):
    sim = ClimbingSim(wall, cameras=[], seed=seed)
    try:
        sim.reset_hanging()
        baseline = sim.get_state()
        for limb in LIMBS:
            source = _source_hold(sim, limb)
            if source is None or not has_hold_contact(sim, limb, source):
                raise RuntimeError(f"Initial support has no physical hold contact: {limb}")

        single_records, discovery = _test_single_moves(sim, baseline, max_targets)
        sequence_records, sequence_moves = _test_sequences(
            sim, baseline, max_branches, single_records
        )
        records = single_records + sequence_records
        sim.set_state(baseline)
        state_restored = _same_state(sim, baseline, sim.get_state())
        if not state_restored or any(not record.get("state_restored", False) for record in records):
            raise RuntimeError("Simulator state restoration check failed")
        return {
            "environment": {
                "wall": wall,
                "seed": seed,
                "python_version": platform.python_version(),
                "mujoco_version": mujoco.__version__,
                "control_dt_s": sim.control_dt,
                "initial_holds": {
                    limb: _source_hold(sim, limb) for limb in LIMBS
                },
                "initial_support_contacts_verified": True,
                "single_move_target_limit_per_limb": max_targets,
                "sequence_branch_limit": max_branches,
                "sequence_seed_moves": sequence_moves,
                "candidate_order": "IK error, initial distance, then hold index",
                "manual_sequence_policy": (
                    "successful single moves first; later move uses lowest-IK-error "
                    "unoccupied geometrically reachable target"
                ),
                "contact_causality": (
                    "contacts concurrent with failures are reported as correlated, "
                    "not asserted as the cause without force/causal evidence"
                ),
                "state_restored_after_each_trial": state_restored,
            },
            "summary": _summary(records, discovery),
            "candidate_discovery": discovery,
            "trials": records,
        }
    finally:
        sim.close()


def run_search_comparison(
    wall="N2", seed=0, max_branches=8, max_moves=8,
    time_limit_s=60.0, settle_s=1.0,
):
    """Compare search strategies from the same MuJoCo snapshot."""
    strategies = (
        "legacy",
        "transfers_first",
        "progress",
        "useful_repositions",
        "combined",
    )
    sim = ClimbingSim(wall, cameras=[], seed=seed)
    try:
        sim.reset_hanging(settle_s=settle_s)
        baseline = sim.get_state()
        start_contacts = {
            limb: sorted(_contacting_holds(sim, limb)) for limb in LIMBS
        }
        trials = []
        for strategy in strategies:
            sim.set_state(baseline)
            result = plan_to_top(
                sim,
                max_moves=max_moves,
                max_branches=max_branches,
                time_limit_s=time_limit_s,
                search_strategy=strategy,
            )
            trials.append({
                "strategy": strategy,
                "success": result["success"],
                "status": result["status"],
                "reason": result["reason"],
                "elapsed_s": result.get("profile", {}).get(
                    "total_s", result.get("elapsed_s")
                ),
                "profile": dict(result.get("profile", {})),
                "branches_explored": result["branches_explored"],
                "nodes_explored": result.get("nodes_explored", 0),
                "transfer_candidates_tested": result.get(
                    "transfer_candidates_tested", 0
                ),
                "transfers_validated": result.get("transfers_validated", 0),
                "reposition_candidates_tested": result.get(
                    "reposition_candidates_tested", 0
                ),
                "repositions_validated": result.get("repositions_validated", 0),
                "max_progress_row": result.get("max_progress_row"),
                "transfer_plan_length": len(result["plan"]),
                "execution_sequence_length": len(result.get("execution_actions", [])),
                "failure_reasons": result.get("failure_reasons", {}),
            })
            sim.set_state(baseline)
            if not _same_state(sim, baseline, sim.get_state()):
                raise RuntimeError(
                    f"State restoration failed after strategy {strategy}"
                )
        return {
            "environment": {
                "wall": wall,
                "seed": seed,
                "settle_s": settle_s,
                "max_branches": max_branches,
                "max_moves": max_moves,
                "soft_time_limit_s": time_limit_s,
                "initial_contacts": start_contacts,
                "same_initial_snapshot_per_strategy": True,
            },
            "trials": trials,
        }
    finally:
        sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", default="N2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-targets", type=int, default=2)
    parser.add_argument("--max-branches", type=int, default=8)
    parser.add_argument(
        "--compare-search", action="store_true",
        help="compare five search strategies from the same initial MuJoCo state",
    )
    parser.add_argument("--max-moves", type=int, default=8)
    parser.add_argument("--search-time-limit-s", type=float, default=60.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--output-prefix", type=Path, default=Path("planner/benchmark"))
    args = parser.parse_args()
    if args.max_targets < 1 or args.max_branches < 1:
        parser.error("--max-targets and --max-branches must be positive")

    if args.compare_search:
        report = run_search_comparison(
            args.wall,
            args.seed,
            max_branches=args.max_branches,
            max_moves=args.max_moves,
            time_limit_s=args.search_time_limit_s,
            settle_s=args.settle_s,
        )
    else:
        report = run_benchmark(
            args.wall, args.seed, args.max_targets, args.max_branches
        )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, default=_json_value) + "\n")
    rows = report["trials"]
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, default=_json_value)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            })
    print(json.dumps(report.get("summary", report["trials"]), indent=2))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
