"""Deterministic bottle execution and feedback for the local tabletop API."""

import numpy as np

from .g1_multiview import pose_matrix

FAULTS = ("none", "grasp_unconfirmed")


def simulate_stages(preview, fault, emit, checkpoint):
    if preview["state"] != "preview":
        return preview

    candidate = preview["candidates"][0]
    stage_context = {
        "candidate_id": candidate["candidate_id"],
        "scene_revision": preview["scene_revision"],
    }
    geometry = preview["target_geometry"]
    initial_bottle = np.eye(4)
    initial_bottle[:3, 3] = geometry["center_m"]
    bottle = initial_bottle.copy()
    grip_point = (*geometry["center_m"][:2], geometry["grip_height_m"])
    tcp = np.eye(4)
    attached = None
    gripper_closed = False
    shake_cycles = 0
    results = []

    for stage in candidate["stages"]:
        checkpoint()
        stage_id = stage["stage_id"]
        emit("tabletop_sim_stage_started", {**stage_context, "stage_id": stage_id})
        waypoints = stage["tcp_waypoints_base"]
        applied_waypoints = 0
        for waypoint in waypoints:
            checkpoint()
            tcp = pose_matrix(waypoint)
            if attached is not None:
                bottle = tcp @ attached
            applied_waypoints += 1
        reached = applied_waypoints > 0

        if stage_id == "close":
            gripper_closed = True
            if fault != "grasp_unconfirmed" and np.allclose(tcp[:3, 3], grip_point, atol=1e-5):
                attached = np.linalg.inv(tcp) @ bottle
        elif stage_id == "release" and np.allclose(bottle, initial_bottle, atol=1e-5):
            gripper_closed = False
            attached = None
        elif stage_id == "shake" and attached is not None:
            shake_cycles += applied_waypoints // 3

        lift_m = float(bottle[2, 3] - initial_bottle[2, 3])
        at_origin = bool(np.allclose(bottle, initial_bottle, atol=1e-5))
        held = gripper_closed and attached is not None
        evidence = stage["completion_evidence"]
        verified = {
            "arm_reached": reached,
            "target_held": held,
            "target_followed_off_table": reached and held and lift_m >= 0.015,
            "target_supported": reached and held and at_origin,
            "target_released_at_origin": not gripper_closed and not held and at_origin,
        }[evidence]
        row = {
            **stage_context,
            "stage_id": stage_id,
            "status": "verified" if verified else "rejected",
            "evidence": evidence,
            "held_object": preview["object_id"] if held else None,
            "tcp_position_m": tcp[:3, 3].tolist(),
            "bottle_position_m": bottle[:3, 3].tolist(),
            "bottle_lift_m": lift_m,
            "shake_cycles": shake_cycles,
            "feedback_source": "deterministic_tabletop_simulator",
        }
        results.append(row)
        emit("tabletop_sim_stage_result", row)
        if not verified:
            break

    complete = all(row["status"] == "verified" for row in results)
    return {
        **preview,
        "state": "complete" if complete else "held",
        "reason": "simulated_task_verified" if complete else "simulated_stage_unconfirmed",
        "simulated_candidate_id": candidate["candidate_id"],
        "simulated_stage_results": results,
        "simulated_verified_stages": sum(row["status"] == "verified" for row in results),
        "shake_cycles": shake_cycles,
    }
