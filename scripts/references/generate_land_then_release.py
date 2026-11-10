"""Generate a geometric two-foot landing followed by hand withdrawal.

This is an IK seed. It must pass an Isaac rollout before use as a teacher.
"""

import argparse
import json
from pathlib import Path

import coal
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from generate_foot_reach_dataset import Workspace
from probe_multi_target_support import solve_case


def smoothstep(u):
    return 10*u**3 - 15*u**4 + 6*u**5


def segment(a, b, count):
    u = np.linspace(0., 1., count+1)
    return a[None] + smoothstep(u)[:, None]*(b-a)[None]


def solve_arm_release(w, q_land, distance):
    pin.framesForwardKinematics(w.model, w.data, q_land)
    original = w.data.oMf[w.hand].copy()
    target = original.translation.copy()
    target[1] -= distance
    names = [name for name in w.names if name.startswith((
        "left_shoulder", "left_elbow", "left_wrist"))]
    indices = [w.model.joints[w.model.getJointId(name)].idx_q for name in names]
    lo = w.model.lowerPositionLimit[indices] + 1e-4
    hi = w.model.upperPositionLimit[indices] - 1e-4

    def residual(x):
        q = q_land.copy()
        q[indices] = x
        pin.framesForwardKinematics(w.model, w.data, q)
        hand = w.data.oMf[w.hand]
        return np.r_[100*(hand.translation-target),
                     2*pin.log3(original.rotation.T@hand.rotation),
                     .2*(x-q_land[indices])]

    solved = least_squares(residual, q_land[indices], bounds=(lo, hi),
                           max_nfev=100)
    q = q_land.copy()
    q[indices] = solved.x
    pin.framesForwardKinematics(w.model, w.data, q)
    error = np.linalg.norm(w.data.oMf[w.hand].translation-target)
    if not solved.success or error > .0005:
        raise ValueError(f"Hand withdrawal IK error {error:.6f} m")
    return q, float(error)


def check_geometry(w, q, wall_plane):
    pin.framesForwardKinematics(w.model, w.data, q)
    poses = {name: coal.Transform3s(w.data.oMf[frame].rotation,
                                   w.data.oMf[frame].translation)
             for name, frame in w.frame_ids.items()}
    min_distance = np.inf
    for first, second in w.pairs:
        result = coal.DistanceResult()
        distance = coal.distance(w.hulls[first], poses[first],
                                 w.hulls[second], poses[second],
                                 w.distance_request, result)
        min_distance = min(min_distance, distance)
        if distance < .008:
            return "self_collision", min_distance
    for name in ("left_ankle_roll_link", "left_ankle_pitch_link",
                 "left_knee_link"):
        pose = w.data.oMf[w.frame_ids[name]]
        vertices = w.vertices[name]@pose.rotation.T + pose.translation
        if vertices[:, 2].min() < -.002 or vertices[:, 1].max() > wall_plane-.012:
            return "environment_collision", min_distance
    return None, min_distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--landing-xyz", type=float, nargs=3,
                   default=(0., .06, 0.))
    p.add_argument("--hand-withdrawal", type=float, default=.06)
    p.add_argument("--root-transfer-y", type=float, default=0.,
                   help="Shift the root after landing while keeping both feet and hand fixed.")
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.fps != 50 or not 0 < args.hand_withdrawal <= .1
            or not -.05 <= args.root_transfer_y <= .05):
        p.error("Expected 50 Hz, hand withdrawal within (0, 0.1] m, root shift within +/-5 cm")
    source = json.loads(args.reference.read_text())
    w = Workspace(source)
    q0 = w.q0.copy()
    target = np.asarray(args.landing_xyz, dtype=float)
    q_land = w.solve(target, q0)
    if q_land is None:
        raise ValueError("Left-foot landing IK failed")
    q_transfer = q_land.copy()
    transfer_report = None
    if args.root_transfer_y:
        w_land = Workspace(dict(source, q_pinocchio=q_land.tolist()))
        transfer_report = solve_case(w_land, "root",
                                     np.array([0., args.root_transfer_y, 0.]),
                                     check_dynamics=False)
        if (not transfer_report["solver_success"]
                or transfer_report["root_target_error_m"] > .005
                or transfer_report["max_fixed_contact_pose_error"] > .015):
            raise ValueError(f"Root load-transfer IK failed: {transfer_report}")
        q_transfer = np.asarray(transfer_report["q_pinocchio"])
    q_release, hand_error = solve_arm_release(w, q_transfer, args.hand_withdrawal)
    # 2 s foot landing, 1 s load-transfer hold, 2 s hand withdrawal,
    # 1 s hand-off-wall hold. Adjacent segments share one endpoint.
    land = segment(q0, q_land, 2*args.fps)
    transfer = segment(q_land, q_transfer, args.fps)[1:]
    release = segment(q_transfer, q_release, 2*args.fps)[1:]
    hold = np.repeat(q_release[None], args.fps, axis=0)
    q = np.concatenate((land, transfer, release, hold))
    time = np.arange(len(q))/args.fps
    wall = float(json.loads(args.reference.read_text())["wall_plane"]["position_m"])
    min_clearance = np.inf
    sole_positions = []
    hand_positions = []
    for i, pose in enumerate(q):
        reason, clearance = check_geometry(w, pose, wall)
        if reason:
            raise ValueError(f"Geometry failure at frame {i}: {reason}")
        min_clearance = min(min_clearance, clearance)
        sole_positions.append(w.data.oMf[w.foot].translation.copy())
        hand_positions.append(w.data.oMf[w.hand].translation.copy())
    sole_positions = np.asarray(sole_positions)
    hand_positions = np.asarray(hand_positions)
    joint_delta = np.max(np.abs(np.diff(q[:, 7:], axis=0)), axis=1)*args.fps
    if sole_positions[:, 2].min() < -.002:
        raise ValueError("Landing trajectory penetrates ground")
    report = {"reference": str(args.reference), "frames": len(q),
              "fps": args.fps, "duration_s": float(time[-1]),
              "landing_xyz_m": target.tolist(),
              "landing_sole_error_m": float(np.linalg.norm(sole_positions[2*args.fps]-target)),
              "root_transfer_y_m": float(q_transfer[1]-q_land[1]),
              "root_transfer_ik": ({k: v for k, v in transfer_report.items()
                                    if k != "q_pinocchio"} if transfer_report else None),
              "hand_withdrawal_m": float(hand_positions[0, 1]-hand_positions[-1, 1]),
              "hand_ik_error_m": hand_error,
              "min_self_clearance_m": float(min_clearance),
              "min_left_sole_height_m": float(sole_positions[:, 2].min()),
              "max_joint_speed_rad_s": float(joint_delta.max()),
              "note": "Geometric IK seed only; contact dynamics and Isaac tracking unverified."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, time_s=time.astype("f4"),
                        joint_names=np.asarray(w.names),
                        root_pos=q[:, :3].astype("f4"),
                        root_rot=q[:, 3:7].astype("f4"),
                        dof_pos=q[:, 7:].astype("f4"),
                        left_sole_world=sole_positions.astype("f4"),
                        hand_world=hand_positions.astype("f4"),
                        report_json=np.asarray(json.dumps(report)))
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
