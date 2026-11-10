"""Generate collision- and contact-checked G1 root/hand target trajectories.

Each trajectory starts from the verified wall-supported anchor. The right sole
and raised left sole remain fixed while root or wall-hand target moves smoothly.
The output uses the existing 50 Hz MotionDataset NPZ schema.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from generate_foot_reach_dataset import Workspace
from probe_multi_target_support import solve_case


def quintic(t):
    return 10*t**3 - 15*t**4 + 6*t**5


def collision_reason(w, q):
    pin.framesForwardKinematics(w.model, w.data, q)
    original = w.contact_poses
    w.contact_poses = [w.data.oMf[f].copy() for f in (w.support, w.hand)]
    try:
        return w.geometry(q)
    finally:
        w.contact_poses = original


def trajectory(w, goal, offset, fps, seconds, hold):
    q0 = w.q0.copy()
    count = round(seconds*fps)
    knot_ids = np.unique(np.r_[np.arange(0, count+1, 5), count])
    knot_t = knot_ids/fps
    knots = []
    last = q0
    for t in knot_t:
        solved = solve_case(w, goal, offset*quintic(t/seconds),
                            initial_q=last, check_dynamics=False)
        if (not solved["solver_success"] or
                solved["max_fixed_contact_pose_error"] > 2e-4 or
                solved["hand_target_error_m"] > 2e-4 or
                solved["root_target_error_m"] > 2e-4):
            raise ValueError(f"IK failure at {t:.3f}s: {solved}")
        last = np.asarray(solved["q_pinocchio"])
        knots.append(np.r_[last[:3], last[7:]])
    knots = np.stack(knots)
    dim = knots.shape[1]
    spline = CubicSpline(knot_t, knots,
                         bc_type=((1, np.zeros(dim)), (1, np.zeros(dim))))
    times = np.arange(count+1)/fps
    moving = spline(times)
    moving_vel = spline(times, 1)
    moving_acc = spline(times, 2)
    hold_count = max(2, round(hold*fps))
    x = np.concatenate((moving, np.tile(moving[-1], (hold_count, 1))))
    xvel = np.concatenate((moving_vel, np.zeros((hold_count, dim))))
    xacc = np.concatenate((moving_acc, np.zeros((hold_count, dim))))
    q = np.tile(q0, (len(x), 1))
    q[:, :3], q[:, 7:] = x[:, :3], x[:, 3:]
    v = np.zeros((len(x), w.model.nv))
    a = np.zeros_like(v)
    # Free-flyer linear velocity is expressed in the root frame in Pinocchio.
    v[:, :3] = xvel[:, :3] @ w.root_R
    a[:, :3] = xacc[:, :3] @ w.root_R
    v[:, 6:], a[:, 6:] = xvel[:, 3:], xacc[:, 3:]
    return q, v, a


def export_checked(w, path, q, v, a, goal, offset, fps, seconds):
    pin.framesForwardKinematics(w.model, w.data, w.q0)
    support0 = w.data.oMf[w.support].copy()
    hand0 = w.data.oMf[w.hand].copy()
    swing0 = w.data.oMf[w.foot].copy()
    all_body_pos, all_body_rot = [], []
    torques, forces = [], []
    worst_contact_m = worst_goal_m = 0.
    for i, (qi, vi, ai) in enumerate(zip(q, v, a)):
        reason = collision_reason(w, qi)
        if reason:
            return {"failure": reason, "frame": i}
        pin.computeJointJacobians(w.model, w.data, qi)
        pin.updateFramePlacements(w.model, w.data)
        stance = w.data.oMf[w.support]
        hand = w.data.oMf[w.hand]
        swing = w.data.oMf[w.foot]
        fixed_error = max(
            np.linalg.norm(pin.log6(support0.inverse()*stance).vector),
            np.linalg.norm(pin.log6(swing0.inverse()*swing).vector))
        fraction = quintic(min(i/fps/seconds, 1.))
        hand_target = hand0.translation + (offset*fraction if goal == "left_hand" else 0.)
        root_target = w.q0[:3] + (offset*fraction if goal == "root" else 0.)
        goal_error = max(np.linalg.norm(hand.translation-hand_target),
                         np.linalg.norm(qi[:3]-root_target))
        worst_contact_m = max(worst_contact_m, fixed_error)
        worst_goal_m = max(worst_goal_m, goal_error)
        if fixed_error > .0005 or goal_error > .0005:
            return {"failure": "interpolated_target_drift", "frame": i,
                    "fixed_error_m": fixed_error, "goal_error_m": goal_error}
        w.jt = np.column_stack((
            pin.getFrameJacobian(w.model, w.data, w.support,
                                 pin.LOCAL_WORLD_ALIGNED).T,
            pin.getFrameJacobian(w.model, w.data, w.hand,
                                 pin.LOCAL_WORLD_ALIGNED)[:3].T))
        dynamics = w.dynamics(qi, vi, ai)
        if dynamics is None:
            return {"failure": "contact_dynamics", "frame": i}
        torques.append(dynamics[0])
        forces.append(dynamics[1])
        all_body_pos.append([w.data.oMf[f].translation.copy() for f in w.body_ids])
        all_body_rot.append([w.data.oMf[f].rotation.copy() for f in w.body_ids])
    wp = np.asarray(all_body_pos)
    wr = np.asarray(all_body_rot)
    local_pos = (wp-q[:, None, :3]) @ w.root_R
    local_rot = np.einsum("ij,tbjk->tbik", w.root_R.T, wr)
    local_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(
        len(q), len(w.bodies), 4)
    np.savez_compressed(
        path, fps=np.array(fps), joint_names=np.array(w.names),
        body_names=np.array(w.bodies), root_pos=q[:, :3].astype("f4"),
        root_rot=q[:, 3:7].astype("f4"), dof_pos=q[:, 7:].astype("f4"),
        local_body_pos=local_pos.astype("f4"),
        local_body_rot=local_quat.astype("f4"),
        joint_vel=v[:, 6:].astype("f4"), joint_acc=a[:, 6:].astype("f4"),
        torques=np.asarray(torques, dtype="f4"),
        contact_forces=np.asarray(forces, dtype="f4"),
    )
    return {"frames": len(q), "worst_fixed_contact_error_m": worst_contact_m,
            "worst_goal_error_m": worst_goal_m,
            "max_torque_ratio": w.max_torque_ratio}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/teaching"))
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--seconds", type=float, default=2.5)
    p.add_argument("--hold", type=float, default=.5)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    w = Workspace(json.loads(args.reference.read_text()))
    cases = []
    for goal, axis in (("root", 0), ("root", 1),
                       ("left_hand", 0), ("left_hand", 2)):
        for signed_magnitude in (-.015, -.0075, .0075, .015):
            offset = np.zeros(3)
            offset[axis] = signed_magnitude
            cases.append((goal, offset))
    manifest = {"fps": args.fps, "joint_names": w.names,
                "body_names": w.bodies, "source": str(args.reference),
                "clips": [], "failures": [],
                "validation": "full-frame collision, contact pose, target, and inverse dynamics"}
    for index, (goal, offset) in enumerate(cases):
        split = "val" if index % 4 == 0 else "train"
        folder = args.output/"motions"/split
        folder.mkdir(parents=True, exist_ok=True)
        q, v, a = trajectory(w, goal, offset, args.fps, args.seconds, args.hold)
        name = f"{goal}_{index:02d}.npz"
        report = export_checked(w, folder/name, q, v, a, goal, offset,
                                args.fps, args.seconds)
        record = {"name": name, "split": split, "goal": goal,
                  "offset_m": offset.tolist(), **report}
        manifest["failures" if "failure" in report else "clips"].append(record)
        print(record, flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GENERATED", len(manifest["clips"]), "FAILED", len(manifest["failures"]), flush=True)


if __name__ == "__main__":
    main()
