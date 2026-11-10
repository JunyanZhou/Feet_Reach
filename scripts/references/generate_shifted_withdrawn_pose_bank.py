"""Generate statically screened single-foot poses with shifted pelvis targets.

Each accepted pose holds the right foot fixed and the left hand away from the
wall; these are only offline candidates until checked in Isaac physics.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from generate_foot_reach_dataset import Workspace
from probe_multi_target_support import solve_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--release", type=Path,
                   default=Path("artifacts/unified_support_reach/support_release.npz"))
    p.add_argument("--hand-withdrawal", type=float,
                   help="Solve a static hand-off-wall pose this far from the wall-supported reference, "
                        "instead of reading a completed release trajectory")
    p.add_argument("--source-manifest", type=Path,
                   default=Path("artifacts/foot_reach/manifest.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--offsets", type=float, nargs="+",
                   default=[-.04, -.03, -.02, -.01, 0., .01, .02, .03, .04])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.reference.read_text())
    if args.hand_withdrawal is None:
        with np.load(args.release) as release:
            active = np.flatnonzero(release["hand_release_fraction"] >= .999)
            q0 = release["q_pinocchio"][active[len(active)//2]].astype(float)
    else:
        if not .02 <= args.hand_withdrawal <= .14:
            raise ValueError("hand-withdrawal must be between 2 and 14 cm")
        wall_workspace = Workspace(source)
        guess = wall_workspace.q0.copy()
        for fraction in np.linspace(.25, 1., 4):
            result = solve_case(wall_workspace, "left_hand",
                                np.array([0., -args.hand_withdrawal*fraction, 0.]),
                                initial_q=guess, check_dynamics=False)
            if not result["solver_success"] or result["hand_target_error_m"] > .002:
                raise RuntimeError(f"Hand withdrawal IK failed at {fraction}: {result}")
            guess = np.asarray(result["q_pinocchio"])
        q0 = guess
    w = Workspace(dict(source, q_pinocchio=q0.tolist()))
    clips = []
    report = []
    for offset in args.offsets:
        outcome = solve_case(w, "root", np.array([0., offset, 0.]),
                             check_dynamics=False)
        item = {"requested_root_y_offset_m": offset,
                "ik_success": outcome["solver_success"],
                "root_error_m": outcome["root_target_error_m"],
                "fixed_pose_error_m": outcome["max_fixed_contact_pose_error"],
                "max_joint_delta_rad": outcome["max_joint_delta_rad"]}
        if not outcome["solver_success"] or outcome["root_target_error_m"] > .006 or \
                outcome["max_fixed_contact_pose_error"] > .015:
            item["accepted"] = False
            report.append(item)
            continue
        q = np.asarray(outcome["q_pinocchio"])
        pin.framesForwardKinematics(w.model, w.data, q)
        pin.computeJointJacobians(w.model, w.data, q)
        pin.updateFramePlacements(w.model, w.data)
        right = w.data.oMf[w.support].translation.copy()
        left = w.data.oMf[w.foot].translation.copy()
        hand = w.data.oMf[w.hand].translation.copy()
        v = np.zeros(w.model.nv)
        h = pin.rnea(w.model, w.data, q, v, v)
        J = pin.getFrameJacobian(w.model, w.data, w.support,
                                 pin.LOCAL_WORLD_ALIGNED)
        wrench = np.linalg.solve(J[:, :6].T, h[:6])
        tau = h[6:] - J[:, 6:].T@wrench
        normal = wrench[2]
        torque_ratio = np.max(np.abs(tau)/np.maximum(.85*w.model.effortLimit[6:], 1e-6))
        friction_ratio = np.linalg.norm(wrench[:2])/max(normal, 1e-6)
        moment_ratio = max(abs(wrench[3]/normal)/.028,
                           abs(wrench[5]/normal)/.018,
                           (-wrench[4]/normal)/.11,
                           (wrench[4]/normal)/.045)
        old_contacts = w.contact_poses
        w.contact_poses = [w.data.oMf[w.support].copy(), w.data.oMf[w.hand].copy()]
        collision = w.geometry(q)
        w.contact_poses = old_contacts
        item.update(hand_wall_distance_m=float(w.wall_y-hand[1]),
                    swing_foot_height_m=float(left[2]),
                    right_foot_normal_N=float(normal),
                    max_torque_ratio=float(torque_ratio),
                    friction_ratio=float(friction_ratio),
                    foot_moment_ratio=float(moment_ratio),
                    collision=collision)
        item["accepted"] = bool(normal > 0 and friction_ratio < .7
                                and torque_ratio < 1 and moment_ratio < 1
                                and w.wall_y-hand[1] > .04 and not collision)
        if item["accepted"]:
            wp = np.array([w.data.oMf[f].translation for f in w.body_ids])
            wr = np.array([w.data.oMf[f].rotation for f in w.body_ids])
            root_R = pin.Quaternion(q[3:7]).matrix()
            local_pos = (wp-q[:3])@root_R
            local_rot = np.einsum("ij,bjk->bik", root_R.T, wr)
            local_quat = Rotation.from_matrix(local_rot).as_quat()
            foot_b = root_R.T@(left-q[:3])
            clip = {
                "fps": np.array(50),
                "joint_names": np.array(w.names), "body_names": np.array(w.bodies),
                "root_pos": np.repeat(q[None, :3].astype("f4"), args.frames, axis=0),
                "root_rot": np.repeat(q[None, 3:7].astype("f4"), args.frames, axis=0),
                "dof_pos": np.repeat(q[None, 7:].astype("f4"), args.frames, axis=0),
                "local_body_pos": np.repeat(local_pos[None].astype("f4"), args.frames, axis=0),
                "local_body_rot": np.repeat(local_quat[None].astype("f4"), args.frames, axis=0),
                "target_foot_pos_b": np.repeat(foot_b[None].astype("f4"), args.frames, axis=0),
                "joint_vel": np.zeros((args.frames, len(w.names)), dtype="f4"),
                "joint_acc": np.zeros((args.frames, len(w.names)), dtype="f4"),
                "torques": np.repeat(tau[None].astype("f4"), args.frames, axis=0),
                "contact_forces": np.repeat(np.r_[wrench, np.zeros(3)][None].astype("f4"),
                                             args.frames, axis=0),
            }
            name = f"shift_y_{len(clips):02d}.npz"
            for split in ("train", "val"):
                path = args.output/"motions"/split
                path.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path/name, **clip)
            item["clip_index"] = len(clips)
            item["clip_name"] = name
            clips.append(item)
        report.append(item)
        print("POSE", json.dumps(item), flush=True)
    if not clips:
        raise RuntimeError("No candidate pose passed the offline screen")
    source_manifest = json.loads(args.source_manifest.read_text())
    manifest = {key: source_manifest[key] for key in
                ("seed", "reference", "joint_names", "body_names", "sole_offset")}
    manifest.update(fps=50, static_frames=args.frames, frames=2*args.frames*len(clips),
                    hand_withdrawal_m=args.hand_withdrawal,
                    source_release_trajectory=(str(args.release) if
                                               args.hand_withdrawal is None else None),
                    clips=[{"file": f"motions/{split}/{item['clip_name']}",
                            "frames": args.frames, "split": split}
                           for split in ("train", "val") for item in clips])
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output/"report.json").write_text(json.dumps({
        "hand_withdrawal_m": args.hand_withdrawal,
        "source_release_trajectory": (str(args.release) if args.hand_withdrawal is None
                                      else None),
        "poses": report}, indent=2)+"\n")
    print("ACCEPTED", len(clips), "/", len(report), flush=True)


if __name__ == "__main__":
    main()
