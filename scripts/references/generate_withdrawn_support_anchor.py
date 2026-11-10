"""Export a statically checked right-foot-only pose with the left hand off wall."""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from generate_foot_reach_dataset import Workspace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--release", type=Path,
                   default=Path("artifacts/unified_support_reach/support_release.npz"))
    p.add_argument("--source-manifest", type=Path,
                   default=Path("artifacts/foot_reach/manifest.json"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/withdrawn_anchor"))
    p.add_argument("--frames", type=int, default=500)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.frames < 100:
        raise ValueError("Need a sustained reference of at least 100 frames")
    w = Workspace(json.loads(args.reference.read_text()))
    trajectory = np.load(args.release)
    fraction = trajectory["hand_release_fraction"]
    selected = np.flatnonzero(fraction >= .999)[len(np.flatnonzero(fraction >= .999))//2]
    q = trajectory["q_pinocchio"][selected]
    pin.framesForwardKinematics(w.model, w.data, q)
    pin.computeJointJacobians(w.model, w.data, q)
    pin.updateFramePlacements(w.model, w.data)
    right = w.data.oMf[w.support].translation
    left = w.data.oMf[w.foot].translation
    hand = w.data.oMf[w.hand].translation
    if hand[1] > w.wall_y-.04:
        raise RuntimeError("Hand is not far enough from the wall for a no-contact reset")
    v = np.zeros(w.model.nv)
    h = pin.rnea(w.model, w.data, q, v, v)
    J = pin.getFrameJacobian(w.model, w.data, w.support,
                             pin.LOCAL_WORLD_ALIGNED)
    wrench = np.linalg.solve(J[:, :6].T, h[:6])
    tau = h[6:] - J[:, 6:].T@wrench
    torque_ratio = np.max(np.abs(tau)/np.maximum(.85*w.model.effortLimit[6:], 1e-6))
    if wrench[2] <= 0 or np.linalg.norm(wrench[:2])/wrench[2] > .7 or torque_ratio > 1.:
        raise RuntimeError("Right-foot-only static wrench is not feasible")
    wp = np.array([w.data.oMf[f].translation for f in w.body_ids])
    wr = np.array([w.data.oMf[f].rotation for f in w.body_ids])
    root_R = pin.Quaternion(q[3:7]).matrix()
    local_pos = (wp-q[:3])@root_R
    local_rot = np.einsum("ij,bjk->bik", root_R.T, wr)
    local_quat = Rotation.from_matrix(local_rot).as_quat()
    target_foot_b = root_R.T@(left-q[:3])
    clip = {
        "fps": np.array(50),
        "joint_names": np.array(w.names), "body_names": np.array(w.bodies),
        "root_pos": np.repeat(q[None, :3].astype("f4"), args.frames, axis=0),
        "root_rot": np.repeat(q[None, 3:7].astype("f4"), args.frames, axis=0),
        "dof_pos": np.repeat(q[None, 7:].astype("f4"), args.frames, axis=0),
        "local_body_pos": np.repeat(local_pos[None].astype("f4"), args.frames, axis=0),
        "local_body_rot": np.repeat(local_quat[None].astype("f4"), args.frames, axis=0),
        "target_foot_pos_b": np.repeat(target_foot_b[None].astype("f4"), args.frames, axis=0),
        "joint_vel": np.zeros((args.frames, len(w.names)), dtype="f4"),
        "joint_acc": np.zeros((args.frames, len(w.names)), dtype="f4"),
        "torques": np.repeat(tau[None].astype("f4"), args.frames, axis=0),
        "contact_forces": np.repeat(np.r_[wrench, np.zeros(3)][None].astype("f4"),
                                     args.frames, axis=0),
    }
    for split in ("train", "val"):
        directory = args.output/"motions"/split
        directory.mkdir(parents=True)
        np.savez_compressed(directory/"withdrawn_support.npz", **clip)
    manifest = json.loads(args.source_manifest.read_text())
    manifest = {key: manifest[key] for key in
                ("seed", "reference", "joint_names", "body_names", "sole_offset")}
    manifest.update(fps=50, static_frames=args.frames, frames=2*args.frames,
                    clips=[{"file": f"motions/{split}/withdrawn_support.npz",
                            "frames": args.frames, "split": split}
                           for split in ("train", "val")],
                    source_release_trajectory=str(args.release.resolve()),
                    hand_wall_distance_m=float(w.wall_y-hand[1]),
                    max_static_torque_ratio=float(torque_ratio))
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    report = {"frames": args.frames, "hand_wall_distance_m": float(w.wall_y-hand[1]),
              "right_foot_world_m": right.tolist(), "left_foot_world_m": left.tolist(),
              "right_foot_normal_N": float(wrench[2]),
              "max_static_torque_ratio": float(torque_ratio)}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
