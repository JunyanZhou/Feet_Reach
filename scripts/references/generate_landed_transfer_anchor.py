"""Create a double-foot, wall-hand geometric reset pose for Isaac screening."""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from generate_foot_reach_dataset import Workspace
from generate_land_then_release import check_geometry, solve_arm_release


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, required=True)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--source-manifest", type=Path,
                   default=Path("artifacts/foot_reach/manifest.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frame", type=int, default=150)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--hand-preload-m", type=float, default=0.,
                   help="Move the left hand into the wall by this amount (m).")
    args = p.parse_args()
    if args.output.exists() or args.frames < 100 or not 0 <= args.hand_preload_m <= .02:
        p.error("Output must be new, have at least 100 frames, and use <=2 cm preload")
    source = json.loads(args.reference.read_text())
    w = Workspace(source)
    trajectory = np.load(args.trajectory.resolve(strict=True))
    q = np.r_[trajectory["root_pos"][args.frame],
              trajectory["root_rot"][args.frame],
              trajectory["dof_pos"][args.frame]].astype(float)
    if args.hand_preload_m:
        q, _ = solve_arm_release(w, q, -args.hand_preload_m)
    reason, clearance = check_geometry(w, q, source["wall_plane"]["position_m"])
    if reason:
        raise ValueError(f"Anchor geometry failed: {reason}")
    pin.framesForwardKinematics(w.model, w.data, q)
    positions = np.asarray([w.data.oMf[frame].translation for frame in w.body_ids])
    rotations = np.asarray([w.data.oMf[frame].rotation for frame in w.body_ids])
    root_R = pin.Quaternion(q[3:7]).matrix()
    local_pos = (positions-q[:3])@root_R
    local_rot = np.einsum("ij,bjk->bik", root_R.T, rotations)
    local_quat = Rotation.from_matrix(local_rot).as_quat()
    foot = w.data.oMf[w.foot].translation
    right = w.data.oMf[w.support].translation
    hand = w.data.oMf[w.hand].translation
    com = pin.centerOfMass(w.model, w.data, q)
    foot_b = root_R.T@(foot-q[:3])
    def repeat(value):
        return np.repeat(np.asarray(value, dtype="f4")[None], args.frames, axis=0)
    clip = {
        "fps": np.asarray(50), "joint_names": np.asarray(w.names),
        "body_names": np.asarray(w.bodies),
        "root_pos": repeat(q[:3]), "root_rot": repeat(q[3:7]),
        "dof_pos": repeat(q[7:]), "local_body_pos": repeat(local_pos),
        "local_body_rot": repeat(local_quat), "target_foot_pos_b": repeat(foot_b),
        "joint_vel": np.zeros((args.frames, len(w.names)), dtype="f4"),
        "joint_acc": np.zeros((args.frames, len(w.names)), dtype="f4"),
        "torques": np.zeros((args.frames, len(w.names)), dtype="f4"),
        "contact_forces": np.zeros((args.frames, 9), dtype="f4"),
    }
    for split in ("train", "val"):
        dest = args.output/"motions"/split
        dest.mkdir(parents=True)
        np.savez_compressed(dest/"landed_transfer.npz", **clip)
    original = json.loads(args.source_manifest.read_text())
    manifest = {key: original[key] for key in
                ("seed", "reference", "joint_names", "body_names", "sole_offset")}
    manifest.update(fps=50, static_frames=args.frames,
                    source_trajectory=str(args.trajectory), source_frame=args.frame,
                    clips=[{"file": f"motions/{split}/landed_transfer.npz",
                            "frames": args.frames, "split": split}
                           for split in ("train", "val")])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    report = {"trajectory": str(args.trajectory), "frame": args.frame,
              "hand_preload_m": args.hand_preload_m,
              "root_xyz_m": q[:3].tolist(), "com_xyz_m": com.tolist(),
              "left_sole_xyz_m": foot.tolist(), "right_sole_xyz_m": right.tolist(),
              "hand_xyz_m": hand.tolist(), "min_clearance_m": clearance,
              "note": "Geometric reset pose only; no validated static wrench or Isaac support."}
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
