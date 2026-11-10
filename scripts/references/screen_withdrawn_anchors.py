"""Rank supported foot-reach poses for a statically feasible hand-off-wall reset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from generate_foot_reach_dataset import Workspace
from probe_multi_target_support import solve_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path,
                   default=Path("artifacts/wall_support/reference.json"))
    p.add_argument("--clips", type=Path,
                   default=Path("artifacts/foot_reach/motions/val"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/withdrawn_anchor_screen.json"))
    p.add_argument("--frame", type=int, default=100)
    p.add_argument("--withdrawal-m", type=float, default=.06)
    args = p.parse_args()
    source = json.loads(args.reference.read_text())
    results = []
    for clip_path in sorted(args.clips.glob("*v0.npz")):
        with np.load(clip_path) as clip:
            if list(clip["joint_names"]) != source["joint_names"]:
                raise ValueError(f"Joint order mismatch in {clip_path}")
            if len(clip["dof_pos"]) <= args.frame:
                continue
            q0 = np.r_[clip["root_pos"][args.frame], clip["root_rot"][args.frame],
                       clip["dof_pos"][args.frame]].astype(float)
        ref = dict(source, q_pinocchio=q0.tolist())
        w = Workspace(ref)
        outcome = solve_case(w, "left_hand", np.array([0., -args.withdrawal_m, 0.]),
                             check_dynamics=False)
        entry = {"source_clip": str(clip_path), "source_frame": args.frame,
                 "solver_success": outcome["solver_success"],
                 "hand_error_m": outcome["hand_target_error_m"],
                 "fixed_contact_error_m": outcome["max_fixed_contact_pose_error"]}
        if outcome["solver_success"] and outcome["hand_target_error_m"] < .001:
            q = np.array(outcome["q_pinocchio"])
            pin.framesForwardKinematics(w.model, w.data, q)
            pin.computeJointJacobians(w.model, w.data, q)
            pin.updateFramePlacements(w.model, w.data)
            hand_y = w.data.oMf[w.hand].translation[1]
            left_z = w.data.oMf[w.foot].translation[2]
            h = pin.rnea(w.model, w.data, q, np.zeros(w.model.nv),
                         np.zeros(w.model.nv))
            J = pin.getFrameJacobian(w.model, w.data, w.support,
                                     pin.LOCAL_WORLD_ALIGNED)
            wrench = np.linalg.solve(J[:, :6].T, h[:6])
            tau = h[6:] - J[:, 6:].T@wrench
            normal = wrench[2]
            cop_ratio = max(abs(wrench[3]/normal)/.028,
                            abs(wrench[5]/normal)/.018,
                            (-wrench[4]/normal)/.11,
                            (wrench[4]/normal)/.045)
            old = w.contact_poses
            w.contact_poses = [w.data.oMf[w.support].copy(),
                               w.data.oMf[w.hand].copy()]
            collision = w.geometry(q)
            w.contact_poses = old
            entry.update(hand_wall_distance_m=float(w.wall_y-hand_y),
                         swing_foot_height_m=float(left_z),
                         foot_normal_N=float(normal),
                         foot_moment_ratio=float(cop_ratio),
                         max_torque_ratio=float(np.max(np.abs(tau)
                                                       /np.maximum(.85*w.model.effortLimit[6:],1e-6))),
                         friction_ratio=float(np.linalg.norm(wrench[:2])/normal),
                         collision=collision, q_pinocchio=q.tolist())
        results.append(entry)
        print({k: v for k, v in entry.items() if k != "q_pinocchio"}, flush=True)
    results.sort(key=lambda x: x.get("foot_moment_ratio", float("inf")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
