"""Generate a bimanual reach/pinch/pull curriculum from the trained G1 pose.

Only the arm joints are optimized; the root, stance foot, and pinning foot stay
at the validated original reference. This is a kinematic training target, not
a claim of successful physical extraction.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from active_adaptation.utils.motion import MotionData

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "deploy/mujoco_whole_body/g1/g1_29dof_rev_1_0.xml"


def smooth(u):
    u = np.clip(u, 0., 1.)
    return u * u * (3. - 2. * u)


def interp_target(t, keys):
    times = [row[0] for row in keys]
    i = min(max(np.searchsorted(times, t, side="right") - 1, 0), len(keys) - 2)
    a, b = keys[i], keys[i + 1]
    u = smooth((t - a[0]) / (b[0] - a[0]))
    return (1. - u) * np.asarray(a[1]) + u * np.asarray(b[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "dataset/table_retrieval_trained_motion/train")
    p.add_argument("--dest", type=Path, default=ROOT / "dataset/foot_pin_bimanual/train")
    p.add_argument("--duration", type=float, default=8.)
    p.add_argument("--contact-x", type=float, default=.40)
    p.add_argument("--contact-z", type=float, default=.86)
    p.add_argument("--pull-x", type=float, default=.18)
    p.add_argument("--foot-shift-x", type=float, default=.07)
    p.add_argument("--sequential", action="store_true",
                   help="Keep left hand on support until the right hand reaches the part")
    args = p.parse_args()
    source = MotionData.load_memmap(str(args.source))
    meta = json.loads((args.source / "meta_motion.json").read_text())
    names = meta["body_names"]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    root_pos = source.root_pos_w[0].numpy()
    root_quat = source.root_quat_w[0].numpy()
    q_start = source.joint_pos[0].numpy().copy()
    # SciPy requires the initial guess to lie strictly inside the IK bounds.
    # Some source clips sit exactly at the margin after float32 rounding.
    limits_all = model.jnt_range[1:]
    q_start = np.clip(q_start, limits_all[:, 0] + .003,
                      limits_all[:, 1] - .003)
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:] = q_start
    mujoco.mj_forward(model, data)

    def palm(side):
        body = data.body(side + "_wrist_yaw_link")
        return body.xpos + body.xmat.reshape(3, 3) @ np.array([.116, 0., 0.])
    left_start = palm("left").copy()
    right_start = palm("right").copy()
    left_keys = [(0., left_start), (1., left_start),
                 (2.5, [args.contact_x-.01, .14, args.contact_z]),
                 (3.5, [args.contact_x, .085, args.contact_z]),
                 (6.5, [args.pull_x, .085, args.contact_z]),
                 (8., [args.pull_x, .085, args.contact_z])]
    right_keys = [(0., right_start), (1., right_start),
                  (2.5, [args.contact_x-.01, -.14, args.contact_z]),
                  (3.5, [args.contact_x, -.085, args.contact_z]),
                  (6.5, [args.pull_x, -.085, args.contact_z]),
                  (8., [args.pull_x, -.085, args.contact_z])]
    if args.sequential:
        left_keys = [(0., left_start), (2.5, left_start),
                     (4.0, [args.contact_x-.01, .14, args.contact_z]),
                     (4.5, [args.contact_x, .085, args.contact_z]),
                     (7.5, [args.pull_x, .085, args.contact_z]),
                     (9., [args.pull_x, .085, args.contact_z])]
        right_keys = [(0., right_start), (1., right_start),
                      (2.5, [args.contact_x-.01, -.14, args.contact_z]),
                      (3.0, [args.contact_x, -.085, args.contact_z]),
                      (4.5, [args.contact_x, -.085, args.contact_z]),
                      (7.5, [args.pull_x, -.085, args.contact_z]),
                      (9., [args.pull_x, -.085, args.contact_z])]
    left_foot_start = data.body("left_ankle_roll_link").xpos.copy()
    foot_keys = [(0., left_foot_start), (1., left_foot_start),
                 (2.5, left_foot_start + [args.foot_shift_x, 0., 0.]),
                 (8., left_foot_start + [args.foot_shift_x, 0., 0.])]
    key_times = np.arange(0., args.duration + 1e-9, .1)
    solved = []
    q_previous = q_start.copy()
    for t in key_times:
        q = q_previous.copy()
        if t > 0:
            foot_target = interp_target(t, foot_keys)
            limits = model.jnt_range[1:7]
            def leg_residual(leg):
                data.qpos[7:] = q
                data.qpos[7:13] = leg
                mujoco.mj_forward(model, data)
                return np.r_[15. * (data.body("left_ankle_roll_link").xpos - foot_target),
                             .05 * (leg - q_previous[:6]),
                             .02 * (leg - q_start[:6])]
            result = least_squares(leg_residual, q[:6],
                                   bounds=(limits[:, 0] + .002, limits[:, 1] - .002),
                                   max_nfev=60)
            q[:6] = result.x
            for side, span, target in (("left", (15, 22), interp_target(t, left_keys)),
                                       ("right", (22, 29), interp_target(t, right_keys))):
                a, b = span
                limits = model.jnt_range[a + 1:b + 1]
                def residual(arm):
                    data.qpos[7:] = q
                    data.qpos[7 + a:7 + b] = arm
                    mujoco.mj_forward(model, data)
                    return np.r_[15. * (palm(side) - target),
                                 .07 * (arm - q_previous[a:b]),
                                 .02 * (arm - q_start[a:b])]
                result = least_squares(residual, q[a:b],
                                       bounds=(limits[:, 0] + .002, limits[:, 1] - .002),
                                       max_nfev=60)
                q[a:b] = result.x
        solved.append(q)
        q_previous = q
    solved = np.asarray(solved)
    times = np.arange(round(args.duration * 50) + 1, dtype=np.float64) / 50.
    joints = np.stack([np.interp(times, key_times, solved[:, k])
                       for k in range(solved.shape[1])], axis=1).astype(np.float32)
    body_pos = np.empty((len(times), len(names), 3), np.float32)
    body_quat = np.empty((len(times), len(names), 4), np.float32)
    for i, q in enumerate(joints):
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat
        data.qpos[7:] = q
        mujoco.mj_forward(model, data)
        for k, name in enumerate(names):
            if name.endswith("_hand_mimic"):
                side = name.split("_")[0]
                body = data.body(side + "_wrist_yaw_link")
                body_pos[i, k] = palm(side)
                body_quat[i, k] = body.xquat
            elif name == "head_mimic":
                body = data.body("torso_link")
                body_pos[i, k] = body.xpos + body.xmat.reshape(3, 3) @ np.array([.01, 0., .41])
                body_quat[i, k] = body.xquat
            else:
                body = data.body(name)
                body_pos[i, k] = body.xpos
                body_quat[i, k] = body.xquat
    inv_rot = Rotation.from_quat(root_quat[[1, 2, 3, 0]]).inv().as_matrix()
    body_pos_b = np.einsum("ij,tbj->tbi", inv_rot, body_pos - root_pos[None, None]).astype(np.float32)
    n = len(times)
    output = MotionData(
        motion_id=torch.zeros(n, dtype=torch.int32),
        step=torch.arange(n, dtype=torch.int32),
        root_pos_w=torch.from_numpy(np.repeat(root_pos[None], n, axis=0).copy()),
        root_quat_w=torch.from_numpy(np.repeat(root_quat[None], n, axis=0).copy()),
        root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3),
        joint_pos=torch.from_numpy(joints),
        joint_vel=torch.from_numpy(np.gradient(joints, .02, axis=0).astype(np.float32)),
        body_pos_w=torch.from_numpy(body_pos),
        body_pos_b=torch.from_numpy(body_pos_b),
        body_quat_w=torch.from_numpy(body_quat), batch_size=[n])
    args.dest.mkdir(parents=True, exist_ok=True)
    output.memmap(str(args.dest))
    meta = dict(meta, starts=[0], ends=[n], info={})
    (args.dest / "meta_motion.json").write_text(json.dumps(meta))
    (args.dest / "source_files.json").write_text(json.dumps(["bimanual arm IK from validated first-task initial posture"]))
    for t in (0., 2.5, 3.5, 6.5, 8.):
        i = int(t * 50)
        print(f"t={t:.1f} left={body_pos[i,names.index('left_hand_mimic')]} "
              f"right={body_pos[i,names.index('right_hand_mimic')]}")
    print(f"FOOT_PIN_BIMANUAL_REFERENCE_READY frames={n} dest={args.dest}")


if __name__ == "__main__":
    main()
