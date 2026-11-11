"""Prepend a double-stance hand-supported phase to a trained foot motion.

The reference is for curriculum training. It is not a physics rollout.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/table_retrieval_trained_motion/train")
    parser.add_argument("--dest", type=Path, default=ROOT / "dataset/table_retrieval_standing_prelude/train")
    parser.add_argument("--hold", type=int, default=50)
    parser.add_argument("--lift", type=int, default=100)
    args = parser.parse_args()
    source = MotionData.load_memmap(str(args.source))
    source_meta = json.loads((args.source / "meta_motion.json").read_text())
    n_pre = args.hold + args.lift
    n_src = len(source)
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    names = source_meta["body_names"]
    q_ref = source.joint_pos[0].numpy().copy()
    q_stand = q_ref.copy()
    q_stand[:6] = q_ref[6:12]
    # Rearward stance avoids spawning the left foot inside the target block.
    q_stand[0] = 0.1
    root_pos = source.root_pos_w[0].numpy()
    root_quat = source.root_quat_w[0].numpy()
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:] = q_stand
    mujoco.mj_forward(model, data)
    ankle = data.body("left_ankle_roll_link")
    lift_target = ankle.xpos.copy()
    lift_target[2] = .22
    stand_rot = ankle.xmat.reshape(3, 3).copy()
    limits = model.jnt_range[1:7]
    def mid_residual(left_leg):
        data.qpos[7:13] = left_leg
        mujoco.mj_forward(model, data)
        body = data.body("left_ankle_roll_link")
        orient = Rotation.from_matrix(body.xmat.reshape(3, 3) @ stand_rot.T).as_rotvec()
        return np.r_[20. * (body.xpos - lift_target), 2. * orient,
                     .1 * (left_leg - q_stand[:6])]
    middle_leg = least_squares(mid_residual, q_stand[:6],
                               bounds=(limits[:, 0] + .002, limits[:, 1] - .002),
                               max_nfev=100).x
    q_middle = q_stand.copy()
    q_middle[:6] = middle_leg
    body_pos = np.empty((n_pre, len(names), 3), dtype=np.float32)
    body_quat = np.empty((n_pre, len(names), 4), dtype=np.float32)
    q_pre = np.empty((n_pre, len(q_ref)), dtype=np.float32)
    for i in range(n_pre):
        if i < args.hold:
            q_pre[i] = q_stand
        elif i < args.hold + args.lift // 2:
            a = smooth((i - args.hold) / (args.lift // 2))
            q_pre[i] = (1. - a) * q_stand + a * q_middle
        else:
            a = smooth((i - args.hold - args.lift // 2) / (args.lift - args.lift // 2))
            q_pre[i] = (1. - a) * q_middle + a * q_ref
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat
        data.qpos[7:] = q_pre[i]
        mujoco.mj_forward(model, data)
        for k, name in enumerate(names):
            if name.endswith("_hand_mimic"):
                side = name.split("_")[0]
                body = data.body(side + "_wrist_yaw_link")
                rot = body.xmat.reshape(3, 3)
                body_pos[i, k] = body.xpos + rot @ np.array([.116, 0., 0.])
                body_quat[i, k] = body.xquat
            elif name == "head_mimic":
                body = data.body("torso_link")
                body_pos[i, k] = body.xpos + body.xmat.reshape(3, 3) @ np.array([.01, 0., .41])
                body_quat[i, k] = body.xquat
            else:
                body = data.body(name)
                body_pos[i, k] = body.xpos
                body_quat[i, k] = body.xquat
    root_pre = np.repeat(root_pos[None], n_pre, axis=0)
    quat_pre = np.repeat(root_quat[None], n_pre, axis=0)
    rot_inv = Rotation.from_quat(root_quat[[1, 2, 3, 0]]).inv().as_matrix()
    body_pos_b = np.einsum("ij,tbj->tbi", rot_inv, body_pos - root_pre[:, None]).astype(np.float32)
    q_all = np.concatenate((q_pre, source.joint_pos.numpy()), axis=0)
    joint_vel = np.gradient(q_all, .02, axis=0).astype(np.float32)
    def cat(name, first):
        return torch.as_tensor(np.concatenate((first, getattr(source, name).numpy()), axis=0).copy())
    fields = dict(
        motion_id=torch.zeros(n_pre + n_src, dtype=torch.int32),
        step=torch.arange(n_pre + n_src, dtype=torch.int32),
        root_pos_w=cat("root_pos_w", root_pre),
        root_quat_w=cat("root_quat_w", quat_pre),
        root_lin_vel_w=cat("root_lin_vel_w", np.zeros((n_pre, 3), np.float32)),
        root_ang_vel_w=cat("root_ang_vel_w", np.zeros((n_pre, 3), np.float32)),
        joint_pos=torch.as_tensor(q_all.copy()),
        joint_vel=torch.as_tensor(joint_vel),
        body_pos_w=cat("body_pos_w", body_pos),
        body_pos_b=cat("body_pos_b", body_pos_b),
        body_quat_w=cat("body_quat_w", body_quat),
    )
    output = MotionData(**fields, batch_size=[n_pre + n_src])
    args.dest.mkdir(parents=True, exist_ok=True)
    output.memmap(str(args.dest))
    meta = dict(source_meta, starts=[0], ends=[n_pre + n_src], info={})
    (args.dest / "meta_motion.json").write_text(json.dumps(meta))
    (args.dest / "source_files.json").write_text(json.dumps(["original trained motion with double-stance prelude"]))
    print(f"STANDING_PRELUDE_READY frames={n_pre+n_src} stand={args.hold} lift={args.lift} dest={args.dest}")


if __name__ == "__main__":
    main()
