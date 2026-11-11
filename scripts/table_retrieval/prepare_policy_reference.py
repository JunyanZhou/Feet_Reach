"""Convert the kinematic table demo to a 50 Hz motion reference for policy training.

The source is used only as a curriculum reference. Evaluation always runs a
free-base articulated G1 and integrates the object's rigid-body dynamics.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from active_adaptation.utils.motion import MotionDataset


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/table_retrieval/motion.npz"
RAW = ROOT / "artifacts/table_policy/reference_50hz.npz"
DEST = ROOT / "dataset/table_retrieval/train"


def interp_quat(times, source_times, quats_wxyz):
    xyzw = quats_wxyz[..., [1, 2, 3, 0]]
    rotations = Rotation.from_quat(xyzw.reshape(-1, 4))
    n = xyzw.shape[1] if xyzw.ndim == 3 else 1
    rotations = rotations.as_quat().reshape(len(source_times), n, 4)
    out = np.empty((len(times), n, 4))
    for i in range(n):
        out[:, i] = Slerp(source_times, Rotation.from_quat(rotations[:, i]))(times).as_quat()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.raw = args.raw.resolve()
    args.dest = args.dest.resolve()
    source = dict(np.load(args.source, allow_pickle=False))
    # The first 16 s cover standing, table support, foot retrieval and foot landing.
    t = np.arange(801, dtype=np.float64) / 50.0
    ts = source["time"]
    def linear(values):
        flat = values.reshape(len(ts), -1)
        return np.stack([np.interp(t, ts, flat[:, i]) for i in range(flat.shape[1])], -1).reshape((len(t),) + values.shape[1:])

    q = linear(source["qpos"])
    body_pos = linear(source["body_pos"])
    root_quat = interp_quat(t, ts, source["qpos"][:, None, 3:7])[:, 0]
    body_quat = interp_quat(t, ts, source["body_quat_wxyz"])
    names = list(source["body_names"])
    # The MuJoCo asset fuses the rubber hands into the wrist bodies; the Isaac
    # asset exposes fixed hand_mimic bodies at these offsets.
    for side, y in (("left", 0.000016093254), ("right", -0.000016093254)):
        idx = names.index(f"{side}_wrist_yaw_link")
        offset = np.array([0.116, y, 0.0000046119094])
        hand = body_pos[:, idx] + Rotation.from_quat(body_quat[:, idx]).apply(offset)
        names.append(f"{side}_hand_mimic")
        body_pos = np.concatenate((body_pos, hand[:, None]), axis=1)
        body_quat = np.concatenate((body_quat, body_quat[:, idx:idx+1]), axis=1)
    torso = names.index("torso_link")
    head = body_pos[:, torso] + Rotation.from_quat(body_quat[:, torso]).apply([0.01, 0.0, 0.41])
    names.append("head_mimic")
    body_pos = np.concatenate((body_pos, head[:, None]), axis=1)
    body_quat = np.concatenate((body_quat, body_quat[:, torso:torso+1]), axis=1)
    q[:, 3:7] = root_quat[:, [3, 0, 1, 2]]
    rot = Rotation.from_quat(root_quat)
    root_inv = rot.inv().as_matrix()
    local_pos = np.einsum("tij,tbj->tbi", root_inv, body_pos - q[:, None, :3])
    world_body_rot = Rotation.from_quat(body_quat.reshape(-1, 4)).as_matrix().reshape(len(t), -1, 3, 3)
    local_rot = np.einsum("tij,tbjk->tbik", root_inv, world_body_rot)
    local_quat = Rotation.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(len(t), -1, 4)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.raw, fps=50, root_pos=q[:, :3].astype(np.float32),
                        root_rot=root_quat.astype(np.float32),
                        dof_pos=q[:, 7:].astype(np.float32),
                        local_body_pos=local_pos.astype(np.float32),
                        local_body_rot=local_quat.astype(np.float32),
                        joint_names=source["joint_names"], body_names=np.array(names))
    MotionDataset.create_from_path(str(args.raw), target_fps=50, mem_path=str(args.dest),
                                   segment_len=2000, storage_float_dtype=torch.float32)
    ds = MotionDataset.create_from_path_lazy(str(args.dest))
    assert ds.num_steps == len(t)
    print(f"TABLE_REFERENCE_READY frames={ds.num_steps} bodies={len(ds.body_names)}")


if __name__ == "__main__":
    main()
