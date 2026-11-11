"""Evaluate many sampled trained-policy rollouts in one Isaac simulation.

Every saved candidate contains the actual articulated robot and object states
from physics; the policy chooses all joint actions. Env origins are subtracted
only when exporting the independent per-environment trajectories.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")

import hydra
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
import active_adaptation.learning


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="G1/G1_table_retrieval_standing_tall_rl")
    p.add_argument("--dataset", default="table_retrieval_standing_prelude_liftfirst/train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=510)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-top", type=int, default=8)
    p.add_argument("--mode", choices=("sample", "mean"), default="sample")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[f"task={args.task}", "+exp=finetune"])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = args.num_envs
    cfg.task.max_episode_length = args.steps + 20
    cfg.task.randomize_episode_length = False
    cfg.task.termination = {}
    cfg.task.command.dataset.mem_paths = [args.dataset]
    cfg.task.command.anchor_init_prob = 1.0
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    cfg.algo.symmetry_augmentation = False
    cfg.vecnorm = "eval"
    cfg.wandb.mode = "disabled"
    cfg.seed = args.seed
    cfg.app = {"headless": True, "enable_cameras": False}
    torch.manual_seed(args.seed)
    app = AppLauncher(headless=True, device="cuda:0").app
    exit_code = 0
    try:
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval()
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.0
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        env.reset()
        base.episode_length_buf.zero_()
        origins = base.scene.env_origins.detach().cpu().numpy()
        obj = base.scene["target_object"]
        hand_sensor = base.scene["ee_contact_forces"]
        hand_id = hand_sensor.find_bodies("left_hand_mimic")[0][0]
        records = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "object_pos",
                                   "object_quat", "left_hand_pos", "left_foot_pos",
                                   "right_foot_pos", "hand_force", "right_foot_force", "action")}
        def append(key, tensor):
            records[key].append(tensor.detach().cpu().numpy().astype(np.float32, copy=True))
        exploration = ExplorationType.RANDOM if args.mode == "sample" else ExplorationType.MODE
        for i in range(args.steps):
            raw = TensorDict({}, batch_size=[args.num_envs], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.full((args.num_envs, 1), i == 0,
                                           device=base.device, dtype=torch.bool)
            with torch.inference_mode(), set_exploration_type(exploration):
                actor(carry)
                base.step(carry)
            robot = cmd.asset.data
            append("root_pos", robot.root_pos_w)
            append("root_quat", robot.root_quat_w)
            append("joint_pos", robot.joint_pos)
            append("object_pos", obj.data.root_pos_w)
            append("object_quat", obj.data.root_quat_w)
            append("left_hand_pos", robot.body_pos_w[:, cmd.hand_asset])
            append("left_foot_pos", robot.body_pos_w[:, cmd.left_asset])
            append("right_foot_pos", robot.body_pos_w[:, cmd.right_asset])
            append("hand_force", hand_sensor.data.net_forces_w[:, hand_id])
            append("right_foot_force", base.scene["contact_forces"].data.net_forces_w[:, cmd.right_foot_sensor_id])
            append("action", carry["action"])
            if i % 100 == 0:
                print(f"BATCH step={i} object_x_range={records['object_pos'][-1][:,0].min():.3f},{records['object_pos'][-1][:,0].max():.3f}", flush=True)
        arrays = {key: np.stack(values, axis=0) for key, values in records.items()}
        for key in ("root_pos", "object_pos", "left_hand_pos", "left_foot_pos", "right_foot_pos"):
            arrays[key] -= origins[None, :, :]
        boundary = float(getattr(cfg.task, "retrieval_boundary_x", .28))
        half_x = float(getattr(cfg.task, "object_half_x", .045))
        dt = float(base.step_dt)
        x0 = arrays["object_pos"][0, :, 0]
        x1 = arrays["object_pos"][-1, :, 0]
        root_z = arrays["root_pos"][:, :, 2]
        hand_mag = np.linalg.norm(arrays["hand_force"], axis=-1)
        right_mag = np.linalg.norm(arrays["right_foot_force"], axis=-1)
        rows = []
        for j in range(args.num_envs):
            hand_s = float(np.count_nonzero(hand_mag[:, j] > 10) * dt)
            stance_s = float(np.count_nonzero(right_mag[:, j] > 100) * dt)
            row = dict(env_id=j, checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                       mode=args.mode, duration_s=(args.steps - 1) * dt,
                       object_initial_x_m=float(x0[j]), object_final_x_m=float(x1[j]),
                       object_outward_displacement_m=float(x0[j]-x1[j]),
                       object_outside_table=bool(x1[j]+half_x < boundary),
                       root_min_height_m=float(root_z[:, j].min()),
                       root_final_height_m=float(root_z[-1, j]),
                       hand_support_duration_s=hand_s,
                       right_stance_duration_s=stance_s)
            row["task_success"] = bool(row["object_outside_table"]
                                       and row["root_min_height_m"] > .55
                                       and hand_s > .5 and stance_s > .5)
            rows.append(row)
        rows.sort(key=lambda r: (r["task_success"], r["root_min_height_m"] > .55,
                                 r["object_outward_displacement_m"]), reverse=True)
        (args.output / "batch_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        selected = rows[:args.save_top]
        time = np.arange(args.steps, dtype=np.float32) * dt
        for rank, row in enumerate(selected):
            j = row["env_id"]
            out = {key: arr[:, j] for key, arr in arrays.items()}
            out["time"] = time
            out["joint_names"] = np.asarray(cmd.asset.joint_names)
            folder = args.output / f"candidate_{rank:02d}_env_{j:03d}"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / "policy_rollout.npz", **out)
            (folder / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
        success_count = sum(r["task_success"] for r in rows)
        print(f"BATCH_RESULT success={success_count}/{args.num_envs} best={json.dumps(rows[0])}", flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
