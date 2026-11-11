"""Sample many free-base policy rollouts and retain real physics trajectories."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="G1/G1_foot_pin_bimanual_v0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-top", type=int, default=6)
    parser.add_argument("--mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--save-bc", action="store_true",
                        help="Save policy observations and reference joint actions for optional arm imitation")
    parser.add_argument("--right-head", type=Path, default=None)
    parser.add_argument("--left-head", type=Path, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[f"task={args.task}", "+exp=finetune"])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = args.num_envs
    cfg.task.max_episode_length = args.steps + 20
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.algo.symmetry_augmentation = False
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.wandb.mode = "disabled"
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
        actor = agent.get_rollout_policy("eval")
        heads = []
        for path in (args.right_head, args.left_head):
            if path is not None:
                from scripts.foot_pin_bimanual.train_arm_head import ArmHead
                state = torch.load(path, map_location=base.device, weights_only=False)
                head = ArmHead(state["observation_dim"]).to(base.device)
                head.load_state_dict(state["model"])
                head.eval()
                heads.append((head, state["joint_ids"]))
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        env.reset()
        base.episode_length_buf.zero_()
        origins = base.scene.env_origins.detach().cpu().numpy()
        base_object, part_object = base.scene["pin_base"], base.scene["extract_part"]
        hands = base.scene["ee_contact_forces"]
        feet = base.scene["contact_forces"]
        fields = ("root_pos", "root_quat", "joint_pos", "base_pos", "base_quat",
                  "part_pos", "part_quat", "left_hand_pos", "right_hand_pos",
                  "left_foot_pos", "right_foot_pos", "left_hand_force",
                  "right_hand_force", "left_foot_force", "right_foot_force", "action")
        records = {key: [] for key in fields}
        bc_records = {key: [] for key in ("observation", "target_action", "old_loc", "root_z", "joint_pos")}
        def append(key, value):
            records[key].append(value.detach().cpu().numpy().astype(np.float32, copy=True))
        mode = ExplorationType.RANDOM if args.mode == "sample" else ExplorationType.MODE
        for i in range(args.steps):
            raw = TensorDict({}, batch_size=[args.num_envs], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            if args.save_bc:
                bc_records["observation"].append(carry["policy"].detach().cpu().numpy().astype(np.float32, copy=True))
                manager = base.action_manager
                assert [cmd.dataset.joint_names[i] for i in cmd.joint_idx_motion.tolist()] == manager.joint_names
                target_action = ((cmd.target_joint_pos_obs()
                                  - manager.default_joint_pos[:, manager.joint_ids])
                                 / manager.action_scaling)
                bc_records["target_action"].append(target_action.detach().cpu().numpy().astype(np.float32, copy=True))
            carry["is_init"] = torch.full((args.num_envs, 1), i == 0,
                                           device=base.device, dtype=torch.bool)
            with torch.inference_mode(), set_exploration_type(mode):
                actor(carry)
                if args.save_bc:
                    bc_records["old_loc"].append(carry["loc"].detach().cpu().numpy().astype(np.float32, copy=True))
                for head, joint_ids in heads:
                    carry["action"][:, joint_ids] = head(carry["policy"])
                base.step(carry)
            robot = cmd.asset.data
            if args.save_bc:
                bc_records["root_z"].append(robot.root_pos_w[:, 2].detach().cpu().numpy().astype(np.float32, copy=True))
                bc_records["joint_pos"].append(robot.joint_pos.detach().cpu().numpy().astype(np.float32, copy=True))
            data = dict(root_pos=robot.root_pos_w, root_quat=robot.root_quat_w,
                        joint_pos=robot.joint_pos,
                        base_pos=base_object.data.root_pos_w, base_quat=base_object.data.root_quat_w,
                        part_pos=part_object.data.root_pos_w, part_quat=part_object.data.root_quat_w,
                        left_hand_pos=robot.body_pos_w[:, cmd.hand_asset],
                        right_hand_pos=robot.body_pos_w[:, cmd.right_hand_asset],
                        left_foot_pos=robot.body_pos_w[:, cmd.left_asset],
                        right_foot_pos=robot.body_pos_w[:, cmd.right_asset],
                        left_hand_force=hands.data.net_forces_w[:, cmd.hand_sensor_id],
                        right_hand_force=hands.data.net_forces_w[:, cmd.right_hand_sensor_id],
                        left_foot_force=feet.data.net_forces_w[:, cmd.left_foot_sensor_id],
                        right_foot_force=feet.data.net_forces_w[:, cmd.right_foot_sensor_id],
                        action=carry["action"])
            for key, value in data.items():
                append(key, value)
            if i % 100 == 0:
                print(f"BATCH step={i} root_z_min={robot.root_pos_w[:,2].min():.3f}", flush=True)
        arrays = {key: np.stack(value) for key, value in records.items()}
        if args.save_bc:
            np.savez_compressed(args.output / "arm_bc_dataset.npz",
                                **{key: np.stack(value) for key, value in bc_records.items()},
                                default_joint_pos=base.action_manager.default_joint_pos[0, base.action_manager.joint_ids].detach().cpu().numpy(),
                                action_scaling=base.action_manager.action_scaling.detach().cpu().numpy())
        for key in ("root_pos", "base_pos", "part_pos", "left_hand_pos",
                    "right_hand_pos", "left_foot_pos", "right_foot_pos"):
            arrays[key] -= origins[None]
        relative = arrays["part_pos"][:, :, 0] - arrays["base_pos"][:, :, 0]
        base_drift = np.linalg.norm(arrays["base_pos"] - arrays["base_pos"][0], axis=-1)
        force = {key: np.linalg.norm(arrays[key], axis=-1) for key in
                 ("left_hand_force", "right_hand_force", "left_foot_force", "right_foot_force")}
        dt = float(base.step_dt)
        left_part = ((force["left_hand_force"] > 10) &
                     (np.linalg.norm(arrays["left_hand_pos"] - arrays["part_pos"], axis=-1) < .17))
        right_part = ((force["right_hand_force"] > 10) &
                      (np.linalg.norm(arrays["right_hand_pos"] - arrays["part_pos"], axis=-1) < .17))
        foot_front = ((force["left_foot_force"] > 20) &
                      (np.abs(arrays["left_foot_pos"][:, :, 0] -
                              (arrays["base_pos"][:, :, 0] - cmd.base_half_x)) < .13) &
                      (np.abs(arrays["left_foot_pos"][:, :, 1] -
                              arrays["base_pos"][:, :, 1]) < .18) &
                      (arrays["left_foot_pos"][:, :, 2] > .1))
        rows = []
        for j in range(args.num_envs):
            row = dict(env_id=j, checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                       mode=args.mode, duration_s=float((args.steps - 1) * dt),
                       right_head=str(args.right_head.resolve()) if args.right_head else None,
                       left_head=str(args.left_head.resolve()) if args.left_head else None,
                       extraction_m=float(relative[0, j] - relative[-1, j]),
                       maximum_extraction_m=float(relative[0, j] - relative[:, j].min()),
                       base_final_drift_m=float(base_drift[-1, j]),
                       base_max_drift_m=float(base_drift[:, j].max()),
                       root_min_height_m=float(arrays["root_pos"][:, j, 2].min()),
                       root_final_height_m=float(arrays["root_pos"][-1, j, 2]),
                       left_foot_contact_s=float(np.count_nonzero(force["left_foot_force"][:, j] > 20) * dt),
                       right_foot_contact_s=float(np.count_nonzero(force["right_foot_force"][:, j] > 100) * dt),
                       left_hand_contact_s=float(np.count_nonzero(force["left_hand_force"][:, j] > 10) * dt),
                       right_hand_contact_s=float(np.count_nonzero(force["right_hand_force"][:, j] > 10) * dt),
                       foot_base_contact_s=float(np.count_nonzero(foot_front[:, j]) * dt),
                       both_hands_part_contact_s=float(np.count_nonzero((left_part & right_part)[:, j]) * dt),
                       three_way_contact_s=float(np.count_nonzero((left_part & right_part & foot_front)[:, j]) * dt))
            row["task_success"] = bool(row["extraction_m"] >= .15 and
                row["base_final_drift_m"] < .05 and row["root_min_height_m"] > .55 and
                row["right_foot_contact_s"] > .5 and
                row["three_way_contact_s"] > .5)
            rows.append(row)
        rows.sort(key=lambda row: (row["task_success"],
                                   row["root_min_height_m"] > .55,
                                   row["three_way_contact_s"],
                                   row["maximum_extraction_m"]), reverse=True)
        (args.output / "batch_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        for rank, row in enumerate(rows[:args.save_top]):
            j = row["env_id"]
            trajectory = {key: value[:, j] for key, value in arrays.items()}
            trajectory["time"] = np.arange(args.steps, dtype=np.float32) * dt
            trajectory["joint_names"] = np.asarray(cmd.asset.joint_names)
            folder = args.output / f"candidate_{rank:02d}_env_{j:03d}"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / "policy_rollout.npz", **trajectory)
            (folder / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"BATCH_RESULT success={sum(r['task_success'] for r in rows)}/{args.num_envs} "
              f"best={json.dumps(rows[0])}", flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
