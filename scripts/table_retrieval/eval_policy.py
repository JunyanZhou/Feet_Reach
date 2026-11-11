"""Evaluate a joint-action student on the dynamic Isaac table retrieval task.

The saved NPZ is a record of the physics rollout, suitable for independent
MuJoCo visualization. It is never a prescribed robot trajectory in evaluation.
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
import active_adaptation.learning  # register PPO Hydra config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=850)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="table_retrieval/train")
    p.add_argument("--task", default="G1/G1_table_retrieval")
    p.add_argument("--direct-reference", action="store_true", help="Diagnostic only: use IK joint targets instead of policy")
    p.add_argument("--stochastic", action="store_true", help="Sample actions from the trained policy distribution")
    p.add_argument("--action-clip", type=float, default=None, help="Optional symmetric actuator command safety bound")
    p.add_argument("--foot-retract-extra", type=float, default=0.0, help="Extra rearward Cartesian foot goal in metres during the pull phase")
    p.add_argument("--hand-preload-y", type=float, default=0.0, help="Move table hand target toward the support side during contact")
    p.add_argument("--root-shift-y", type=float, default=0.0, help="Move root target toward the support side during contact")
    p.add_argument("--object-x", type=float)
    p.add_argument("--object-y", type=float)
    p.add_argument("--object-height", type=float)
    p.add_argument("--object-mass", type=float)
    p.add_argument("--object-friction", type=float)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[f"task={args.task}", "+exp=finetune"])
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = args.steps + 20
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.task.command.anchor_init_prob = 1.0
    cfg.task.command.dataset.mem_paths = [args.dataset]
    for obj in cfg.task.objects:
        if obj.name == "target_object":
            if args.object_x is not None:
                obj.pos[0] = args.object_x
            if args.object_y is not None:
                obj.pos[1] = args.object_y
            if args.object_height is not None:
                obj.size[2] = args.object_height
                obj.pos[2] = args.object_height / 2.0 + .001
            if args.object_mass is not None:
                obj.mass = args.object_mass
            if args.object_friction is not None:
                obj.friction = args.object_friction
    cfg.task.command.foot_force.enabled = False
    cfg.algo.symmetry_augmentation = False
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
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
        print(f"INIT reference_step={cmd.t[0].item()} root={cmd.asset.data.root_pos_w[0].tolist()}", flush=True)
        object_asset = base.scene["target_object"]
        hand_sensor = base.scene["ee_contact_forces"]
        hand_id = hand_sensor.find_bodies("left_hand_mimic")[0][0]
        records = dict(time=[], root_pos=[], root_quat=[], joint_pos=[], object_pos=[],
                       object_quat=[], left_hand_pos=[], left_foot_pos=[], right_foot_pos=[],
                       hand_force=[], right_foot_force=[], action=[])
        is_init = True
        def record(action):
            robot = cmd.asset.data
            o = object_asset.data
            sensors = hand_sensor.data.net_forces_w
            foot_forces = base.scene["contact_forces"].data.net_forces_w
            def append(name, value):
                records[name].append(np.asarray(value.detach().cpu(), dtype=np.float32).copy())
            records["time"].append(float(len(records["time"]) * base.step_dt))
            append("root_pos", robot.root_pos_w[0])
            append("root_quat", robot.root_quat_w[0])
            append("joint_pos", robot.joint_pos[0])
            append("object_pos", o.root_pos_w[0])
            append("object_quat", o.root_quat_w[0])
            append("left_hand_pos", robot.body_pos_w[0, cmd.hand_asset])
            append("left_foot_pos", robot.body_pos_w[0, cmd.left_asset])
            append("right_foot_pos", robot.body_pos_w[0, cmd.right_asset])
            append("hand_force", sensors[0, hand_id] if sensors is not None else torch.zeros(3,device=base.device))
            append("right_foot_force", foot_forces[0, cmd.right_foot_sensor_id] if foot_forces is not None else torch.zeros(3,device=base.device))
            append("action", action)
        for i in range(args.steps):
            phase = i * base.step_dt
            def smooth01(x):
                u = max(0.0, min(1.0, x))
                return u * u * (3.0 - 2.0 * u)
            if args.foot_retract_extra:
                envelope = smooth01((phase - 0.8) / 1.4) * smooth01((5.4 - phase) / 1.4)
                foot_goal = cmd._reference_world("left_foot").clone()
                foot_goal[:, 0] -= args.foot_retract_extra * envelope
                cmd.set_goal_world("left_foot", foot_goal)
            if args.hand_preload_y:
                hand_goal = cmd._reference_world("left_hand").clone()
                hand_goal[:, 1] += args.hand_preload_y * smooth01((phase - 3.3) / 1.2)
                cmd.set_goal_world("left_hand", hand_goal)
            if args.root_shift_y:
                root_goal = cmd._reference_world("root").clone()
                root_goal[:, 1] += args.root_shift_y * smooth01((phase - 3.3) / 1.2)
                cmd.set_goal_world("root", root_goal)
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.tensor([[is_init]], device=base.device, dtype=torch.bool)
            mode = ExplorationType.RANDOM if args.stochastic else ExplorationType.MODE
            with torch.inference_mode(), set_exploration_type(mode):
                if args.direct_reference:
                    manager = base.action_manager
                    carry["action"] = ((cmd.target_joint_pos_obs()
                        - manager.default_joint_pos[:, manager.joint_ids])
                        / manager.action_scaling)
                else:
                    actor(carry)
                    if args.action_clip is not None:
                        carry["action"].clamp_(-args.action_clip, args.action_clip)
                base.step(carry)
            is_init = False
            record(carry["action"][0])
            if i % 100 == 0:
                print(f"EVAL step={i} object_x={records['object_pos'][-1][0]:.3f} root_z={records['root_pos'][-1][2]:.3f}", flush=True)
            if records["root_pos"][-1][2] < 0.38:
                break
        out = {k: np.asarray(v) for k, v in records.items()}
        out["joint_names"] = np.asarray(cmd.asset.joint_names)
        np.savez_compressed(args.output / "policy_rollout.npz", **out)
        x0 = float(out["object_pos"][0, 0]); x1 = float(out["object_pos"][-1, 0])
        boundary_x = float(getattr(cfg.task, "retrieval_boundary_x", 0.28))
        object_half_x = float(getattr(cfg.task, "object_half_x", 0.045))
        hand_mag = np.linalg.norm(out["hand_force"], axis=1)
        right_mag = np.linalg.norm(out["right_foot_force"], axis=1)
        dt = float(base.step_dt)
        hand_support_duration = float(np.count_nonzero(hand_mag > 10.0) * dt)
        right_stance_duration = float(np.count_nonzero(right_mag > 100.0) * dt)
        require_foot_recovery = bool(getattr(cfg.task, "require_foot_recovery", True))
        metrics = dict(checkpoint=str(args.checkpoint.resolve()), control="direct_reference" if args.direct_reference else ("policy_stochastic" if args.stochastic else "policy_mode"),
                       action_clip=args.action_clip, foot_retract_extra_m=args.foot_retract_extra,
                       hand_preload_y_m=args.hand_preload_y, root_shift_y_m=args.root_shift_y,
                       object_setup=dict(x=args.object_x, y=args.object_y,
                                         height=args.object_height, mass=args.object_mass,
                                         friction=args.object_friction),
                       steps=len(out["time"]),
                       duration_s=float(out["time"][-1]), object_initial_x_m=x0,
                       object_final_x_m=x1, object_outward_displacement_m=x0-x1,
                       object_outside_table=bool(x1 + object_half_x < boundary_x),
                       retrieval_boundary_x_m=boundary_x,
                       root_min_height_m=float(out["root_pos"][:, 2].min()),
                       hand_force_peak_N=float(hand_mag.max()),
                       hand_force_mean_N=float(hand_mag.mean()),
                       hand_support_duration_s=hand_support_duration,
                       right_foot_force_mean_N=float(right_mag.mean()),
                       right_stance_duration_s=right_stance_duration,
                       recovered_stance=bool(out["root_pos"][-1,2]>.55 and out["left_foot_pos"][-1,2]<.12))
        metrics["stable_completion"] = bool(out["root_pos"][-1,2] > .55)
        metrics["task_success"] = bool(metrics["object_outside_table"] and metrics["stable_completion"]
                                       and hand_support_duration > .5 and right_stance_duration > .5
                                       and (not require_foot_recovery or metrics["recovered_stance"]))
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        # Isaac Kit can keep background threads alive after headless evaluation.
        # Artifacts and diagnostics are already flushed; release GPU resources.
        os._exit(exit_code)


if __name__ == "__main__":
    main()
