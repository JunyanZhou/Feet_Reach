"""Physics rollout for the foot-pin, two-hand extraction prototype."""
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
import active_adaptation.learning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/foot_pin_bimanual/right_hand_warmstart.pt")
    parser.add_argument("--task", default="G1/G1_foot_pin_bimanual_v0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct-reference", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--arm-checkpoint", type=Path, default=None,
                        help="Diagnostic: retain base policy lower body and use a second trained policy for arms")
    parser.add_argument("--right-head", type=Path, default=None)
    parser.add_argument("--left-head", type=Path, default=None)
    parser.add_argument("--foot-extra", type=float, default=0.,
                        help="Additional forward Cartesian pin-foot target after 1 s")
    parser.add_argument("--left-head-after", type=float, default=0.,
                        help="Diagnostic policy-head transfer start time")
    parser.add_argument("--left-head-ramp", type=float, default=0.,
                        help="Seconds to interpolate from baseline arm action to trained head")
    parser.add_argument("--right-goal-extra-x", type=float, default=0.,
                        help="Shift the trained right-hand Cartesian goal toward the part")
    args = parser.parse_args()
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
        assert [cmd.dataset.joint_names[i] for i in cmd.joint_idx_motion.tolist()] == base.action_manager.joint_names
        actor = agent.get_rollout_policy("eval")
        heads = []
        for head_path in (args.right_head, args.left_head):
            if head_path is not None:
                from scripts.foot_pin_bimanual.train_arm_head import ArmHead
                state = torch.load(head_path, map_location=base.device, weights_only=False)
                head = ArmHead(state["observation_dim"]).to(base.device)
                head.load_state_dict(state["model"])
                head.eval()
                heads.append((head, state["joint_ids"]))
        arm_actor = None
        arm_ids = None
        if args.arm_checkpoint is not None:
            import copy
            arm_agent = copy.deepcopy(agent)
            arm_state = torch.load(args.arm_checkpoint, map_location=base.device,
                                   weights_only=False)["policy"]
            arm_agent.load_state_dict(arm_state)
            arm_agent.eval()
            arm_actor = arm_agent.get_rollout_policy("eval")
            arm_ids = [index for index, name in enumerate(agent.joint_names)
                       if any(word in name for word in ("shoulder", "elbow", "wrist"))]
            assert len(arm_ids) == 14
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        env.reset()
        base.episode_length_buf.zero_()
        print(f"INIT root={cmd.asset.data.root_pos_w[0].tolist()}", flush=True)
        base_object = base.scene["pin_base"]
        part_object = base.scene["extract_part"]
        hand_sensor = base.scene["ee_contact_forces"]
        foot_sensor = base.scene["contact_forces"]
        records = {k: [] for k in ("time", "root_pos", "root_quat", "joint_pos", "base_pos",
                   "base_quat", "part_pos", "part_quat", "left_hand_pos", "right_hand_pos",
                   "left_foot_pos", "right_foot_pos", "left_hand_force", "right_hand_force",
                   "left_foot_force", "right_foot_force", "action")}

        def append(name, tensor):
            records[name].append(np.asarray(tensor.detach().cpu(), dtype=np.float32).copy())

        is_init = True
        for i in range(args.steps):
            if args.foot_extra:
                phase = i * float(base.step_dt)
                u = min(1., max(0., (phase - 1.) / 1.5))
                fraction = u*u*(3.-2.*u)
                foot_target = cmd._reference_world("left_foot").clone()
                foot_target[:, 0] += args.foot_extra * fraction
                cmd.set_goal_world("left_foot", foot_target)
            if args.right_goal_extra_x:
                phase = i * float(base.step_dt)
                u = min(1., max(0., (phase - 1.) / 1.5))
                fraction = u*u*(3.-2.*u)
                hand_target = cmd._reference_world("right_hand").clone()
                hand_target[:, 0] += args.right_goal_extra_x * fraction
                cmd.set_goal_world("right_hand", hand_target)
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.tensor([[is_init]], device=base.device, dtype=torch.bool)
            mode = ExplorationType.RANDOM if args.stochastic else ExplorationType.MODE
            with torch.inference_mode(), set_exploration_type(mode):
                if args.direct_reference:
                    manager = base.action_manager
                    assert [cmd.dataset.joint_names[i] for i in cmd.joint_idx_motion.tolist()] == manager.joint_names
                    carry["action"] = ((cmd.target_joint_pos_obs()
                        - manager.default_joint_pos[:, manager.joint_ids])
                        / manager.action_scaling)
                else:
                    actor(carry)
                    if arm_actor is not None:
                        arm_carry = carry.select("policy", "is_init").clone()
                        arm_actor(arm_carry)
                        carry["action"][:, arm_ids] = arm_carry["action"][:, arm_ids]
                    for head, joint_ids in heads:
                        proposed = head(carry["policy"])
                        if joint_ids == [11, 15, 19, 21, 23, 25, 27]:
                            phase = i * float(base.step_dt)
                            if args.left_head_ramp > 0:
                                u = min(1., max(0., (phase - args.left_head_after) /
                                               args.left_head_ramp))
                                weight = u*u*(3.-2.*u)
                            else:
                                weight = float(phase >= args.left_head_after)
                            baseline = carry["action"][:, joint_ids].clamp(-10., 10.)
                            proposed = (1.-weight)*baseline + weight*proposed
                        carry["action"][:, joint_ids] = proposed
                base.step(carry)
            is_init = False
            robot = cmd.asset.data
            hand_forces = hand_sensor.data.net_forces_w
            foot_forces = foot_sensor.data.net_forces_w
            records["time"].append(i * float(base.step_dt))
            for key, value in (("root_pos", robot.root_pos_w[0]),
                               ("root_quat", robot.root_quat_w[0]),
                               ("joint_pos", robot.joint_pos[0]),
                               ("base_pos", base_object.data.root_pos_w[0]),
                               ("base_quat", base_object.data.root_quat_w[0]),
                               ("part_pos", part_object.data.root_pos_w[0]),
                               ("part_quat", part_object.data.root_quat_w[0]),
                               ("left_hand_pos", robot.body_pos_w[0, cmd.hand_asset]),
                               ("right_hand_pos", robot.body_pos_w[0, cmd.right_hand_asset]),
                               ("left_foot_pos", robot.body_pos_w[0, cmd.left_asset]),
                               ("right_foot_pos", robot.body_pos_w[0, cmd.right_asset]),
                               ("left_hand_force", hand_forces[0, cmd.hand_sensor_id]),
                               ("right_hand_force", hand_forces[0, cmd.right_hand_sensor_id]),
                               ("left_foot_force", foot_forces[0, cmd.left_foot_sensor_id]),
                               ("right_foot_force", foot_forces[0, cmd.right_foot_sensor_id]),
                               ("action", carry["action"][0])):
                append(key, value)
            if i % 50 == 0:
                print(f"step={i} root_z={records['root_pos'][-1][2]:.3f} "
                      f"base_x={records['base_pos'][-1][0]:.3f} "
                      f"part_x={records['part_pos'][-1][0]:.3f}", flush=True)
            if records["root_pos"][-1][2] < .38:
                break
        out = {name: np.asarray(values) for name, values in records.items()}
        out["joint_names"] = np.asarray(cmd.asset.joint_names)
        np.savez_compressed(args.output / "rollout.npz", **out)
        relative = out["part_pos"][:, 0] - out["base_pos"][:, 0]
        base_drift = np.linalg.norm(out["base_pos"] - out["base_pos"][0], axis=1)
        force = lambda key: np.linalg.norm(out[key], axis=1)
        left_part = ((force("left_hand_force") > 10) &
                     (np.linalg.norm(out["left_hand_pos"] - out["part_pos"], axis=1) < .17))
        right_part = ((force("right_hand_force") > 10) &
                      (np.linalg.norm(out["right_hand_pos"] - out["part_pos"], axis=1) < .17))
        foot_front = ((force("left_foot_force") > 20) &
                      (np.abs(out["left_foot_pos"][:, 0] - (out["base_pos"][:, 0] - cmd.base_half_x)) < .13) &
                      (np.abs(out["left_foot_pos"][:, 1] - out["base_pos"][:, 1]) < .18) &
                      (out["left_foot_pos"][:, 2] > .1))
        metrics = dict(control="direct_reference" if args.direct_reference else
                       ("trained_policy_with_arm_heads" if heads else
                        ("two_trained_policy_heads" if arm_actor is not None else
                         ("policy_stochastic" if args.stochastic else "policy_mode"))),
                       checkpoint=str(args.checkpoint.resolve()), steps=len(out["time"]),
                       arm_checkpoint=str(args.arm_checkpoint.resolve()) if args.arm_checkpoint else None,
                       right_head=str(args.right_head.resolve()) if args.right_head else None,
                       left_head=str(args.left_head.resolve()) if args.left_head else None,
                       foot_extra_m=args.foot_extra,
                       left_head_after_s=args.left_head_after,
                       left_head_ramp_s=args.left_head_ramp,
                       right_goal_extra_x_m=args.right_goal_extra_x,
                       duration_s=float(out["time"][-1]),
                       extraction_m=float(relative[0] - relative[-1]),
                       maximum_extraction_m=float(relative[0] - relative.min()),
                       base_final_drift_m=float(base_drift[-1]),
                       base_max_drift_m=float(base_drift.max()),
                       root_min_height_m=float(out["root_pos"][:, 2].min()),
                       left_foot_contact_s=float(np.count_nonzero(force("left_foot_force") > 20) * base.step_dt),
                       right_foot_contact_s=float(np.count_nonzero(force("right_foot_force") > 100) * base.step_dt),
                       left_hand_contact_s=float(np.count_nonzero(force("left_hand_force") > 10) * base.step_dt),
                       right_hand_contact_s=float(np.count_nonzero(force("right_hand_force") > 10) * base.step_dt),
                       foot_base_contact_s=float(np.count_nonzero(foot_front) * base.step_dt),
                       both_hands_part_contact_s=float(np.count_nonzero(left_part & right_part) * base.step_dt),
                       three_way_contact_s=float(np.count_nonzero(left_part & right_part & foot_front) * base.step_dt))
        metrics["task_success"] = bool(metrics["extraction_m"] >= .15
            and metrics["base_final_drift_m"] < .05
            and metrics["root_min_height_m"] > .55
            and metrics["right_foot_contact_s"] > .5
            and metrics["three_way_contact_s"] > .5)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
