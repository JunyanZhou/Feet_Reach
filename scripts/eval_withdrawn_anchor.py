"""Check whether a right-foot-only, hand-off-wall reset remains stable in Isaac."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(ROOT/"dataset"))

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import active_adaptation.learning
from isaaclab.app import AppLauncher

LEG_JOINTS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
              "left_ankle_pitch_joint", "right_hip_roll_joint", "right_knee_joint",
              "right_ankle_pitch_joint")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("policy", "direct_hold", "direct_gravity",
                                      "direct_balance"), required=True)
    p.add_argument("--torque-gain", type=float, default=1.)
    p.add_argument("--balance-gain", type=float, default=1.5,
                   help="Matched hip/ankle roll feedback in rad per meter of root-y error")
    p.add_argument("--balance-damping", type=float, default=0.,
                   help="Additional matched hip/ankle roll feedback in rad per m/s of root-y velocity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seconds", type=float, default=3.)
    p.add_argument("--leg-offsets", type=float, nargs=7, default=[0.]*7,
                   help="Assisted leg joint-position offsets in LEG_JOINTS order (radians).")
    p.add_argument("--dataset", default="unified_withdrawn_support/val")
    p.add_argument("--torque-file", type=Path,
                   default=ROOT/"artifacts/unified_support_reach/withdrawn_anchor/motions/val/withdrawn_support.npz")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve(strict=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.algo.geometry_points = 0
    cfg.checkpoint_path = str(checkpoint)
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = 1000000
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    if "aux_support_force" in cfg.task.command:
        cfg.task.command.aux_support_force.enabled = False
    cfg.task.command.dataset.mem_paths = [args.dataset]
    cfg.task.command.dataset.path_weights = [1.]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.command.disable_motion_finish = True
    cfg.task.proactive_contact = True
    cfg.task.command.contact_compliance = {"enabled": True, "force_cap_N": 80.}
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
    del state
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        from isaaclab.sim import SimulationContext
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval();env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.
        env.reset()
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm();normalizer.eval()
        joint_names = list(cmd.dataset.joint_names)
        index = [joint_names.index(name) for name in base.action_manager.joint_names]
        target_q = cmd._motion.joint_pos[0, 0, index]
        if args.mode in ("direct_gravity", "direct_balance"):
            with np.load(args.torque_file) as clip:
                names = list(clip["joint_names"])
                tau = torch.as_tensor(
                    clip["torques"][0, [names.index(name) for name in base.action_manager.joint_names]],
                    dtype=target_q.dtype, device=target_q.device)
            stiffness = cmd.asset.data.joint_stiffness[0, base.action_manager.joint_ids]
            target_q = target_q + args.torque_gain*tau/stiffness
        direct_action = ((target_q-base.action_manager.default_joint_pos[
            0, base.action_manager.joint_ids])/base.action_manager.action_scaling)[None]
        leg_ids = [base.action_manager.joint_names.index(name) for name in LEG_JOINTS]
        leg_offsets = (torch.as_tensor(args.leg_offsets, device=base.device,
                                       dtype=target_q.dtype)
                       /base.action_manager.action_scaling[leg_ids])
        if args.mode == "direct_balance":
            hip = base.action_manager.joint_names.index("right_hip_roll_joint")
            ankle = base.action_manager.joint_names.index("right_ankle_roll_joint")
            y_reference = cmd.asset.data.root_pos_w[0, 1].clone()
        sensor = base.scene["ee_wall_contact_forces"]

        def measure(t):
            data = cmd.asset.data
            joint_error = (data.joint_pos[0, base.action_manager.joint_ids]-target_q).abs()
            matrix = sensor.data.force_matrix_w
            wall = (matrix.reshape(1, -1, 3).sum(dim=1).norm(dim=-1).item()
                    if matrix is not None else 0.)
            feet = cmd.foot_contact_sensor.data.net_forces_w
            return {"time_s": t, "wall_load_N": wall,
                    "wall_peak_physics_N": cmd.contact_peak_N.max().item(),
                    "hand_y_m": data.body_pos_w[0, cmd.hand_asset, 1].item(),
                    "right_foot_load_N": feet[:, cmd.right_foot_sensor_id].norm(dim=-1).item(),
                    "left_foot_load_N": feet[:, cmd.left_foot_sensor_id].norm(dim=-1).item(),
                    "right_foot_normal_N": feet[:, cmd.right_foot_sensor_id, 2].clamp_min(0.).item(),
                    "left_foot_normal_N": feet[:, cmd.left_foot_sensor_id, 2].clamp_min(0.).item(),
                    "root_height_m": data.root_pos_w[0, 2].item(),
                    "projected_gravity_z": data.projected_gravity_b[0, 2].item(),
                    "root_x_m": data.root_pos_w[0, 0].item(),
                    "root_y_m": data.root_pos_w[0, 1].item(),
                    "joint_max_error_rad": joint_error.max().item(),
                    "joint_mean_error_rad": joint_error.mean().item()}

        samples = [measure(0.)]
        failure = None
        for i in range(round(args.seconds/base.step_dt)):
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.tensor([[i == 0]], dtype=torch.bool,
                                            device=base.device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                if args.mode == "policy":
                    actor(carry)
                    if any(args.leg_offsets):
                        fraction = min(1., (i+1)*base.step_dt/.5)
                        carry["action"][:, leg_ids] += leg_offsets[None]*fraction
                else:
                    if args.mode == "direct_balance":
                        error = y_reference-cmd.asset.data.root_pos_w[0, 1]
                        velocity = cmd.asset.data.root_lin_vel_w[0, 1]
                        correction = (-args.balance_gain*error
                                      +args.balance_damping*velocity).clamp(-.35, .35)
                        q_command = target_q.clone()
                        q_command[hip] += correction
                        q_command[ankle] -= correction
                        carry["action"] = ((q_command-base.action_manager.default_joint_pos[
                            0, base.action_manager.joint_ids])
                            /base.action_manager.action_scaling)[None]
                    else:
                        carry["action"] = direct_action
                base.step(carry)
            m = measure((i+1)*base.step_dt)
            samples.append(m)
            if (m["root_height_m"] < .55 or m["projected_gravity_z"] > -.7
                    or abs(m["root_y_m"]-samples[0]["root_y_m"]) > .15):
                failure = f"instability_at_{m['time_s']:.2f}s"
                break
        final = [s for s in samples if s["time_s"] >= max(0., args.seconds-1.)]
        result = {"checkpoint": str(checkpoint), "mode": args.mode,
                  "dataset": args.dataset,
                  "leg_offsets_rad": args.leg_offsets,
                  "policy_only": args.mode == "policy" and not any(args.leg_offsets),
                  "torque_gain": args.torque_gain if args.mode in
                  ("direct_gravity", "direct_balance") else None,
                  "balance_gain": args.balance_gain if args.mode == "direct_balance" else None,
                  "balance_damping": args.balance_damping if args.mode == "direct_balance" else None,
                  "frames_completed": len(samples)-1, "stable": failure is None,
                  "failure": failure,
                  "initial_wall_load_N": samples[0]["wall_load_N"],
                  "max_wall_load_N": max(s["wall_load_N"] for s in samples),
                  "max_wall_peak_physics_N": max(s["wall_peak_physics_N"] for s in samples),
                  "direct_action_max_abs": direct_action.abs().max().item(),
                  "direct_action_clipped_joints": int((direct_action.abs() > 10.).sum().item()),
                  "max_joint_error_rad": max(s["joint_max_error_rad"] for s in samples),
                  "reset_left_foot_load_N": samples[0]["left_foot_load_N"],
                  "max_left_foot_load_after_step_N": max(
                      (s["left_foot_load_N"] for s in samples[1:]), default=0.),
                  "first_wall_contact_time_s": next((s["time_s"] for s in samples
                                                      if s["wall_load_N"] > 1.), None),
                  "final_second_mean_wall_load_N": (sum(s["wall_load_N"] for s in final)
                                                    /len(final) if final else None),
                  "final_second_hand_y_drift_m": (sum(s["hand_y_m"]-samples[0]["hand_y_m"]
                                                   for s in final)/len(final) if final else None),
                  "min_root_height_m": min(s["root_height_m"] for s in samples),
                  "note": "Physical reset/hold screen; not support establishment or task success."}
        result["final_second_left_contact_fraction_50N"] = (
            sum(s["left_foot_normal_N"] >= 50 for s in final)/len(final)
            if final else None)
        result["final_second_left_normal_N"] = (
            sum(s["left_foot_normal_N"] for s in final)/len(final)
            if final else None)
        result["final_second_right_normal_N"] = (
            sum(s["right_foot_normal_N"] for s in final)/len(final)
            if final else None)
        result["final_second_root_y_shift_m"] = (
            sum(s["root_y_m"]-samples[0]["root_y_m"] for s in final)/len(final)
            if final else None)
        result["two_foot_maintenance_success"] = bool(
            failure is None and len(samples)-1 == round(args.seconds/base.step_dt)
            and result["final_second_left_normal_N"] is not None
            and result["final_second_left_normal_N"] >= 50
            and result["final_second_left_contact_fraction_50N"] >= .8
            and result["final_second_right_normal_N"] >= 50)
        args.output.mkdir(parents=True)
        (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
        with (args.output/"samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0]))
            writer.writeheader(); writer.writerows(samples)
        print("WITHDRAWN_ANCHOR", json.dumps(result), flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush();sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()
