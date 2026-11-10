"""Screen small goal-conditioned joint corrections in paired Isaac environments.

The learned checkpoint stays fixed. Each candidate receives both signs of the
same 15 mm goal, and its joint correction changes sign with the goal. This is
an action-space diagnostic, not a trained student checkpoint.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(ROOT/"dataset"))

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
import active_adaptation.learning


JOINTS = {
    "root_y": ["right_hip_roll_joint", "right_ankle_roll_joint",
               "left_hip_roll_joint", "waist_roll_joint"],
    "hand_z": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
               "left_shoulder_yaw_joint", "left_elbow_joint",
               "left_wrist_pitch_joint"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--goal", choices=JOINTS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--std-rad", type=float, default=.06)
    p.add_argument("--max-rad", type=float, default=.18)
    p.add_argument("--settle", type=float, default=1.)
    p.add_argument("--reach", type=float, default=2.)
    p.add_argument("--hold", type=float, default=1.)
    p.add_argument("--wall-friction", type=float,
                   help="Override wall friction for a diagnostic run")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.candidates < 2 or args.std_rad <= 0 or args.max_rad <= 0:
        p.error("Need at least two candidates and positive correction scales")
    ckpt = args.checkpoint.resolve(strict=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(ckpt)
    cfg.task.num_envs = 2*args.candidates
    cfg.task.max_episode_length = 1000000
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    if "aux_support_force" in cfg.task.command:
        cfg.task.command.aux_support_force.enabled = False
    cfg.task.command.dataset.mem_paths = ["wall_foot_reach/val"]
    cfg.task.command.dataset.path_weights = [1.]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.command.disable_motion_finish = True
    if args.wall_friction is not None:
        if not 0 <= args.wall_friction <= 2:
            p.error("wall friction must be in [0,2]")
        cfg.task.objects[0].friction = args.wall_friction
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
    del state

    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        from isaaclab.sim import SimulationContext
        from active_adaptation.utils.math import quat_apply
        from scripts.utils.helpers import make_env_policy

        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval()
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.
        env.reset()
        base.episode_length_buf.zero_()
        workspace = np.load(ROOT/"artifacts/foot_reach/workspace.npz")
        anchor = torch.as_tensor(workspace["points_root"][0],
                                 device=base.device, dtype=torch.float32)
        cmd.set_target_foot_pos_b(anchor)
        dt = base.step_dt
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        manager = base.action_manager
        joint_ids = [manager.joint_names.index(name) for name in JOINTS[args.goal]]
        scaling = manager.action_scaling[joint_ids]
        rng = torch.Generator(device=base.device).manual_seed(args.seed+800)
        q = (torch.randn(args.candidates, len(joint_ids), generator=rng,
                         device=base.device)*args.std_rad).clamp(
                             -args.max_rad, args.max_rad)
        q[0] = 0.
        # Add single-joint sweeps so a useful direction is not lost in a
        # high-dimensional random draw.
        for j in range(len(joint_ids)):
            for k, value in enumerate((-.10, -.05, .05, .10)):
                idx = 1+4*j+k
                if idx < args.candidates:
                    q[idx] = 0.
                    q[idx, j] = value
        sign = torch.cat((torch.ones(args.candidates, device=base.device),
                          -torch.ones(args.candidates, device=base.device)))
        correction = torch.cat((q, -q), dim=0)/scaling
        n = 2*args.candidates
        first = True

        def step(fraction):
            nonlocal first
            raw = TensorDict({}, batch_size=[n], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.full((n, 1), first, dtype=torch.bool,
                                          device=base.device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                actor(carry)
                if fraction:
                    carry["action"][:, joint_ids] += correction*fraction
                base.step(carry)
            first = False

        for _ in range(round(args.settle/dt)):
            step(0.)
        data = cmd.asset.data
        body = "root" if args.goal == "root_y" else "left_hand"
        goal_name = "root" if args.goal == "root_y" else "left_hand"
        axis = 1 if args.goal == "root_y" else 2
        def body_pos():
            return (data.root_pos_w if body == "root"
                    else data.body_pos_w[:, cmd.hand_asset])
        initial_body = body_pos().clone()
        initial_joint = data.joint_pos[:, manager.joint_ids[joint_ids]].clone()
        initial_root = data.root_pos_w.clone()
        initial_stance = (data.body_pos_w[:, cmd.right_asset]
                          +quat_apply(data.body_quat_w[:, cmd.right_asset],
                                      cmd.sole_offset.expand(n, -1))).clone()
        target = cmd.goal_world(goal_name).clone()
        target[:, axis] += sign*.015
        cmd.set_goal_world(goal_name, target)
        max_root_drift = torch.zeros(n, device=base.device)
        max_stance_slip = torch.zeros_like(max_root_drift)
        max_hand_load = torch.zeros_like(max_root_drift)
        max_swing_error = torch.zeros_like(max_root_drift)
        min_height = torch.full_like(max_root_drift, 10.)
        max_gravity_z = torch.full_like(max_root_drift, -10.)
        sum_motion = torch.zeros_like(max_root_drift)
        sum_goal_error = torch.zeros_like(max_root_drift)
        sum_joint_motion = torch.zeros(n, len(joint_ids), device=base.device)
        hold_count = 0
        total = round((args.reach+args.hold)/dt)
        for k in range(total):
            phase = min((k+1)*dt/args.reach, 1.)
            fraction = phase*phase*(3.-2.*phase)
            step(fraction)
            position = body_pos()
            stance = (data.body_pos_w[:, cmd.right_asset]
                      +quat_apply(data.body_quat_w[:, cmd.right_asset],
                                  cmd.sole_offset.expand(n, -1)))
            swing = (data.body_pos_w[:, cmd.left_asset]
                     +quat_apply(data.body_quat_w[:, cmd.left_asset],
                                 cmd.sole_offset.expand(n, -1)))
            foot_target = (data.root_pos_w
                           +quat_apply(data.root_quat_w, cmd.target_foot_pos_b()))
            force = cmd.hand_contact_sensor.data.net_forces_w
            if force is not None:
                max_hand_load = torch.maximum(max_hand_load,
                    force[:, cmd.hand_sensor_id].norm(dim=-1))
            max_root_drift = torch.maximum(max_root_drift,
                (data.root_pos_w-initial_root).norm(dim=-1))
            max_stance_slip = torch.maximum(max_stance_slip,
                (stance-initial_stance).norm(dim=-1))
            max_swing_error = torch.maximum(max_swing_error,
                (swing-foot_target).norm(dim=-1))
            min_height = torch.minimum(min_height, data.root_pos_w[:, 2])
            max_gravity_z = torch.maximum(max_gravity_z,
                data.projected_gravity_b[:, 2])
            if k >= round(args.reach/dt):
                sum_motion += sign*(position[:, axis]-initial_body[:, axis])
                sum_goal_error += (position-target).norm(dim=-1)
                sum_joint_motion += (data.joint_pos[:, manager.joint_ids[joint_ids]]
                                     -initial_joint)
                hold_count += 1
        motion = sum_motion/hold_count
        joint_motion = sum_joint_motion/hold_count
        error = sum_goal_error/hold_count
        stable = ((min_height > .55) & (max_gravity_z < -.7)
                  & (max_root_drift < .12) & (max_stance_slip < .06)
                  & (max_swing_error < .06) & torch.isfinite(motion))
        # A paired candidate must work with both target signs. Loss counts
        # body tracking first, then collateral foot, root, and contact effects.
        penalty = (error + .2*max_swing_error + .1*max_stance_slip
                   + .00005*max_hand_load)
        pair_loss = penalty[:args.candidates]+penalty[args.candidates:]
        pair_valid = stable[:args.candidates] & stable[args.candidates:]
        pair_loss = torch.where(pair_valid, pair_loss, torch.full_like(pair_loss, 1e3))
        order = torch.argsort(pair_loss).tolist()
        rows = []
        for i in range(args.candidates):
            rows.append({"index": i, "joint_offset_rad": q[i].tolist(),
                         "joint_motion_positive_rad": joint_motion[i].tolist(),
                         "joint_motion_negative_rad": joint_motion[i+args.candidates].tolist(),
                         "stable_positive": bool(stable[i]),
                         "stable_negative": bool(stable[i+args.candidates]),
                         "motion_positive_mm": float(motion[i]*1000),
                         "motion_negative_mm": float(motion[i+args.candidates]*1000),
                         "goal_error_positive_mm": float(error[i]*1000),
                         "goal_error_negative_mm": float(error[i+args.candidates]*1000),
                         "max_root_drift_m": float(torch.maximum(
                             max_root_drift[i], max_root_drift[i+args.candidates])),
                         "max_stance_slip_m": float(torch.maximum(
                             max_stance_slip[i], max_stance_slip[i+args.candidates])),
                         "max_swing_error_m": float(torch.maximum(
                             max_swing_error[i], max_swing_error[i+args.candidates])),
                         "max_hand_load_N": float(torch.maximum(
                             max_hand_load[i], max_hand_load[i+args.candidates])),
                         "pair_loss": float(pair_loss[i])})
        best = rows[order[0]]
        args.output.mkdir(parents=True)
        with (args.output/"candidates.csv").open("w", newline="") as file:
            columns = [k for k in rows[0] if k not in (
                "joint_offset_rad", "joint_motion_positive_rad",
                "joint_motion_negative_rad")]
            columns += [f"{name}_rad" for name in JOINTS[args.goal]]
            writer = csv.DictWriter(file, columns)
            writer.writeheader()
            for row in rows:
                value = {k: v for k, v in row.items() if k != "joint_offset_rad"}
                value.update(zip(columns[-len(JOINTS[args.goal]):], row["joint_offset_rad"]))
                writer.writerow(value)
        report = {"checkpoint": str(ckpt), "goal": args.goal,
                  "joint_names": JOINTS[args.goal], "candidates": args.candidates,
                  "wall_friction": args.wall_friction,
                  "seed": args.seed, "protocol": f"{args.settle}s settle + {args.reach}s ramp + {args.hold}s hold; paired +/-15mm",
                  "best": best, "baseline": rows[0],
                  "paired_stable_count": int(pair_valid.sum()),
                  "note": "Action residual search only; no trained-policy improvement claim."}
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
        print("SEARCH_COMPLETE", json.dumps({"best": best, "baseline": rows[0],
                                             "paired_stable_count": int(pair_valid.sum())}),
              flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
    finally:
        app.close()


if __name__ == "__main__":
    main()
