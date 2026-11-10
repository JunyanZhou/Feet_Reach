"""Batched Isaac search for a grounded left-foot load correction.

Each candidate adds a small fixed joint-position offset after landing. This is
an action diagnostic around a fixed student, not a trained policy or teacher.
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
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
import active_adaptation.learning


JOINTS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
          "left_ankle_pitch_joint", "right_hip_roll_joint", "right_knee_joint",
          "right_ankle_pitch_joint")


def ease(u):
    u = min(max(u, 0.), 1.)
    return 10*u**3 - 15*u**4 + 6*u**5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory", type=Path,
                   default=ROOT/"artifacts/unified_support_reach/land_then_release_ik.npz")
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--std-rad", type=float, default=.05)
    p.add_argument("--max-rad", type=float, default=.12)
    p.add_argument("--center", type=float, nargs=7,
                   help="Search around these seven joint offsets (radians).")
    p.add_argument("--landing-depth", type=float, default=0.)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.candidates < 32 or args.candidates > 256 or not 0 <= args.landing_depth <= .02:
        p.error("Expected 32-256 candidates and landing depth in [0,.02]")
    trajectory = np.load(args.trajectory.resolve(strict=True))
    ckpt = args.checkpoint.resolve(strict=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(ckpt)
    cfg.algo.geometry_points = 0
    cfg.task.num_envs = args.candidates
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
    cfg.task.proactive_contact = True
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
        n = args.candidates
        device = base.device
        cmd.zero_init_prob = 1.
        env.reset(); base.episode_length_buf.zero_()
        foot0 = torch.as_tensor(trajectory["left_sole_world"][0],
                                dtype=torch.float32, device=device)
        cmd.set_goal_world("left_foot", foot0[None]+base.scene.env_origins)
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm(); normalizer.eval()
        manager = base.action_manager
        joint_ids = [manager.joint_names.index(name) for name in JOINTS]
        scales = manager.action_scaling[joint_ids]
        rng = torch.Generator(device=device).manual_seed(args.seed+971)
        center = torch.as_tensor(args.center if args.center is not None else [0.]*7,
                                 dtype=torch.float32, device=device)
        offsets = (center[None]+torch.randn(n, len(JOINTS), generator=rng,
                                            device=device)*args.std_rad).clamp(
                                                -args.max_rad, args.max_rad)
        offsets[0] = 0.
        offsets[1] = center
        offsets[2] = center*.5
        offsets[3] = (center*1.5).clamp(-args.max_rad, args.max_rad)
        for j in range(len(JOINTS)):
            values = ((-.04, -.02, .02, .04) if args.center is not None
                      else (-.08, -.04, .04, .08))
            for k, value in enumerate(values):
                idx = 4+4*j+k
                offsets[idx] = center
                offsets[idx, j] = float(np.clip(float(center[j])+value,
                                                -args.max_rad, args.max_rad))
        sensor = cmd.foot_contact_sensor
        wall_sensor = base.scene["ee_wall_contact_forces"]
        first = True

        def sample():
            data = cmd.asset.data
            left = data.body_pos_w[:, cmd.left_asset]+quat_apply(
                data.body_quat_w[:, cmd.left_asset], cmd.sole_offset.expand(n, -1))
            right = data.body_pos_w[:, cmd.right_asset]+quat_apply(
                data.body_quat_w[:, cmd.right_asset], cmd.sole_offset.expand(n, -1))
            forces = sensor.data.net_forces_w
            wall = wall_sensor.data.force_matrix_w.reshape(n, -1, 3).sum(dim=1).norm(dim=-1)
            return (data.root_pos_w.clone(), data.projected_gravity_b[:, 2].clone(),
                    left.clone(), right.clone(),
                    forces[:, cmd.left_foot_sensor_id, 2].clamp_min(0.).clone(),
                    forces[:, cmd.right_foot_sensor_id, 2].clamp_min(0.).clone(),
                    wall.clone())

        def step(fraction=0.):
            nonlocal first
            raw = TensorDict({}, batch_size=[n], device=device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.full((n, 1), first, dtype=torch.bool, device=device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                actor(carry)
                if fraction:
                    carry["action"][:, joint_ids] += offsets/scales*fraction
                base.step(carry)
            first = False
            return sample()

        for _ in range(50):
            step()
        settled = sample()
        start = cmd.goal_world("left_foot").clone()
        landing = (torch.as_tensor(trajectory["left_sole_world"][100],
                                   dtype=torch.float32, device=device)[None]
                   +base.scene.env_origins)
        landing[:, 2] -= args.landing_depth
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        first_failure = torch.full((n,), -1, dtype=torch.int, device=device)
        total_left = torch.zeros(n, device=device)
        total_right = torch.zeros(n, device=device)
        total_wall = torch.zeros(n, device=device)
        total_error = torch.zeros(n, device=device)
        contact_count = torch.zeros(n, device=device)
        peak_left = torch.zeros(n, device=device)
        peak_wall = torch.zeros(n, device=device)
        min_height = torch.full((n,), 10., device=device)
        for i in range(301):
            t = i*.02
            f = ease(t/2.)
            cmd.set_goal_world("left_foot", (1.-f)*start+f*landing)
            correction_fraction = ease((t-2.)/.5)
            root, gravity, left, right, left_n, right_n, wall = step(correction_fraction)
            peak_left = torch.maximum(peak_left, left_n)
            peak_wall = torch.maximum(peak_wall, wall)
            min_height = torch.minimum(min_height, root[:, 2])
            drift = (root[:, :2]-settled[0][:, :2]).abs().amax(dim=-1)
            slip = (right[:, :2]-settled[3][:, :2]).norm(dim=-1)
            bad = ((root[:, 2] < .55) | (gravity > -.7) | (drift > .1)
                   | (slip > .05))
            first_failure = torch.where(bad & ~failed,
                                         torch.full_like(first_failure, i), first_failure)
            failed |= bad
            if t >= 3.:
                valid = (~failed).float()/151
                total_left += valid*left_n
                total_right += valid*right_n
                total_wall += valid*wall
                total_error += valid*(left-cmd.goal_world("left_foot")).norm(dim=-1)
                contact_count += valid*(left_n >= 50).float()
        stable = ~failed
        passed = (stable & (contact_count >= .8) & (total_left >= 50)
                  & (total_right >= 200) & (total_error <= .03)
                  & (peak_left <= 300) & (peak_wall <= 150))
        rows = []
        for j in range(n):
            row = {"index": j, "joint_offset_rad": offsets[j].tolist(),
                   "stable": bool(stable[j]),
                   "first_failure_s": None if stable[j] else float(first_failure[j])*.02,
                   "left_contact_fraction_50N": float(contact_count[j]),
                   "mean_left_normal_N": float(total_left[j]),
                   "mean_right_normal_N": float(total_right[j]),
                   "mean_wall_N": float(total_wall[j]),
                   "mean_left_sole_error_m": float(total_error[j]),
                   "peak_left_control_N": float(peak_left[j]),
                   "peak_wall_control_N": float(peak_wall[j]),
                   "min_root_height_m": float(min_height[j]),
                   "landing_support_success": bool(passed[j])}
            rows.append(row)
        rows.sort(key=lambda r: (r["landing_support_success"], r["stable"],
                                 r["left_contact_fraction_50N"],
                                 -r["mean_left_sole_error_m"]), reverse=True)
        args.output.mkdir(parents=True)
        with (args.output/"candidates.csv").open("w", newline="") as file:
            fields = [k for k in rows[0] if k != "joint_offset_rad"] + list(JOINTS)
            writer = csv.DictWriter(file, fields)
            writer.writeheader()
            for row in rows:
                flat = {k: v for k, v in row.items() if k != "joint_offset_rad"}
                flat.update(zip(JOINTS, row["joint_offset_rad"]))
                writer.writerow(flat)
        report = {"checkpoint": str(ckpt), "seed": args.seed,
                  "landing_depth_m": args.landing_depth, "candidates": n,
                  "center_rad": center.tolist(), "std_rad": args.std_rad,
                  "stable_count": int(stable.sum()), "success_count": int(passed.sum()),
                  "joint_names": JOINTS, "top10": rows[:10], "baseline": rows[0] if rows[0]["index"] < 4 else next(r for r in rows if r["index"] == 0),
                  "note": "Action residual search only; no trained-policy improvement claim."}
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
        print("LANDING_LOAD_SEARCH", json.dumps({"stable_count": report["stable_count"],
              "success_count": report["success_count"], "best": rows[0]}), flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
