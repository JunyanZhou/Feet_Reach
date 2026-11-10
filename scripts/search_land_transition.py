"""Batched Isaac diagnostic for foot landing, load transfer, then arm release.

This searches scripted targets and arm IK assistance around a fixed student.
It does not train a policy and must not be reported as student-only success.
"""

import argparse
import csv
import itertools
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


def ease(u):
    u = min(max(u, 0.), 1.)
    return 10*u**3 - 15*u**4 + 6*u**5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectory", type=Path,
                   default=ROOT / "artifacts/unified_support_reach/land_then_release_ik.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--depths", type=float, nargs="+", default=[0., .004, .008, .012])
    p.add_argument("--root-shifts", type=float, nargs="+", default=[-.02, 0., .02, .04])
    p.add_argument("--arm-gains", type=float, nargs="+", default=[0., .2, .4, .6])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidates = list(itertools.product(args.depths, args.root_shifts, args.arm_gains))
    if not candidates or len(candidates) > 256:
        p.error("Expected 1 to 256 candidates")
    trajectory = np.load(args.trajectory.resolve(strict=True))
    times = trajectory["time_s"]
    if len(times) != 301 or not np.allclose(np.diff(times), .02, atol=1e-6):
        raise ValueError("Expected 301 frames at 50 Hz")
    ckpt = args.checkpoint.resolve(strict=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(ckpt)
    cfg.algo.geometry_points = 0
    cfg.task.num_envs = len(candidates)
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
        n = len(candidates)
        device = base.device
        cmd.zero_init_prob = 1.
        env.reset()
        base.episode_length_buf.zero_()
        foot0 = trajectory["left_sole_world"][0]
        cmd.set_goal_world("left_foot", torch.as_tensor(foot0,
            dtype=torch.float32, device=device)[None]+base.scene.env_origins)
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm(); normalizer.eval()
        manager = base.action_manager
        names = list(trajectory["joint_names"])
        index = [names.index(name) for name in manager.joint_names]
        q = torch.as_tensor(trajectory["dof_pos"][:, index], dtype=torch.float32, device=device)
        arm_mask = torch.as_tensor([name.startswith(("left_shoulder", "left_elbow", "left_wrist"))
                                    for name in manager.joint_names], device=device)
        depth = torch.tensor([x[0] for x in candidates], device=device)
        root_shift = torch.tensor([x[1] for x in candidates], device=device)
        arm_gain = torch.tensor([x[2] for x in candidates], device=device)
        feet = cmd.foot_contact_sensor
        wall_sensor = base.scene["ee_wall_contact_forces"]
        first = True

        def sample():
            data = cmd.asset.data
            sole_l = data.body_pos_w[:, cmd.left_asset] + quat_apply(
                data.body_quat_w[:, cmd.left_asset], cmd.sole_offset.expand(n, -1))
            sole_r = data.body_pos_w[:, cmd.right_asset] + quat_apply(
                data.body_quat_w[:, cmd.right_asset], cmd.sole_offset.expand(n, -1))
            wall_matrix = wall_sensor.data.force_matrix_w
            wall = wall_matrix.reshape(n, -1, 3).sum(dim=1).norm(dim=-1)
            forces = feet.data.net_forces_w
            return {
                "root": data.root_pos_w.clone(),
                "gravity_z": data.projected_gravity_b[:, 2].clone(),
                "hand_y": data.body_pos_w[:, cmd.hand_asset, 1].clone(),
                "sole_l": sole_l.clone(), "sole_r": sole_r.clone(),
                "wall": wall.clone(),
                "left_normal": forces[:, cmd.left_foot_sensor_id, 2].clamp_min(0.).clone(),
                "right_normal": forces[:, cmd.right_foot_sensor_id, 2].clamp_min(0.).clone(),
            }

        def step(arm_action=None):
            nonlocal first
            raw = TensorDict({}, batch_size=[n], device=device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.full((n, 1), first, dtype=torch.bool, device=device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                actor(carry)
                if arm_action is not None:
                    carry["action"][:, arm_mask] = arm_action
                base.step(carry)
            first = False
            return sample()

        for _ in range(50):
            step()
        settled = sample()
        left_ref = cmd.goal_world("left_foot").clone()
        hand_ref = cmd.goal_world("left_hand").clone()
        root_ref = cmd.goal_world("root").clone()
        landing = (torch.as_tensor(trajectory["left_sole_world"][100],
                                   dtype=torch.float32, device=device)[None]
                   + base.scene.env_origins)
        landing[:, 2] -= depth
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        first_fail = torch.full((n,), -1, dtype=torch.int, device=device)
        transfer_left = torch.zeros(n, device=device)
        transfer_wall = torch.zeros(n, device=device)
        release_left = torch.zeros(n, device=device)
        release_right = torch.zeros(n, device=device)
        release_wall = torch.zeros(n, device=device)
        release_hand = torch.zeros(n, device=device)
        release_sole_error = torch.zeros(n, device=device)
        peak_wall = torch.zeros(n, device=device)
        arm_anchor = None
        for i, t in enumerate(times):
            if i == 150:
                arm_anchor = cmd.asset.data.joint_pos[:, manager.joint_ids][:, arm_mask].clone()
            f = ease(float(t)/2.)
            cmd.set_goal_world("left_foot", (1-f)*left_ref+f*landing)
            hand_delta = trajectory["hand_world"][i]-trajectory["hand_world"][0]
            cmd.set_goal_world("left_hand", hand_ref+torch.as_tensor(
                hand_delta, dtype=torch.float32, device=device))
            root_goal = root_ref.clone()
            root_goal[:, 1] += root_shift*ease((float(t)-2.)/1.)
            cmd.set_goal_world("root", root_goal)
            arm_action = None
            if arm_anchor is not None:
                target = arm_anchor+arm_gain[:, None]*(q[i, arm_mask]-q[150, arm_mask])[None]
                default = manager.default_joint_pos[:, manager.joint_ids][:, arm_mask]
                arm_action = (target-default)/manager.action_scaling[arm_mask]
            m = step(arm_action)
            slip = (m["sole_r"][:, :2]-settled["sole_r"][:, :2]).norm(dim=-1)
            drift = (m["root"][:, :2]-settled["root"][:, :2]).abs().amax(dim=-1)
            bad = ((m["root"][:, 2] < .55) | (m["gravity_z"] > -.7)
                   | (drift > .1) | (slip > .05))
            first_fail = torch.where(bad & ~failed, torch.full_like(first_fail, i), first_fail)
            failed |= bad
            valid = (~failed).float()
            peak_wall = torch.maximum(peak_wall, m["wall"])
            if 2.5 <= t < 3.:
                transfer_left += valid*m["left_normal"]/25
                transfer_wall += valid*m["wall"]/25
            if 5. <= t <= 6.:
                release_left += valid*m["left_normal"]/51
                release_right += valid*m["right_normal"]/51
                release_wall += valid*m["wall"]/51
                release_hand += valid*(settled["hand_y"]-m["hand_y"])/51
                release_sole_error += valid*(m["sole_l"]-cmd.goal_world("left_foot")).norm(dim=-1)/51
        # Require full rollout and sustained double-foot support, true hand
        # withdrawal, and low wall force. Force sums are zeroed after failure.
        success = ((~failed) & (transfer_left >= 50) & (release_left >= 50)
                   & (release_right >= 50) & (release_wall <= 5)
                   & (release_hand >= .03) & (release_sole_error <= .03))
        fields = ["index", "depth_m", "root_shift_y_m", "arm_gain",
                  "stable", "first_failure_s", "transfer_left_normal_N",
                  "transfer_wall_N", "released_left_normal_N", "released_right_normal_N",
                  "released_wall_N", "released_hand_withdrawal_m",
                  "released_left_sole_error_m", "peak_wall_N", "support_switch_success"]
        rows = []
        for j, (d, s, g) in enumerate(candidates):
            rows.append(dict(zip(fields, (j, d, s, g, bool(~failed[j]),
                                          float(first_fail[j])*.02 if failed[j] else None,
                                          float(transfer_left[j]), float(transfer_wall[j]),
                                          float(release_left[j]), float(release_right[j]),
                                          float(release_wall[j]), float(release_hand[j]),
                                          float(release_sole_error[j]), float(peak_wall[j]),
                                          bool(success[j])))))
        rows.sort(key=lambda r: (r["support_switch_success"], r["stable"],
                                 r["released_hand_withdrawal_m"] >= .03,
                                 r["released_left_normal_N"],
                                 -r["released_wall_N"]), reverse=True)
        args.output.mkdir(parents=True)
        with (args.output/"candidates.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fields)
            writer.writeheader(); writer.writerows(rows)
        report = {"checkpoint": str(ckpt), "trajectory": str(args.trajectory),
                  "seed": args.seed, "candidates": n,
                  "stable_count": int((~failed).sum()),
                  "success_count": int(success.sum()), "top10": rows[:10],
                  "note": "Scripted IK arm action; no student-only support-switch claim."}
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
        print("SEARCH_LAND", json.dumps({k: report[k] for k in
              ("candidates", "stable_count", "success_count", "top10")}), flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
