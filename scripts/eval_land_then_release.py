"""Isaac screen of left-foot landing before withdrawing wall-hand support."""

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
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
import active_adaptation.learning

LEG_JOINTS = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
              "left_ankle_pitch_joint", "right_hip_roll_joint", "right_knee_joint",
              "right_ankle_pitch_joint")


def ease(u):
    u = min(max(u, 0.), 1.)
    return 10*u**3-15*u**4+6*u**5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectory", type=Path,
                   default=ROOT/"artifacts/unified_support_reach/land_then_release_ik.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("policy", "residual", "direct", "direct_relative", "blend"),
                   default="policy")
    p.add_argument("--gain", type=float, default=1.)
    p.add_argument("--residual-joints", choices=("all", "arm_release", "arm_direct"), default="all",
                   help="Apply only the left-arm release motion after the landing hold.")
    p.add_argument("--root-shift-y", type=float, default=0.,
                   help="Lateral root goal shift, ramped during load transfer (m).")
    p.add_argument("--root-from-trajectory", action="store_true",
                   help="Follow the offline root transfer in the root target.")
    p.add_argument("--transfer-ik-gain", type=float, default=0.,
                   help="Add lower-body IK joint differences after landing (assisted).")
    p.add_argument("--landing-depth", type=float, default=0.,
                   help="Lower the final left-foot goal by this depth (m).")
    p.add_argument("--initial-foot-world", action="store_true",
                   help="Hold the offline initial sole goal fixed in the world during settling.")
    p.add_argument("--hold-hand", action="store_true",
                   help="Keep the left-hand wall target fixed to isolate foot landing and loading.")
    p.add_argument("--leg-offsets", type=float, nargs=7, default=[0.]*7,
                   metavar="RAD", help="Post-landing position offsets in LEG_JOINTS order.")
    p.add_argument("--lateral-feedback-gain", type=float, default=0.,
                   help="Assisted right hip/ankle roll response to root-y target error (rad/m).")
    p.add_argument("--lateral-feedback-limit", type=float, default=.15,
                   help="Absolute roll correction limit (rad).")
    p.add_argument("--knee-force-target", type=float,
                   help="Enable feedback on left knee offset to track this normal force (N).")
    p.add_argument("--knee-force-gain", type=float, default=.0002,
                   help="Knee offset update in rad per N per 20 ms step.")
    p.add_argument("--knee-force-alpha", type=float, default=.2,
                   help="EMA coefficient for measured left-foot normal force.")
    p.add_argument("--knee-force-min-offset", type=float, default=-.15)
    p.add_argument("--knee-force-max-offset", type=float, default=.05)
    p.add_argument("--blend-weight", type=float, default=.25)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (not 0 <= args.gain <= 3 or not 0 <= args.blend_weight <= 1
            or not -.2 <= args.root_shift_y <= .2
            or not 0. <= args.landing_depth <= .05
            or not 0 <= args.transfer_ik_gain <= 3
            or (args.knee_force_target is not None and args.knee_force_target <= 0)
            or args.knee_force_gain < 0 or not 0 < args.knee_force_alpha <= 1
            or abs(args.lateral_feedback_gain) > 10
            or not 0 < args.lateral_feedback_limit <= .35
            or not -.3 <= args.knee_force_min_offset < args.knee_force_max_offset <= .2):
        p.error("gain and blend weight outside allowed range")
    trajectory = np.load(args.trajectory.resolve(strict=True))
    times = trajectory["time_s"]
    if len(times) != 301 or not np.allclose(np.diff(times), .02, atol=1e-6):
        raise ValueError("Expected 301 frames at 50 Hz")
    checkpoint = args.checkpoint.resolve(strict=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(checkpoint)
    cfg.algo.geometry_points = 0
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = 1000000
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    if "aux_support_force" in cfg.task.command:
        cfg.task.command.aux_support_force.enabled = False
    # Force monitors run at the 200 Hz physics rate. In eval mode these
    # optional command features change no goal or applied force.
    cfg.task.command.landing_cycle = {"enabled": True}
    cfg.task.command.contact_compliance = {"enabled": True}
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
        cmd.zero_init_prob = 1.
        env.reset()
        base.episode_length_buf.zero_()
        left0 = trajectory["left_sole_world"][0]
        root0 = trajectory["root_pos"][0]
        root_R = Rotation.from_quat(trajectory["root_rot"][0]).as_matrix()
        target_b = root_R.T@(left0-root0)
        if args.initial_foot_world:
            cmd.set_goal_world("left_foot", torch.as_tensor(left0,
                dtype=torch.float32, device=base.device)[None]+base.scene.env_origins)
        else:
            cmd.set_target_foot_pos_b(torch.as_tensor(target_b, dtype=torch.float32,
                                                      device=base.device))
        manager = base.action_manager
        names = list(trajectory["joint_names"])
        index = [names.index(name) for name in manager.joint_names]
        q = torch.as_tensor(trajectory["dof_pos"][:, index],
                            dtype=torch.float32, device=base.device)
        if args.hold_hand:
            q = q.clone()
            q[151:] = q[150]
        residual = args.gain*(q-q[0])/manager.action_scaling
        arm_mask = torch.as_tensor([name.startswith(("left_shoulder", "left_elbow",
                                                   "left_wrist"))
                                    for name in manager.joint_names], device=base.device)
        leg_ids = [manager.joint_names.index(name) for name in LEG_JOINTS]
        transfer_mask = torch.as_tensor([name.startswith(("left_hip", "left_knee",
                            "left_ankle", "right_hip", "right_knee", "right_ankle",
                            "waist")) for name in manager.joint_names], device=base.device)
        leg_offsets = torch.as_tensor(args.leg_offsets, dtype=torch.float32, device=base.device)
        right_hip_roll = manager.joint_names.index("right_hip_roll_joint")
        right_ankle_roll = manager.joint_names.index("right_ankle_roll_joint")
        knee_offset = float(args.leg_offsets[2])
        knee_force_filtered = 0.
        if args.residual_joints in ("arm_release", "arm_direct"):
            residual = args.gain*(q-q[150])/manager.action_scaling
            residual[:, ~arm_mask] = 0.
            residual[:150] = 0.
        direct = (q-manager.default_joint_pos[0, manager.joint_ids]) / manager.action_scaling
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        sensor = base.scene["ee_wall_contact_forces"]
        feet_id_r = cmd.right_foot_sensor_id
        feet_id_l = cmd.left_foot_sensor_id
        dt = base.step_dt
        if abs(dt-.02) > 1e-8:
            raise ValueError(f"Unexpected policy step {dt}")
        first = True

        def measure():
            data = cmd.asset.data
            matrix = sensor.data.force_matrix_w
            wall = (matrix.reshape(1, -1, 3).sum(dim=1).norm(dim=-1).item()
                    if matrix is not None else 0.)
            feet = cmd.foot_contact_sensor.data.net_forces_w
            sole_l = (data.body_pos_w[:, cmd.left_asset]
                      +quat_apply(data.body_quat_w[:, cmd.left_asset],
                                  cmd.sole_offset.expand(1, -1)))
            sole_r = (data.body_pos_w[:, cmd.right_asset]
                      +quat_apply(data.body_quat_w[:, cmd.right_asset],
                                  cmd.sole_offset.expand(1, -1)))
            return {"root_x_m": data.root_pos_w[0, 0].item(),
                    "root_y_m": data.root_pos_w[0, 1].item(),
                    "root_goal_y_m": cmd.goal_world("root")[0, 1].item(),
                    "root_height_m": data.root_pos_w[0, 2].item(),
                    "root_quat_wxyz": data.root_quat_w[0].tolist(),
                    "gravity_z": data.projected_gravity_b[0, 2].item(),
                    "hand_y_m": data.body_pos_w[0, cmd.hand_asset, 1].item(),
                    "wall_load_N": wall,
                    "wall_peak_physics_N": cmd.contact_peak_N[0].item(),
                    "right_foot_load_N": feet[:, feet_id_r].norm(dim=-1).item(),
                    "left_foot_load_N": feet[:, feet_id_l].norm(dim=-1).item(),
                    "right_foot_normal_N": feet[:, feet_id_r, 2].clamp_min(0.).item(),
                    "left_foot_normal_N": feet[:, feet_id_l, 2].clamp_min(0.).item(),
                    "left_foot_peak_physics_N": cmd.landing_foot_peak_N[0].item(),
                    "left_sole_error_m": (sole_l-cmd.goal_world("left_foot")).norm().item(),
                    "left_sole_x_m": sole_l[0, 0].item(),
                    "left_sole_y_m": sole_l[0, 1].item(),
                    "left_sole_height_m": sole_l[0, 2].item(),
                    "right_sole_x_m": sole_r[0, 0].item(),
                    "right_sole_y_m": sole_r[0, 1].item()}

        def step(correction=None, direct_action=None, absolute_arm_action=None,
                 leg_action=None, roll_correction_rad=0.):
            nonlocal first
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.tensor([[first]], dtype=torch.bool,
                                             device=base.device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                cmd.landing_foot_peak_N.zero_()
                if args.mode in ("direct", "direct_relative") and direct_action is not None:
                    carry["action"] = direct_action[None]
                else:
                    actor(carry)
                    if correction is not None:
                        carry["action"] += correction[None]
                    if absolute_arm_action is not None:
                        carry["action"][:, arm_mask] = absolute_arm_action[None]
                    if args.mode == "blend" and direct_action is not None:
                        carry["action"] = ((1-args.blend_weight)*carry["action"]
                                           +args.blend_weight*direct_action[None])
                if leg_action is not None:
                    carry["action"][:, leg_ids] += leg_action[None]
                if roll_correction_rad:
                    carry["action"][:, right_hip_roll] += (
                        roll_correction_rad/manager.action_scaling[right_hip_roll])
                    carry["action"][:, right_ankle_roll] -= (
                        roll_correction_rad/manager.action_scaling[right_ankle_roll])
                base.step(carry)
            first = False
            return measure()

        for _ in range(round(1./dt)):
            step()
        settled = measure()
        if args.mode == "direct_relative":
            joint_anchor = cmd.asset.data.joint_pos[0, manager.joint_ids].clone()
            direct = ((joint_anchor[None]+q-q[0]
                       -manager.default_joint_pos[0, manager.joint_ids])
                      /manager.action_scaling)
        left_ref = cmd.goal_world("left_foot").clone()
        hand_ref = cmd.goal_world("left_hand").clone()
        root_ref = cmd.goal_world("root").clone()
        landing_world = torch.as_tensor(trajectory["left_sole_world"][2*50],
                                        dtype=torch.float32, device=base.device)
        landing_world[2] -= args.landing_depth
        cmd.set_goal_world("left_foot", left_ref)
        rows = []
        failure = None
        arm_anchor = None
        for i, t in enumerate(times):
            if i == 150 and args.residual_joints == "arm_direct":
                arm_anchor = cmd.asset.data.joint_pos[0, manager.joint_ids][arm_mask].clone()
            # The first target is root-relative during settling. Once the
            # rollout begins, interpolate to an absolute ground target; using
            # the offline foot displacement alone would retain any rise caused
            # by the live root's lean and leave the foot above the floor.
            landing_fraction = min(max(float(t)/2., 0.), 1.)
            landing_fraction = (10*landing_fraction**3
                                -15*landing_fraction**4+6*landing_fraction**5)
            delta_hand = (np.zeros(3) if args.hold_hand else
                          trajectory["hand_world"][i]-trajectory["hand_world"][0])
            cmd.set_goal_world("left_foot", (1-landing_fraction)*left_ref
                               +landing_fraction*landing_world)
            cmd.set_goal_world("left_hand", hand_ref+torch.as_tensor(
                delta_hand, dtype=torch.float32, device=base.device))
            transfer_fraction = min(max((float(t)-2.)/1., 0.), 1.)
            transfer_fraction = (10*transfer_fraction**3
                                 -15*transfer_fraction**4+6*transfer_fraction**5)
            root_goal = root_ref.clone()
            root_goal[:, 1] += args.root_shift_y*transfer_fraction
            if args.root_from_trajectory:
                root_goal += torch.as_tensor(
                    trajectory["root_pos"][i]-trajectory["root_pos"][0],
                    dtype=torch.float32, device=base.device)
            cmd.set_goal_world("root", root_goal)
            arm_action = None
            if arm_anchor is not None and args.mode == "residual":
                target_arm = arm_anchor + args.gain*(q[i, arm_mask]-q[150, arm_mask])
                arm_action = ((target_arm-manager.default_joint_pos[0, manager.joint_ids][arm_mask])
                              /manager.action_scaling[arm_mask])
            leg_fraction = ease((float(t)-2.)/.5)
            if args.knee_force_target is not None and t >= 2.:
                knee_offset = float(np.clip(
                    knee_offset-args.knee_force_gain*(args.knee_force_target-knee_force_filtered),
                    args.knee_force_min_offset, args.knee_force_max_offset))
            leg_offsets[2] = knee_offset
            leg_action = (leg_offsets/manager.action_scaling[leg_ids]*leg_fraction
                          if leg_fraction else None)
            correction = (residual[i].clone() if args.mode == "residual" and
                          args.residual_joints != "arm_direct" else None)
            if args.transfer_ik_gain and i >= 100:
                if correction is None:
                    correction = torch.zeros_like(q[i])
                correction[transfer_mask] += (args.transfer_ik_gain
                    *(q[i, transfer_mask]-q[100, transfer_mask])
                    /manager.action_scaling[transfer_mask])
            roll_correction = (float(np.clip(
                args.lateral_feedback_gain*(cmd.asset.data.root_pos_w[0, 1].item()
                                            -root_goal[0, 1].item()),
                -args.lateral_feedback_limit, args.lateral_feedback_limit))
                *transfer_fraction)
            m = step(correction,
                     direct[i] if args.mode in ("direct", "direct_relative", "blend") else None,
                     arm_action, leg_action, roll_correction)
            m["left_knee_offset_rad"] = knee_offset*leg_fraction
            m["right_roll_feedback_rad"] = roll_correction
            if args.knee_force_target is not None:
                knee_force_filtered = ((1.-args.knee_force_alpha)*knee_force_filtered
                                       +args.knee_force_alpha*m["left_foot_normal_N"])
            m["time_s"] = float(t)
            m["hand_withdrawal_m"] = settled["hand_y_m"]-m["hand_y_m"]
            rows.append(m)
            slip = ((m["right_sole_x_m"]-settled["right_sole_x_m"])**2
                    +(m["right_sole_y_m"]-settled["right_sole_y_m"])**2)**.5
            if (m["root_height_m"] < .55 or m["gravity_z"] > -.7
                    or abs(m["root_x_m"]-settled["root_x_m"]) > .1
                    or abs(m["root_y_m"]-settled["root_y_m"]) > .1
                    or slip > .05 or not all(np.isfinite(v).all() for v in m.values())):
                failure = f"instability_at_{t:.2f}s"
                break
        transfer = [r for r in rows if 2.5 <= r["time_s"] < 3.]
        landed = [r for r in rows if 3. <= r["time_s"] <= 6.]
        released = [r for r in rows if 5. <= r["time_s"] <= 6.]
        def mean(rows, key):
            return float(np.mean([r[key] for r in rows])) if rows else None
        policy_only = (args.mode == "policy" and args.knee_force_target is None
                       and args.transfer_ik_gain == 0
                       and args.lateral_feedback_gain == 0
                       and not any(abs(v) > 0 for v in args.leg_offsets))
        report = {"checkpoint": str(checkpoint),
                  "trajectory": str(args.trajectory.resolve()),
                  "mode": args.mode, "gain": args.gain,
                  "residual_joints": args.residual_joints,
                  "root_shift_y_m": args.root_shift_y,
                  "root_from_trajectory": args.root_from_trajectory,
                  "transfer_ik_gain": args.transfer_ik_gain,
                  "landing_depth_m": args.landing_depth,
                  "initial_foot_world": args.initial_foot_world,
                  "hold_hand": args.hold_hand,
                  "leg_offsets_rad": args.leg_offsets,
                  "lateral_feedback_gain_rad_per_m": args.lateral_feedback_gain,
                  "lateral_feedback_limit_rad": args.lateral_feedback_limit,
                  "observed_roll_feedback_range_rad": [
                      min(r["right_roll_feedback_rad"] for r in rows),
                      max(r["right_roll_feedback_rad"] for r in rows)],
                  "knee_force_target_N": args.knee_force_target,
                  "knee_force_gain": args.knee_force_gain,
                  "knee_force_alpha": args.knee_force_alpha,
                  "knee_force_offset_bounds_rad": [args.knee_force_min_offset,
                                                    args.knee_force_max_offset],
                  "observed_knee_offset_range_rad": [
                      min(r["left_knee_offset_rad"] for r in rows),
                      max(r["left_knee_offset_rad"] for r in rows)],
                  "policy_only": policy_only,
                  "blend_weight": args.blend_weight if args.mode == "blend" else None,
                  "seed": args.seed, "frames_completed": len(rows),
                  "stable": failure is None and len(rows) == len(times),
                  "failure": failure,
                  "settled": settled,
                  "settled_left_foot_goal_world_m": left_ref[0].tolist(),
                  "transfer_left_foot_load_N": mean(transfer, "left_foot_load_N"),
                  "transfer_left_normal_N": mean(transfer, "left_foot_normal_N"),
                  "transfer_left_contact_fraction_50N": (
                      float(np.mean([r["left_foot_normal_N"] >= 50 for r in transfer]))
                      if transfer else None),
                  "transfer_wall_load_N": mean(transfer, "wall_load_N"),
                  "landed_left_normal_N": mean(landed, "left_foot_normal_N"),
                  "landed_left_contact_fraction_50N": (
                      float(np.mean([r["left_foot_normal_N"] >= 50 for r in landed]))
                      if landed else None),
                  "landed_right_normal_N": mean(landed, "right_foot_normal_N"),
                  "landed_left_sole_error_m": mean(landed, "left_sole_error_m"),
                  "landed_root_y_shift_m": (
                      mean(landed, "root_y_m")-settled["root_y_m"] if landed else None),
                  "landed_root_y_target_error_m": (
                      float(np.mean([abs(r["root_y_m"]-r["root_goal_y_m"])
                                     for r in landed])) if landed else None),
                  "released_left_foot_load_N": mean(released, "left_foot_load_N"),
                  "released_left_normal_N": mean(released, "left_foot_normal_N"),
                  "released_left_contact_fraction_50N": (
                      float(np.mean([r["left_foot_normal_N"] >= 50 for r in released]))
                      if released else None),
                  "released_right_foot_load_N": mean(released, "right_foot_load_N"),
                  "released_right_normal_N": mean(released, "right_foot_normal_N"),
                  "released_wall_load_N": mean(released, "wall_load_N"),
                  "released_hand_withdrawal_m": mean(released, "hand_withdrawal_m"),
                  "released_left_sole_error_m": mean(released, "left_sole_error_m"),
                  "peak_wall_load_N": max(r["wall_load_N"] for r in rows),
                  "peak_wall_physics_N": max(r["wall_peak_physics_N"] for r in rows),
                  "peak_left_foot_physics_N": max(r["left_foot_peak_physics_N"] for r in rows),
                  "note": "Contact peaks sampled at 200 Hz. Arm/leg IK or force feedback is reported as assisted control."}
        report["landing_support_success"] = bool(
            report["stable"] and report["landed_left_normal_N"] is not None
            and report["landed_left_normal_N"] >= 50
            and report["landed_left_contact_fraction_50N"] >= .8
            and report["landed_right_normal_N"] >= 50
            and report["landed_left_sole_error_m"] <= .03)
        report["support_switch_success"] = bool(
            report["stable"] and report["transfer_left_foot_load_N"] is not None
            and report["transfer_left_normal_N"] >= 50
            and report["transfer_left_contact_fraction_50N"] >= .8
            and report["released_left_normal_N"] >= 50
            and report["released_left_contact_fraction_50N"] >= .8
            and report["released_right_normal_N"] >= 50
            and report["released_wall_load_N"] <= 5
            and report["released_hand_withdrawal_m"] >= .03
            and report["released_left_sole_error_m"] <= .03)
        args.output.mkdir(parents=True)
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
        with (args.output/"samples.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("LAND_RELEASE", json.dumps(report), flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
