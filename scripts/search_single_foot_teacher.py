"""Batched Isaac search for a contact-free right-foot stance controller.

The controller holds an offline IK pose with gravity feedforward, constant
hip/ankle/waist offsets, and root-lateral-position/velocity feedback. All
candidates are evaluated in the same physical rollout; no policy is trained.
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
import active_adaptation.learning
from isaaclab.app import AppLauncher

PARAMS = ("torque_gain", "hip_offset", "ankle_offset", "waist_offset",
          "hip_p", "hip_d", "ankle_p", "ankle_d",
          "hip_pitch_offset", "ankle_pitch_offset",
          "hip_pitch_p", "hip_pitch_d", "ankle_pitch_p", "ankle_pitch_d",
          "shoulder_roll_offset", "hand_y_p", "hand_y_d",
          "elbow_offset", "hand_elbow_p", "hand_elbow_d")


def candidates(count, seed, center_path, pitch_only, hand_only):
    rng = np.random.default_rng(seed)
    if center_path:
        center = json.loads(center_path.read_text())["best"]["parameters"]
        rows = np.tile(np.array([center.get(k, 0.) for k in PARAMS], dtype="f4"),
                       (count, 1))
        scales = np.array([.15, .08, .08, .05, 1.5, .55, 1.5, .55,
                           .08, .08, 3., 1.2, 3., 1.2, .10, 3., 1.,
                           .10, 3., 1.], dtype="f4")
        if pitch_only:
            scales[:8] = 0.; scales[14:] = 0.
        if hand_only:
            scales[:14] = 0.
        rows[1:] += rng.normal(size=(count-1, len(PARAMS))).astype("f4")*scales
    else:
        rows = np.column_stack([
            rng.uniform(.6, 1.4, count),
            rng.uniform(-.28, .28, count),
            rng.uniform(-.28, .28, count),
            rng.uniform(-.12, .12, count),
            rng.uniform(-6, 6, count),
            rng.uniform(-2.5, 2.5, count),
            rng.uniform(-6, 6, count),
            rng.uniform(-2.5, 2.5, count),
            rng.uniform(-.12, .12, count),
            rng.uniform(-.12, .12, count),
            rng.uniform(-6, 6, count),
            rng.uniform(-2.5, 2.5, count),
            rng.uniform(-6, 6, count),
            rng.uniform(-2.5, 2.5, count),
            rng.uniform(-.2, .2, count),
            rng.uniform(-8, 8, count),
            rng.uniform(-2.5, 2.5, count),
            rng.uniform(-.2, .2, count),
            rng.uniform(-8, 8, count),
            rng.uniform(-2.5, 2.5, count),
        ]).astype("f4")
        rows[0] = [1., 0., 0., 0., 1.5, 1.5, -1.5, -1.5,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        rows[1] = [1., 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    rows[:, 0] = np.clip(rows[:, 0], .3, 1.6)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "artifacts/unified_support_reach/v2_goal_offset_x10.pt")
    p.add_argument("--torque-file", type=Path,
                   default=ROOT / "artifacts/unified_support_reach/withdrawn_anchor/motions/val/withdrawn_support.npz")
    p.add_argument("--dataset", default="unified_withdrawn_support/val")
    p.add_argument("--motion-id", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--seconds", type=float, default=3.)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--center", type=Path)
    p.add_argument("--pitch-only", action="store_true",
                   help="When centered, vary only pitch parameters")
    p.add_argument("--hand-only", action="store_true",
                   help="When centered, vary only left-shoulder hand-y feedback")
    p.add_argument("--reset-drop", type=float, default=0.,
                   help="Lower root after dataset reset; 0.04 m removes the default reset lift")
    p.add_argument("--wall-shift-y", type=float, default=0.,
                   help="Counterfactual: move wall away in +y to test dependence on wall force")
    p.add_argument("--clamp-joint-targets", action="store_true",
                   help="Clamp position targets to simulated joint limits")
    p.add_argument("--feedback-state", choices=("root", "com"), default="root")
    p.add_argument("--com-target", choices=("initial", "center"), default="initial")
    p.add_argument("--pitch-state", choices=("root", "com"), default="com")
    p.add_argument("--pitch-target", choices=("initial", "center"), default="initial")
    p.add_argument("--trace-index", type=int, default=-1,
                   help="Save a per-step diagnostic CSV for one candidate")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.count < 2:
        raise ValueError("count must be >= 2")
    if (args.pitch_only or args.hand_only) and not args.center:
        raise ValueError("--pitch-only and --hand-only require --center")
    if args.pitch_only and args.hand_only:
        raise ValueError("Choose at most one restricted search mode")
    matrix_np = candidates(args.count, args.seed, args.center,
                           args.pitch_only, args.hand_only)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.algo.geometry_points = 0
    cfg.checkpoint_path = str(args.checkpoint.resolve(strict=True))
    cfg.task.num_envs = args.count
    cfg.task.max_episode_length = 1000000
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    if "aux_support_force" in cfg.task.command:
        cfg.task.command.aux_support_force.enabled = False
    cfg.task.command.dataset.mem_paths = [args.dataset]
    cfg.task.command.dataset.path_weights = [1.]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = args.motion_id
    cfg.task.command.disable_motion_finish = True
    cfg.task.proactive_contact = True
    cfg.task.command.contact_compliance = {"enabled": True, "force_cap_N": 80.}
    if args.wall_shift_y:
        for obj in cfg.task.objects:
            if obj.type == "wall":
                obj.pos[1] += args.wall_shift_y
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
    del state

    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        from isaaclab.sim import SimulationContext
        from scripts.utils.helpers import make_env_policy
        env, agent, _, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval(); env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.
        with torch.inference_mode():
            env.reset()
            if args.reset_drop:
                root_state = cmd.asset.data.root_state_w.clone()
                root_state[:, 2] -= args.reset_drop
                cmd.asset.write_root_state_to_sim(root_state)
        data = cmd.asset.data
        action = base.action_manager
        names = list(cmd.dataset.joint_names)
        idx = [names.index(name) for name in action.joint_names]
        q_reference = cmd._motion.joint_pos[0, 0, idx]
        reset_error = (data.joint_pos[0, action.joint_ids]-q_reference).abs()
        masses = data.default_mass.to(base.device)
        com = (data.body_com_pos_w*masses.unsqueeze(-1)).sum(dim=1)/masses.sum(dim=1, keepdim=True)
        right_ankle = data.body_pos_w[:, cmd.right_asset]
        print("RESET", {"root_z": data.root_pos_w[0, 2].item(),
              "joint_error_max": reset_error.max().item(),
              "joint_error_name": action.joint_names[reset_error.argmax().item()],
              "right_ankle_z": data.body_pos_w[0, cmd.asset.body_names.index(
                  "right_ankle_roll_link"), 2].item(),
              "mass_kg": masses[0].sum().item(),
              "com_minus_right_ankle_y_m": (com[0, 1]-right_ankle[0, 1]).item()},
              flush=True)
        with np.load(args.torque_file) as clip:
            torque_names = list(clip["joint_names"])
            torque = torch.as_tensor(clip["torques"][0, [torque_names.index(n)
                                  for n in action.joint_names]], device=base.device,
                                  dtype=q_reference.dtype)
        params = torch.as_tensor(matrix_np, device=base.device)
        hip_id = action.joint_names.index("right_hip_roll_joint")
        ankle_id = action.joint_names.index("right_ankle_roll_joint")
        waist_id = action.joint_names.index("waist_roll_joint")
        hip_pitch_id = action.joint_names.index("right_hip_pitch_joint")
        ankle_pitch_id = action.joint_names.index("right_ankle_pitch_joint")
        shoulder_roll_id = action.joint_names.index("left_shoulder_roll_joint")
        elbow_id = action.joint_names.index("left_elbow_joint")
        stiffness = data.joint_stiffness[:, action.joint_ids]
        joint_limits = data.joint_pos_limits[:, action.joint_ids].to(base.device)
        limit_lo, limit_hi = joint_limits[..., 0], joint_limits[..., 1]
        q_base = q_reference[None] + params[:, 0, None]*torque[None]/stiffness
        q_base[:, hip_id] += params[:, 1]
        q_base[:, ankle_id] += params[:, 2]
        q_base[:, waist_id] += params[:, 3]
        q_base[:, hip_pitch_id] += params[:, 8]
        q_base[:, ankle_pitch_id] += params[:, 9]
        q_base[:, shoulder_roll_id] += params[:, 14]
        q_base[:, elbow_id] += params[:, 17]
        origins = base.scene.env_origins
        initial_root = data.root_pos_w.clone()-origins
        initial_right_ankle = right_ankle.clone()-origins
        initial_com_support_y = com[:, 1]-right_ankle[:, 1]
        initial_com_support_x = com[:, 0]-right_ankle[:, 0]
        initial_hand_y = data.body_pos_w[:, cmd.hand_asset, 1].clone()-origins[:, 1]
        sensor = base.scene["ee_wall_contact_forces"]
        foot_sensor = cmd.foot_contact_sensor
        failures = torch.full((args.count,), -1, dtype=torch.long, device=base.device)
        max_left = torch.zeros(args.count, device=base.device)
        max_wall = torch.zeros_like(max_left)
        max_action = torch.zeros_like(max_left)
        max_drift = torch.zeros_like(max_left)
        max_x_drift = torch.zeros_like(max_left)
        max_foot_slip = torch.zeros_like(max_left)
        min_height = torch.full_like(max_left, 100.)
        max_joint_err = torch.zeros_like(max_left)
        max_limit_excess = torch.zeros_like(max_left)
        trace = []
        steps = round(args.seconds/base.step_dt)
        for step in range(steps):
            with torch.no_grad():
                if args.feedback_state == "com":
                    com = ((data.body_com_pos_w*masses.unsqueeze(-1)).sum(dim=1)
                           /masses.sum(dim=1, keepdim=True))
                    com_vel = ((data.body_com_lin_vel_w*masses.unsqueeze(-1)).sum(dim=1)
                               /masses.sum(dim=1, keepdim=True))
                    lateral = com[:, 1]-data.body_pos_w[:, cmd.right_asset, 1]
                    if args.com_target == "initial":
                        lateral = lateral-initial_com_support_y
                    velocity = com_vel[:, 1]-data.body_lin_vel_w[:, cmd.right_asset, 1]
                else:
                    lateral = data.root_pos_w[:, 1]-origins[:, 1]-initial_root[:, 1]
                    velocity = data.root_lin_vel_w[:, 1]
                if args.pitch_state == "com":
                    if args.feedback_state != "com":
                        com = ((data.body_com_pos_w*masses.unsqueeze(-1)).sum(dim=1)
                               /masses.sum(dim=1, keepdim=True))
                        com_vel = ((data.body_com_lin_vel_w*masses.unsqueeze(-1)).sum(dim=1)
                                   /masses.sum(dim=1, keepdim=True))
                    sagittal = com[:, 0]-data.body_pos_w[:, cmd.right_asset, 0]
                    if args.pitch_target == "initial":
                        sagittal = sagittal-initial_com_support_x
                    sagittal_velocity = (com_vel[:, 0]-
                                         data.body_lin_vel_w[:, cmd.right_asset, 0])
                else:
                    sagittal = data.root_pos_w[:, 0]-origins[:, 0]-initial_root[:, 0]
                    sagittal_velocity = data.root_lin_vel_w[:, 0]
                q_cmd = q_base.clone()
                q_cmd[:, hip_id] += params[:, 4]*lateral+params[:, 5]*velocity
                q_cmd[:, ankle_id] += params[:, 6]*lateral+params[:, 7]*velocity
                q_cmd[:, hip_pitch_id] += (params[:, 10]*sagittal
                                           +params[:, 11]*sagittal_velocity)
                q_cmd[:, ankle_pitch_id] += (params[:, 12]*sagittal
                                             +params[:, 13]*sagittal_velocity)
                hand_y_error = (data.body_pos_w[:, cmd.hand_asset, 1]
                                -origins[:, 1]-initial_hand_y)
                hand_vy = data.body_lin_vel_w[:, cmd.hand_asset, 1]
                q_cmd[:, shoulder_roll_id] += (params[:, 15]*hand_y_error
                                                +params[:, 16]*hand_vy)
                q_cmd[:, elbow_id] += (params[:, 18]*hand_y_error
                                       +params[:, 19]*hand_vy)
                active = failures < 0
                limit_excess = torch.maximum(limit_lo-q_cmd, q_cmd-limit_hi)
                limit_excess = limit_excess.clamp_min(0.).max(dim=1).values
                max_limit_excess = torch.where(active, torch.maximum(
                    max_limit_excess, limit_excess), max_limit_excess)
                if args.clamp_joint_targets:
                    q_cmd = q_cmd.clamp(limit_lo, limit_hi)
                raw_action = ((q_cmd-action.default_joint_pos[:, action.joint_ids])
                              /action.action_scaling)
                max_action = torch.where(active, torch.maximum(
                    max_action, raw_action.abs().max(dim=1).values), max_action)
                carry = TensorDict({"action": raw_action}, batch_size=[args.count],
                                   device=base.device)
                base.step(carry)
                wall_matrix = sensor.data.force_matrix_w
                wall = (wall_matrix.reshape(args.count, -1, 3).sum(dim=1).norm(dim=-1)
                        if wall_matrix is not None else torch.zeros_like(max_left))
                feet = foot_sensor.data.net_forces_w
                left = feet[:, cmd.left_foot_sensor_id].norm(dim=-1)
                root = data.root_pos_w-origins
                drift = (root[:, 1]-initial_root[:, 1]).abs()
                x_drift = (root[:, 0]-initial_root[:, 0]).abs()
                right_ankle_now = data.body_pos_w[:, cmd.right_asset]-origins
                foot_slip = (right_ankle_now[:, :2]-initial_right_ankle[:, :2]).norm(dim=-1)
                max_left = torch.where(active, torch.maximum(max_left, left), max_left)
                max_wall = torch.where(active, torch.maximum(max_wall, wall), max_wall)
                max_drift = torch.where(active, torch.maximum(max_drift, drift), max_drift)
                max_x_drift = torch.where(active, torch.maximum(max_x_drift, x_drift),
                                          max_x_drift)
                max_foot_slip = torch.where(active, torch.maximum(max_foot_slip, foot_slip),
                                             max_foot_slip)
                min_height = torch.where(active, torch.minimum(min_height, root[:, 2]),
                                         min_height)
                max_joint_err = torch.where(active, torch.maximum(max_joint_err,
                    (data.joint_pos[:, action.joint_ids]-q_cmd).abs().max(dim=1).values),
                    max_joint_err)
                failed = ((root[:, 2] < .55) | (data.projected_gravity_b[:, 2] > -.7)
                          | (drift > .15) | (x_drift > .15))
                failures[(failures < 0) & failed] = step+1
                if args.trace_index >= 0:
                    j = args.trace_index
                    if not 0 <= j < args.count:
                        raise ValueError("trace-index outside candidate range")
                    com_now = ((data.body_com_pos_w[j]*masses[j, :, None]).sum(dim=0)
                               /masses[j].sum())
                    com_vel_now = ((data.body_com_lin_vel_w[j]*masses[j, :, None]).sum(dim=0)
                                   /masses[j].sum())
                    errors = (data.joint_pos[j, action.joint_ids]-q_cmd[j]).abs()
                    trace.append({
                        "time_s": float((step+1)*base.step_dt),
                        "root_y_m": float(root[j, 1].item()),
                        "root_x_m": float(root[j, 0].item()),
                        "root_z_m": float(root[j, 2].item()),
                        "root_vy_m_s": float(data.root_lin_vel_w[j, 1].item()),
                        "root_vx_m_s": float(data.root_lin_vel_w[j, 0].item()),
                        "right_ankle_x_m": float(right_ankle_now[j, 0].item()),
                        "right_ankle_y_m": float(right_ankle_now[j, 1].item()),
                        "right_foot_slip_m": float(foot_slip[j].item()),
                        "hand_y_m": float((data.body_pos_w[j, cmd.hand_asset, 1]
                                            -origins[j, 1]).item()),
                        "com_minus_ankle_y_m": float((com_now[1]-
                            data.body_pos_w[j, cmd.right_asset, 1]).item()),
                        "com_minus_ankle_x_m": float((com_now[0]-
                            data.body_pos_w[j, cmd.right_asset, 0]).item()),
                        "com_vy_m_s": float(com_vel_now[1].item()),
                        "wall_N": float(wall[j].item()),
                        "left_foot_N": float(left[j].item()),
                        "right_foot_N": float(feet[j, cmd.right_foot_sensor_id].norm().item()),
                        "projected_gravity_z": float(data.projected_gravity_b[j, 2].item()),
                        "max_joint_error_rad": float(errors.max().item()),
                        "joint_limit_excess_rad": float(limit_excess[j].item()),
                        "worst_joint": action.joint_names[errors.argmax().item()],
                        "hip_target_rad": float(q_cmd[j, hip_id].item()),
                        "ankle_target_rad": float(q_cmd[j, ankle_id].item()),
                        "hip_pitch_target_rad": float(q_cmd[j, hip_pitch_id].item()),
                        "ankle_pitch_target_rad": float(q_cmd[j, ankle_pitch_id].item()),
                        "shoulder_roll_target_rad": float(q_cmd[j, shoulder_roll_id].item()),
                        "shoulder_roll_actual_rad": float(data.joint_pos[j,
                            action.joint_ids[shoulder_roll_id]].item()),
                        "elbow_target_rad": float(q_cmd[j, elbow_id].item()),
                        "elbow_actual_rad": float(data.joint_pos[j,
                            action.joint_ids[elbow_id]].item()),
                        "waist_target_rad": float(q_cmd[j, waist_id].item()),
                    })
            if (step+1) % 50 == 0:
                print("STEP", step+1, "upright", int((failures < 0).sum().item()),
                      flush=True)

        failure_cpu = failures.cpu().numpy()
        left_cpu = max_left.cpu().numpy()
        wall_cpu = max_wall.cpu().numpy()
        drift_cpu = max_drift.cpu().numpy()
        x_drift_cpu = max_x_drift.cpu().numpy()
        foot_slip_cpu = max_foot_slip.cpu().numpy()
        action_cpu = max_action.cpu().numpy()
        height_cpu = min_height.cpu().numpy()
        err_cpu = max_joint_err.cpu().numpy()
        excess_cpu = max_limit_excess.cpu().numpy()
        rows = []
        for j, candidate in enumerate(matrix_np):
            survival = args.seconds if failure_cpu[j] < 0 else failure_cpu[j]*base.step_dt
            gate = (failure_cpu[j] < 0 and left_cpu[j] < 5 and wall_cpu[j] < 5
                    and drift_cpu[j] < .05 and x_drift_cpu[j] < .05
                    and foot_slip_cpu[j] < .02 and action_cpu[j] <= 10.
                    and excess_cpu[j] < .02)
            score = (survival - 4*max(0, drift_cpu[j]-.05)
                     - 4*max(0, x_drift_cpu[j]-.05)
                     - 4*max(0, foot_slip_cpu[j]-.02)
                     - .02*left_cpu[j] - .02*wall_cpu[j]
                     - 2*excess_cpu[j])
            row = {"index": j, **dict(zip(PARAMS, map(float, candidate))),
                   "survival_s": float(survival), "gate_pass": bool(gate),
                   "max_wall_N": float(wall_cpu[j]), "max_left_foot_N": float(left_cpu[j]),
                   "max_root_y_drift_m": float(drift_cpu[j]),
                   "max_root_x_drift_m": float(x_drift_cpu[j]),
                   "max_right_foot_slip_m": float(foot_slip_cpu[j]),
                   "min_root_height_m": float(height_cpu[j]),
                   "max_abs_action": float(action_cpu[j]),
                   "max_joint_error_rad": float(err_cpu[j]), "score": float(score)}
            row["max_joint_limit_excess_rad"] = float(excess_cpu[j])
            rows.append(row)
        ranked = sorted(rows, key=lambda row: (row["gate_pass"], row["score"]),
                        reverse=True)
        args.output.mkdir(parents=True)
        with (args.output/"candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        if trace:
            with (args.output/"trace.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
                writer.writeheader(); writer.writerows(trace)
        summary = {"checkpoint": str(args.checkpoint.resolve()),
                   "dataset": args.dataset, "motion_id": args.motion_id,
                   "seconds": args.seconds,
                   "seed": args.seed, "count": args.count,
                   "reset_drop_m": args.reset_drop,
                   "wall_shift_y_m": args.wall_shift_y,
                   "feedback_state": args.feedback_state,
                   "com_target": args.com_target if args.feedback_state == "com" else None,
                   "pitch_state": args.pitch_state,
                   "pitch_target": args.pitch_target if args.pitch_state == "com" else None,
                   "gate": "3 s upright; wall and swing-foot load <5 N; "
                           "root x/y drift <5 cm; right-foot slip <2 cm; "
                           "|action|<=10; joint-limit excess <0.02 rad",
                   "clamp_joint_targets": args.clamp_joint_targets,
                   "passed": sum(row["gate_pass"] for row in rows),
                   "best": {"index": ranked[0]["index"],
                            "parameters": {k: ranked[0][k] for k in PARAMS},
                            "metrics": {k: v for k, v in ranked[0].items()
                                        if k not in ("index", *PARAMS)}},
                   "top": ranked[:12]}
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print("SEARCH", json.dumps({"passed": summary["passed"],
                                   "best": summary["best"]}), flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()
