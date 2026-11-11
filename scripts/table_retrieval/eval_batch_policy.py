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
from scripts.table_retrieval.contact_feedback import ACTION_JOINTS, contact_correction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="G1/G1_table_retrieval_supported_rl")
    p.add_argument("--dataset", default="table_retrieval_supported/train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=510)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-top", type=int, default=8)
    p.add_argument("--mode", choices=("sample", "mean"), default="sample")
    p.add_argument("--object-x", type=float, default=None)
    p.add_argument("--object-y", type=float, default=None)
    p.add_argument("--object-width-x", type=float, default=None)
    p.add_argument("--object-width-y", type=float, default=None)
    p.add_argument("--object-height", type=float, default=None)
    p.add_argument("--object-mass", type=float, default=None)
    p.add_argument("--object-friction", type=float, default=None)
    p.add_argument("--without-table-top", action="store_true")
    p.add_argument("--noise-scale", type=float, default=None,
                   help="Override loaded actor std for sampled-action evaluation")
    p.add_argument("--residual-json", type=Path, default=None,
                   help="Learned three-phase leg residual parameters from CEM")
    p.add_argument("--foot-follow-after", type=float, default=None,
                   help="After this time, set the policy's left-foot goal relative to the moving object")
    p.add_argument("--foot-follow-offset", type=float, nargs=3, default=(.05, .10, -.03),
                   metavar=("DX", "DY", "DZ"))
    p.add_argument("--foot-follow-sweep", type=Path, default=None,
                   help="JSON [num_envs,4] of per-env [start_s, dx, dy, dz] for a goal sweep")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sweep_values = None
    if args.foot_follow_sweep is not None:
        sweep_values = np.asarray(json.loads(args.foot_follow_sweep.read_text()), dtype=np.float32)
        if sweep_values.shape != (args.num_envs, 4):
            raise ValueError("foot-follow-sweep must have shape [num_envs,4]")
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
    if args.without_table_top:
        cfg.task.objects = [obj for obj in cfg.task.objects if obj.name != "table_top"]
    target_size = None
    for obj in cfg.task.objects:
        if obj.name == "target_object":
            if args.object_x is not None:
                obj.pos[0] = args.object_x
            if args.object_y is not None:
                obj.pos[1] = args.object_y
            if args.object_width_x is not None:
                obj.size[0] = args.object_width_x
            if args.object_width_y is not None:
                obj.size[1] = args.object_width_y
            if args.object_height is not None:
                obj.size[2] = args.object_height
                obj.pos[2] = args.object_height/2 + .001
            if args.object_mass is not None:
                obj.mass = args.object_mass
            if args.object_friction is not None:
                obj.friction = args.object_friction
            target_size = tuple(float(v) for v in obj.size)
    cfg.algo.symmetry_augmentation = False
    if args.noise_scale is not None:
        cfg.algo.load_noise_scale = args.noise_scale
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
        residual = None
        feedback_params = None
        feedback_version = 1
        if args.residual_json is not None:
            payload = json.loads(args.residual_json.read_text())
            residual = torch.tensor(payload["parameters"], device=base.device)
            residual_indices = [base.action_manager.joint_names.index(name)
                                for name in payload["joint_names"]]
            if residual.shape not in ((3, len(residual_indices)),
                                      (4, len(residual_indices)),
                                      (5, len(residual_indices)),
                                      (6, len(residual_indices)),
                                      (7, len(residual_indices))):
                raise ValueError("Expected three to seven phases of residual joint actions")
            if "contact_feedback" in payload:
                feedback_version = int(payload["contact_feedback"].get("version", 1))
                feedback_params = torch.tensor(
                    payload["contact_feedback"]["parameters"], device=base.device)
                if feedback_params.shape != (5,):
                    raise ValueError("contact_feedback.parameters must contain five values")
                feedback_indices = [base.action_manager.joint_names.index(name)
                                    for name in ACTION_JOINTS]
        def smooth(v):
            v = max(0.0, min(1.0, v))
            return v*v*(3.0-2.0*v)
        def phase_weights(t):
            if residual is not None and residual.shape[0] in (5, 6, 7):
                weights = (smooth(t/.30)*(1-smooth((t-1.5)/.5)),
                        smooth((t-1.2)/.5)*(1-smooth((t-4.2)/.5)),
                        smooth((t-3.9)/.5),
                        smooth((t-3.5)/.4)*(1-smooth((t-5.6)/.4)),
                        smooth((t-5.3)/.4))
                if residual.shape[0] == 7:
                    return weights + (smooth((t-6.1)/.4), smooth((t-7.2)/.5))
                return weights + (smooth((t-6.1)/.4),) if residual.shape[0] == 6 else weights
            if residual is not None and residual.shape[0] == 4:
                return (smooth(t/.30)*(1-smooth((t-1.5)/.5)),
                        smooth((t-1.2)/.5)*(1-smooth((t-4.2)/.5)),
                        smooth((t-3.9)/.5)*(1-smooth((t-5.8)/.4)),
                        smooth((t-5.6)/.4))
            return (smooth(t/.30)*(1-smooth((t-1.5)/.5)),
                    smooth((t-1.2)/.5)*(1-smooth((t-4.2)/.5)),
                    smooth((t-3.9)/.5))
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        env.reset()
        base.episode_length_buf.zero_()
        origins = base.scene.env_origins.detach().cpu().numpy()
        obj = base.scene["target_object"]
        sweep_tensor = (torch.tensor(sweep_values, device=base.device)
                        if sweep_values is not None else None)
        initial_object_x = obj.data.root_pos_w[:, 0].clone()
        hand_sensor = base.scene["ee_contact_forces"]
        hand_id = hand_sensor.find_bodies("left_hand_mimic")[0][0]
        records = {k: [] for k in ("root_pos", "root_quat", "joint_pos", "object_pos",
                                   "object_quat", "left_hand_pos", "left_foot_pos",
                                   "right_foot_pos", "hand_force", "right_foot_force", "action")}
        def append(key, tensor):
            records[key].append(tensor.detach().cpu().numpy().astype(np.float32, copy=True))
        exploration = ExplorationType.RANDOM if args.mode == "sample" else ExplorationType.MODE
        for i in range(args.steps):
            if sweep_tensor is not None:
                active = (i*base.step_dt >= sweep_tensor[:, 0])
                relative_goal = obj.data.root_pos_w + sweep_tensor[:, 1:4]
                reference_goal = cmd._reference_world("left_foot")
                cmd.set_goal_world("left_foot", torch.where(
                    active[:, None], relative_goal, reference_goal))
            elif args.foot_follow_after is not None and i*base.step_dt >= args.foot_follow_after:
                foot_goal = obj.data.root_pos_w + torch.tensor(
                    args.foot_follow_offset, device=base.device)
                cmd.set_goal_world("left_foot", foot_goal)
            raw = TensorDict({}, batch_size=[args.num_envs], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.full((args.num_envs, 1), i == 0,
                                           device=base.device, dtype=torch.bool)
            with torch.inference_mode(), set_exploration_type(exploration):
                actor(carry)
                if residual is not None:
                    weights = torch.tensor(phase_weights(i*base.step_dt), device=base.device)
                    carry["action"][:, residual_indices] += (weights[:, None]*residual).sum(0)
                if feedback_params is not None:
                    correction = contact_correction(
                        feedback_params, cmd.asset.data.body_pos_w[:, cmd.left_asset],
                        obj.data.root_pos_w, initial_object_x, i*base.step_dt,
                        version=feedback_version)
                    carry["action"][:, feedback_indices] += correction
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
        assert target_size is not None
        hx, hy, hz = (v/2 for v in target_size)
        quats = arrays["object_quat"][-1]
        qw, qx, qy, qz = (quats[:, j] for j in range(4))
        back_extent_x = (np.abs(1-2*(qy*qy+qz*qz))*hx
                         + np.abs(2*(qx*qy-qz*qw))*hy
                         + np.abs(2*(qx*qz+qy*qw))*hz)
        dt = float(base.step_dt)
        x0 = arrays["object_pos"][0, :, 0]
        x1 = arrays["object_pos"][-1, :, 0]
        root_z = arrays["root_pos"][:, :, 2]
        hand_mag = np.linalg.norm(arrays["hand_force"], axis=-1)
        hand_up = arrays["hand_force"][:, :, 2] > 5.0
        right_mag = np.linalg.norm(arrays["right_foot_force"], axis=-1)
        right_up = arrays["right_foot_force"][:, :, 2] > 100.0
        rows = []
        for j in range(args.num_envs):
            hand_s = float(np.count_nonzero(hand_mag[:, j] > 10) * dt)
            stance_s = float(np.count_nonzero(right_mag[:, j] > 100) * dt)
            hand_up_s = float(np.count_nonzero(hand_up[:, j]) * dt)
            support_overlap = hand_up[:, j] & right_up[:, j] & (root_z[:, j] > .50)
            support_overlap_s = float(np.count_nonzero(support_overlap) * dt)
            object_moving_out = np.r_[False, np.diff(arrays["object_pos"][:, j, 0]) < -.0001]
            foot_delta = arrays["left_foot_pos"][:, j] - arrays["object_pos"][:, j]
            foot_near_object = ((np.abs(foot_delta[:, 0]) < .18)
                                & (np.abs(foot_delta[:, 1]) < .15)
                                & (np.abs(foot_delta[:, 2]) < .20))
            supported_foot_push_s = float(np.count_nonzero(
                object_moving_out & foot_near_object & support_overlap) * dt)
            row = dict(env_id=j, checkpoint=str(args.checkpoint.resolve()), seed=args.seed,
                       mode=args.mode, duration_s=(args.steps - 1) * dt,
                       foot_follow_after_s=(float(sweep_values[j, 0]) if sweep_values is not None
                                            else args.foot_follow_after),
                       foot_follow_offset_m=(sweep_values[j, 1:4].tolist() if sweep_values is not None
                                             else args.foot_follow_offset
                                             if args.foot_follow_after is not None else None),
                       residual_json=str(args.residual_json.resolve()) if args.residual_json else None,
                       table_top_present=not args.without_table_top,
                       object_config=dict(x_m=args.object_x, y_m=args.object_y,
                                          width_x_m=args.object_width_x,
                                          width_y_m=args.object_width_y,
                                          height_m=args.object_height,
                                          mass_kg=args.object_mass,
                                          friction=args.object_friction),
                       object_initial_x_m=float(x0[j]), object_final_x_m=float(x1[j]),
                       object_final_backmost_x_m=float(x1[j]+back_extent_x[j]),
                       object_final_x_half_extent_m=float(back_extent_x[j]),
                       object_outward_displacement_m=float(x0[j]-x1[j]),
                       object_center_past_front_edge=bool(x1[j] < boundary),
                       front_face_protrusion_m=float(boundary-(x1[j]-back_extent_x[j])),
                       object_outside_table=bool(x1[j]+back_extent_x[j] < boundary),
                       root_min_height_m=float(root_z[:, j].min()),
                       root_final_height_m=float(root_z[-1, j]),
                       hand_support_duration_s=hand_s,
                       hand_upward_support_duration_s=hand_up_s,
                       simultaneous_hand_and_foot_support_s=support_overlap_s,
                       supported_foot_push_s=supported_foot_push_s,
                       right_stance_duration_s=stance_s)
            row["task_success"] = bool(row["object_outside_table"]
                                       and row["root_min_height_m"] > .50
                                       and hand_up_s > .5 and stance_s > .5
                                       and support_overlap_s > 5.0
                                       and supported_foot_push_s > .08)
            row["front_edge_crossing_success"] = bool(
                row["object_center_past_front_edge"]
                and row["root_min_height_m"] > .50
                and support_overlap_s > 5.0
                and supported_foot_push_s > .08)
            rows.append(row)
        rows.sort(key=lambda r: (r["task_success"], r["front_edge_crossing_success"],
                                 r["root_min_height_m"] > .50,
                                 r["supported_foot_push_s"],
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
