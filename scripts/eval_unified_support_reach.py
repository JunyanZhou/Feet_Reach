"""Paired Isaac evaluation of root, support-hand, and stance-foot target response."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import active_adaptation.learning
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--foot-point-index", type=int, default=0,
                        help="Index of a verified left-foot workspace target")
    parser.add_argument("--settle", type=float, default=1.)
    parser.add_argument("--reach", type=float, default=2.)
    parser.add_argument("--hold", type=float, default=1.)
    parser.add_argument("--aux-force", action="store_true",
                        help="Apply randomized forces at the hand, torso, and pelvis")
    parser.add_argument("--aux-force-min", type=float, default=5.)
    parser.add_argument("--aux-force-max", type=float, default=20.)
    parser.add_argument("--direct-ik", action="store_true",
                        help="After settling, command the validated IK joint target directly")
    parser.add_argument("--ik-residual", action="store_true",
                        help="Add a fixed IK joint correction to the running student action")
    parser.add_argument("--ik-gain", type=float, default=1.,
                        help="Scale the IK action correction in --ik-residual mode")
    parser.add_argument("--include-negative-x", action="store_true",
                        help="Also test held-out -15 mm root and hand x targets")
    parser.add_argument("--include-negative-axes", action="store_true",
                        help="Also test -15 mm root y and hand z targets")
    args = parser.parse_args()
    if args.direct_ik and args.ik_residual:
        parser.error("Choose either --direct-ik or --ik-residual")
    if (args.include_negative_x or args.include_negative_axes) and (
            args.direct_ik or args.ik_residual):
        parser.error("Negative-axis extras currently apply to student policy trials")
    if not 0 < args.ik_gain <= 10:
        parser.error("--ik-gain must be in (0, 10]")
    if not 0 <= args.aux_force_min <= args.aux_force_max <= 20:
        parser.error("Auxiliary force range must be within 0..20 N")
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    if not cfg.task.command._target_.endswith(".UnifiedSupportReachCommand"):
        raise ValueError("Expected a unified support reaching checkpoint")
    cfg.algo.geometry_points = 0
    cfg.checkpoint_path = str(checkpoint)
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = 1000000
    cfg.task.termination = {}
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    if args.aux_force:
        cfg.task.command.aux_support_force = {
            "enabled": True, "train_only": False,
            "body_names": ["left_hand_mimic", "torso_link", "pelvis"],
            "application_radius_m": .02,
            "force_range": [args.aux_force_min, args.aux_force_max],
            "ramp_up_range": [10, 20], "hold_range": [15, 50],
            "ramp_down_range": [10, 20], "rest_range": [10, 20],
            "zero_prob": 0.,
        }
    elif "aux_support_force" in cfg.task.command:
        cfg.task.command.aux_support_force.enabled = False
    cfg.task.command.dataset.mem_paths = ["wall_foot_reach/val"]
    cfg.task.command.dataset.path_weights = [1.]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.command.disable_motion_finish = True
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
    del state

    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        from isaaclab.sim import SimulationContext
        from active_adaptation.utils.math import quat_apply, quat_apply_inverse
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval()
        env.base_env.eval()
        agent.eval()
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        workspace = __import__("numpy").load(ROOT / "artifacts/foot_reach/workspace.npz")
        points = workspace["points_root"]
        if not 0 <= args.foot_point_index < len(points):
            raise ValueError(f"foot-point-index must be in [0, {len(points)-1}]")
        anchor = torch.as_tensor(points[args.foot_point_index],
                                 device=base.device, dtype=torch.float32)
        dt = base.step_dt
        trials = [
            ("baseline", None, (0., 0., 0.)),
            ("root_forward", "root", (.015, 0., 0.)),
            ("root_side", "root", (0., .015, 0.)),
            ("hand_forward", "left_hand", (.015, 0., 0.)),
            ("hand_up", "left_hand", (0., 0., .015)),
            ("stance_forward", "right_foot", (.01, 0., 0.)),
            ("stance_side", "right_foot", (0., -.01, 0.)),
        ]
        if args.include_negative_x:
            trials.extend((
                ("root_backward", "root", (-.015, 0., 0.)),
                ("hand_backward", "left_hand", (-.015, 0., 0.)),
            ))
        if args.include_negative_axes:
            trials.extend((
                ("root_opposite_side", "root", (0., -.015, 0.)),
                ("hand_down", "left_hand", (0., 0., -.015)),
            ))
        results = []

        def measure():
            data = cmd.asset.data
            sole_r = data.body_pos_w[:, cmd.right_asset] + quat_apply(
                data.body_quat_w[:, cmd.right_asset], cmd.sole_offset.expand(1, -1))
            sole_l = data.body_pos_w[:, cmd.left_asset] + quat_apply(
                data.body_quat_w[:, cmd.left_asset], cmd.sole_offset.expand(1, -1))
            foot_target_w = data.root_pos_w + quat_apply(data.root_quat_w, cmd.target_foot_pos_b())
            hand_force = cmd.hand_contact_sensor.data.net_forces_w
            hand_load = (hand_force[:, cmd.hand_sensor_id].norm(dim=-1).item()
                         if hand_force is not None else 0.)
            result = {
                "root_error_m": (data.root_pos_w - cmd.goal_world("root")).norm(dim=-1).item(),
                "hand_error_m": (data.body_pos_w[:, cmd.hand_asset]
                                 - cmd.goal_world("left_hand")).norm(dim=-1).item(),
                "stance_error_m": (sole_r - cmd.goal_world("right_foot")).norm(dim=-1).item(),
                "swing_error_m": (sole_l - foot_target_w).norm(dim=-1).item(),
                "hand_load_N": hand_load,
                "aux_force_N": cmd.aux_force_ramp.force.norm(dim=-1).item(),
                "root_height_m": data.root_pos_w[0, 2].item(),
                "gravity_z": data.projected_gravity_b[0, 2].item(),
            }
            for name, position in (("root", data.root_pos_w[0]),
                                   ("hand", data.body_pos_w[0, cmd.hand_asset]),
                                   ("stance", sole_r[0])):
                for axis, value in zip("xyz", position):
                    result[f"{name}_{axis}_m"] = value.item()
            return result

        def step(is_init, direct_action=None, residual_action=None):
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            normalized = normalizer(raw)
            carry = normalized.select("policy")
            carry["is_init"] = torch.tensor([[is_init]], dtype=torch.bool, device=base.device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                if direct_action is None:
                    actor(carry)
                    if residual_action is not None:
                        carry["action"] += residual_action
                else:
                    carry["action"] = direct_action
                base.step(carry)
            return measure()

        if args.direct_ik or args.ik_residual:
            teaching = ROOT/"artifacts/unified_support_reach/teaching"
            manifest = json.loads((teaching/"manifest.json").read_text())
            direct_actions = {}
            ik_deltas = {}
            action_manager = base.action_manager
            for record in manifest["clips"]:
                if record["goal"] not in ("root", "left_hand"):
                    continue
                if max(record["offset_m"]) != .015:
                    continue
                clip = __import__("numpy").load(teaching/"motions"/record["split"]
                                                 /record["name"])
                names = list(clip["joint_names"])
                order = [names.index(name) for name in action_manager.joint_names]
                target_q = torch.tensor(clip["dof_pos"][-1, order], device=base.device)
                direct_actions[(record["goal"], tuple(record["offset_m"]))] = (
                    (target_q-action_manager.default_joint_pos[0,
                         action_manager.joint_ids])/action_manager.action_scaling).unsqueeze(0)
            anchor_clip = __import__("numpy").load(
                teaching/"motions"/manifest["clips"][0]["split"]
                /manifest["clips"][0]["name"])
            names = list(anchor_clip["joint_names"])
            order = [names.index(name) for name in action_manager.joint_names]
            q0 = torch.tensor(anchor_clip["dof_pos"][0, order], device=base.device)
            direct_actions[(None, (0., 0., 0.))] = (
                (q0-action_manager.default_joint_pos[0,
                    action_manager.joint_ids])/action_manager.action_scaling).unsqueeze(0)
            for key, absolute_action in direct_actions.items():
                ik_deltas[key] = args.ik_gain*(
                    absolute_action-direct_actions[(None, (0., 0., 0.))])
            trials = [trial for trial in trials if trial[1] in (None, "root", "left_hand")]

        for label, goal, delta in trials:
            env.set_seed(args.seed)
            env.reset()
            base.episode_length_buf.zero_()
            cmd.set_target_foot_pos_b(anchor)
            reference = {name: cmd.goal_world(name).clone()
                         for name in ("root", "left_hand", "right_foot")}
            first = True
            for _ in range(round(args.settle / dt)):
                step(first)
                first = False
            settled = measure()
            if goal is not None:
                cmd.set_goal_world(goal, reference[goal] + torch.tensor(delta, device=base.device))
            samples = []
            trajectory = []
            failure = None
            direct_action = (direct_actions[(goal, tuple(delta))]
                             if args.direct_ik else None)
            residual_action = (ik_deltas[(goal, tuple(delta))]
                               if args.ik_residual else None)
            for i in range(round((args.reach + args.hold) / dt)):
                m = step(first, direct_action, residual_action)
                trajectory.append(m)
                first = False
                if (m["root_height_m"] < .55 or m["gravity_z"] > -.7
                        or not all(__import__("math").isfinite(v) for v in m.values())):
                    failure = f"instability_at_{i*dt:.2f}s"
                    break
                for body, limit in (("root", .20), ("hand", .15), ("stance", .12)):
                    drift = sum((m[f"{body}_{axis}_m"]
                                 - settled[f"{body}_{axis}_m"])**2
                                for axis in "xyz")**.5
                    if drift > limit:
                        failure = f"{body}_drift_{drift:.3f}m_at_{i*dt:.2f}s"
                        break
                if failure:
                    break
                if i >= round(args.reach / dt):
                    samples.append(m)
            result = {"label": label, "target": goal, "offset_m": delta,
                      "stable": failure is None, "failure": failure,
                      "hold_samples": len(samples)}
            result["peak_aux_force_N"] = max(m["aux_force_N"] for m in trajectory)
            result["peak_hand_load_N"] = max(m["hand_load_N"] for m in trajectory)
            result["min_hand_load_N"] = min(m["hand_load_N"] for m in trajectory)
            result["failure_aux_force_N"] = (trajectory[-1]["aux_force_N"]
                                              if failure is not None else None)
            for key in ("root_error_m", "hand_error_m", "stance_error_m",
                        "swing_error_m", "hand_load_N", "aux_force_N",
                        *(f"{name}_{axis}_m" for name in ("root", "hand", "stance")
                          for axis in "xyz")):
                result[f"mean_{key}"] = (sum(m[key] for m in samples) / len(samples)
                                          if samples else None)
            result["hand_contact_fraction"] = (
                sum(m["hand_load_N"] > 1. for m in samples) / len(samples)
                if samples else None)
            results.append(result)
            print("TRIAL", json.dumps(result), flush=True)

        report = {"checkpoint": str(checkpoint), "seed": args.seed,
                  "foot_point_index": args.foot_point_index,
                  "foot_target_root_xyz_m": anchor.tolist(),
                  "aux_force_enabled": args.aux_force,
                  "aux_force_range_N": ([args.aux_force_min, args.aux_force_max]
                                        if args.aux_force else None),
                  "control_mode": ("direct IK joint target" if args.direct_ik else
                                   "student plus fixed IK action correction" if args.ik_residual
                                   else "student policy"),
                  "ik_gain": args.ik_gain if args.ik_residual else None,
                  "protocol": ("same validation anchor, world-fixed offsets, "
                               f"{args.settle:g}s settle + {args.reach:g}s reach + "
                               f"{args.hold:g}s hold"),
                  "note": "Local screen; instability includes root/hand/stance drift limits of 0.20/0.15/0.12 m. Not task success or general feasibility proof.",
                  "trials": results}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("UNIFIED_EVAL_COMPLETE", args.output, flush=True)
        SimulationContext.instance()._disable_app_control_on_stop_handle = True
        env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()
