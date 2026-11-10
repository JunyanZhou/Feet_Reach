"""Isaac closed-loop screen of unloading, withdrawing, and replacing wall support."""

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
import active_adaptation.learning
from isaaclab.app import AppLauncher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--trajectory", type=Path,
                   default=ROOT/"artifacts/unified_support_reach/support_release.npz")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("policy", "residual", "direct", "blend"),
                   default="residual")
    p.add_argument("--gain", type=float, default=1.)
    p.add_argument("--blend-weight", type=float, default=.5)
    p.add_argument("--blend-time", type=float, default=1.)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0. <= args.gain <= 5.:
        raise ValueError("Gain must be in [0, 5]")
    if not 0. <= args.blend_weight <= 1. or args.blend_time <= 0.:
        raise ValueError("Blend weight must be in [0, 1] and time positive")
    checkpoint = args.checkpoint.resolve(strict=True)
    trajectory = np.load(args.trajectory.resolve(strict=True))
    times = trajectory["time_s"]
    fraction = trajectory["hand_release_fraction"]
    if len(times) != 251 or not np.allclose(np.diff(times), .02):
        raise ValueError("Expected a 251-frame, 50 Hz support-release trajectory")
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
        workspace = np.load(ROOT/"artifacts/foot_reach/workspace.npz")
        cmd.set_target_foot_pos_b(torch.tensor(workspace["points_root"][0],
                                               device=base.device))
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm(); normalizer.eval()
        manager = base.action_manager
        names = list(trajectory["joint_names"])
        index = [names.index(name) for name in manager.joint_names]
        q = torch.tensor(trajectory["dof_pos"][:, index],
                         dtype=torch.float32, device=base.device)
        q0 = q[0]
        residual = args.gain*(q-q0)/manager.action_scaling
        direct = (q-manager.default_joint_pos[0, manager.joint_ids])/manager.action_scaling
        sensor = base.scene["ee_wall_contact_forces"]
        right_id = cmd.right_foot_sensor_id
        left_id = cmd.left_foot_sensor_id
        dt = base.step_dt
        if abs(dt-.02) > 1e-8:
            raise ValueError(f"Expected 20 ms controller step, got {dt}")

        def measure():
            data = cmd.asset.data
            matrix = sensor.data.force_matrix_w
            wall = (matrix.reshape(1, -1, 3).sum(dim=1).norm(dim=-1).item()
                    if matrix is not None else 0.)
            feet = cmd.foot_contact_sensor.data.net_forces_w
            sole_l = (data.body_pos_w[:, cmd.left_asset]
                      + quat_apply(data.body_quat_w[:, cmd.left_asset],
                                   cmd.sole_offset.expand(1, -1)))
            sole_target = cmd.goal_world("left_foot")
            return {
                "wall_load_N": wall,
                "wall_peak_physics_N": (cmd.contact_peak_N.max().item()
                                        if hasattr(cmd, "contact_peak_N") else wall),
                "right_foot_load_N": feet[:, right_id].norm(dim=-1).item(),
                "left_foot_load_N": feet[:, left_id].norm(dim=-1).item(),
                "hand_y_m": data.body_pos_w[0, cmd.hand_asset, 1].item(),
                "root_height_m": data.root_pos_w[0, 2].item(),
                "root_y_m": data.root_pos_w[0, 1].item(),
                "projected_gravity_z": data.projected_gravity_b[0, 2].item(),
                "swing_error_m": (sole_l-sole_target).norm(dim=-1).item(),
            }

        def step(is_init, correction=None, direct_action=None, blend_weight=None):
            raw = TensorDict({}, batch_size=[1], device=base.device)
            base._compute_observation(raw)
            carry = normalizer(raw).select("policy")
            carry["is_init"] = torch.tensor([[is_init]], dtype=torch.bool,
                                            device=base.device)
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                if direct_action is None or blend_weight is not None:
                    actor(carry)
                    if correction is not None:
                        carry["action"] += correction[None]
                    if blend_weight is not None:
                        carry["action"] = ((1.-blend_weight)*carry["action"]
                                           + blend_weight*direct_action[None])
                else:
                    carry["action"] = direct_action[None]
                base.step(carry)
            return measure()

        first = True
        for _ in range(round(1./dt)):
            step(first)
            first = False
        settled = measure()
        hand_ref = cmd.goal_world("left_hand").clone()
        cmd.set_goal_world("left_foot", cmd.goal_world("left_foot").clone())
        distance = json.loads(trajectory["report_json"].item())["distance_m"]
        rows = []
        failure = None
        for i, t in enumerate(times):
            cmd.set_goal_world("left_hand", hand_ref + torch.tensor(
                [0., -distance*float(fraction[i]), 0.], device=base.device))
            m = step(first,
                     residual[i] if args.mode == "residual" else None,
                     direct[i] if args.mode in ("direct", "blend") else None,
                     (args.blend_weight*min(1., float(t)/args.blend_time)
                      if args.mode == "blend" else None))
            first = False
            m.update(time_s=float(t), release_fraction=float(fraction[i]),
                     hand_withdrawal_m=settled["hand_y_m"]-m["hand_y_m"])
            rows.append(m)
            if (m["root_height_m"] < .55 or m["projected_gravity_z"] > -.7
                    or abs(m["root_y_m"]-settled["root_y_m"]) > .2
                    or not all(np.isfinite(v) for v in m.values())):
                failure = f"instability_at_{t:.2f}s"
                break
        released = [r for r in rows if 2. <= r["time_s"] < 3.]
        returned = [r for r in rows if 4.8 <= r["time_s"] <= 5.]
        summary = {
            "checkpoint": str(checkpoint), "trajectory": str(args.trajectory.resolve()),
            "mode": args.mode, "gain": args.gain,
            "blend_weight": args.blend_weight if args.mode == "blend" else None,
            "blend_time": args.blend_time if args.mode == "blend" else None,
            "seed": args.seed,
            "frames_completed": len(rows), "stable": failure is None, "failure": failure,
            "min_root_height_m": min(r["root_height_m"] for r in rows),
            "max_swing_error_m": max(r["swing_error_m"] for r in rows),
            "max_hand_withdrawal_m": max(r["hand_withdrawal_m"] for r in rows),
            "released_phase_mean_hand_withdrawal_m": (
                float(np.mean([r["hand_withdrawal_m"] for r in released])) if released else None),
            "released_phase_mean_wall_load_N": (
                float(np.mean([r["wall_load_N"] for r in released])) if released else None),
            "released_phase_contact_fraction": (
                float(np.mean([r["wall_load_N"] > 1. for r in released])) if released else None),
            "released_phase_wall_peak_N": (
                max(r["wall_peak_physics_N"] for r in released) if released else None),
            "returned_phase_mean_wall_load_N": (
                float(np.mean([r["wall_load_N"] for r in returned])) if returned else None),
            "returned_phase_contact_fraction": (
                float(np.mean([r["wall_load_N"] > 1. for r in returned])) if returned else None),
            "returned_phase_wall_peak_N": (
                max(r["wall_peak_physics_N"] for r in returned) if returned else None),
            "note": "Isaac local screen; residual, direct, and blend modes use teacher actions and are not student-only success.",
        }
        summary["support_cycle_success"] = bool(
            failure is None and len(rows) == len(times)
            and summary["released_phase_mean_hand_withdrawal_m"] is not None
            and summary["released_phase_mean_hand_withdrawal_m"] >= .03
            and summary["released_phase_contact_fraction"] <= .1
            and summary["returned_phase_contact_fraction"] is not None
            and summary["returned_phase_contact_fraction"] >= .8
            and summary["max_swing_error_m"] <= .02)
        args.output.mkdir(parents=True)
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        with (args.output/"samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print("SUPPORT_RELEASE", json.dumps(summary), flush=True)
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
