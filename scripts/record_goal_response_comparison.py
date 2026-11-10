"""Record matched headless Isaac G1 rollouts for a root/hand/foot comparison video."""

import argparse
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
import active_adaptation.learning
from isaaclab.app import AppLauncher


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.algo.geometry_points = 0
    cfg.checkpoint_path = str(args.checkpoint.resolve())
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
        env.eval(); env.base_env.eval(); agent.eval(); env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm(); normalizer.eval()
        workspace = np.load(ROOT/"artifacts/foot_reach/workspace.npz")
        anchor = torch.as_tensor(workspace["points_root"][0], dtype=torch.float32,
                                 device=base.device)
        foot_goal = torch.as_tensor(workspace["points_root"][20], dtype=torch.float32,
                                    device=base.device)
        frames = round(4./base.step_dt)
        body_names = list(cmd.asset.body_names)
        result = {}
        for trial in ("root_x", "hand_x", "foot_reach"):
            with torch.inference_mode():
                env.reset()
            if hasattr(cmd, "set_goal_world"):
                cmd.set_goal_world("root", None)
                cmd.set_goal_world("left_hand", None)
            cmd.set_target_foot_pos_b(anchor)
            root0 = cmd.asset.data.root_pos_w[0].clone()
            hand0 = cmd.asset.data.body_pos_w[0, cmd.hand_asset].clone()
            positions = []
            root = []
            hand = []
            foot = []
            foot_target = []
            wall_load = []
            target_root = []
            target_hand = []
            with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
                for i in range(frames):
                    active = i*base.step_dt >= 1.
                    if trial == "root_x" and hasattr(cmd, "set_goal_world"):
                        goal = root0 + torch.tensor([.015, 0., 0.], device=base.device)
                        cmd.set_goal_world("root", goal if active else None)
                    if trial == "hand_x" and hasattr(cmd, "set_goal_world"):
                        goal = hand0 + torch.tensor([.015, 0., 0.], device=base.device)
                        cmd.set_goal_world("left_hand", goal if active else None)
                    cmd.set_target_foot_pos_b(foot_goal if trial == "foot_reach" and active
                                              else anchor)
                    raw = TensorDict({}, batch_size=[1], device=base.device)
                    base._compute_observation(raw)
                    carry = normalizer(raw).select("policy")
                    carry["is_init"] = torch.tensor([[i == 0]], dtype=torch.bool,
                                                    device=base.device)
                    actor(carry)
                    base.step(carry)
                    data = cmd.asset.data
                    sole = data.body_pos_w[:, cmd.left_asset] + quat_apply(
                        data.body_quat_w[:, cmd.left_asset], cmd.sole_offset.expand(1, -1))
                    target_w = data.root_pos_w + quat_apply(data.root_quat_w,
                                                             cmd.target_foot_pos_b())
                    load = 0.
                    if hasattr(cmd, "hand_contact_sensor"):
                        hand_force = cmd.hand_contact_sensor.data.net_forces_w
                        if hand_force is not None:
                            load = hand_force[:, cmd.hand_sensor_id].norm(dim=-1).item()
                    positions.append(data.body_pos_w[0].cpu().numpy().copy())
                    root.append(data.root_pos_w[0].cpu().numpy().copy())
                    hand.append(data.body_pos_w[0, cmd.hand_asset].cpu().numpy().copy())
                    foot.append(sole[0].cpu().numpy().copy())
                    foot_target.append(target_w[0].cpu().numpy().copy())
                    wall_load.append(load)
                    target_root.append((root0 + (torch.tensor([.015, 0., 0.], device=base.device)
                                                if trial == "root_x" and active else 0.)).cpu().numpy().copy())
                    target_hand.append((hand0 + (torch.tensor([.015, 0., 0.], device=base.device)
                                                if trial == "hand_x" and active else 0.)).cpu().numpy().copy())
            result[f"{trial}_bodies"] = np.asarray(positions, dtype="f4")
            for name, values in (("root", root), ("hand", hand), ("foot", foot),
                                 ("foot_target", foot_target), ("root_target", target_root),
                                 ("hand_target", target_hand), ("wall_load_N", wall_load)):
                result[f"{trial}_{name}"] = np.asarray(values, dtype="f4")
            print("RECORDED", trial, len(positions), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, fps=np.array(round(1./base.step_dt)),
                            body_names=np.asarray(body_names), **result)
        metadata = {"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
                    "trials": ["root_x", "hand_x", "foot_reach"],
                    "protocol": "4 s each; 1 s baseline then +15 mm x root or hand goal, "
                                "or foot workspace point 20; old policy has no root/hand input",
                    "camera_render": False, "source": "Isaac Lab physical rollout at 50 Hz"}
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
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
