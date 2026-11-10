"""Supervise only new goal input columns from supported Pinocchio IK poses.

The old student network stays frozen. At the shared wall-supported anchor,
each goal offset is paired with the full-body joint change of a validated IK
trajectory. This warm start learns a local action correction; Isaac rollouts
must still verify tracking, contact, and balance.
"""

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


def load_cases(source, split, action_names, scaling, device,
               x_only=False, correction_gain=1.):
    manifest = json.loads((source/"manifest.json").read_text())
    cases = []
    for record in manifest["clips"]:
        if record["split"] != split:
            continue
        if x_only and (record["offset_m"][0] == 0 or
                       record["offset_m"][1] != 0 or
                       record["offset_m"][2] != 0):
            continue
        clip = np.load(source/"motions"/split/record["name"])
        names = list(clip["joint_names"])
        index = [names.index(name) for name in action_names]
        delta = (clip["dof_pos"][-1, index] - clip["dof_pos"][0, index])
        cases.append((record["goal"], record["offset_m"],
                      correction_gain*torch.as_tensor(delta, device=device)/scaling))
    if not cases:
        raise ValueError(f"No {split} cases in {source}")
    return cases


def load_release_cases(path, action_names, scaling, device, correction_gain):
    """Use the outbound portion of an offline-screened hand-release motion."""
    trajectory = np.load(path)
    names = list(trajectory["joint_names"])
    index = [names.index(name) for name in action_names]
    fraction = trajectory["hand_release_fraction"]
    joint_pos = trajectory["dof_pos"][:, index]
    distance = json.loads(trajectory["report_json"].item())["distance_m"]
    train, val = [], []
    for i in range(5, 101, 5):
        target = (0., -distance*float(fraction[i]), 0.)
        delta = correction_gain*torch.as_tensor(
            joint_pos[i]-joint_pos[0], device=device)/scaling
        case = ("left_hand", target, delta)
        (val if (i//5) % 4 == 1 else train).append(case)
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path,
                        default=ROOT/"artifacts/unified_support_reach/teaching")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x-only", action="store_true",
                        help="Train only root and wall-hand x targets verified in Isaac")
    parser.add_argument("--release-trajectory", type=Path,
                        help="Supervise only the hand-away-from-wall goal column")
    parser.add_argument("--correction-gain", type=float, default=1.)
    args = parser.parse_args()
    if args.release_trajectory and args.x_only:
        parser.error("Choose either x-only or release-trajectory")
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.steps < 1 or args.num_envs < 1 or args.lr <= 0
            or not 0 < args.correction_gain <= 10):
        raise ValueError("steps, num-envs, lr and correction-gain must be positive")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = args.num_envs
    cfg.task.command.student_train = True
    cfg.task.command.foot_force.enabled = False
    if "aux_support_force" in cfg.task.command:
        cfg.task.command.aux_support_force.enabled = False
    cfg.task.command.dataset.mem_paths = ["wall_foot_reach/val"]
    cfg.task.command.dataset.path_weights = [1.]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = 0
    cfg.task.termination = {}
    cfg.vecnorm = "eval"
    cfg.wandb.mode = "disabled"
    cfg.seed = args.seed
    cfg.app = {"headless": True, "enable_cameras": False}
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        from isaaclab.sim import SimulationContext
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval(); agent.eval()
        env.set_seed(args.seed)
        cmd = env.base_env.command_manager
        cmd.zero_init_prob = 1.
        obs = env.reset()
        base = env.base_env
        manager = base.action_manager
        scaling = manager.action_scaling
        if args.release_trajectory:
            train_cases, val_cases = load_release_cases(
                args.release_trajectory, manager.joint_names, scaling,
                base.device, args.correction_gain)
        else:
            train_cases = load_cases(args.source, "train", manager.joint_names,
                                     scaling, base.device, args.x_only,
                                     args.correction_gain)
            val_cases = load_cases(args.source, "val", manager.joint_names,
                                   scaling, base.device, args.x_only,
                                   args.correction_gain)
        first = dict(agent.actor_student.named_parameters())[
            "module.0.module.1.module.0.weight"]
        policy_dim = int(obs["policy"].shape[-1])
        if policy_dim not in (253, 262):
            raise ValueError("Checkpoint has no appended goal columns")
        goal_columns = (253, 262) if policy_dim == 262 else (241, 250)
        if args.release_trajectory:
            if policy_dim != 262:
                raise ValueError("Support-release warmstart needs persistent goal offsets")
            goal_columns = (257, 258)  # left-hand lateral offset in policy obs
        for param in agent.parameters():
            param.requires_grad_(False)
        first.requires_grad_(True)
        mask = torch.zeros_like(first)
        mask[:, goal_columns[0]:goal_columns[1]] = 1.
        first.register_hook(lambda gradient: gradient*mask)
        optimizer = torch.optim.Adam([first], lr=args.lr)
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        is_init = obs["is_init"].clone()
        root0 = cmd.goal_world("root").clone()
        hand0 = cmd.goal_world("left_hand").clone()

        def policy_obs():
            raw = TensorDict({}, batch_size=[args.num_envs], device=base.device)
            base._compute_observation(raw)
            raw = normalizer(raw)
            raw["is_init"] = is_init
            return raw.select("policy", "is_init").clone()

        cmd.set_goal_world("root", None)
        cmd.set_goal_world("left_hand", None)
        baseline_obs = policy_obs()
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            original = actor(baseline_obs.clone())["loc"].detach()
        cases = train_cases
        goals = [c[0] for c in cases]
        is_root_goal = torch.tensor([goal == "root" for goal in goals],
                                    device=base.device)
        offsets = torch.tensor([c[1] for c in cases], device=base.device)
        deltas = torch.stack([c[2] for c in cases])
        loss_history = []
        with set_exploration_type(ExplorationType.MODE):
            for step in range(args.steps):
                selected = torch.randint(len(cases), (args.num_envs,),
                                         device=base.device)
                chosen = offsets[selected]
                root_mask = is_root_goal[selected][:, None]
                hand_mask = ~root_mask
                cmd.set_goal_world("root", root0 + chosen*root_mask)
                cmd.set_goal_world("left_hand", hand0 + chosen*hand_mask)
                target_obs = policy_obs()
                baseline_loc = actor(baseline_obs.clone())["loc"]
                target_loc = actor(target_obs)["loc"]
                correction = target_loc - baseline_loc
                correction_loss = (correction-deltas[selected]).square().mean()
                baseline_loss = (baseline_loc-original).square().mean()
                loss = (correction_loss + 10.*baseline_loss
                        + 1e-5*first[:,goal_columns[0]:goal_columns[1]].square().mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 100 == 0 or step == args.steps-1:
                    row = {"step": step, "correction_mse": correction_loss.item(),
                           "baseline_mse": baseline_loss.item()}
                    loss_history.append(row)
                    print("WARMSTART", json.dumps(row), flush=True)
        cmd.set_goal_world("root", None)
        cmd.set_goal_world("left_hand", None)
        assert torch.isfinite(first).all()
        def score_cases(items):
            errors = []
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                base_loc = actor(baseline_obs.clone())["loc"]
                for goal, offset, expected in items:
                    delta = torch.tensor(offset, device=base.device)
                    cmd.set_goal_world("root", root0 + delta if goal == "root" else root0)
                    cmd.set_goal_world("left_hand", hand0 + delta if goal == "left_hand" else hand0)
                    actual = actor(policy_obs())["loc"] - base_loc
                    errors.append((actual-expected).square().mean().item())
            return sum(errors)/len(errors)
        train_mse = score_cases(train_cases)
        val_mse = score_cases(val_cases)
        print("WARMSTART_SCORE", json.dumps({"train_mse": train_mse,
                                             "heldout_mse": val_mse}), flush=True)
        # Save only if the supervised correction loss improved materially.
        if loss_history[-1]["correction_mse"] >= loss_history[0]["correction_mse"]:
            raise RuntimeError("Goal-column supervision failed to reduce correction loss")
        state["policy"] = agent.state_dict()
        metadata = {
            "source_checkpoint": str(args.checkpoint.resolve()),
            "train_cases": len(train_cases), "heldout_cases": len(val_cases),
            "x_only": args.x_only, "correction_gain": args.correction_gain,
            "train_loss": loss_history, "train_mse": train_mse,
            "heldout_mse": val_mse, "trained_columns": list(goal_columns),
            "method": "IK joint delta supervision at fixed supported anchor",
        }
        if args.release_trajectory:
            metadata["release_trajectory"] = str(args.release_trajectory.resolve())
            state["support_release_warmstart"] = metadata
        else:
            state["goal_warmstart"] = metadata
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.output)
        print("WARMSTART_SAVED", args.output, flush=True)
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
