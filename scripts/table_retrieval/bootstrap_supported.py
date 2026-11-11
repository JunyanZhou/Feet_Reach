"""Bootstrap a student from a physically rolled-out low-table joint reference.

The reference is only a training label.  Saved checkpoints contain a student
that maps its public observations to all 29 joint actions at inference.
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
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from isaaclab.app import AppLauncher
from omegaconf import OmegaConf
import active_adaptation.learning


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expert-steps", type=int, default=400)
    p.add_argument("--bc-steps", type=int, default=1500)
    p.add_argument("--dagger-rounds", type=int, default=3)
    p.add_argument("--dagger-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[
            "task=G1/G1_table_retrieval_supported_rl", "+exp=finetune"])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = max(args.expert_steps, args.dagger_steps) + 50
    cfg.task.termination = {}
    cfg.task.command.dataset.mem_paths = ["table_retrieval_supported/train"]
    cfg.task.command.anchor_init_prob = 1.0
    cfg.task.command.foot_force.enabled = False
    cfg.algo.symmetry_augmentation = False
    cfg.vecnorm = "eval"
    cfg.wandb.mode = "disabled"
    cfg.seed = args.seed
    cfg.app = {"headless": True, "enable_cameras": False}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    AppLauncher(headless=True, device="cuda:0").app
    exit_code = 0
    try:
        from scripts.utils.helpers import make_env_policy
        env, agent, vecnorm, _ = make_env_policy(cfg)
        env.eval(); env.base_env.eval()
        env.set_seed(args.seed)
        base = env.base_env
        cmd = base.command_manager
        cmd.zero_init_prob = 1.0
        actor = agent.get_rollout_policy("eval")
        normalizer = vecnorm.to_observation_norm()
        normalizer.eval()
        observations, labels = [], []

        def collect(use_student, steps):
            env.reset()
            base.episode_length_buf.zero_()
            actor.eval()
            root_min = 10.0
            object_x0 = float(base.scene["target_object"].data.root_pos_w[0, 0])
            count = 0
            for i in range(steps):
                raw = TensorDict({}, batch_size=[1], device=base.device)
                base._compute_observation(raw)
                carry = normalizer(raw).select("policy")
                carry["is_init"] = torch.tensor([[i == 0]], device=base.device)
                reference_action = ((cmd.target_joint_pos_obs()
                    - base.action_manager.default_joint_pos[:, base.action_manager.joint_ids])
                    / base.action_manager.action_scaling)
                observations.append(carry["policy"][0].detach().cpu().clone())
                labels.append(reference_action[0].detach().cpu().clone())
                with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                    if use_student:
                        actor(carry)
                    else:
                        carry["action"] = reference_action
                    base.step(carry)
                count += 1
                root_min = min(root_min, float(cmd.asset.data.root_pos_w[0, 2]))
                if use_student and root_min < 0.5:
                    break
            object_x1 = float(base.scene["target_object"].data.root_pos_w[0, 0])
            print(f"COLLECT student={use_student} steps={count} root_min={root_min:.3f} "
                  f"object_dx={object_x0-object_x1:.3f}", flush=True)
            return {"student": use_student, "steps": count,
                    "root_min_m": root_min, "object_outward_m": object_x0-object_x1}

        def fit(steps, stage):
            x = torch.stack(observations).to(base.device)
            y = torch.stack(labels).to(base.device)
            actor.train()
            opt = torch.optim.Adam(actor.parameters(), lr=2e-4)
            losses = []
            for i in range(steps):
                ids = torch.randint(len(x), (min(128, len(x)),), device=base.device)
                td = TensorDict({"policy": x[ids],
                                 "is_init": torch.zeros(len(ids), 1, dtype=torch.bool,
                                                        device=base.device)},
                                batch_size=[len(ids)], device=base.device)
                with set_exploration_type(ExplorationType.MODE):
                    actor(td)
                loss = F.mse_loss(td["loc"], y[ids])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                opt.step()
                if i % 100 == 0 or i == steps - 1:
                    value = float(loss.detach())
                    losses.append([i, value])
                    print(f"BC {stage} step={i} loss={value:.6f}", flush=True)
            actor.eval()
            source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            source["policy"] = agent.state_dict()
            source["bootstrap"] = {"stage": stage, "expert": "direct joint reference in Isaac",
                                   "training_samples": len(x), "loss": losses,
                                   "dataset": "table_retrieval_supported/train"}
            path = args.output / f"{stage}.pt"
            torch.save(source, path)
            print(f"BOOTSTRAP_CHECKPOINT {path}", flush=True)
            return {"stage": stage, "checkpoint": str(path), "loss": losses[-1][1]}

        report = {"source_checkpoint": str(args.checkpoint.resolve()), "rollouts": [], "fits": []}
        report["rollouts"].append(collect(False, args.expert_steps))
        report["fits"].append(fit(args.bc_steps, "bc_expert"))
        for j in range(args.dagger_rounds):
            report["rollouts"].append(collect(True, args.dagger_steps))
            report["fits"].append(fit(max(args.bc_steps // 2, 300), f"dagger_{j+1}"))
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
