"""Optional supervised arm adaptation from observed task states and IK targets.

Only the policy's learned actor is updated. Lower-body action means are
distilled from the source policy, preserving its balance decisions. The
reference targets are kinematic supervision; success still needs physics eval.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="G1/G1_foot_pin_bimanual_v1")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-clip", type=float, default=6.)
    parser.add_argument("--reset-arm-output", action="store_true",
                        help="Start arm output rows near the physical reference rather than saturated PPO values")
    args = parser.parse_args()
    torch.manual_seed(42)
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        cfg = hydra.compose("train", overrides=[f"task={args.task}", "+exp=finetune"])
    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.task.num_envs = 1
    cfg.algo.symmetry_augmentation = False
    cfg.vecnorm = "eval"
    cfg.wandb.mode = "disabled"
    cfg.app = {"headless": True, "enable_cameras": False}
    app = AppLauncher(headless=True, device="cuda:0").app
    exit_code = 0
    try:
        from scripts.utils.helpers import make_env_policy
        env, agent, _, _ = make_env_policy(cfg)
        names = agent.joint_names
        arm_ids = [i for i, name in enumerate(names) if any(
            word in name for word in ("shoulder", "elbow", "wrist"))]
        lower_ids = [i for i in range(len(names)) if i not in arm_ids]
        assert len(arm_ids) == 14 and len(lower_ids) == 15, names
        data = np.load(args.dataset)
        steps, num_envs, dim = data["observation"].shape
        time = np.arange(steps)[:, None] * float(env.base_env.step_dt)
        mask = (data["root_z"] > .55) & (time <= 8.5)
        sample_time = torch.as_tensor(np.broadcast_to(time, mask.shape)[mask], device="cuda:0")
        obs = torch.as_tensor(data["observation"][mask], device="cuda:0")
        target = torch.as_tensor(data["target_action"][mask], device="cuda:0")
        old_loc = torch.as_tensor(data["old_loc"][mask], device="cuda:0")
        target.clamp_(-args.target_clip, args.target_clip)
        print(f"BC samples={len(obs)} policy_dim={dim} arm_ids={arm_ids} "
              f"target_arm_abs_p95={target[:,arm_ids].abs().quantile(.95).item():.3f}", flush=True)
        assert len(obs) > args.batch_size
        for module in (agent.adapt_module, agent.adapt_joint_module, agent.critic):
            module.requires_grad_(False)
            module.eval()
        agent.actor_student.train()
        if args.reset_arm_output:
            output_layer = agent.actor_student.module[0].module[2].module.actor_mean
            with torch.no_grad():
                output_layer.weight[arm_ids] = 0.
                output_layer.bias[arm_ids] = target[sample_time < .1][:, arm_ids].mean(dim=0)
        optim = torch.optim.Adam(agent.actor_student.parameters(), lr=args.lr)
        actor = agent.get_rollout_policy("eval")
        validation = torch.randperm(len(obs), device="cuda:0")
        val_idx = validation[:max(512, len(obs)//10)]
        train_idx = validation[len(val_idx):]
        train_weights = torch.where(sample_time[train_idx] < 1.5, 3., 1.)

        def predict(indices):
            batch = TensorDict({"policy": obs[indices],
                                "is_init": torch.zeros(len(indices), 1, dtype=torch.bool,
                                                       device="cuda:0")},
                               batch_size=[len(indices)], device="cuda:0")
            with set_exploration_type(ExplorationType.MODE):
                actor(batch)
            return batch["loc"]

        with torch.no_grad():
            before = predict(val_idx)
            before_arm = (before[:, arm_ids] - target[val_idx][:, arm_ids]).square().mean().item()
            before_lower = (before[:, lower_ids] - old_loc[val_idx][:, lower_ids]).square().mean().item()
        print(f"BC before arm_mse={before_arm:.4f} lower_mse={before_lower:.4f}", flush=True)
        arm_mse, lower_mse = before_arm, before_lower
        for update in range(args.updates):
            choices = torch.multinomial(train_weights, args.batch_size, replacement=True)
            indices = train_idx[choices]
            loc = predict(indices)
            arm_loss = (loc[:, arm_ids] - target[indices][:, arm_ids]).square().mean()
            lower_loss = (loc[:, lower_ids] - old_loc[indices][:, lower_ids]).square().mean()
            old_arm_loss = (loc[:, arm_ids] - old_loc[indices][:, arm_ids]).square().mean()
            loss = arm_loss + 5. * lower_loss + .02 * old_arm_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor_student.parameters(), 10.)
            optim.step()
            if (update + 1) % 100 == 0:
                with torch.no_grad():
                    loc_val = predict(val_idx)
                    arm_mse = (loc_val[:, arm_ids] - target[val_idx][:, arm_ids]).square().mean().item()
                    lower_mse = (loc_val[:, lower_ids] - old_loc[val_idx][:, lower_ids]).square().mean().item()
                print(f"BC update={update+1} arm_mse={arm_mse:.4f} lower_mse={lower_mse:.4f}", flush=True)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint["policy"] = agent.state_dict()
        checkpoint["arm_imitation"] = dict(source=str(args.checkpoint.resolve()),
                                            dataset=str(args.dataset.resolve()),
                                            updates=args.updates,
                                            reset_arm_output=args.reset_arm_output,
                                            arm_ids=arm_ids, lower_ids=lower_ids,
                                            validation_arm_mse_before=before_arm,
                                            validation_arm_mse_after=arm_mse,
                                            validation_lower_mse_after=lower_mse)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output)
        print(f"ARM_IMITATION_READY {args.output}", flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
