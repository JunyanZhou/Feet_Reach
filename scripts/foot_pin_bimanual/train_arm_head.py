"""Train a small goal-conditioned arm action head on stable physics states."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


ARM_IDS = {
    "left": [11, 15, 19, 21, 23, 25, 27],
    "right": [12, 16, 20, 22, 24, 26, 28],
}


class ArmHead(nn.Module):
    def __init__(self, observation_dim=265):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.Mish(),
            nn.Linear(256, 7),
        )

    def forward(self, observation):
        return self.net(observation)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmstart", type=Path, default=None)
    parser.add_argument("--feedback-gain", type=float, default=0.)
    parser.add_argument("--support-until", type=float, default=None,
                        help="For left arm, imitate stable physical pose until this time")
    parser.add_argument("--support-transition-end", type=float, default=4.5)
    parser.add_argument("--support-source", choices=("action", "pose"), default="action")
    parser.add_argument("--target-clip", type=float, default=6.)
    args = parser.parse_args()
    torch.manual_seed(42)
    data = np.load(args.dataset)
    steps, num_envs, dim = data["observation"].shape
    mask = data["root_z"] > .55
    observation = torch.as_tensor(data["observation"][mask], device="cuda")
    target = torch.as_tensor(data["target_action"][mask][:, ARM_IDS[args.side]], device="cuda")
    time = torch.as_tensor(np.broadcast_to(np.arange(steps)[:, None] * .02,
                                            mask.shape)[mask], device="cuda")
    if args.support_until is not None:
        assert args.side == "left" and args.support_transition_end > args.support_until
        if args.support_source == "action":
            hold_action = torch.as_tensor(data["old_loc"][mask][:, ARM_IDS[args.side]],
                                          device="cuda").clamp(-10., 10.)
        else:
            actual_q = torch.as_tensor(data["joint_pos"][mask][:, ARM_IDS[args.side]], device="cuda")
            default_q = torch.as_tensor(data["default_joint_pos"][ARM_IDS[args.side]], device="cuda")
            action_scale = torch.as_tensor(data["action_scaling"][ARM_IDS[args.side]], device="cuda")
            hold_action = (actual_q - default_q) / action_scale
        u = ((time - args.support_until) /
             (args.support_transition_end - args.support_until)).clamp(0., 1.)
        mix = u*u*(3.-2.*u)
        target = (1.-mix[:, None])*hold_action + mix[:, None]*target
    if args.feedback_gain:
        actual_q = torch.as_tensor(data["joint_pos"][mask][:, ARM_IDS[args.side]], device="cuda")
        default_q = torch.as_tensor(data["default_joint_pos"][ARM_IDS[args.side]], device="cuda")
        action_scale = torch.as_tensor(data["action_scaling"][ARM_IDS[args.side]], device="cuda")
        actual_action = (actual_q - default_q) / action_scale
        target = target + args.feedback_gain * (target - actual_action)
    target.clamp_(-args.target_clip, args.target_clip)
    permutation = torch.randperm(len(observation), device="cuda")
    split = max(512, len(permutation)//10)
    val_ids, train_ids = permutation[:split], permutation[split:]
    head = ArmHead(dim).to("cuda")
    if args.warmstart is not None:
        state = torch.load(args.warmstart, map_location="cuda", weights_only=False)
        assert state["side"] == args.side and state["observation_dim"] == dim
        head.load_state_dict(state["model"])
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    weights = torch.where(time[train_ids] < 3., 2., 1.)
    for update in range(args.updates):
        indices = train_ids[torch.multinomial(weights, args.batch_size, replacement=True)]
        pred = head(observation[indices])
        loss = (pred - target[indices]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
        optimizer.step()
        if update == 0 or (update + 1) % 200 == 0:
            head.eval()
            with torch.no_grad():
                val_pred = head(observation[val_ids])
                val_mse = (val_pred - target[val_ids]).square().mean().item()
                early = time[val_ids] < 3.
                early_mse = (val_pred[early] - target[val_ids][early]).square().mean().item()
            head.train()
            print(f"HEAD update={update+1} train_mse={loss.item():.5f} "
                  f"val_mse={val_mse:.5f} early_mse={early_mse:.5f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": head.cpu().state_dict(), "side": args.side,
                "joint_ids": ARM_IDS[args.side], "observation_dim": dim,
                "source_dataset": str(args.dataset.resolve()),
                "warmstart": str(args.warmstart.resolve()) if args.warmstart else None,
                "feedback_gain": args.feedback_gain,
                "support_until": args.support_until,
                "support_transition_end": args.support_transition_end,
                "support_source": args.support_source,
                "validation_mse": val_mse,
                "validation_early_mse": early_mse}, args.output)
    print(f"ARM_HEAD_READY {args.output}", flush=True)


if __name__ == "__main__":
    main()
