"""Warm-start a 3D right-hand goal channel from a trained unified student.

The new channel is appended to the public policy observation. New first-layer
weights are zero, so the transferred model initially preserves the old actor.
The channel becomes trainable during bimanual fine-tuning.
"""
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def insert_zero_columns(weight, at, count=3):
    assert weight.ndim == 2 and 0 <= at <= weight.shape[1]
    zeros = weight.new_zeros((weight.shape[0], count))
    return torch.cat((weight[:, :at], zeros, weight[:, at:]), dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "artifacts/table_policy/table_retrieval_policy.pt")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/foot_pin_bimanual/right_hand_warmstart.pt")
    args = p.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    policy = checkpoint["policy"]
    stats = checkpoint["vecnorm"]["_extra_state"]
    old_dim = stats["policy_sum"].numel()
    assert old_dim == 262, f"Expected validated public observation size 262, got {old_dim}"
    for module in ("adapt_module", "adapt_joint_module"):
        key = "module.0.0.weight"
        weight = policy[module][key]
        assert weight.shape[1] == old_dim
        policy[module][key] = insert_zero_columns(weight, old_dim)
    actor_key = "module.0.module.1.module.0.weight"
    for module in ("actor_student", "actor_teacher"):
        weight = policy[module][actor_key]
        assert weight.shape[1] == old_dim + 256 + 29
        policy[module][actor_key] = insert_zero_columns(weight, old_dim)
    critic_key = "module.1.module.0.0.weight"
    critic_weight = policy["critic"][critic_key]
    assert critic_weight.shape[1] > old_dim
    policy["critic"][critic_key] = insert_zero_columns(critic_weight, old_dim)
    symmetry = policy["obs_transform"]
    assert symmetry["perm"].numel() == old_dim and symmetry["signs"].numel() == old_dim
    symmetry["perm"] = torch.cat((symmetry["perm"], torch.arange(old_dim, old_dim + 3,
                                                                   dtype=symmetry["perm"].dtype)))
    symmetry["signs"] = torch.cat((symmetry["signs"], symmetry["signs"].new_ones(3)))
    count = stats["policy_count"].reshape(-1)[0]
    stats["policy_sum"] = torch.cat((stats["policy_sum"], stats["policy_sum"].new_zeros(3)))
    stats["policy_ssq"] = torch.cat((stats["policy_ssq"], count.expand(3).to(stats["policy_ssq"])))
    checkpoint["transfer_note"] = dict(source=str(args.source.resolve()),
                                        old_policy_dim=old_dim, new_policy_dim=old_dim + 3,
                                        new_feature="target_right_hand_pos_b",
                                        inserted_at=old_dim)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"RIGHT_HAND_WARMSTART_READY {args.output}")


if __name__ == "__main__":
    main()
