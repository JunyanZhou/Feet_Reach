"""Zero-extend the v2 foot student for root/hand/right-foot/load observations.

All new first-layer columns are zero. At zero goal jitter the migrated policy
therefore has the same actor output as the source for identical old observations.
The new inputs become trainable during subsequent finetuning.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


NEW_FIELDS = (
    "target_root_pos_b",
    "target_left_hand_pos_b",
    "target_right_foot_pos_b",
    "support_contact_loads",
)
OFFSET_FIELDS = (
    "goal_offset_root_b",
    "goal_offset_left_hand_b",
    "goal_offset_right_foot_b",
)
FIRST_LAYERS = {
    "adapt_module": "module.0.0.weight",
    "adapt_joint_module": "module.0.0.weight",
    "actor_teacher": "module.0.module.1.module.0.weight",
    "actor_student": "module.0.module.1.module.0.weight",
    "critic": "module.1.module.0.0.weight",
}


def migrate(source, output, learning_rate=None, goal_norm_file=None,
            include_offsets=False):
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(source, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    state["cfg"] = cfg
    policy = state["policy"]
    norm = state["vecnorm"]["_extra_state"]
    if (cfg.task.name != "G1WallFootReach" or policy["last_phase"] != "finetune"
            or list(cfg.task.observation.policy) != [
                "target_foot_pos_b", "root_ang_vel_history", "projected_gravity_history",
                "joint_pos_history", "prev_actions"]):
        raise ValueError("Expected the v2 wall-foot-reach finetune checkpoint")
    old_dim = norm["policy_sum"].numel()
    if old_dim != 241:
        raise ValueError(f"Expected 241 public inputs, got {old_dim}")
    fields = NEW_FIELDS + (OFFSET_FIELDS if include_offsets else ())
    extra_dim = 3*len(fields)
    for group, key in FIRST_LAYERS.items():
        weight = policy[group][key]
        if weight.ndim != 2 or weight.shape[1] < old_dim:
            raise ValueError(f"Unexpected first layer {group}.{key}: {weight.shape}")
        # The actor/critic concatenate the public policy vector before their
        # latent/privileged vectors. Insert columns at that boundary.
        policy[group][key] = torch.cat((
            weight[:, :old_dim], weight.new_zeros(weight.shape[0], extra_dim),
            weight[:, old_dim:]), dim=1)
    symmetry = policy["obs_transform"]
    symmetry["perm"] = torch.cat((symmetry["perm"],
                                    torch.arange(old_dim, old_dim + extra_dim,
                                                 dtype=symmetry["perm"].dtype)))
    symmetry["signs"] = torch.cat((symmetry["signs"],
                                     symmetry["signs"].new_ones(extra_dim)))
    count = norm["policy_count"]
    if goal_norm_file is None:
        mean = norm["policy_sum"].new_zeros(extra_dim)
        std = mean.new_ones(extra_dim)
    else:
        stats = json.loads(goal_norm_file.read_text())
        mean = torch.as_tensor(stats["mean"], dtype=norm["policy_sum"].dtype)
        std = torch.as_tensor(stats["std"], dtype=norm["policy_sum"].dtype)
        if (mean.shape != (12,) or std.shape != (12,)
                or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
                or (std <= 0).any()):
            raise ValueError("goal_norm_file must contain 12 finite means and positive stds")
        if include_offsets:
            mean = torch.cat((mean, mean.new_zeros(9)))
            std = torch.cat((std, std.new_full((9,), float(stats["position_std_m"]))))
    norm["policy_sum"] = torch.cat((norm["policy_sum"], count*mean))
    norm["policy_ssq"] = torch.cat((norm["policy_ssq"], count*(mean.square()+std.square())))
    OmegaConf.set_struct(cfg, False)
    cfg.task.name = "G1UnifiedSupportReachV2" if include_offsets else "G1UnifiedSupportReach"
    # The saved v2 checkpoint predates the optional PointNet setting in PPOPolicy.
    cfg.algo.geometry_points = 0
    cfg.task.command._target_ = (
        "active_adaptation.envs.mdp.commands.unified_support_reach.UnifiedSupportReachCommand")
    cfg.task.command.goal_jitter_m = {
        "root": [0., 0., 0.],
        "left_hand": [0., 0., 0.],
        "right_foot": [0., 0., 0.],
    }
    cfg.task.command.goal_jitter_one_goal = True
    for field in fields:
        cfg.task.observation.policy[field] = {}
    cfg.checkpoint_path = None
    if learning_rate is not None:
        if not (0. < learning_rate <= 1e-3):
            raise ValueError("learning_rate must be in (0, 1e-3]")
        cfg.algo.lr = learning_rate
    # This is a new curriculum, not a continuation of the old 8-GPU schedule.
    policy["_meta"].update(current_lr=float(cfg.algo.lr), progress=0.,
                            num_updates=0)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    state["migration"] = {
        "source_sha256": source_hash,
        "method": "zero new first-layer columns and configured normalization of appended policy inputs",
        "old_policy_dim": old_dim,
        "new_policy_dim": old_dim + extra_dim,
        "new_fields": list(fields),
        "goal_normalizer": str(goal_norm_file) if goal_norm_file else "unit",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    return state["migration"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--learning-rate", type=float, default=None,
                   help="Reset the imported PPO learning rate for conservative fine-tuning")
    p.add_argument("--goal-norm-file", type=Path,
                   help="JSON with 12 means and stds for appended policy inputs")
    p.add_argument("--include-offsets", action="store_true",
                   help="Also append persistent root, hand, and stance goal offsets")
    args = p.parse_args()
    print(migrate(args.source, args.output, args.learning_rate,
                  args.goal_norm_file, args.include_offsets))
