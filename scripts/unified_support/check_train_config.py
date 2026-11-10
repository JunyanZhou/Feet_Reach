"""Reject curriculum configurations incompatible with a student checkpoint."""

import argparse
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch

import active_adaptation.learning  # Registers the PPO Hydra configs.


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = state["cfg"]
    with hydra.initialize_config_dir(config_dir=str(ROOT / "cfg"), version_base=None):
        target = hydra.compose(config_name="train", overrides=args.overrides)
    OmegaConf.resolve(target)

    if target.algo.phase != "finetune" or target.vecnorm != "eval":
        raise ValueError("Curriculum continuation requires the finetune phase and frozen VecNorm")
    if not target.task.command._target_.endswith(".UnifiedSupportReachCommand"):
        raise ValueError("Selected task does not use the unified command interface")
    for group in ("policy", "joint_target", "priv", "priv_critic"):
        old = list(source.task.observation[group])
        new = list(target.task.observation[group])
        if old != new:
            raise ValueError(f"Observation {group} differs from checkpoint: {old} != {new}")
    for field in ("_target_", "latent_dim", "layer_norm", "geometry_points"):
        if source.algo.get(field) != target.algo.get(field):
            raise ValueError(f"Model architecture field {field} differs from checkpoint")
    if source.task.action._target_ != target.task.action._target_:
        raise ValueError("Action type differs from checkpoint")
    for path in target.task.command.dataset.mem_paths:
        schema = ROOT / "dataset" / path / "_tensordict/meta.json"
        if not schema.is_file():
            raise FileNotFoundError(f"Missing curriculum dataset: {schema}")
    if Path(target.checkpoint_path).resolve() != args.checkpoint.resolve():
        raise ValueError("Hydra checkpoint_path differs from the checked checkpoint")
    print(json.dumps({"checkpoint": str(args.checkpoint),
                      "task": target.task.name,
                      "phase": target.algo.phase,
                      "policy_inputs": state["vecnorm"]["_extra_state"]["policy_sum"].numel(),
                      "datasets": list(target.task.command.dataset.mem_paths)}, indent=2))


if __name__ == "__main__":
    main()
