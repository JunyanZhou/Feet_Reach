"""Combine independently screened goal directions without changing the base policy."""

import argparse
from pathlib import Path

import torch


WEIGHT = "module.0.module.1.module.0.weight"
ROOT_Y = 254
HAND_Z = 258


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root-y-gain", type=float, default=1.)
    p.add_argument("--hand-z-gain", type=float, default=1.)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0. < args.root_y_gain <= 3. or not 0. < args.hand_z_gain <= 3.:
        raise ValueError("Goal-column gains must be in (0, 3]")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    a, b = base["policy"], source["policy"]
    if a.keys() != b.keys():
        raise ValueError("Policy state keys differ")
    for module in a:
        if not hasattr(a[module], "items"):
            if a[module] != b[module]:
                raise ValueError(f"Policy metadata differs: {module}")
            continue
        for key, value in a[module].items():
            other = b[module][key]
            if module == "actor_student" and key == WEIGHT:
                if value.shape != other.shape or value.shape[1] < 262:
                    raise ValueError("Expected compatible 262-dimensional student prefix")
                if not torch.equal(value[:, :253], other[:, :253]):
                    raise ValueError("Original input columns differ")
            elif isinstance(value, torch.Tensor) and not torch.equal(value, other):
                raise ValueError(f"Policy tensor differs: {module}.{key}")
            elif not isinstance(value, torch.Tensor) and value != other:
                raise ValueError(f"Policy value differs: {module}.{key}")
    for key, value in base["vecnorm"].items():
        other = source["vecnorm"][key]
        if isinstance(value, torch.Tensor) and not torch.equal(value, other):
            raise ValueError(f"Normalization tensor differs: {key}")
    weight = a["actor_student"][WEIGHT]
    from_weight = b["actor_student"][WEIGHT]
    weight[:, ROOT_Y] = from_weight[:, ROOT_Y] * args.root_y_gain
    weight[:, HAND_Z] = from_weight[:, HAND_Z] * args.hand_z_gain
    base["goal_column_composition"] = {
        "base": str(args.base.resolve()), "source": str(args.source.resolve()),
        "root_y_gain": args.root_y_gain, "hand_z_gain": args.hand_z_gain,
        "columns": {"root_y": ROOT_Y, "hand_z": HAND_Z},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
