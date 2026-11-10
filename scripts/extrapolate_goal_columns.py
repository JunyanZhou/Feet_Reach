"""Scale learned goal-input column changes while preserving a validated policy."""

import argparse
from pathlib import Path

import torch


KEY = "module.0.module.1.module.0.weight"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--columns", type=int, nargs="+", required=True)
    p.add_argument("--gain", type=float, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.columns or len(set(args.columns)) != len(args.columns):
        p.error("columns must be unique and nonempty")
    if not -20 <= args.gain <= 20:
        p.error("gain must be in [-20,20]")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    trained = torch.load(args.trained, map_location="cpu", weights_only=False)
    actor_base = base["policy"]["actor_student"]
    actor_trained = trained["policy"]["actor_student"]
    if actor_base.keys() != actor_trained.keys():
        raise ValueError("Actor architecture differs")
    for name, value in actor_base.items():
        other = actor_trained[name]
        if name == KEY:
            if value.shape != other.shape:
                raise ValueError("First-layer shape differs")
            if any(i < 0 or i >= value.shape[1] for i in args.columns):
                raise ValueError("Column index outside first-layer input")
            mask = torch.ones(value.shape[1], dtype=torch.bool)
            mask[args.columns] = False
            if not torch.equal(value[:, mask], other[:, mask]):
                raise ValueError("Trained checkpoint changed unselected actor columns")
        elif isinstance(value, torch.Tensor) and not torch.equal(value, other):
            raise ValueError(f"Trained checkpoint changed actor tensor {name}")
    for module in ("adapt_module", "adapt_joint_module"):
        for name, value in base["policy"][module].items():
            other = trained["policy"][module][name]
            if isinstance(value, torch.Tensor) and not torch.equal(value, other):
                raise ValueError(f"Trained checkpoint changed {module}.{name}")
    columns = args.columns
    actor_base[KEY][:, columns] += args.gain*(actor_trained[KEY][:, columns]
                                                -actor_base[KEY][:, columns])
    base["goal_column_extrapolation"] = {
        "base": str(args.base.resolve()), "trained": str(args.trained.resolve()),
        "columns": columns, "gain": args.gain,
        "note": "Requires independent Isaac closed-loop evaluation before promotion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
