"""Scale only supervised persistent-goal input weights for a closed-loop gain sweep."""

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--factor", type=float, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.factor <= 3:
        raise ValueError("factor must be in (0, 3]")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = (state.get("support_release_warmstart")
                or state.get("goal_warmstart", {}))
    columns = metadata.get("trained_columns")
    if columns not in ([253, 262], [257, 258]):
        raise ValueError("Expected a supervised persistent-offset checkpoint")
    weight = state["policy"]["actor_student"]["module.0.module.1.module.0.weight"]
    weight[:, columns[0]:columns[1]] *= args.factor
    state["goal_gain_scale"] = {"source": str(args.checkpoint.resolve()),
                                "factor": args.factor,
                                "trained_columns": columns}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
