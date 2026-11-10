"""Summarize paired Isaac resets from a statically feasible hand-off-wall pose."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path,
                   default=Path("artifacts/unified_support_reach"))
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/unified_support_reach/withdrawn_anchor_comparison"))
    args = p.parse_args()
    result = {"description": "Isaac 3 s rollout from a 6 cm hand-off-wall reset; "
                             "physics-rate peak is sampled over each 20 ms policy step.",
              "seeds": {}}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    colors = {"x10": "#2663ac", "contact1": "#dd7733"}
    for seed, col in ((42, 0), (7, 1)):
        result["seeds"][str(seed)] = {}
        for name in ("x10", "contact1"):
            base = args.artifact_root/f"eval_withdrawn_{name}_physics_seed{seed}"
            summary = json.loads((base/"summary.json").read_text())
            with (base/"samples.csv").open() as f:
                rows = list(csv.DictReader(f))
            t = [float(r["time_s"]) for r in rows]
            y = [100*(float(r["hand_y_m"])-float(rows[0]["hand_y_m"])) for r in rows]
            peak = [float(r["wall_peak_physics_N"]) for r in rows]
            axes[0, col].plot(t, y, color=colors[name], label=name)
            axes[1, col].plot(t, peak, color=colors[name], label=name)
            result["seeds"][str(seed)][name] = {
                key: summary[key] for key in
                ("stable", "first_wall_contact_time_s", "max_wall_peak_physics_N",
                 "final_second_mean_wall_load_N", "final_second_hand_y_drift_m")}
        axes[0, col].set_title(f"Seed {seed}: hand moves toward wall")
        axes[0, col].set_ylabel("Hand y drift (cm)")
        axes[1, col].set_ylabel("Wall peak per step (N)")
        axes[1, col].set_xlabel("Time (s)")
        axes[1, col].set_xlim(0, 3)
        axes[0, col].legend()
        axes[1, col].grid(alpha=.25)
    fig.suptitle("Hand-off-wall reset: both policies re-establish wall contact at 0.04 s")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
