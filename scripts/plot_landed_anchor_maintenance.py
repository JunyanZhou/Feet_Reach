"""Plot physical load maintenance after resetting to a two-foot IK anchor."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    keys = ("time_s", "left_foot_normal_N", "right_foot_normal_N",
            "wall_load_N", "root_y_m")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--assisted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.3), sharex=True,
                             layout="constrained")
    for label, path, color in (("Policy", args.policy, "#202020"),
                               ("Policy + fixed leg offset", args.assisted,
                                "#2876a5")):
        d = read(path)
        t = d["time_s"]
        axes[0].plot(t, d["left_foot_normal_N"], color=color,
                     label=label, linewidth=1.2)
        axes[1].plot(t, 1000*(d["root_y_m"]-d["root_y_m"][0]),
                     color=color, linewidth=1.2)
        axes[2].plot(t, d["wall_load_N"], color=color, linewidth=1.2)
    axes[0].axhline(50, linestyle="--", color="#777777", linewidth=.9)
    axes[0].set_ylabel("Left foot normal (N)")
    axes[1].set_ylabel("Pelvis lateral shift (mm)")
    axes[2].set_ylabel("Wall force (N)")
    axes[2].set_xlabel("Time after reset (s)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Isaac Sim: maintaining a two-foot IK anchor")
    for axis in axes:
        axis.grid(alpha=.15)
        axis.set_xlim(0, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
