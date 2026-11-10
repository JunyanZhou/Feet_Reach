"""Plot paired Isaac landing rollouts with identical foot and root targets."""

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
    return {key: np.asarray([float(row[key]) for row in rows])
            for key in ("time_s", "left_foot_normal_N", "wall_load_N", "root_y_m")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", nargs=2, action="append", metavar=("LABEL", "CSV"),
                   required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.2), sharex=True,
                             layout="constrained")
    colors = ("#222222", "#2876a5", "#b04c44", "#5c8f58")
    for i, (label, file) in enumerate(args.series):
        data = read(Path(file))
        t = data["time_s"]
        root_shift = 1000*(data["root_y_m"]-data["root_y_m"][0])
        for axis, value in zip(axes,
                               (data["left_foot_normal_N"], root_shift,
                                data["wall_load_N"])):
            axis.plot(t, value, label=label, color=colors[i % len(colors)],
                      linewidth=1.25, alpha=.9)
    axes[0].axhline(50, color="#888888", linestyle="--", linewidth=.9)
    axes[0].text(6.03, 50, "50 N", va="center", fontsize=8, color="#666666")
    axes[0].set_ylabel("Left foot normal (N)")
    axes[1].set_ylabel("Pelvis y shift (mm)")
    axes[2].set_ylabel("Wall force (N)")
    axes[2].set_xlabel("Time after landing command (s)")
    for axis in axes:
        axis.axvline(2, color="#bbbbbb", linewidth=.8)
        axis.axvline(3, color="#bbbbbb", linewidth=.8)
        axis.grid(alpha=.15)
        axis.set_xlim(0, 6.35)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[0].set_title("Ground-level foot landing; 4 cm pelvis target after 2 s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
