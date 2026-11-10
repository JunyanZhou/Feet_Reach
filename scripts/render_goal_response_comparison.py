"""Render matched Isaac state rollouts as a side-by-side policy comparison MP4."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


TRIALS = (
    ("root_x", "Pelvis +15 mm", "root", "Pelvis x (mm)"),
    ("hand_x", "Left hand +15 mm", "hand", "Left hand x (mm)"),
    ("foot_reach", "Left foot reach", "foot", "Left foot x (mm)"),
)
EDGES = (
    ("pelvis", "torso_link"), ("torso_link", "head_mimic"),
    ("pelvis", "left_hip_roll_link"), ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"), ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_hand_mimic"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_hand_mimic"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    recordings = [np.load(args.original), np.load(args.selected)]
    names = [str(x) for x in recordings[0]["body_names"]]
    assert names == [str(x) for x in recordings[1]["body_names"]]
    indices = {name: names.index(name) for name in names}
    edges = [(indices[a], indices[b]) for a, b in EDGES]
    source_fps = int(recordings[0]["fps"])
    assert source_fps == int(recordings[1]["fps"]) == 50
    assert all(len(data["root_x_bodies"]) == 200 for data in recordings)

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "normal"})
    fig = plt.figure(figsize=(12.6, 8.4), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[3.5, 1.35], hspace=0.16,
                           left=0.065, right=0.97, top=0.90, bottom=0.09)
    poses = [fig.add_subplot(grid[0, j], projection="3d") for j in range(2)]
    graph = fig.add_subplot(grid[1, :])
    chapter_text = fig.text(0.5, 0.965, "", ha="center", va="center", fontsize=15,
                            color="#202733")

    stick_lines, body_markers, goal_markers, status_texts = [], [], [], []
    colors = ["#64748b", "#2463ca"]
    for j, (ax, label) in enumerate(zip(poses, ["v2 finetune", "Goal-conditioned"])):
        ax.set_title(label, fontsize=12, color=colors[j], pad=3)
        ax.set_xlim(-0.42, 0.72); ax.set_ylim(-0.48, 0.38); ax.set_zlim(0, 1.65)
        ax.set_xlabel("x (m)", labelpad=1); ax.set_ylabel("y (m)", labelpad=1)
        ax.set_zlabel("z (m)", labelpad=1)
        ax.set_box_aspect((1.14, .86, 1.65), zoom=.88)
        ax.view_init(elev=18, azim=-62)
        ax.tick_params(labelsize=8, pad=0)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        wall = Poly3DCollection([[
            (-.45, .25, 0), (.55, .25, 0), (.55, .25, 1.5), (-.45, .25, 1.5)
        ]], facecolor="#b9bec8", edgecolor="#a2aab6", alpha=.09)
        ax.add_collection3d(wall)
        ax.plot([-.42, .72], [-.48, -.48], [0, 0], color="#b0bac8", alpha=.65)
        stick_lines.append([ax.plot([], [], [], color=colors[j], lw=2.9,
                                    solid_capstyle="round")[0] for _ in edges])
        markers = {}
        for name, color in (("root", "#11a6aa"), ("hand", "#e68a21"),
                            ("foot", "#25a65a")):
            markers[name] = ax.plot([], [], [], marker="o", ms=8, color=color,
                                    markeredgecolor="white", markeredgewidth=1.1,
                                    linestyle="None")[0]
        body_markers.append(markers)
        goal_markers.append(ax.plot([], [], [], marker="*", ms=15,
                                    markerfacecolor="#e94654", markeredgecolor="white",
                                    markeredgewidth=.8, linestyle="None")[0])
        status_texts.append(ax.text2D(.04, .93, "", transform=ax.transAxes, fontsize=10,
                                      color=colors[j]))

    def relative_series(data, trial, body):
        values = data[f"{trial}_{body}"][:, 0]
        reference = values[35:50].mean()
        targets = data[f"{trial}_{body}_target"][:, 0]
        return (values-reference)*1000, (targets-reference)*1000

    line_old = graph.plot([], [], color=colors[0], lw=2.2, label="v2 finetune")[0]
    line_new = graph.plot([], [], color=colors[1], lw=2.5, label="Goal-conditioned")[0]
    line_goal = graph.plot([], [], color="#e94654", lw=1.8, ls="--",
                           label="Target")[0]
    cursor = graph.axvline(0, color="#334155", lw=1, alpha=.5)
    graph.legend(ncol=3, loc="upper left", frameon=False, fontsize=9)
    graph.axvline(1, color="#9aa5b5", lw=1, ls=":")
    graph.grid(alpha=.18)
    graph.set_xlim(0, 4)
    graph.set_xlabel("Time (s)")

    def set_xyz(artist, point):
        artist.set_data_3d([point[0]], [point[1]], [point[2]])

    def draw(global_frame):
        trial_index, index = divmod(global_frame, 200)
        trial, title, body, ylabel = TRIALS[trial_index]
        chapter_text.set_text(title)
        graph.set_ylabel(ylabel)
        graph.set_ylim((-8, 28) if body != "foot" else (-15, 180))
        time = np.arange(index+1)/source_fps
        goal_curve = None
        for j, data in enumerate(recordings):
            xyz = data[f"{trial}_bodies"][index]
            for artist, (a, b) in zip(stick_lines[j], edges):
                artist.set_data_3d(xyz[[a,b], 0], xyz[[a,b], 1], xyz[[a,b], 2])
            for marker_body, marker in body_markers[j].items():
                set_xyz(marker, data[f"{trial}_{marker_body}"][index])
            set_xyz(goal_markers[j], data[f"{trial}_{body}_target"][index])
            actual, goal = relative_series(data, trial, body)
            (line_old if j == 0 else line_new).set_data(time, actual[:index+1])
            if j == 1:
                goal_curve = goal
            current_error = np.linalg.norm(data[f"{trial}_{body}"][index] -
                                           data[f"{trial}_{body}_target"][index])*1000
            status_texts[j].set_text(f"Error: {current_error:.1f} mm")
        line_goal.set_data(np.arange(200)/source_fps, goal_curve)
        cursor.set_xdata([index/source_fps]*2)
        return fig

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        draw(145)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=110)
        print("PREVIEW", args.preview, flush=True)
    writer = FFMpegWriter(fps=25, codec="libx264", bitrate=3000,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(args.output), dpi=110):
        for frame in range(0, 600, 2):
            draw(frame)
            writer.grab_frame()
            if frame % 100 == 0:
                print(f"FRAME {frame}/600", flush=True)
    plt.close(fig)
    print("VIDEO", args.output, flush=True)


if __name__ == "__main__":
    main()
