"""Compare paired Isaac goal-response summaries using measured body displacement."""

import argparse
import json
from pathlib import Path


AXES = {
    "root_forward": ("root", "x", 1),
    "root_backward": ("root", "x", -1),
    "root_side": ("root", "y", 1),
    "root_opposite_side": ("root", "y", -1),
    "hand_forward": ("hand", "x", 1),
    "hand_backward": ("hand", "x", -1),
    "hand_up": ("hand", "z", 1),
    "hand_down": ("hand", "z", -1),
}


def summarize(path):
    data = json.loads(path.read_text())
    trials = {row["label"]: row for row in data["trials"]}
    baseline = trials["baseline"]
    rows = {}
    for label, (body, axis, sign) in AXES.items():
        row = trials[label]
        key = f"mean_{body}_{axis}_m"
        movement = (row[key] - baseline[key])*sign*1000 if row["stable"] else None
        rows[label] = {
            "stable": row["stable"],
            "failure": row["failure"],
            "signed_motion_mm": round(movement, 3) if movement is not None else None,
            "goal_error_mm": (round(row[f"mean_{body}_error_m"]*1000, 3)
                              if row["stable"] else None),
            "swing_error_mm": (round(row["mean_swing_error_m"]*1000, 3)
                               if row["stable"] else None),
            "peak_hand_load_N": round(row["peak_hand_load_N"], 3),
        }
    return data, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, old_rows = summarize(args.baseline)
    new, new_rows = summarize(args.candidate)
    if (old["seed"] != new["seed"] or
            old["foot_point_index"] != new["foot_point_index"] or
            old["protocol"] != new["protocol"]):
        raise ValueError("Paired evaluations must share seed, foot point, and protocol")
    comparison = {}
    for label in AXES:
        a, b = old_rows[label], new_rows[label]
        delta = (round(b["signed_motion_mm"]-a["signed_motion_mm"], 3)
                 if a["stable"] and b["stable"] else None)
        comparison[label] = {"baseline": a, "candidate": b,
                             "signed_motion_change_mm": delta}
        print(label, "stable", b["stable"], "motion_mm", b["signed_motion_mm"],
              "change_mm", delta)
    report = {"baseline_checkpoint": old["checkpoint"],
              "candidate_checkpoint": new["checkpoint"],
              "seed": old["seed"], "foot_point_index": old["foot_point_index"],
              "comparison": comparison,
              "note": "Positive signed motion is toward the target. This screen does not establish task success."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
