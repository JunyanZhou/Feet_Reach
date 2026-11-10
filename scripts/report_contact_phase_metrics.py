"""Compare paired Isaac wall-contact runs by phase and physics-rate peaks."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


PHASES = {
    "settle": {"settle"},
    "reach": {"reach"},
    "hold": {"hold"},
    "controlled": {"reach", "hold"},
    "all": {"settle", "reach", "hold"},
}


def summarize(path):
    summary = json.loads((path/"summary.json").read_text())
    samples = list(csv.DictReader((path/"samples.csv").open()))
    if not samples or "left_hand_wall_peak_physics_N" not in samples[0]:
        raise ValueError(f"Missing physics-rate wall force measurements: {path}")
    result = {"path": str(path.resolve()), "success_rate": summary["success_rate"],
              "planned_targets": summary["planned_targets"], "phases": {}}
    for name, accepted in PHASES.items():
        rows = [r for r in samples if r["phase"] in accepted]
        grouped = {}
        for row in rows:
            grouped.setdefault(int(row["point"]), []).append(
                float(row["left_hand_wall_peak_physics_N"]))
        if name != "settle" and len(grouped) != summary["planned_targets"]:
            raise ValueError(f"Phase {name} did not cover all targets: {path}")
        peaks = {str(key): max(value) for key, value in grouped.items()}
        values = sorted(float(r["left_hand_wall_peak_physics_N"]) for r in rows)
        result["phases"][name] = {
            "observed_targets": len(grouped),
            "mean_per_target_peak_N": mean(peaks.values()),
            "worst_peak_N": max(peaks.values()),
            "targets_over_100N": sum(v > 100. for v in peaks.values()),
            "sample_p95_peak_N": values[int(.95*(len(values)-1))],
            "per_target_peak_N": peaks,
        }
    hold = [r for r in samples if r["phase"] == "hold"]
    result["hold_mean_foot_error_m"] = mean(float(r["error_m"]) for r in hold)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                   metavar=("BASELINE", "CANDIDATE"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = []
    for baseline_path, candidate_path in args.pair:
        baseline = summarize(baseline_path)
        candidate = summarize(candidate_path)
        if baseline["planned_targets"] != candidate["planned_targets"]:
            raise ValueError("Target counts differ")
        baseline_targets = json.loads((baseline_path/"summary.json").read_text())["per_point"]
        candidate_targets = json.loads((candidate_path/"summary.json").read_text())["per_point"]
        if [(x["point"], x["target_root_xyz_m"]) for x in baseline_targets] != [
                (x["point"], x["target_root_xyz_m"]) for x in candidate_targets]:
            raise ValueError("Paired world targets differ")
        delta = {}
        for phase in PHASES:
            b = baseline["phases"][phase]["per_target_peak_N"]
            c = candidate["phases"][phase]["per_target_peak_N"]
            if b.keys() != c.keys():
                raise ValueError(f"Paired target ids differ in phase {phase}")
            delta[phase] = {
                "mean_paired_peak_change_N": mean(c[k]-b[k] for k in b),
                "worst_peak_change_N": (candidate["phases"][phase]["worst_peak_N"]
                                         - baseline["phases"][phase]["worst_peak_N"]),
            }
        report.append({"baseline": baseline, "candidate": candidate,
                       "candidate_minus_baseline": delta})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"pairs": report}, indent=2)+"\n")
    for pair in report:
        print({"baseline": pair["baseline"]["path"],
               "candidate": pair["candidate"]["path"],
               "success": (pair["baseline"]["success_rate"],
                           pair["candidate"]["success_rate"]),
               "hold_foot_mm": (round(pair["baseline"]["hold_mean_foot_error_m"]*1000, 2),
                                round(pair["candidate"]["hold_mean_foot_error_m"]*1000, 2)),
               "controlled_peak_delta_N": round(pair["candidate_minus_baseline"]
                                                  ["controlled"]["mean_paired_peak_change_N"], 2),
               "settle_worst_delta_N": round(pair["candidate_minus_baseline"]
                                              ["settle"]["worst_peak_change_N"], 2)})


if __name__ == "__main__":
    main()
