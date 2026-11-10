"""Compare paired Isaac multi-body force rollouts without hiding failures."""

import argparse
import json
from pathlib import Path


MEASURES = ("mean_root_error_m", "mean_hand_error_m",
            "mean_stance_error_m", "mean_swing_error_m",
            "peak_hand_load_N", "peak_aux_force_N")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", metavar=("BASE", "CANDIDATE"),
                   required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    results = []
    for baseline_path, candidate_path in args.pair:
        a = json.loads(Path(baseline_path).read_text())
        b = json.loads(Path(candidate_path).read_text())
        for key in ("seed", "foot_point_index", "protocol",
                    "aux_force_enabled", "aux_force_range_N"):
            if a[key] != b[key]:
                raise ValueError(f"Paired run differs on {key}: {a[key]} vs {b[key]}")
        if not a["aux_force_enabled"]:
            raise ValueError("Expected auxiliary-force rollouts")
        base = {row["label"]: row for row in a["trials"]}
        candidate = {row["label"]: row for row in b["trials"]}
        if base.keys() != candidate.keys():
            raise ValueError("Paired goal trials differ")
        for label in base:
            old, new = base[label], candidate[label]
            results.append({
                "seed": a["seed"], "foot_point_index": a["foot_point_index"],
                "force_range_N": a["aux_force_range_N"], "goal": label,
                "baseline_stable": old["stable"],
                "candidate_stable": new["stable"],
                "baseline_failure": old["failure"],
                "candidate_failure": new["failure"],
                **{f"baseline_{key}": old[key] for key in MEASURES},
                **{f"candidate_{key}": new[key] for key in MEASURES},
            })
    stable_old = sum(row["baseline_stable"] for row in results)
    stable_new = sum(row["candidate_stable"] for row in results)
    paired_stable = [row for row in results
                     if row["baseline_stable"] and row["candidate_stable"]]
    metrics = {}
    for key in MEASURES:
        values = [(row[f"candidate_{key}"]-row[f"baseline_{key}"])
                  for row in paired_stable if row[f"baseline_{key}"] is not None
                  and row[f"candidate_{key}"] is not None]
        metrics[key] = sum(values)/len(values) if values else None
    report = {"pairs": args.pair, "trials": len(results),
              "baseline_stable": stable_old, "candidate_stable": stable_new,
              "paired_stable": len(paired_stable),
              "paired_stable_mean_candidate_minus_baseline": metrics,
              "rows": results,
              "note": "Tracking deltas use only trials stable for both checkpoints; count failures separately."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in (
        "trials", "baseline_stable", "candidate_stable", "paired_stable",
        "paired_stable_mean_candidate_minus_baseline")}, indent=2))


if __name__ == "__main__":
    main()
