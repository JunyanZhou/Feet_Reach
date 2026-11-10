"""Fail the runnable bundle check if the two policy evaluations are incomplete."""

import json
from pathlib import Path
import sys


def main(run_dir):
    foot = json.loads((run_dir / "foot_targets/summary.json").read_text())
    goal = json.loads((run_dir / "goal_response/summary.json").read_text())
    if not foot.get("all_targets_evaluated") or foot.get("attempted_targets") != 20:
        raise RuntimeError("The 20-target foot evaluation did not finish")
    if not foot.get("sequence_completed"):
        raise RuntimeError("The foot policy failed one or more target trials")
    trials = goal.get("trials", [])
    if len(trials) != 7 or not all(trial.get("stable") for trial in trials):
        raise RuntimeError("The seven unified goal-response trials did not all remain stable")
    print(json.dumps({"run_dir": str(run_dir),
                      "foot_targets_completed": foot["attempted_targets"],
                      "foot_success_rate": foot["success_rate"],
                      "stable_goal_trials": len(trials)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python verify_run.py RUN_DIRECTORY")
    main(Path(sys.argv[1]))
