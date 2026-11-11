"""Search seeded policy rollouts and retain every physics trajectory and metric.

This is the transparent cherry-pick protocol: the selected video must trace to
one checkpoint, one seed, and one unmodified Isaac rollout.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="G1/G1_table_retrieval_standing_tall_rl")
    p.add_argument("--dataset", default="table_retrieval_standing_prelude_liftfirst/train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--steps", type=int, default=510)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--stop-on-success", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    env.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")
    env.setdefault("MEMPATH", str(ROOT / "dataset"))
    for seed in range(args.seed_start, args.seed_start + args.count):
        run_dir = args.output / f"seed_{seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/table_retrieval/eval_policy.py"),
                   "--checkpoint", str(args.checkpoint.resolve()),
                   "--task", args.task, "--dataset", args.dataset,
                   "--steps", str(args.steps), "--seed", str(seed),
                   "--output", str(run_dir)]
        if args.stochastic:
            command.append("--stochastic")
        with (run_dir / "eval.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300)
        metric_path = run_dir / "metrics.json"
        if result.returncode != 0 or not metric_path.exists():
            row = dict(seed=seed, error=f"exit code {result.returncode}")
        else:
            metrics = json.loads(metric_path.read_text())
            row = dict(seed=seed, success=metrics["task_success"],
                       duration_s=metrics["duration_s"],
                       object_outward_displacement_m=metrics["object_outward_displacement_m"],
                       object_final_x_m=metrics["object_final_x_m"],
                       root_min_height_m=metrics["root_min_height_m"],
                       hand_support_duration_s=metrics["hand_support_duration_s"])
        rows.append(row)
        (args.output / "sweep_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if row.get("success") and args.stop_on_success:
            break


if __name__ == "__main__":
    main()
