# Unified support controller: runnable Isaac evaluation

This repository includes the selected trained G1 student checkpoint, its
observation normalizer, the G1 asset, the 8,400-frame validation set, the
curriculum training datasets, and the foot-target workspace. No image or video
is needed for evaluation or further training. The checkpoint controls 29 joint-position targets from the current
left-foot target plus local root, left-hand, and right-foot goals.

## Run on an Isaac Lab machine

Use an environment with Isaac Sim 4.5.0, Isaac Lab 2.2.0, Python 3.10, a CUDA
GPU, and the Python packages in `setup.py`. From the repository root:

```bash
export PYTHON_BIN=/path/to/isaaclab-environment/bin/python
export ISAACLAB_PATH=/path/to/IsaacLab
"$PYTHON_BIN" -m pip install -e .
python3 scripts/unified_support/check_bundle.py
bash scripts/unified_support/run.sh
```

If Isaac Lab is already importable in that Python environment, `ISAACLAB_PATH`
may be omitted. `check_bundle.py` uses only Python's standard library and
checks the checkpoint/workspace hashes and every validation memmap size.
`run.sh` additionally loads the PyTorch checkpoint, then performs two Isaac
physics evaluations with the **student policy**:

1. Twenty consecutive left-foot targets in the wall-supported posture.
2. Seven root, left-hand, and stance-foot goal-response trials.

The command exits with an error if either evaluation is incomplete, a foot
target fails, or a goal-response trial becomes unstable. Results are written
under `artifacts/unified_support_reach/run_<UTC timestamp>/` and are ignored
by Git. Set `UNIFIED_RUN_DIR=/absolute/output/path` to choose another directory.
The prior local seed-42 result was 20/20 foot targets and 7/7 stable goal
trials; numerical agreement on another simulator installation is an
evaluation result, not an assumption of this package.

## Scope

This is the currently selected **wall-supported, single-leg reaching**
checkpoint. It tracks local root and support-hand goals only over the small
tested range. The four object-manipulation videos were scripted demonstrations;
this checkpoint has not been shown to perform those tasks. The optional
`--direct-ik` and `--ik-residual` flags of the low-level evaluation script use
additional teaching trajectories and are outside this student-only run.

## Continue curriculum fine-tuning

The repository bundles all datasets referenced by the current unified task
configs: `wall_foot_reach/train`, `unified_support/train`,
`unified_static_support/train`, and `unified_withdrawn_support/train`, plus the
validation set. With the same Isaac environment:

```bash
python3 scripts/unified_support/check_bundle.py --training
bash scripts/unified_support/train_next.sh teaching
```

`train_next.sh` loads `v2_goal_offset_x10.pt` by default, uses the `finetune`
student phase, checks observation and model compatibility before launching,
and writes checkpoints under `outputs/unified-curriculum/`. Its default budget
is 1,048,576 environment frames on one GPU with 1,024 environments. For
example, to change the budget and continue from a newer stage:

```bash
CHECKPOINT=/path/to/previous/checkpoint_final.pt \
TOTAL_FRAMES=2097152 NUM_ENVS=512 \
bash scripts/unified_support/train_next.sh landing
```

Available stages are `reach`, `teaching`, `multiaxis`, `missing_axes`, `aux10`,
`contact`, `contact_v2`, `landing`, `landing_root`, `landing_root_force`,
`release`, `release_static`, `release_cycle`, and `no_contact`. Hydra overrides
can be appended to the command. Offline W&B logging is the default; set
`WANDB_MODE=online` when desired. The printed final checkpoint can be supplied
as `CHECKPOINT` for the next stage.

This continues the **weights and observation normalization**. The PPO
optimizer is recreated for each new curriculum run, so this is fine-tuning
from a checkpoint rather than an exact interrupted-run resume. Training reward
alone does not establish physical support switching; use the evaluation
scripts and contact gates before selecting a new checkpoint.
