#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
stage="${1:-}"
if [[ -z "$stage" ]]; then
  echo "Usage: bash scripts/unified_support/train_next.sh STAGE [Hydra overrides]" >&2
  exit 2
fi
shift
case "$stage" in
  reach) task=G1/G1_unified_support_reach_v2 ;;
  teaching|aux10|contact|contact_v2|multiaxis|missing_axes|landing|landing_root|landing_root_force|release|release_static|release_cycle|no_contact)
    task="G1/G1_unified_support_$stage" ;;
  *) echo "Unknown curriculum stage: $stage" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python3}"
checkpoint="${CHECKPOINT:-$repo_root/artifacts/unified_support_reach/v2_goal_offset_x10.pt}"
checkpoint="$(realpath "$checkpoint")"
export MEMPATH="$repo_root/dataset"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${ISAACLAB_PATH:-}" ]]; then
  export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH"
fi

"$python_bin" scripts/unified_support/check_bundle.py --training --torch --checkpoint "$checkpoint"

run_dir="${UNIFIED_TRAIN_DIR:-$repo_root/outputs/unified-curriculum/${stage}_$(date -u +%Y%m%d_%H%M%S)}"
overrides=(
  "task=$task"
  "+exp=finetune"
  "checkpoint_path=$checkpoint"
  "total_frames=${TOTAL_FRAMES:-1048576}"
  "task.num_envs=${NUM_ENVS:-1024}"
  "algo.symmetry_augmentation=false"
  "save_interval=${SAVE_INTERVAL:-10}"
  "wandb.mode=${WANDB_MODE:-offline}"
  "wandb.project=${WANDB_PROJECT:-unified-support-curriculum}"
  "hydra.run.dir=$run_dir"
  "$@"
)
"$python_bin" scripts/unified_support/check_train_config.py --checkpoint "$checkpoint" "${overrides[@]}"

if [[ -e "$run_dir" ]]; then
  echo "Training output directory already exists: $run_dir" >&2
  exit 1
fi
"$python_bin" scripts/train.py "${overrides[@]}"
mapfile -t final_checkpoints < <(find "$run_dir" -type f -name checkpoint_final.pt)
if [[ ${#final_checkpoints[@]} -ne 1 ]]; then
  echo "Expected one final checkpoint under $run_dir, found ${#final_checkpoints[@]}" >&2
  exit 1
fi
echo "Training output: $run_dir"
echo "Next-stage checkpoint: ${final_checkpoints[0]}"
