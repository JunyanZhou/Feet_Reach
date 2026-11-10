#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON_BIN:-python3}"
export MEMPATH="$repo_root/dataset"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${ISAACLAB_PATH:-}" ]]; then
  export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$PYTHONPATH"
fi

"$python_bin" scripts/unified_support/check_bundle.py --torch

run_dir="${UNIFIED_RUN_DIR:-$repo_root/artifacts/unified_support_reach/run_$(date -u +%Y%m%d_%H%M%S)}"
if [[ -e "$run_dir" ]]; then
  echo "Output directory already exists: $run_dir" >&2
  exit 1
fi

"$python_bin" scripts/eval_foot_reach.py \
  --checkpoint "$repo_root/artifacts/unified_support_reach/v2_goal_offset_x10.pt" \
  --points 20 --output "$run_dir/foot_targets"

"$python_bin" scripts/eval_unified_support_reach.py \
  --checkpoint "$repo_root/artifacts/unified_support_reach/v2_goal_offset_x10.pt" \
  --output "$run_dir/goal_response"

"$python_bin" scripts/unified_support/verify_run.py "$run_dir"
