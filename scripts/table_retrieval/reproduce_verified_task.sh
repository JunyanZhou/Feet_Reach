#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
OUTPUT_DIR="${1:-artifacts/table_policy/reproduced_final_policy_seed57}"
mkdir -p "$OUTPUT_DIR"

OMNI_KIT_ACCEPT_EULA=Y \
ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE=1 \
MEMPATH="$PROJECT_ROOT/dataset" \
  .venv-isaac/bin/python scripts/table_retrieval/eval_batch_policy.py \
  --checkpoint artifacts/table_policy/supported_bootstrap_dagger/dagger_2.pt \
  --residual-json artifacts/table_policy/late_two_phase_cem_y095/success_gen07.json \
  --output "$OUTPUT_DIR" --num-envs 32 --steps 450 --seed 57 --mode mean \
  --object-x 0.315 --object-y 0.095 --object-width-x 0.06 \
  --object-width-y 0.09 --object-height 0.12 \
  --object-mass 0.08 --object-friction 0.15

ENV_ID="$(python3 - "$OUTPUT_DIR/batch_metrics.json" <<'PY'
import json
import sys
rows = json.load(open(sys.argv[1]))
winner = next((row for row in rows if row["task_success"]), None)
if winner is None:
    raise SystemExit("No strict success in this evaluation batch")
print(f'{winner["env_id"]:03d}')
PY
)"
ROLLOUT="$OUTPUT_DIR/candidate_00_env_${ENV_ID}/policy_rollout.npz"
if [[ ! -f "$ROLLOUT" ]]; then
  echo "Expected successful rollout missing: $ROLLOUT" >&2
  exit 1
fi
MUJOCO_GL=egl .venv-isaac/bin/python scripts/table_retrieval/render_policy_rollout.py \
  "$ROLLOUT" --output "$OUTPUT_DIR/policy_video.mp4" \
  --object-height 0.12 --object-width-x 0.06 --object-width-y 0.09 \
  --fps 30 --width 1280 --height 720
