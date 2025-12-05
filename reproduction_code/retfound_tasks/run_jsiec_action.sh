#!/usr/bin/env bash
# Run the JSIEC action-based StratCP reproduction.
# Update RESULTS_DIR to point at your JSIEC predictions directory
# and SIM_FILE to the action-overlap similarity matrix (npy).

set -euo pipefail

RESULTS_DIR="/Users/yjinstat/Desktop/Research/collaborations/cp-diagnosis/StratCP/data/retfound_tasks/JSIEC"
SIM_FILE="/Users/yjinstat/Desktop/Research/collaborations/cp-diagnosis/StratCP/data/retfound_tasks/JSIEC/action_similarity.npy"

python3 "$(dirname "$0")/jsiec_action.py" \
  --results_dir "$RESULTS_DIR" \
  --sim_file "$SIM_FILE" \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0 \
  "$@"
