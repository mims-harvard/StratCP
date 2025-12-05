#!/usr/bin/env bash
# Run the diabetic retinopathy reproduction pipeline.
# Update RESULTS_DIR to point at your RETFound predictions directory
# containing predicted_probabilities.npy and true_labels.npy.

set -euo pipefail

RESULTS_DIR="/Users/yjinstat/Desktop/Research/collaborations/cp-diagnosis/StratCP/data/retfound_tasks/diabetic_retinopathy"

python3 "$(dirname "$0")/diabetic_retinopacy.py" \
  --results_dir "$RESULTS_DIR" \
  --cp_methods aps \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0 \
  --eligibility per_class \
  "$@"
