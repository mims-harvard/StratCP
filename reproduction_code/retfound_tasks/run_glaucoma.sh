#!/usr/bin/env bash
# Run the glaucoma reproduction pipeline.
# By default this uses data under the current StratCP repo.
# Override RESULTS_DIR to run on a custom dataset directory
# containing predicted_probabilities.npy and true_labels.npy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/data/retfound_tasks/glaucoma}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/glaucoma.py" \
  --results_dir "$RESULTS_DIR" \
  --cp_methods aps \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0 \
  --eligibility per_class \
  "$@"
