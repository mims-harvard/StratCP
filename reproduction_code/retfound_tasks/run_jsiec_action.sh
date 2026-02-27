#!/usr/bin/env bash
# Run the JSIEC action-based StratCP reproduction.
# By default this uses data under the current StratCP repo.
# Override RESULTS_DIR and/or SIM_FILE to use custom paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/data/retfound_tasks/JSIEC}"
SIM_FILE="${SIM_FILE:-${RESULTS_DIR}/action_similarity.npy}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/jsiec_action.py" \
  --results_dir "$RESULTS_DIR" \
  --sim_file "$SIM_FILE" \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0 \
  "$@"
