# RETFound Retinal Tasks Reproduction Guide

This folder contains the reproduction code for the retinal tasks in the StratCP paper/package:

- Diabetic retinopathy (APTOS-style 5-class setting)
- Glaucoma (3-class setting)
- JSIEC eye condition classification (action-similarity utility setting)

The scripts in this folder generate the core CSVs used to build manuscript retinal figures.

## 1. Files in This Folder

- `diabetic_retinopathy.py`: Diabetic retinopathy reproduction
- `glaucoma.py`: Glaucoma reproduction
- `jsiec_action.py`: JSIEC action-similarity reproduction
- `run_diabetic.sh`, `run_glaucoma.sh`, `run_jsiec_action.sh`: wrapper scripts with default arguments

## 2. Required Inputs

For each task, `results_dir` must contain:

- `predicted_probabilities.npy` with shape `(n_samples, n_classes)`
- `true_labels.npy` with shape `(n_samples,)`

JSIEC additionally needs:

- `action_similarity.npy` with shape `(n_classes, n_classes)`

Default data locations in this repo:

- `data/retfound_tasks/diabetic_retinopathy`
- `data/retfound_tasks/glaucoma`
- `data/retfound_tasks/JSIEC`

## 3. Environment / How to Run

From repo root (`StratCP`), install and activate any Python environment
that has StratCP and dependencies available:

```bash
python -m pip install -e .
export PYTHONPATH=$(pwd)/src
```

If you already installed StratCP as an editable package, `PYTHONPATH` is optional.

### Option A: use wrapper scripts

```bash
bash reproduction_code/retfound_tasks/run_diabetic.sh
bash reproduction_code/retfound_tasks/run_glaucoma.sh
bash reproduction_code/retfound_tasks/run_jsiec_action.sh
```

Optional path overrides:

```bash
RESULTS_DIR=/path/to/diabetic_data bash reproduction_code/retfound_tasks/run_diabetic.sh
RESULTS_DIR=/path/to/glaucoma_data bash reproduction_code/retfound_tasks/run_glaucoma.sh
RESULTS_DIR=/path/to/jsiec_data SIM_FILE=/path/to/action_similarity.npy \
  bash reproduction_code/retfound_tasks/run_jsiec_action.sh
```

### Option B: run Python scripts directly (recommended for explicit reproducibility)

```bash
python reproduction_code/retfound_tasks/diabetic_retinopathy.py \
  --results_dir data/retfound_tasks/diabetic_retinopathy \
  --cp_methods aps \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0 \
  --eligibility per_class

python reproduction_code/retfound_tasks/glaucoma.py \
  --results_dir data/retfound_tasks/glaucoma \
  --cp_methods aps \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0 \
  --eligibility per_class

python reproduction_code/retfound_tasks/jsiec_action.py \
  --results_dir data/retfound_tasks/JSIEC \
  --sim_file data/retfound_tasks/JSIEC/action_similarity.npy \
  --alphas 0.025 0.05 0.1 0.2 \
  --alpha_fixed 0.05 \
  --n_runs 500 \
  --calib_frac 0.5 \
  --random_state 0
```

Wrapper scripts already use the manuscript alpha grid (`0.025, 0.05, 0.1, 0.2`).

## 4. Output Files

### 4.1 Diabetic task

Output dir: `{results_dir}/stratcp_eval_results_diabetic/`

- `baseline_eval.csv`
- `vanilla_eval.csv`
- `stratcp_eval.csv`
- `conditional_eval.csv`
- `summary_df.csv`
- `cond_summary.csv`

### 4.2 Glaucoma task

Output dir: `{results_dir}/stratcp_eval_results_glaucoma/`

- `baseline_eval.csv`
- `vanilla_eval.csv`
- `stratcp_eval.csv`
- `conditional_eval.csv`
- `summary_df.csv`
- `cond_summary.csv`

### 4.3 JSIEC task

Output dir: `{results_dir}/stratcp_eval_results_jsiec_action/`

- `stratified_action_eval.csv`
- `stratified_size_sim.csv`
- `stratified_size_sim_unselected.csv`
- `summary_df.csv`

## 5. Mapping Outputs to Manuscript Figure Data

This section explains how to map CSV columns to retinal manuscript plot data.

## 5.1 Diabetic + Glaucoma

Use methods/labels:

- `top1` -> Top-1
- `thresh` -> Threshold baseline
- `aps` with `source=vanilla_cp` -> CP
- `aps` with `source=stratified_cp` -> StratCP

### A) Marginal coverage curve (target coverage vs empirical coverage)

- Data source: `baseline_eval.csv`, `vanilla_eval.csv`, `stratcp_eval.csv`
- X axis: `1 - alpha`
- Y axis: `mgn_cov`

### B) Marginal size curve

- Data source: same as above
- X axis: `1 - alpha`
- Y axis: `mgn_size`

### C) Deferred/unsure metrics at alpha=0.05

- Data source: `summary_df.csv` (or run-level eval CSVs filtered at `alpha=0.05`)
- Coverage of deferred sets: `unselected_coverage`
- Number deferred: `num_unsel`
- Deferred set size: `unselected_set_size`

### D) Decision-conditional coverage/count bars at alpha=0.05

- Data source: `cond_summary.csv` (or `conditional_eval.csv` grouped by `source/method/alpha`)
- Diabetic coverage columns:
  - `cov_normal`, `cov_mild`, `cov_modr`, `cov_sever`, `cov_prol`, `cov_unselected`
- Diabetic count columns:
  - `num_normal`, `num_mild`, `num_modr`, `num_sever`, `num_prol`, `num_unselected`
- Glaucoma coverage columns:
  - `cov_mild`, `cov_early`, `cov_advanced`, `cov_unselected`
- Glaucoma count columns:
  - `num_mild`, `num_early`, `num_advanced`, `num_unselected`

## 5.2 JSIEC

Primary data file: `stratified_action_eval.csv`.

Columns include:

- `cov`: overall marginal coverage
- `scov`: coverage among selected/confident predictions
- `nscov`: coverage among deferred/unselected predictions
- `n_sel`: number of selected/confident predictions
- `size`: average prediction-set size over all samples
- `sim_avg`: average within-set similarity

### A) JSIEC marginal coverage/size curves

- Group by `method`, `conformal`, `alpha` and average across `run`
- Coverage curve uses `cov`
- Size curve uses `size`

### B) JSIEC confident prediction bars at alpha=0.05

- Coverage: `scov`
- Number of confident predictions: `n_sel`

### C) JSIEC deferred/unsure bars at alpha=0.05

Let `m` be test-set size per split.

- Deferred coverage: `unsel_cov = nscov`
- Deferred count: `unsel_num = m - n_sel`
- Deferred set size:
  - `unsel_size = (size * m - n_sel) / unsel_num`

### D) JSIEC action-similarity plot

- Data source: `stratified_size_sim.csv`
- Typical filter used for manuscript utility plot:
  - `conformal == 'stratified'`
  - `alpha == 0.05`
  - selected methods such as `aps` and `expand_greedy`
- Plot `average_sim` vs `size` (weighted by `count` if needed)

## 6. Reproducibility Notes

- Use fixed `random_state` and `n_runs=500` for manuscript-level stability.
- Small differences across environments are possible due to randomized conformal tie handling.
- `jsiec_action.py` is expected to output all runs (`run=0..n_runs-1`) in `stratified_action_eval.csv`.
