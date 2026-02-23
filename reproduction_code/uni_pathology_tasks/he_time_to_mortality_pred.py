"""
he_time_to_mortality_pred.py

Reproduction analysis script for StratCP on the H&E-based time-to-mortality prediction
task from whole-slide imaging (WSI) model outputs.

This module evaluates and summarizes error-controlled decision-making pipelines using
the StratCP framework (mims-harvard/StratCP) for a time-to-event (survival) prediction
setting with right censoring. Given cached per-slide model outputs (e.g., predicted
location/scale parameters such as `mu_pred` and `log_sigma`) and survival metadata,
the script constructs or reuses stratified case-level calibration/test splits and runs
split-wise conformal evaluation over an alpha grid.

The script applies a fixed administrative censoring horizon (in years), computes IPCW
(inverse probability of censoring weighting) using cross-fitted Cox models on the
calibration split, and evaluates methods relative to a clinically meaningful favorable
survival threshold (e.g., survival beyond 1.5 years).

For each split, the script computes:
    1) Baseline methods for time-to-event prediction (e.g., top-1 / threshold-style
       survival decision summaries),
    2) Vanilla conformal prediction for the survival setting, and
    3) Stratified conformal prediction for survival via a two-stage StratCP procedure
       with IPCW-adjusted calibration/evaluation.

In practical terms for this script:
    - The selected-side (action-arm) guarantee is evaluated for the pooled selected
      predictions under a pre-specified nominal error budget alpha, using IPCW-adjusted
      surrogates/metrics due to right censoring (e.g., `selected_coverage_ipcw`).
    - The unselected (deferred) cases are handled by conformal prediction on the
      survival output, and summarized with deferred-side metrics (e.g.,
      `unselected_coverage_ipcw`, `unselected_set_size`).

To support reproducible analysis and efficient reruns, the script caches split-level
evaluation outputs to disk, then aggregates results across splits (mean and standard
error over a user-specified alpha range) and prints final summary tables at a target
alpha (with nearest-alpha matching if needed).

Typical use case:
    - Reproduce and compare baseline, vanilla CP, and StratCP behavior on the H&E
      time-to-mortality WSI task using precomputed survival model outputs.
    - Evaluate IPCW-adjusted selected/unselected performance under alpha sweeps.
    - Study the impact of administrative censoring horizon and favorable survival
      threshold (e.g., 1.5 years within a 5-year study horizon).

Example executions:
    # Default run (5-year administrative censoring, favorable threshold = 1.5 years)
    python he_time_to_mortality_pred.py \
        --results_dir ../../data/uni_pathology_tasks/he_time_to_mortality_pred \
        --n_splits 10 \
        --alpha_fixed 0.05
"""

from __future__ import annotations

# Standard library imports
import argparse
import math
import os
import pickle
from typing import Any, Dict

# Third-party imports
import numpy as np
import pandas as pd

# Project imports
from stratcp.eval_utils import (
    aggregate_conformal_results,
    extract_split_arrays_tte,
    load_or_create_splits,
    summarize_methods_at_alpha,
    apply_administrative_censoring,
    get_ipcw_weights,
    compute_baselines_for_split_tte,
    run_vanilla_cp_for_split_tte,
    compute_stratcp_survival_for_split
)

# Constants
BASELINE_CACHE_TEMPLATE = "top1_thresh_results_split_{split_idx}_of_{n_splits}.pkl"
VANILLA_CP_CACHE_TEMPLATE = "cp_vanilla_results_split_{split_idx}_of_{n_splits}.pkl"
STRATCP_CACHE_TEMPLATE = "stratcp_results_split_{split_idx}_of_{n_splits}.pkl"

GLOBAL_BASELINE_CACHE = "split_to_baseline_top_1_thresh_results.pkl"
GLOBAL_VANILLA_CP_CACHE = "split_to_cp_vanilla_results.pkl"
GLOBAL_STRATCP_CACHE = "split_to_stratcp_results.pkl"


# CLI parsing
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for WSI multiclass StratCP evaluation."""
    parser = argparse.ArgumentParser(description="WSI multiclass evaluation with StratCP (method-selective).")

    # I/O and bookkeeping
    parser.add_argument(
        "--results_dir",
        default="../../data/uni_pathology_tasks/he_time_to_mortality_pred",
        help="Root directory for predictions and where evaluation outputs are saved.",
    )

    # Splitting
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Base RNG seed for stratified splits (each split uses random_state + split_idx).",
    )
    parser.add_argument(
        "--calib_prop",
        type=float,
        default=0.20,
        help="Proportion of calibration cases among (calib + test).",
    )
    parser.add_argument(
        "--test_prop",
        type=float,
        default=0.15,
        help="Proportion of test cases among (calib + test).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Number of independent case-level stratified splits.",
    )

    # CP configuration
    parser.add_argument(
        "--cp_methods",
        nargs="+",
        default=["aps"],
        help="CP methods to run (space-separated): choices are 'tps', 'aps', 'raps'.",
    )
    parser.add_argument(
        "--alpha_fixed",
        type=float,
        default=0.05,
        help="Alpha at which to print the final comparison table.",
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.0375,
        help="Minimum alpha value for the evaluation grid.",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.30,
        help="Maximum alpha value for the evaluation grid.",
    )
    parser.add_argument(
        "--alpha_points",
        type=int,
        default=22,
        help="Number of alpha values in the evaluation grid (linspace).",
    )

    # Aggregation / summary
    parser.add_argument(
        "--alpha_aggr_min",
        type=float,
        default=0.01,
        help="Lower bound for α-range aggregation (inclusive).",
    )
    parser.add_argument(
        "--alpha_aggr_max",
        type=float,
        default=0.20,
        help="Upper bound for α-range aggregation (inclusive).",
    )
    parser.add_argument(
        "--include_se",
        action="store_true",
        default=True,
        help="Include standard-error columns in the summary table.",
    )
    parser.add_argument(
        "--nearest_tol",
        type=float,
        default=5e-3,
        help="Tolerance for nearest-α lookup if fixed α is not exactly on the grid.",
    )

    # Caching controls
    parser.add_argument(
        "--overwrite_split_cache",
        action="store_true",
        default=False,
        help="If set, recompute splits even if a split cache already exists.",
    )
    parser.add_argument(
        "--overwrite_eval_cache",
        action="store_true",
        default=False,
        help="If set, recompute per-split eval results even if caches exist.",
    )

    # Time-to-event specific arguments
    parser.add_argument(
        "--study_end_time_year",
        type=float,
        default=5.0,
        help="Administrative censoring time in days (default: 5 years).",
    )
    parser.add_argument(
        "--clip_eps",
        type=float,
        default=0.05,
        help="Small constant to clip predicted probabilities for IPCW weight estimation (default: 1e-3).",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds for cross-fitting the Cox model for IPCW weight estimation (default: 10).",
    )
    parser.add_argument(
        "--favorable_thresh_year",
        type=float,
        default=1.5,
        help="Threshold for early favorable survival (in years).",
    )

    return parser.parse_args()


# Utility functions
def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_results_dict(results_path: str) -> Dict[str, Dict[str, Any]]:
    """Load cached per-slide prediction dictionary."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing predictions file: {results_path}")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results_dict_test from {results_path} (n={len(results)})")
    return results


# Main entry point
def main() -> None:
    """Run StratCP evaluation for multiclass WSI classification."""
    args = parse_args()

    # Selected CP methods (normalized to lowercase)
    methods = [m.strip().lower() for m in args.cp_methods]

    # α grid used for all components (baselines, vanilla CP, StratCP)
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)

    # Set up directories
    ensure_directory(args.results_dir)
    eval_dir = os.path.join(
        args.results_dir,
        f"stratcp_eval_results",
    )
    ensure_directory(eval_dir)

    # Load predictions & dataset metadata
    model_preds_path = os.path.join(args.results_dir, "uni_eval_results", "uni_results_dict.pkl")
    model_preds = load_results_dict(model_preds_path)
    test_slide_ids = list(model_preds.keys())
    test_slide_ids_compatible = [
        slide_id.split('-')[0] + '-' + slide_id.split('-')[1] + '-' + slide_id.split('-')[2] for slide_id in test_slide_ids
    ]

    dataset_csv_path = os.path.join(args.results_dir, "time_to_mortality_tcga_metadata.csv")
    dataset_df = pd.read_csv(dataset_csv_path)
    # dataset_df['full_case_id'] = dataset_df['case_submitter_id'].astype(str) + '-' + dataset_df['case_id'].astype(str)

    dataset_id_to_test_slide_id = {}
    for test_id, dataset_id in zip(test_slide_ids, test_slide_ids_compatible):
        if dataset_id not in dataset_id_to_test_slide_id:
            dataset_id_to_test_slide_id[dataset_id] = test_id

    dataset_test_df = dataset_df.loc[
        dataset_df.case_submitter_id.isin(test_slide_ids_compatible)]
    dataset_test_df['slide_id'] = dataset_test_df['case_submitter_id'].apply(
        lambda x: dataset_id_to_test_slide_id.get(x, None)
    )
    
    dataset_test_df['mu_pred'] = [
        float(model_preds[dataset_id_to_test_slide_id[dataset_id]]['mu_pred']) 
        for dataset_id in dataset_test_df.case_submitter_id.values
    ]
    dataset_test_df['log_sigma'] = [
        float(model_preds[dataset_id_to_test_slide_id[dataset_id]]['log_sigma'])
        for dataset_id in dataset_test_df.case_submitter_id.values
    ]
    dataset_test_df['sigma'] = dataset_test_df['log_sigma'].apply(math.exp)
    
    study_end_time_days = args.study_end_time_year * 365.25
    dataset_test_df = apply_administrative_censoring(dataset_test_df, study_end_time_days)

    # Split creation / loading (case-level stratification by pat_id)
    test_size = args.test_prop / (args.test_prop + args.calib_prop)
    splits_path = os.path.join(args.results_dir, f"calib_test_splits_n_{args.n_splits}.pkl")

    # Optionally force re-creation of splits by deleting cached file
    if args.overwrite_split_cache and os.path.exists(splits_path):
        os.remove(splits_path)
        print(f"Removed existing split cache at {splits_path} (overwrite_split_cache=True).")

    split_results = load_or_create_splits(
        dataset_test_df,
        test_size,
        args.n_splits,
        args.random_state,
        splits_path,
        patient_id_col="case_submitter_id",
        label_col="event",
    )

    # Columns to use as covariates in the Cox model for IPCW weight estimation (must be present in df_cal)
    covar_cols = ["age_at_index", "gender_male", "mu_pred"]

    # Get normalized favorable threshold for survival (e.g., 1.5 years → 0.3 if study end time is 5 years)
    favorable_thresh_norm = args.favorable_thresh_year / args.study_end_time_year

    # Evaluate each split (with per-split caching)
    split_to_baseline: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_vanilla_cp: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_stratcp: Dict[int, Dict[str, pd.DataFrame]] = {}

    for split_idx, split_info in split_results.items():
        print("-" * 80)
        print(f"Processing split {split_idx + 1}/{args.n_splits}")
        print("-" * 80)

        # Extract calibration & test arrays for this split
        splits_dict = extract_split_arrays_tte(
            split_info,
            dataset_test_df,
            model_preds,
            patient_id_col="case_submitter_id",
            slide_id_col="slide_id",
        )
        calib_mu_pred = splits_dict["calib_mu_pred"]
        calib_sigma_hat = splits_dict["calib_sigma_hat"]
        calib_labels = splits_dict["calib_labels"]
        calib_cases = splits_dict["calib_case_ids"]

        test_mu_pred = splits_dict["test_mu_pred"]
        test_sigma_hat = splits_dict["test_sigma_hat"]
        test_labels = splits_dict["test_labels"]
        test_cases = splits_dict["test_case_ids"]

        # Get corresponding df_cal and df_test for this split
        dataset_test_df_split = dataset_test_df.copy()
        dataset_test_df_split.set_index('case_submitter_id', inplace=True)

        df_cal = dataset_test_df_split.loc[calib_cases].reset_index(drop=True)
        df_test = dataset_test_df_split.loc[test_cases].reset_index(drop=True)

        w_ipcw, cph_model = get_ipcw_weights(
            df_cal,
            covar_cols=covar_cols,
            n_folds=args.n_folds,
            clip_eps=args.clip_eps,
            random_state=args.random_state + split_idx,
        )

        # Per-split cache paths
        baseline_cache_path = os.path.join(
            eval_dir,
            BASELINE_CACHE_TEMPLATE.format(split_idx=split_idx, n_splits=args.n_splits),
        )
        vanilla_cache_path = os.path.join(
            eval_dir,
            VANILLA_CP_CACHE_TEMPLATE.format(split_idx=split_idx, n_splits=args.n_splits),
        )
        stratcp_cache_path = os.path.join(
            eval_dir,
            STRATCP_CACHE_TEMPLATE.format(split_idx=split_idx, n_splits=args.n_splits),
        )

        
        if (not args.overwrite_eval_cache) and os.path.exists(baseline_cache_path):
            with open(baseline_cache_path, "rb") as f:
                baseline_results = pickle.load(f)
            print(f"  Loaded baselines from {baseline_cache_path}")
        else:
            baseline_results = compute_baselines_for_split_tte(
                alphas=alpha_grid,
                df_test=df_test,
                mu_pred=test_mu_pred,
                sigma_hat=test_sigma_hat,
                favorable_thresh_norm=favorable_thresh_norm,
                censor_model=cph_model,
                covar_cols=covar_cols,
                clip_eps=args.clip_eps,
                study_end_time_year=args.study_end_time_year,
                pbar_desc="Baselines (Top1 + LNQ)",
            )
            with open(baseline_cache_path, "wb") as f:
                pickle.dump(baseline_results, f)
            print(f"  Saved baselines to {baseline_cache_path}")
        split_to_baseline[split_idx] = baseline_results

        if (not args.overwrite_eval_cache) and os.path.exists(vanilla_cache_path):
            with open(vanilla_cache_path, "rb") as f:
                vanilla_results = pickle.load(f)
            print(f"  Loaded vanilla CP from {vanilla_cache_path}")
        else:
            vanilla_results = run_vanilla_cp_for_split_tte(
                alpha_grid,
                calib_labels,
                calib_mu_pred,
                calib_sigma_hat,
                df_test,
                test_mu_pred,
                test_sigma_hat,
                favorable_thresh_norm=favorable_thresh_norm,
                censor_model=cph_model,
                covar_cols=covar_cols,
                study_end_time_year=args.study_end_time_year,
                pbar_desc="Vanilla CP"
            )
            with open(vanilla_cache_path, "wb") as f:
                pickle.dump(vanilla_results, f)
            print(f"  Saved vanilla CP to {vanilla_cache_path}")
        split_to_vanilla_cp[split_idx] = {'tte_cp': vanilla_results}

        # Stratified CP
        if (not args.overwrite_eval_cache) and os.path.exists(stratcp_cache_path):
            with open(stratcp_cache_path, "rb") as f:
                stratcp_results = pickle.load(f)
            print(f"  Loaded StratCP from {stratcp_cache_path}")
        else:
            stratcp_results = compute_stratcp_survival_for_split(
                alpha_grid,
                calib_labels,
                calib_mu_pred,
                calib_sigma_hat,
                df_test,
                test_mu_pred,
                test_sigma_hat,
                favorable_thresh_norm=favorable_thresh_norm,
                censor_model=cph_model,
                covar_cols=covar_cols,
                w_ipcw=w_ipcw,
                study_end_time_year=args.study_end_time_year,
                pbar_desc="StratCP (two-stage survival)"
            )
            with open(stratcp_cache_path, "wb") as f:
                pickle.dump(stratcp_results, f)
            print(f"  Saved StratCP to {stratcp_cache_path}")
        split_to_stratcp[split_idx] = {'tte_cp': stratcp_results}

        # breakpoint()
    # Persist aggregated per-split dictionaries for reuse
    with open(os.path.join(eval_dir, GLOBAL_BASELINE_CACHE), "wb") as f:
        pickle.dump(split_to_baseline, f)
    with open(os.path.join(eval_dir, GLOBAL_VANILLA_CP_CACHE), "wb") as f:
        pickle.dump(split_to_vanilla_cp, f)
    with open(os.path.join(eval_dir, GLOBAL_STRATCP_CACHE), "wb") as f:
        pickle.dump(split_to_stratcp, f)
    print("Saved all split-level results to disk.")

    # Aggregate across splits and summarize at a fixed α
    alpha_range = (float(args.alpha_aggr_min), float(args.alpha_aggr_max))

    aggr_baseline, se_baseline = aggregate_conformal_results(split_to_baseline, method="mean", alpha_range=alpha_range)
    aggr_vanilla, se_vanilla = aggregate_conformal_results(split_to_vanilla_cp, method="mean", alpha_range=alpha_range)
    aggr_stratcp, se_stratcp = aggregate_conformal_results(split_to_stratcp, method="mean", alpha_range=alpha_range)

    summary_sources = [
        ("baseline", aggr_baseline, se_baseline),
        ("vanilla_cp", aggr_vanilla, se_vanilla),
        ("stratified_cp", aggr_stratcp, se_stratcp),
    ]

    # Core metrics common to all methods / groups
    metrics = (
        "mgn_cov_ipcw",
        "mgn_size",
        "selected_coverage_ipcw",
        "unselected_coverage_ipcw",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    )

    summary_df = summarize_methods_at_alpha(
        summary_sources=summary_sources,
        alpha=float(args.alpha_fixed),
        metrics=metrics,
        include_se=bool(args.include_se),
        nearest=True,
        atol=float(args.nearest_tol),
    )

    # Pretty-print final summary
    print(f"===== Final summary at alpha={args.alpha_fixed:.3f} (nearest on grid) =====")
    for _, row in summary_df.iterrows():
        print(f"=== {row['source']:<12} | {row['method']} ===")
        vals = row.drop(
            ["source", "method", "alpha_requested", "alpha_selected"],
            errors="ignore",
        )
        print(vals.to_frame(name="value"))

    return


if __name__ == "__main__":
    main()
