"""
idh_mut_status_pred.py

Reproduction analysis script for StratCP on the IDH mutation status prediction task
from whole-slide imaging (WSI) model outputs.

This module evaluates and summarizes error-controlled decision-making pipelines using
the StratCP framework, as presented in the StratCP repository (mims-harvard/StratCP),
for a binary neuro-oncology classification setting (IDH mutation status). Given cached
per-slide model predictions (class probabilities and labels) and slide-level metadata,
the script constructs or reuses stratified case-level calibration/test splits and runs
split-wise conformal evaluation across an alpha grid.

For each split, the script computes:
    1) Baseline methods (e.g., top-1 / threshold-based summaries),
    2) Vanilla conformal prediction methods (TPS / APS / RAPS), and
    3) Stratified conformal prediction (StratCP), with configurable eligibility.

StratCP eligibility modes and selected-side guarantee interpretation
-------------------------------------------------------------------
This script defaults to:
    --eligibility per_class

Supported eligibility settings (passed to `run_stratified_cp_for_split`) include:
    - "per_class":
        Eligibility/selection is defined separately for each predicted class.
        The selected-side error control target is applied at the class-specific
        selected subset level (i.e., per predicted class), under the nominal
        error budget alpha (subject to the assumptions/guarantees of StratCP).

    - "overall":
        Eligibility/selection is defined globally across all predictions.
        The selected-side error control target is applied to the pooled selected
        predictions as a whole, under a single pre-specified error budget alpha.

In both cases, unselected (deferred) samples are evaluated via conformal prediction
sets (APS/TPS/RAPS), and the script reports selected/unselected metrics accordingly.

To support reproducible analysis and efficient reruns, split-level outputs are cached
to disk and later consolidated into global result dictionaries. The script then
aggregates metrics across splits (mean and standard error over a user-specified alpha
range) and prints final summary tables at a target alpha, including selected/unselected
performance and prediction-set statistics.

Typical use case:
    - Reproduce and compare baseline, vanilla CP, and StratCP behavior on the IDH
      mutation status WSI task using precomputed model probabilities.
    - Inspect coverage, set size, and selection-related metrics under fixed error
      budgets and alpha sweeps.
    - Compare class-specific vs overall StratCP selection behavior via the
      `--eligibility` flag.

Example executions:
    # Default (per-class eligibility; APS; summarize at alpha=0.05)
    python idh_mut_status_pred.py \
        --results_dir ../../data/uni_pathology_tasks/idh_mutation_status_pred \
        --cp_methods aps \
        --n_splits 10 \
        --alpha_fixed 0.05 \
        --eligibility per_class

    # Alternative: overall eligibility with APS + RAPS
    python idh_mut_status_pred.py \
        --results_dir ../../data/uni_pathology_tasks/idh_mutation_status_pred \
        --cp_methods aps raps \
        --n_splits 10 \
        --alpha_fixed 0.05 \
        --eligibility overall
"""

import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stratcp.eval_utils import (
    aggregate_conformal_results,
    compute_baselines_for_split,
    extract_split_arrays,
    load_or_create_splits,
    run_stratified_cp_for_split,
    run_vanilla_cp_for_split,
    summarize_methods_at_alpha,
)

# Convenience constants for binary-label use cases; not strictly required by
# this script, but kept for clarity and downstream compatibility.
CLASS_ZERO, CLASS_ONE = 0, 1

# Templates for per-split cache files.
BASELINE_CACHE_TEMPLATE = "top1_thresh_results_split_{split_idx}_of_{n_splits}.pkl"
VANILLA_CP_CACHE_TEMPLATE = "cp_vanilla_results_split_{split_idx}_of_{n_splits}.pkl"
STRATCP_CACHE_TEMPLATE = "stratcp_results_split_{split_idx}_of_{n_splits}.pkl"

# Filenames for global (all-splits) caches.
GLOBAL_BASELINE_CACHE = "split_to_baseline_top_1_thresh_results.pkl"
GLOBAL_VANILLA_CP_CACHE = "split_to_cp_vanilla_results.pkl"
GLOBAL_STRATCP_CACHE = "split_to_stratcp_results.pkl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for WSI StratCP evaluation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Configurations for WSI conformal prediction evaluation.")

    # Core configuration for I/O and experiment splits
    parser.add_argument(
        "--results_dir",
        default="../../data/uni_pathology_tasks/idh_mutation_status_pred",
        help="Directory containing inputs and where results will be saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility (bookkeeping).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Base random state for stratified splits (split_idx is added).",
    )
    parser.add_argument(
        "--calib_prop",
        type=float,
        default=0.15,
        help="Proportion of calibration cases among (calibration + test).",
    )
    parser.add_argument(
        "--test_prop",
        type=float,
        default=0.20,
        help="Proportion of test cases among (calibration + test).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=10,
        help="Number of stratified case-level splits to evaluate.",
    )

    # CP methods and alpha configuration
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
        help="Fixed alpha value for summary reporting.",
    )

    # α-grid for evaluation
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.025,
        help="Minimum alpha value for the evaluation grid.",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.325,
        help="Maximum alpha value for the evaluation grid.",
    )
    parser.add_argument(
        "--alpha_points",
        type=int,
        default=25,
        help="Number of alpha points in the evaluation grid.",
    )

    # Aggregation / summary α-range
    parser.add_argument(
        "--alpha_aggr_min",
        type=float,
        default=0.025,
        help="Lower bound for alpha-range aggregation.",
    )
    parser.add_argument(
        "--alpha_aggr_max",
        type=float,
        default=0.3,
        help="Upper bound for alpha-range aggregation.",
    )

    # Optional: whether to include per-class metrics in all outputs
    parser.add_argument(
        "--return_per_class_metrics",
        action="store_true",
        help="If set, return per-class metrics in baseline / CP evaluations.",
    )

    # Eligibility mode for StratCP (e.g., 'per_class' or 'overall')
    parser.add_argument(
        "--eligibility",
        type=str,
        default="per_class",
        help="Eligibility criteria for StratCP (default: 'per_class').",
    )

    return parser.parse_args()


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Path to the directory.
    """
    os.makedirs(path, exist_ok=True)


def load_results_dict(results_path: str) -> Dict[str, Dict[str, Any]]:
    """Load cached per-slide prediction results.

    Args:
        results_path: Path to the pickled `uni_results_dict.pkl` file.

    Returns:
        A dictionary mapping slide IDs to a dict with at least:
            {"prob": np.ndarray, "label": int}.

    Raises:
        FileNotFoundError: If `results_path` does not exist.
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing predictions file: {results_path}")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results_dict_test from {results_path}")
    return results


def load_dataset(csv_path: str, test_slide_ids: List[str]) -> pd.DataFrame:
    """Load dataset metadata and restrict to slides that have predictions.

    Args:
        csv_path: Path to the dataset CSV (e.g., tumor_idh_mutation_status.csv).
        test_slide_ids: List of slide IDs for which predictions exist.

    Returns:
        A DataFrame containing only rows whose slide_id is in `test_slide_ids`.

    Raises:
        ValueError: If the filtered dataset is empty.
    """
    dataset_df = pd.read_csv(csv_path)
    dataset_test_df = dataset_df.loc[dataset_df["slide_id"].isin(test_slide_ids)].copy()
    if dataset_test_df.empty:
        raise ValueError("Filtered dataset is empty; verify slide IDs and CSV path.")
    return dataset_test_df


def main() -> None:
    """Entry point: load data, run baselines/CP, aggregate, and summarize."""
    args = parse_args()

    # Normalize method names from CLI
    methods = [m.strip().lower() for m in args.cp_methods]

    # Ensure the root results directory exists
    eval_results_dir = os.path.join(args.results_dir, "stratcp_eval_results")
    ensure_directory(eval_results_dir)

    # Step 1: Load per-slide predictions and dataset metadata
    model_preds_path = os.path.join(args.results_dir, "uni_eval_results", "uni_results_dict.pkl")
    model_preds = load_results_dict(model_preds_path)

    dataset_csv_path = os.path.join(args.results_dir, "tumor_idh_mutation_status.csv")
    dataset_test_df = load_dataset(dataset_csv_path, list(model_preds.keys()))
    # Step 2: Build (or load) stratified calibration/test splits at case level
    # test_size is the fraction of cases reserved for test among calib+test
    test_size = args.test_prop / (args.test_prop + args.calib_prop)
    split_cache_path = os.path.join(args.results_dir, f"calib_test_splits_n_{args.n_splits}.pkl")
    split_results = load_or_create_splits(
        dataset_test_df,
        test_size,
        args.n_splits,
        args.random_state,
        split_cache_path,
    )

    # α-grid to sweep over for baselines and CP methods
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)

    # Containers to collect per-split results
    split_to_baseline: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_vanilla_cp: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_stratcp: Dict[int, Dict[str, pd.DataFrame]] = {}

    # Step 3: Split-wise evaluation with caching
    for split_idx, split_info in split_results.items():
        print("-" * 80)
        print(f"Processing split {split_idx + 1}/{args.n_splits}")
        print("-" * 80)

        # Extract calibration/test arrays for this split
        calib_probs, calib_labels, test_probs, test_labels = extract_split_arrays(
            split_info,
            dataset_test_df,
            model_preds,
        )

        # Construct per-split cache paths
        baseline_cache_path = os.path.join(
            args.results_dir,
            "stratcp_eval_results",
            BASELINE_CACHE_TEMPLATE.format(
                split_idx=split_idx,
                n_splits=args.n_splits,
            ),
        )
        vanilla_cache_path = os.path.join(
            args.results_dir,
            "stratcp_eval_results",
            VANILLA_CP_CACHE_TEMPLATE.format(
                split_idx=split_idx,
                n_splits=args.n_splits,
            ),
        )
        stratcp_cache_path = os.path.join(
            args.results_dir,
            "stratcp_eval_results",
            STRATCP_CACHE_TEMPLATE.format(
                split_idx=split_idx,
                n_splits=args.n_splits,
            ),
        )

        # Baselines (Top-1, naive threshold)
        if os.path.exists(baseline_cache_path):
            with open(baseline_cache_path, "rb") as f:
                baseline_results = pickle.load(f)
            print(f"  Loaded baselines from {baseline_cache_path}")
        else:
            baseline_results = compute_baselines_for_split(
                alpha_grid,
                test_probs,
                test_labels,
                return_per_class_metrics=args.return_per_class_metrics,
            )
            with open(baseline_cache_path, "wb") as f:
                pickle.dump(baseline_results, f)
            print(f"  Saved baselines to {baseline_cache_path}")
        split_to_baseline[split_idx] = baseline_results

        # Vanilla CP (APS/TPS/RAPS)
        if os.path.exists(vanilla_cache_path):
            with open(vanilla_cache_path, "rb") as f:
                vanilla_results = pickle.load(f)
            print(f"  Loaded vanilla CP from {vanilla_cache_path}")
        else:
            vanilla_results = run_vanilla_cp_for_split(
                alpha_grid,
                calib_probs,
                calib_labels,
                test_probs,
                test_labels,
                methods,
                return_per_class_metrics=args.return_per_class_metrics,
            )
            with open(vanilla_cache_path, "wb") as f:
                pickle.dump(vanilla_results, f)
            print(f"  Saved vanilla CP to {vanilla_cache_path}")
        split_to_vanilla_cp[split_idx] = vanilla_results

        # Stratified CP (StratCP)
        if os.path.exists(stratcp_cache_path):
            with open(stratcp_cache_path, "rb") as f:
                stratcp_results = pickle.load(f)
            print(f"  Loaded StratCP from {stratcp_cache_path}")
        else:
            stratcp_results = run_stratified_cp_for_split(
                alpha_grid,
                calib_probs,
                calib_labels,
                test_probs,
                test_labels,
                methods,
                eligibility=args.eligibility,
                return_per_class_metrics=args.return_per_class_metrics,
            )
            with open(stratcp_cache_path, "wb") as f:
                pickle.dump(stratcp_results, f)
            print(f"  Saved StratCP to {stratcp_cache_path}")
        split_to_stratcp[split_idx] = stratcp_results

    # Step 4: Persist aggregated dictionaries for quick reuse
    stratcp_eval_dir = os.path.join(args.results_dir, "stratcp_eval_results")
    with open(os.path.join(stratcp_eval_dir, GLOBAL_BASELINE_CACHE), "wb") as f:
        pickle.dump(split_to_baseline, f)

    with open(os.path.join(stratcp_eval_dir, GLOBAL_VANILLA_CP_CACHE), "wb") as f:
        pickle.dump(split_to_vanilla_cp, f)

    with open(os.path.join(stratcp_eval_dir, GLOBAL_STRATCP_CACHE), "wb") as f:
        pickle.dump(split_to_stratcp, f)

    print("Saved all split-level results to disk.")

    # Step 5: Aggregate and display overall summaries
    alpha_range = (args.alpha_aggr_min, args.alpha_aggr_max)

    # Aggregate baselines
    aggr_results_baseline, se_results_baseline = aggregate_conformal_results(
        split_to_baseline,
        method="mean",
        alpha_range=alpha_range,
    )

    # Aggregate vanilla CP
    aggr_results_vanilla_cp, se_results_vanilla_cp = aggregate_conformal_results(
        split_to_vanilla_cp,
        method="mean",
        alpha_range=alpha_range,
    )

    # Aggregate Stratified CP
    aggr_results_strat_cp, se_results_strat_cp = aggregate_conformal_results(
        split_to_stratcp,
        method="mean",
        alpha_range=alpha_range,
    )

    # Prepare aggregated sources for summary_at_alpha
    summary_sources = [
        ("baseline", aggr_results_baseline, se_results_baseline),
        ("vanilla_cp", aggr_results_vanilla_cp, se_results_vanilla_cp),
        ("stratified_cp", aggr_results_strat_cp, se_results_strat_cp),
    ]

    # Metrics to display at the requested α
    metrics = (
        "mgn_cov",
        "mgn_size",
        "coverage_cls_1_sel",
        "coverage_cls_0_sel",
        "num_sel_cls_1",
        "num_sel_cls_0",
        "unselected_coverage",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    )

    summary_df = summarize_methods_at_alpha(
        summary_sources=summary_sources,
        alpha=args.alpha_fixed,
        metrics=metrics,
        include_se=True,  # set False if you do not want *_se columns
        nearest=True,  # set False to require an exact alpha match
        atol=5e-3,  # tolerance for nearest-match lookup
    )

    # Step 6: Print final summary tables for each (source, method)
    for _, row in summary_df.iterrows():
        print(f"\n=== {row['source']:<12} | {row['method']} ===")
        vals = row.drop(["source", "method", "alpha_requested", "alpha_selected"])
        print(vals.to_frame(name="value"))

    return


if __name__ == "__main__":
    main()
