"""
Evaluate whole-slide image (WSI) classifiers for **IDH mutation status prediction**
using several conformal prediction (CP) settings, including the proposed
Stratified CP (StratCP) pipeline, and summarize performance over α-ranges
and at a fixed α.

Task / labels
-------------
The target task is **IDH mutation status** (e.g., in diffuse glioma).
Labels are assumed to be binary:

    0 → IDH wild-type (non-carrier)
    1 → IDH mutant (carrier)

All metrics and summaries in this script are computed with respect to these
two classes, although many of the utilities generalize to more classes.

Workflow
--------
(A) Load per-slide predictions and dataset metadata.
(B) Build (or load) N stratified splits at the *case* level into calibration
    and test partitions.
(C) For each split:
      - Compute / load baseline metrics (Top-1, naive threshold).
      - Compute / load vanilla CP metrics for requested methods (TPS / APS / RAPS).
      - Compute / load Stratified CP metrics for requested methods.
(D) Persist all split-level results to disk.
(E) Aggregate the per-split results (mean and standard error) over a user-chosen
    α-range.
(F) Summarize metrics at a user-chosen fixed α and print tidy tables.

Command-line arguments (see `parse_args()` for full details)
------------------------------------------------------------
Key flags:
  --results_dir   Root folder containing prediction artifacts and where outputs
                  will be saved.
  --seed          Random seed for experiment bookkeeping (not used in splitting).
  --random_state  Base RNG seed for stratified splits (each split adds split_idx).
  --calib_prop    Proportion of calibration cases among (calib + test).
  --test_prop     Proportion of test cases among (calib + test).
  --n_splits      Number of independent stratified case-level splits.
  --cp_methods    Space-separated list among: tps aps raps.
  --alpha_fixed   α at which to print the final comparison table.
  --alpha_min/max/points  Define the α-grid for evaluation.
  --alpha_aggr_min/max    Define the α-range used for aggregation.
  --return_per_class_metrics  If set, include per-class metrics in outputs.
  --eligibility   Eligibility mode for StratCP (e.g., "per_class" or "overall").

Expected inputs on disk
-----------------------
Under `--results_dir`:
  - uni_eval_results/uni_results_dict.pkl
      A pickled dict mapping:
          slide_id -> {"prob": np.ndarray of shape (n_classes,),
                        "label": int}
  - tumor_idh_mutation_status.csv
      CSV with at least the columns:
          slide_id, case_id, label

Outputs on disk
---------------
Per-split caches under {results_dir}/stratcp_eval_results/:
  - top1_thresh_results_split_{i}_of_{N}.pkl
  - cp_vanilla_results_split_{i}_of_{N}.pkl
  - stratcp_results_split_{i}_of_{N}.pkl

Global (all-splits) caches under {results_dir}/stratcp_eval_results/:
  - split_to_baseline_top_1_thresh_results.pkl
  - split_to_cp_vanilla_results.pkl
  - split_to_stratcp_results.pkl

Assumptions / requirements
--------------------------
  - Binary labels {0, 1} with:
        0 → IDH wild-type
        1 → IDH mutant (carrier)
    Constants CLASS_ZERO / CLASS_ONE are set accordingly, but the underlying
    utilities may support more classes.
  - The `stratcp` package is available and provides:
      * StratifiedCP
      * compute_score_tps / compute_score_aps / compute_score_raps
      * conformal (core CP set constructor)
      * helper functions imported from `stratcp.eval_utils`.
  - The helper functions used for aggregation / summarization are imported from:
      `stratcp.eval_utils` (see imports below).

Example usage
-------------
  python idh_mut_status_pred.py \\
      --results_dir data/uni_pathology_tasks/idh_mutation_status_pred \\
      --random_state 42 \\
      --calib_prop 0.15 --test_prop 0.20 \\
      --n_splits 10 \\
      --cp_methods aps \\
      --alpha_fixed 0.05

Notes
-----
  - Caching is enabled at split-level granularity; re-runs will be fast if
    caches already exist.
  - Aggregation computes split-wise mean and standard error (SE) within a
    specified α-range.
  - The final printed summary uses `--alpha_fixed`; if the exact α is not on
    the grid, the nearest α is used (with a configurable tolerance).
"""

import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from stratcp.eval_utils import (
    aggregate_conformal_results,
    load_or_create_splits,
    summarize_methods_at_alpha,
    extract_split_arrays,
    compute_baselines_for_split,
    run_vanilla_cp_for_split,
    run_stratified_cp_for_split,
)
from stratcp.stratified import StratifiedCP  # imported for completeness / external use

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
    parser = argparse.ArgumentParser(
        description="Configurations for WSI conformal prediction evaluation."
    )

    # Core configuration for I/O and experiment splits
    parser.add_argument(
        "--results_dir",
        default="data/uni_pathology_tasks/idh_mutation_status_pred",
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
    ensure_directory(args.results_dir)

    # Step 1: Load per-slide predictions and dataset metadata
    results_dict_test_path = os.path.join(
        args.results_dir, "uni_eval_results", "uni_results_dict.pkl"
    )
    results_dict_test = load_results_dict(results_dict_test_path)

    dataset_csv_path = os.path.join(
        args.results_dir, "tumor_idh_mutation_status.csv"
    )
    dataset_test_df = load_dataset(dataset_csv_path, list(results_dict_test.keys()))

    # Step 2: Build (or load) stratified calibration/test splits at case level
    # test_size is the fraction of cases reserved for test among calib+test
    test_size = args.test_prop / (args.test_prop + args.calib_prop)
    split_cache_path = os.path.join(
        args.results_dir, f"calib_test_splits_n_{args.n_splits}.pkl"
    )
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
            results_dict_test,
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
        include_se=True,   # set False if you do not want *_se columns
        nearest=True,      # set False to require an exact alpha match
        atol=5e-3,         # tolerance for nearest-match lookup
    )

    # Step 6: Print final summary tables for each (source, method)
    for _, row in summary_df.iterrows():
        print(f"\n=== {row['source']:<12} | {row['method']} ===")
        vals = row.drop(
            ["source", "method", "alpha_requested", "alpha_selected"]
        )
        print(vals.to_frame(name="value"))

    return


if __name__ == "__main__":
    main()
