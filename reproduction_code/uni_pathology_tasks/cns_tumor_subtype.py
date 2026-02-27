"""
cns_tumor_subtype.py

Reproduction analysis script for StratCP on the CNS tumor subtype prediction task
from whole-slide imaging (WSI) model outputs.

This module evaluates and summarizes error-controlled decision-making pipelines using
the StratCP framework (mims-harvard/StratCP) for a multiclass CNS tumor subtype
classification setting. Given cached per-slide model predictions (class probabilities
and labels) and slide-level metadata, the script constructs or reuses stratified
case-level calibration/test splits and performs split-wise conformal evaluation over
an alpha grid.

For each split, the script computes:
    1) Baseline methods (e.g., top-1 / threshold-based summaries),
    2) Vanilla conformal prediction methods (TPS / APS / RAPS), and
    3) Stratified conformal prediction (StratCP), with configurable eligibility.

StratCP eligibility modes and selected-side guarantee interpretation
-------------------------------------------------------------------
This script defaults to:
    --eligibility overall

Supported eligibility settings (passed to `run_stratified_cp_for_split`) include:
    - "overall" (default):
        Eligibility/selection is defined globally across all predicted classes.
        The selected-side error control target is applied to the pooled selected
        predictions under a single pre-specified error budget alpha (subject to
        the assumptions/guarantees of StratCP).

    - "per_class":
        Eligibility/selection is defined separately for each predicted class.
        The selected-side error control target is applied at the class-specific
        selected subset level (i.e., separate control target per predicted class)
        under the nominal error budget alpha.

In both cases, unselected (deferred) samples are handled via conformal prediction
sets, and the script summarizes selected/unselected coverage and set-size behavior.

The script additionally supports optional grade-consistency analyses for multiclass
prediction sets, including:
    - grade-consistent set construction (for compatible score builders), and/or
    - grade-range consistency diagnostics on the unselected (deferred) subset,
      stratified by prediction-set size bins.

To support reproducible analysis and efficient reruns, the script caches split-level
evaluation outputs and can optionally overwrite cached splits/results. It then
aggregates metrics across splits (mean and standard error over a user-specified alpha
range), saves reusable aggregated sources, and prints final summary tables at a target
alpha (with nearest-alpha matching if needed).

Typical use case:
    - Reproduce and compare baseline, vanilla CP, and StratCP behavior on the CNS
      tumor subtype WSI task using precomputed model probabilities.
    - Analyze coverage, set size, and deferred-set behavior under alpha sweeps.
    - Evaluate grade-range consistency diagnostics for deferred prediction sets.
    - Compare global vs class-specific StratCP eligibility via the `--eligibility`
      flag.

Example executions:
    # Default multiclass StratCP (overall eligibility)
    python cns_tumor_subtype.py \
        --results_dir ../../data/uni_pathology_tasks/cns_tumor_subtype \
        --cp_methods aps \
        --n_splits 10 \
        --alpha_fixed 0.05 \
        --eligibility overall

    # Per-class eligibility with grade-consistency diagnostics enabled
    python cns_tumor_subtype.py \
        --results_dir ../../data/uni_pathology_tasks/cns_tumor_subtype \
        --cp_methods aps raps \
        --n_splits 10 \
        --alpha_fixed 0.05 \
        --eligibility per_class \
        --grade_consist_set \
        --grade_consist_eval True

    # Force recomputation of split/eval caches
    python cns_tumor_subtype.py \
        --results_dir ../../data/uni_pathology_tasks/cns_tumor_subtype \
        --cp_methods aps \
        --overwrite_split_cache \
        --overwrite_eval_cache
"""

from __future__ import annotations

# Standard library imports
import argparse
import os
import pickle
from typing import Any, Dict, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd

# Project imports
from stratcp.eval_utils import (
    aggregate_conformal_results,
    compute_baselines_for_split,
    extract_split_arrays,
    load_or_create_splits,
    run_stratified_cp_for_split,
    run_vanilla_cp_for_split,
    summarize_methods_at_alpha,
)

# Constants
BASELINE_CACHE_TEMPLATE = "top1_thresh_results_split_{split_idx}_of_{n_splits}.pkl"
VANILLA_CP_CACHE_TEMPLATE = "cp_vanilla_results_split_{split_idx}_of_{n_splits}.pkl"
STRATCP_CACHE_TEMPLATE = "stratcp_results_split_{split_idx}_of_{n_splits}.pkl"

GLOBAL_BASELINE_CACHE = "split_to_baseline_top_1_thresh_results.pkl"
GLOBAL_VANILLA_CP_CACHE = "split_to_cp_vanilla_results.pkl"
GLOBAL_STRATCP_CACHE = "split_to_stratcp_results.pkl"
GLOBAL_SUMMARY_SOURCES = "summary_sources.pkl"

# Default bins for prediction-set size in grade consistency diagnostics
# Last bin (2, 50) aggregates “all sizes >1” for convenience.
DEFAULT_SIZE_BINS: List[Tuple[int, int]] = [(2, 4), (5, 7), (8, 10), (11, 50), (2, 50)]


# CLI parsing
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for WSI multiclass StratCP evaluation."""
    parser = argparse.ArgumentParser(description="WSI multiclass evaluation with StratCP (method-selective).")

    # I/O and bookkeeping
    parser.add_argument(
        "--results_dir",
        default="../../data/uni_pathology_tasks/cns_tumor_subtype",
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
        default=0.15,
        help="Proportion of calibration cases among (calib + test).",
    )
    parser.add_argument(
        "--test_prop",
        type=float,
        default=0.20,
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
        default=0.01,
        help="Minimum alpha value for the evaluation grid.",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.20,
        help="Maximum alpha value for the evaluation grid.",
    )
    parser.add_argument(
        "--alpha_points",
        type=int,
        default=20,
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

    # Grade-consistency / diagnostics (optional)
    parser.add_argument(
        "--grade_consist_set",
        action="store_true",
        default=False,
        help=(
            "If set, use grade-consistent score builders for APS (utility + "
            "block similarity) and emit grade-range diagnostics."
        ),
    )
    # Run grade-consistency evaluation even if not using grade-consistent set construction (for ablation/comparison).
    parser.add_argument(
        "--grade_consist_eval",
        default=True,
        help=(
            "If set, run grade-range consistency evaluation on the unselected "
            "subset even if not using grade-consistent set construction."
        ),
    )

    # Eligibility mode for StratCP (e.g., 'per_class' or 'overall')
    parser.add_argument(
        "--eligibility",
        type=str,
        default="overall",
        help="Eligibility criterion for StratCP ('per_class' or 'overall'; default: 'per_class').",
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


def build_grade_map(
    dataset_df: pd.DataFrame,
    diagnosis_col: str = "diagnosis",
    grade_col: str = "grade",
) -> Tuple[Dict[Any, List[int]] | None, np.ndarray | None]:
    """Build a mapping grade → list of label IDs, if grade information exists.

    This function assumes that:
      • `diagnosis_col` contains the textual/categorical label per slide.
      • `grade_col` contains the grade for each diagnosis (may have missing entries).

    Returns:
        (grade_to_label_ids, all_labels) where:
          - grade_to_label_ids: Dict[grade, List[int]] mapping each grade to the
            integer class indices belonging to that grade.
          - all_labels: np.ndarray of all label IDs that appear in grade_to_label_ids.

        If either column is missing, returns (None, None).
    """
    if diagnosis_col not in dataset_df.columns or grade_col not in dataset_df.columns:
        return None, None

    # Define a canonical mapping from diagnosis name → class index
    label_names = sorted(dataset_df[diagnosis_col].unique())
    label_to_id = {name: i for i, name in enumerate(label_names)}

    # Map diagnosis → grade (dropping missing grades)
    diag_to_grade = dataset_df.set_index(diagnosis_col)[grade_col].dropna().to_dict()

    grade_to_ids: Dict[Any, List[int]] = {}
    for diagnosis, grade in diag_to_grade.items():
        if diagnosis in label_to_id:
            grade_to_ids.setdefault(grade, []).append(label_to_id[diagnosis])

    # Any diagnosis that never received a grade is grouped into a catch-all grade "X"
    not_graded = set(label_names) - set(diag_to_grade.keys())
    if not_graded:
        grade_to_ids.setdefault("X", []).extend(label_to_id[n] for n in not_graded)

    all_labels: List[int] = []
    for _, ids in grade_to_ids.items():
        all_labels.extend(ids)

    return grade_to_ids, np.array(all_labels, dtype=int)


# Main entry point
def main() -> None:
    """Run StratCP evaluation for multiclass WSI classification."""
    args = parse_args()

    # Selected CP methods (normalized to lowercase)
    methods = [m.strip().lower() for m in args.cp_methods]

    # α grid used for all components (baselines, vanilla CP, StratCP)
    alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)

    # Set up directories
    eval_results_dir = os.path.join(args.results_dir, "stratcp_eval_results")
    ensure_directory(eval_results_dir)
    eval_dir = os.path.join(
        args.results_dir,
        f"stratcp_eval_results_grade_consist_{args.grade_consist_set}",
        # e.g. data/.../stratcp_eval_results_grade_consist_True
    )
    ensure_directory(eval_dir)

    # Load predictions & dataset metadata
    model_preds_path = os.path.join(args.results_dir, "uni_eval_results", "uni_results_dict.pkl")
    model_preds = load_results_dict(model_preds_path)
    test_ids = list(model_preds.keys())

    dataset_csv_path = os.path.join(args.results_dir, "ebrains_annotation.csv")
    dataset_df = pd.read_csv(dataset_csv_path)

    # Restrict dataset to slides for which we have predictions
    dataset_test_df = dataset_df.loc[dataset_df["uuid"].isin(test_ids)].copy()
    if dataset_test_df.empty:
        raise ValueError(
            "Filtered dataset_test_df is empty. Check that 'uuid' in "
            "ebrains_annotation.csv matches keys in results_dict."
        )

    # Optional grade map (only used when grade_consist_set is True)
    grade_map, all_labels = build_grade_map(dataset_test_df)

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
        patient_id_col="pat_id",
        label_col="diagnosis",
    )

    # Evaluate each split (with per-split caching)
    split_to_baseline: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_vanilla_cp: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_stratcp: Dict[int, Dict[str, pd.DataFrame]] = {}

    for split_idx, split_info in split_results.items():
        print("-" * 80)
        print(f"Processing split {split_idx + 1}/{args.n_splits}")
        print("-" * 80)

        # Extract calibration & test arrays for this split
        calib_probs, calib_labels, test_probs, test_labels = extract_split_arrays(
            split_info,
            dataset_df,
            model_preds,
            patient_id_col="pat_id",
            slide_id_col="uuid",
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

        # Baselines (Top-1, Thresh)
        if (not args.overwrite_eval_cache) and os.path.exists(baseline_cache_path):
            with open(baseline_cache_path, "rb") as f:
                baseline_results = pickle.load(f)
            print(f"  Loaded baselines from {baseline_cache_path}")
        else:
            baseline_results = compute_baselines_for_split(
                alpha_grid,
                test_probs,
                test_labels,
                return_per_class_metrics=False,
            )
            with open(baseline_cache_path, "wb") as f:
                pickle.dump(baseline_results, f)
            print(f"  Saved baselines to {baseline_cache_path}")
        split_to_baseline[split_idx] = baseline_results

        # Vanilla CP
        if (not args.overwrite_eval_cache) and os.path.exists(vanilla_cache_path):
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
                return_per_class_metrics=False,
                grade_consist_eval=bool(args.grade_consist_eval),
                grade_map=grade_map,
                size_bins=DEFAULT_SIZE_BINS,
            )
            with open(vanilla_cache_path, "wb") as f:
                pickle.dump(vanilla_results, f)
            print(f"  Saved vanilla CP to {vanilla_cache_path}")
        split_to_vanilla_cp[split_idx] = vanilla_results

        # Stratified CP
        if (not args.overwrite_eval_cache) and os.path.exists(stratcp_cache_path):
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
                return_per_class_metrics=False,
                grade_consist_set=bool(args.grade_consist_set),
                grade_consist_eval=bool(args.grade_consist_eval),
                grade_map=grade_map,
                size_bins=DEFAULT_SIZE_BINS,
            )
            with open(stratcp_cache_path, "wb") as f:
                pickle.dump(stratcp_results, f)
            print(f"  Saved StratCP to {stratcp_cache_path}")
        split_to_stratcp[split_idx] = stratcp_results

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
        "mgn_cov",
        "mgn_size",
        "selected_coverage",
        "unselected_coverage",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    )
    if args.grade_consist_set or args.grade_consist_eval:
        metrics += ("grade_range_consistency",)

    summary_df = summarize_methods_at_alpha(
        summary_sources=summary_sources,
        alpha=float(args.alpha_fixed),
        metrics=metrics,
        include_se=bool(args.include_se),
        nearest=True,
        atol=float(args.nearest_tol),
    )

    # Store summary_sources
    summary_sources_path = os.path.join(eval_dir, GLOBAL_SUMMARY_SOURCES)
    if not os.path.exists(summary_sources_path):
        with open(summary_sources_path, "wb") as f:
            pickle.dump(summary_sources, f)
        print(f"Saved summary_sources to {summary_sources_path}")

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
