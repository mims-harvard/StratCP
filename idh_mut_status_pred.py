"""
This script evaluates whole-slide image (WSI) classifiers under several
conformal prediction (CP) settings including the proposed StratCP pipeline 
and summarizes performance at a fixed alpha.

The workflow is:
  (A) Load per-slide predictions and dataset metadata.
  (B) Build (or load) N stratified splits at the *case* level into
      calibration and test partitions.
  (C) For each split:
       - Compute/cached baselines (Top-1, threshold).
       - Compute/cached vanilla CP metrics for requested methods (TPS/APS/RAPS).
       - Compute/cached Stratified CP metrics for requested methods.
  (D) Persist all split-level results.
  (E) Aggregate results across splits (mean and standard error over α range).
  (F) Summarize metrics at a user-chosen fixed α and print tidy tables.

Inputs / CLI flags (see `parse_args()` for full details):
  --results_dir   Root folder containing prediction artifacts and where outputs will be saved
  --seed          Random seed for experiment bookkeeping (not used in splitting)
  --exp_code      Experiment identifier (used for locating outputs)
  --random_state  Base RNG seed for stratified splits (each split adds split_idx)
  --calib_prop    Proportion of calibration cases among (calib + test)
  --test_prop     Proportion of test cases among (calib + test)
  --n_splits      Number of independent stratified case-level splits
  --cp_methods    Space-separated list among: tps aps raps
  --alpha_fixed   α at which to print the final comparison table

Expected inputs on disk:
  - {results_dir}/uni_eval_results/uni_results_dict.pkl
      A pickled dict: slide_id -> {"prob": np.ndarray, "label": int}
  - {results_dir}/tumor_idh_mutation_status.csv
      CSV with at least: slide_id, case_id, label

Outputs on disk:
  Per-split caches in {results_dir}/stratcp_eval_results/:
    - top1_thresh_results_split_{i}_of_{N}.pkl
    - cp_vanilla_results_split_{i}_of_{N}.pkl
    - stratcp_results_split_{i}_of_{N}.pkl
  Global (all-splits) caches:
    - split_to_baseline_top_1_thresh_results.pkl
    - split_to_cp_vanilla_results.pkl
    - split_to_stratcp_results.pkl

Assumptions / requirements:
  - Binary labels {0,1}. Constants CLASS_ZERO/CLASS_ONE are set accordingly.
  - `stratcp` package is available with:
      * StratifiedCP
      * compute_score_tps / compute_score_aps / compute_score_raps
      * conformal (core CP set constructor)
  - The helper functions used for aggregation/summarization are imported from
    stratcp.eval_utils (see imports below).

Example:
  python idh_mut_status_pred.py \\
      --results_dir data/uni_pathology_tasks/idh_mutation_status_pred \\
      --random_state 42 \\
      --calib_prop 0.15 --test_prop 0.20 \\
      --n_splits 10 \\
      --cp_methods aps \\
      --alpha_fixed 0.05

Notes:
  - Caching is enabled at split and method granularity; re-runs will be fast.
  - Aggregation computes split-wise mean (and SE) within an α-range; the
    final printout shows metrics at `--alpha_fixed` using nearest neighbor
    look-up if the exact α is not on the grid (tolerance configurable).


# [TODOs]
- Add pre-computed 500 split results
- Fix issues with Thresh and Top1 not showing some metrics properly (e.g., coverage_cls_one_sel)
"""

import argparse
import os
import pickle
from typing import Dict, Any, Sequence, Tuple, List

import numpy as np
import pandas as pd
import tqdm

from stratcp.stratified import StratifiedCP
# Import score builders and CP from StratCP
from stratcp.conformal.scores import (
    compute_score_tps,
    compute_score_aps,
    compute_score_raps,
)
from stratcp.conformal.core import conformal
from stratcp.eval_utils import (
    evaluate_top1, 
    evaluate_naive_cumulative, 
    stratified_split_return_case_ids, 
    aggregate_conformal_results, 
    summarize_methods_at_alpha
)

CLASS_ZERO, CLASS_ONE = 0, 1
ALPHA_GRID = np.linspace(0.025, 0.95, 75)[:25]
BASELINE_CACHE_TEMPLATE = "top1_thresh_results_split_{split_idx}_of_{n_splits}.pkl"
VANILLA_CP_CACHE_TEMPLATE = "cp_vanilla_results_split_{split_idx}_of_{n_splits}.pkl"
STRATCP_CACHE_TEMPLATE = "stratcp_results_split_{split_idx}_of_{n_splits}.pkl"
GLOBAL_BASELINE_CACHE = "split_to_baseline_top_1_thresh_results.pkl"
GLOBAL_VANILLA_CP_CACHE = "split_to_cp_vanilla_results.pkl"
GLOBAL_STRATCP_CACHE = "split_to_stratcp_results.pkl"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Configurations for WSI evaluation")
    parser.add_argument("--results_dir", default="data/uni_pathology_tasks/idh_mutation_status_pred", help="Directory for saving results")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--exp_code", type=str, default="data/uni_pathology_tasks/idh_mutation_status_pred", help="Experiment code to locate saved outputs")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for data splits")
    parser.add_argument("--calib_prop", type=float, default=0.15, help="Calibration proportion in overall dataset")
    parser.add_argument("--test_prop", type=float, default=0.20, help="Test proportion in overall dataset")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of stratified splits to evaluate")
    parser.add_argument(
        "--cp_methods",
        nargs="+",
        default=["aps"],
        help="Methods to run for CP (space-separated): choices are 'tps', 'aps', 'raps'",
    )
    parser.add_argument(
        "--alpha_fixed",
        type=float,
        default=0.05,
        help="Fixed alpha value for summary reporting",
    )
    return parser.parse_args()


def ensure_directory(path: str) -> None:
    """Create directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def load_results_dict(results_path: str) -> Dict[str, Dict[str, Any]]:
    """Load cached per-slide predictions."""
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing predictions file: {results_path}")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results_dict_test from {results_path}")
    return results


def load_dataset(csv_path: str, test_slide_ids: List[str]) -> pd.DataFrame:
    """Load dataset metadata restricted to slides present in predictions."""
    dataset_df = pd.read_csv(csv_path)
    dataset_test_df = dataset_df.loc[dataset_df["slide_id"].isin(test_slide_ids)].copy()
    if dataset_test_df.empty:
        raise ValueError("Filtered dataset is empty; verify slide IDs and CSV path.")
    return dataset_test_df


def load_or_create_splits(
    dataset_df: pd.DataFrame,
    test_size: float,
    n_splits: int,
    random_state: int,
    cache_path: str,
) -> Dict[int, Dict[str, Any]]:
    """Load cached splits or create new stratified splits at the **case** level.

    Uses `stratified_split_return_case_ids` to produce calibration/test splits,
    caching the resulting case/label mappings on disk for reproducibility.

    Args:
        dataset_df: DataFrame with columns at least ``'case_id'`` and ``'label'`` (and
            potentially multiple ``'slide_id'`` per case elsewhere in the pipeline).
        test_size: Proportion of unique cases to assign to the **test** split (0, 1].
        n_splits: Number of independent stratified splits to generate.
        random_state: Base RNG seed; each split uses ``random_state + split_idx``.
        cache_path: File path to load from / save to (pickle).

    Returns:
        Dict indexed by split index (0..n_splits-1) with:
            - ``'test_cases'``: pd.Series of case_ids in test split
            - ``'calib_cases'``: pd.Series of case_ids in calibration split
            - ``'test_labels'``: pd.Series of labels aligned to ``test_cases``
            - ``'calib_labels'``: pd.Series of labels aligned to ``calib_cases``

    Notes:
        - If ``cache_path`` exists, its content is returned without recomputing.
        - The cache contains only IDs/labels necessary to reproduce the split.
    """
    # Fast path: read from cache if present.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            splits = pickle.load(f)
        print(f"Loaded {len(splits)} split results from {cache_path}")
        return splits

    # Build each split with a different (but deterministic) seed.
    splits: Dict[int, Dict[str, Any]] = {}
    for split_idx in range(n_splits):
        test_cases, calib_cases, test_labels, calib_labels = stratified_split_return_case_ids(
            dataset_df, test_size, random_state=random_state + split_idx
        )
        splits[split_idx] = {
            "test_cases": test_cases,
            "calib_cases": calib_cases,
            "test_labels": test_labels,
            "calib_labels": calib_labels,
        }
        print(
            f"Split {split_idx + 1}/{n_splits}: "
            f"calib={len(calib_cases)}, test={len(test_cases)}"
        )

    # Persist to cache for later re-use.
    with open(cache_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {n_splits} split results to {cache_path}")
    return splits


def extract_split_arrays(
    split_info: Dict[str, Any],
    dataset_df: pd.DataFrame,
    results_dict: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble calibration/test arrays (probs and labels) for a single split.

    Maps slide-level outputs to case-level splits, assigning each slide to
    calibration or test based on its associated case ID.

    Args:
        split_info: One split entry from ``load_or_create_splits`` containing
            ``'calib_cases'`` and ``'test_cases'`` (pd.Series) and their labels.
        dataset_df: DataFrame with columns ``'slide_id'`` and ``'case_id'`` used
            to map slides to cases.
        results_dict: Mapping ``slide_id -> {'prob': array_like, 'label': int}``
            where ``'prob'`` is per-slide class probabilities and ``'label'`` the
            ground-truth class id for that slide.

    Returns:
        Tuple ``(calib_probs, calib_labels, test_probs, test_labels)`` where:
            - calib_probs: np.ndarray of per-slide probabilities for calibration.
            - calib_labels: np.ndarray of per-slide labels for calibration.
            - test_probs: np.ndarray of per-slide probabilities for test.
            - test_labels: np.ndarray of per-slide labels for test.

    Raises:
        ValueError: If a slide’s case_id is not found in either calib or test sets.

    Notes:
        - If a probability array has an extra leading dimension of size 1, it is
          squeezed to keep shapes consistent (mirrors original behavior).
    """
    # Map each slide to its case id.
    slide_to_case = dataset_df.set_index("slide_id")["case_id"].to_dict()

    # Collectors for each partition.
    calib_probs, test_probs = [], []
    calib_labels, test_labels = [], []

    # Case sets for fast membership testing.
    calib_case_set = set(split_info["calib_cases"].values)
    test_case_set = set(split_info["test_cases"].values)

    # Route each slide to calibration or test by its case_id.
    for slide_id, payload in results_dict.items():
        case_id = slide_to_case[slide_id]
        prob = np.asarray(payload["prob"])
        label = payload["label"]

        if case_id in test_case_set:
            test_probs.append(prob)
            test_labels.append(label)
        elif case_id in calib_case_set:
            calib_probs.append(prob)
            calib_labels.append(label)
        else:
            # Defensive: split definitions must cover all cases in results_dict.
            raise ValueError(f"Case ID {case_id} not assigned to calibration or test split.")

    # Convert to arrays; squeeze singleton leading dim if present (shape (n,1,C)).
    calib_probs_arr = np.squeeze(np.array(calib_probs), axis=1) if np.array(calib_probs).ndim == 3 else np.array(calib_probs)
    test_probs_arr = np.squeeze(np.array(test_probs), axis=1) if np.array(test_probs).ndim == 3 else np.array(test_probs)

    return (
        calib_probs_arr,
        np.asarray(calib_labels),
        test_probs_arr,
        np.asarray(test_labels).flatten(),
    )


def compute_baselines_for_split(
    split_idx: int,
    n_splits: int,
    alphas: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    cache_path: str,
) -> Dict[str, pd.DataFrame]:
    """Compute Top-1 and naive threshold (cumulative) baselines for one split.

    Results are cached per split to avoid recomputation.

    Args:
        split_idx: Current split index (0-based for display).
        n_splits: Total number of splits (for progress messages).
        alphas: Array of alpha values to evaluate.
        test_probs: Test class probabilities, shape (n_test, n_classes).
        test_labels: Test labels, shape (n_test,).
        cache_path: Where to cache the summary DataFrames (pickle).

    Returns:
        Dict mapping method name (``'top1'`` or ``'thresh'``) to a DataFrame
        indexed by alpha with columns:
            ['mgn_cov','mgn_size','cls_cond_cov_cls_one','cls_cond_cov_cls_zero',
             'num_sel_for_cls_one','num_sel_for_cls_zero','unselected_coverage',
             'unselected_set_size','num_unsel','num_total']

    Notes:
        - This relies on ``evaluate_top1`` and ``evaluate_naive_cumulative``.
        - The total count ('num_total') is the same across alphas for a split.
    """
    # Cache load path.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        print(f"Loaded baseline metrics (split {split_idx + 1}/{n_splits})")
        return cached

    # Compute metrics for each alpha.
    alpha_to_metrics: Dict[float, Dict[str, Dict[str, float]]] = {}
    for alpha in tqdm.tqdm(alphas, desc=f"Baselines (split {split_idx + 1})"):
        preds_top1 = np.argmax(test_probs, axis=1)
        top1_metrics = evaluate_top1(preds_top1, test_labels)
        thresh_metrics = evaluate_naive_cumulative(test_probs, test_labels, alpha)

        alpha_to_metrics[alpha] = {
            "top1": top1_metrics,
            "thresh": thresh_metrics,
            "num_total": len(test_labels),
        }

    # Assemble per-method DataFrames with consistent columns.
    methods_baseline = ["top1", "thresh"]
    columns = [
        "mgn_cov",
        "mgn_size",
        "cls_cond_cov_cls_one",
        "cls_cond_cov_cls_zero",
        "num_sel_for_cls_one",
        "num_sel_for_cls_zero",
        "unselected_coverage",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    ]

    summary: Dict[str, pd.DataFrame] = {}
    for method in methods_baseline:
        df = pd.DataFrame(index=alphas, columns=columns, dtype=float)
        for alpha in alphas:
            metrics = alpha_to_metrics[alpha][method]
            # First 9 fields come from the metric dicts.
            df.loc[alpha, columns[:9]] = [metrics[col] for col in columns[:9]]
            # num_total is stored separately.
            df.loc[alpha, ["num_total"]] = alpha_to_metrics[alpha]["num_total"]
        summary[method] = df

    # Save to cache and return.
    with open(cache_path, "wb") as fh:
        pickle.dump(summary, fh)
    return summary


def run_vanilla_cp_for_split(
    split_idx: int,
    n_splits: int,
    alphas: np.ndarray,
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    cache_path: str,
    methods: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Run vanilla conformal prediction (TPS/APS/RAPS) for a split, with caching.

    For each requested method, computes prediction sets across alphas and
    aggregates standard metrics (marginal coverage/size, selected coverage by
    class, counts, and unselected summaries).

    Args:
        split_idx: Current split index (0-based).
        n_splits: Total number of splits.
        alphas: Array of alpha values to evaluate.
        calib_probs: Calibration probabilities, shape (n_calib, n_classes).
        calib_labels: Calibration labels (int class ids), shape (n_calib,).
        test_probs: Test probabilities, shape (n_test, n_classes).
        test_labels: Test labels (int class ids), shape (n_test,).
        cache_path: Where to cache summary DataFrames (pickle).
        methods: Subset of ``{"tps","aps","raps"}`` to run.

    Returns:
        Dict[str, pd.DataFrame]: A mapping of method name to a DataFrame of
        metrics indexed by alpha.

    Raises:
        ValueError: If any method in ``methods`` is not one of {"tps","aps","raps"}.

    Notes:
        - This function relies on external utilities assumed to be available in scope:
          ``compute_score_raps``, ``compute_score_aps``, ``compute_score_tps``,
          ``conformal``, constants ``CLASS_ONE``/``CLASS_ZERO``.
        - For performance, calibration/test scores per method are computed once and reused
          across all alphas.
    """
    methods = tuple(m.lower() for m in methods)
    allowed = {"tps", "aps", "raps"}
    invalid = set(methods) - allowed
    if invalid:
        raise ValueError(f"Unknown method(s): {sorted(invalid)}. Allowed: {sorted(allowed)}")

    # Cache load (callers should ensure cache path uniqueness per configuration).
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"Loaded vanilla CP metrics (split {split_idx + 1}/{n_splits})")
        return cached

    # Basic dimensions.
    m = int(len(test_labels))
    n_classes = int(test_probs.shape[1])

    # Reference masks used by `conformal` (preserving original behavior).
    ones_ref = [np.ones((m, len(calib_labels))) for _ in range(n_classes)]

    # Build method score functions once per method.
    def _scores_for_method(method_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if method_name == "raps":
            return compute_score_raps(calib_probs, test_probs, calib_labels)
        if method_name == "aps":
            return compute_score_aps(calib_probs, test_probs, calib_labels)
        if method_name == "tps":
            return compute_score_tps(calib_probs, test_probs, calib_labels)
        raise RuntimeError("Unexpected method name")  # Should be unreachable.

    scores_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        meth: _scores_for_method(meth) for meth in methods
    }

    def _coverage_singleton_by_class(
        size: np.ndarray, cov: np.ndarray, set_mat: np.ndarray, class_id: int
    ) -> float:
        """Coverage among singleton sets whose predicted singleton label == class_id."""
        idx_single = np.where(size == 1)[0]
        if idx_single.size == 0:
            return np.nan
        # For singleton rows, argmax over columns identifies the chosen class.
        singleton_label = np.argmax(set_mat[idx_single], axis=1)
        idx_class = idx_single[singleton_label == class_id]
        if idx_class.size == 0:
            return np.nan
        return float(np.mean(cov[idx_class]))

    # Gather results for each alpha (across methods).
    alpha_to_results: Dict[float, Dict[str, Any]] = {}

    for alpha in tqdm.tqdm(alphas, desc=f"Vanilla CP (split {split_idx + 1})"):
        # Values common to all methods at this alpha.
        alpha_to_results[alpha] = {
            "num_total": m,
            "num_cls_one": int(np.sum(test_labels)),
            "num_cls_zero": int(m - np.sum(test_labels)),
            "methods": {},
        }

        pred_top1 = np.argmax(test_probs, axis=1)  # used in conformal options below

        for method_name in methods:
            calib_scores, test_scores = scores_by_method[method_name]

            # Construct prediction sets for this alpha using the chosen score function.
            set_mat = conformal(
                calib_scores,
                test_scores,
                calib_labels,
                alpha,
                nonempty=True,
                test_max_id=pred_top1,
                if_in_ref=ones_ref,
                class_conditional=False,
            )

            # Coverage indicator for true labels and set sizes.
            cov = set_mat[np.arange(set_mat.shape[0]), test_labels]
            size = np.sum(set_mat, axis=1)

            # Selected = singleton sets; Unselected = non-singletons (matches original convention).
            singleton_mask = (size == 1)
            unsel_idx = np.where(~singleton_mask)[0]
            sel_cover_sum = float(np.sum(cov[singleton_mask]))

            # Marginal coverage over all samples.
            mgn_cov = (
                (float(np.sum(cov[unsel_idx])) + sel_cover_sum) / float(m)
                if m > 0 else np.nan
            )

            # Marginal size over all samples.
            n_selected = m - cov[unsel_idx].shape[0]
            mgn_size = (
                (float(np.sum(size[unsel_idx])) + float(n_selected)) / float(m)
                if m > 0 else np.nan
            )

            # Counts of singleton selections for each class (binary case).
            num_sel_cls_one = int(np.sum(singleton_mask & (set_mat[:, CLASS_ONE] == True)))
            num_sel_cls_zero = int(np.sum(singleton_mask & (set_mat[:, CLASS_ZERO] == True)))

            # Unselected summaries (non-singletons).
            if unsel_idx.size > 0:
                unselected_coverage = float(np.mean(cov[unsel_idx]))
                unselected_set_size = float(np.mean(size[unsel_idx]))
            else:
                unselected_coverage = np.nan
                unselected_set_size = np.nan
            num_unsel = int(unsel_idx.size)

            # Singleton coverage by class (probability of being correct among those singletons).
            cov_cls_one_singleton = _coverage_singleton_by_class(size, cov, set_mat, CLASS_ONE)
            cov_cls_zero_singleton = _coverage_singleton_by_class(size, cov, set_mat, CLASS_ZERO)

            # Store per-method metrics for this alpha.
            alpha_to_results[alpha]["methods"][method_name] = {
                "mgn_cov": mgn_cov,
                "mgn_size": mgn_size,
                "coverage_cls_one_sel": cov_cls_one_singleton,
                "coverage_cls_zero_sel": cov_cls_zero_singleton,
                "num_sel_cls_one": num_sel_cls_one,
                "num_sel_cls_zero": num_sel_cls_zero,
                "unselected_coverage": unselected_coverage,
                "unselected_set_size": unselected_set_size,
                "num_unsel": num_unsel,
                "num_total": m,
            }

    # Convert per-alpha dicts into per-method DataFrames.
    summary: Dict[str, pd.DataFrame] = {}
    for method_name in methods:
        rows, idx = [], []
        for alpha, results in alpha_to_results.items():
            mres = results["methods"][method_name]
            rows.append(
                {
                    "mgn_cov": mres["mgn_cov"],
                    "mgn_size": mres["mgn_size"],
                    "coverage_cls_one_sel": mres["coverage_cls_one_sel"],
                    "coverage_cls_zero_sel": mres["coverage_cls_zero_sel"],
                    "num_sel_cls_one": mres["num_sel_cls_one"],
                    "num_sel_cls_zero": mres["num_sel_cls_zero"],
                    "num_total": results["num_total"],
                    "unselected_coverage": mres["unselected_coverage"],
                    "unselected_set_size": mres["unselected_set_size"],
                    "num_unsel": mres["num_unsel"],
                }
            )
            idx.append(alpha)
        summary[method_name] = pd.DataFrame(rows, index=idx)

    # Cache and return.
    with open(cache_path, "wb") as f:
        pickle.dump(summary, f)
    print(f"Saved vanilla CP metrics (split {split_idx + 1}/{n_splits})")
    return summary


def _safe_mean(arr: np.ndarray) -> float:
    """Mean handling empty arrays."""
    arr = np.asarray(arr)
    return float(arr.mean()) if arr.size > 0 else np.nan


def _selection_to_indices(selection: Any) -> np.ndarray:
    """Convert selection output (bool mask or indices) to explicit integer indices."""
    arr = np.asarray(selection)
    if arr.dtype == bool:
        return np.flatnonzero(arr)
    return arr.astype(int)


def run_stratified_cp_for_split(
    split_idx: int,
    n_splits: int,
    alphas: np.ndarray,
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    cache_path: str,
    methods: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    """Evaluate Stratified CP methods using a StratifiedCP API (cached).

    For each method name in ``methods`` (e.g., 'aps'), fits a StratifiedCP model
    with both overall and per-class eligibility and aggregates key metrics at
    each alpha.

    Args:
        split_idx: Current split index (0-based).
        n_splits: Total number of splits (for progress messages).
        alphas: Array of alpha values to evaluate.
        calib_probs: Calibration probabilities, shape (n_calib, n_classes).
        calib_labels: Calibration labels, shape (n_calib,).
        test_probs: Test probabilities, shape (n_test, n_classes).
        test_labels: Test labels, shape (n_test,).
        cache_path: Where to cache summary DataFrames (pickle).
        methods: Iterable of StratifiedCP score function names (e.g., ['aps']).

    Returns:
        Dict[str, pd.DataFrame]: Mapping from method name to a DataFrame indexed
        by alpha with columns including:
            ['selection_threshold','num_sel','num_unsel','num_total',
             'sel_false_positive_rate_union','coverage_cls_zero_sel','coverage_cls_one_sel',
             'num_sel_cls_zero','num_sel_cls_one','decision_tau_cls_zero','decision_tau_cls_one',
             'mgn_cov','mgn_size','unselected_coverage','unselected_set_size',
             'cls_cond_cov_cls_one','cls_cond_cov_cls_zero'].

    Notes:
        - Depends on external utilities in scope: ``StratifiedCP``, ``_safe_mean``,
          and ``_selection_to_indices``. These must be provided by your codebase.
        - The function computes both overall and per-class eligibility results at
          the same alpha and merges them into a single record.
    """
    # Cache load.
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        print(f"Loaded Stratified CP metrics (split {split_idx + 1}/{n_splits})")
        return cached

    n_classes = test_probs.shape[1]
    pred_labels = np.argmax(test_probs, axis=1)
    method_rows: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}

    for alpha in tqdm.tqdm(alphas, desc=f"Stratified CP (split {split_idx + 1})"):
        for method in methods:
            # ----- Overall eligibility -----
            scp_overall = StratifiedCP(
                score_fn=method,
                alpha_sel=alpha,
                alpha_cp=alpha,
                eligibility="overall",
                nonempty=True,
                rand=True,
            ).fit(calib_probs, calib_labels)

            overall_results = scp_overall.predict(test_probs, test_labels)
            selected_idx = np.asarray(overall_results["selected_idx"])
            unselected_idx = np.asarray(overall_results["unselected_idx"])
            selection_threshold = float(overall_results["threshold"])

            # Unselected sets and sizes, as returned by the API.
            pred_sets_unsel = np.asarray(overall_results["prediction_sets"], dtype=bool)
            unsel_set_sizes = np.asarray(overall_results["set_sizes"], dtype=float)

            selected_count = selected_idx.size
            unselected_count = unselected_idx.size
            total_count = len(test_labels)

            # Selected (singleton) accuracy / FDR.
            selected_correct = (
                int(np.sum(pred_labels[selected_idx] == test_labels[selected_idx]))
                if selected_count > 0
                else 0
            )
            selected_accuracy = (
                float(selected_correct / selected_count) if selected_count > 0 else np.nan
            )
            selected_fdr = float(1 - selected_accuracy) if selected_count > 0 else np.nan

            # Unselected mean coverage/size (if any unselected).
            if unselected_count > 0:
                unsel_labels = test_labels[unselected_idx]
                unsel_cov = pred_sets_unsel[
                    np.arange(unselected_count), unsel_labels
                ].astype(float)
                unsel_mean_cov = _safe_mean(unsel_cov)
                unsel_mean_size = _safe_mean(unsel_set_sizes)
            else:
                unsel_cov = np.array([])
                unsel_mean_cov = np.nan
                unsel_mean_size = np.nan

            # Marginal coverage/size over all samples.
            covered_total = selected_correct + (float(np.sum(unsel_cov)) if unsel_cov.size > 0 else 0.0)
            marginal_cov = float(covered_total / total_count) if total_count > 0 else np.nan
            marginal_size = float(
                (selected_count + np.sum(unsel_set_sizes)) / total_count
            ) if total_count > 0 else np.nan

            # ----- Per-class eligibility -----
            scp_per_class = StratifiedCP(
                score_fn=method,
                alpha_sel=alpha,
                alpha_cp=alpha,
                eligibility="per_class",
                nonempty=True,
                rand=True,
            ).fit(calib_probs, calib_labels)

            per_class_results = scp_per_class.predict(test_probs, test_labels)
            all_selected = per_class_results["all_selected"]
            tau_list = np.asarray(per_class_results["thresholds"], dtype=float)

            # Count selections and per-class FDR for (binary) classes 0 and 1.
            decision_sel_counts = {}
            decision_fdr = {}
            for cls in range(min(2, n_classes)):
                sel_indices = _selection_to_indices(all_selected[cls])
                decision_sel_counts[cls] = int(len(sel_indices))
                decision_fdr[cls] = (
                    float(np.mean(test_labels[sel_indices] != cls))
                    if len(sel_indices) > 0
                    else np.nan
                )

            # Class-conditional coverage among unselected samples (binary only).
            unsel_true = test_labels[unselected_idx] if unselected_count > 0 else np.array([])
            cls_cond_cov_cls_one = _safe_mean(unsel_cov[unsel_true == 1]) if unselected_count > 0 else np.nan
            cls_cond_cov_cls_zero = _safe_mean(unsel_cov[unsel_true == 0]) if unselected_count > 0 else np.nan

            # Aggregate all fields into one record for this (alpha, method).
            method_rows[method].append(
                {
                    "alpha": alpha,
                    "selection_threshold": selection_threshold,
                    "num_sel": selected_count,
                    "num_unsel": unselected_count,
                    "num_total": total_count,
                    "sel_false_positive_rate_union": selected_fdr,
                    "coverage_cls_zero_sel": 1 - decision_fdr.get(0, np.nan),
                    "coverage_cls_one_sel": 1 - decision_fdr.get(1, np.nan),
                    "num_sel_cls_zero": decision_sel_counts.get(0, 0),
                    "num_sel_cls_one": decision_sel_counts.get(1, 0),
                    "decision_tau_cls_zero": tau_list[0] if tau_list.size > 0 else np.nan,
                    "decision_tau_cls_one": tau_list[1] if tau_list.size > 1 else np.nan,
                    "mgn_cov": marginal_cov,
                    "mgn_size": marginal_size,
                    "unselected_coverage": unsel_mean_cov,
                    "unselected_set_size": unsel_mean_size,
                    "cls_cond_cov_cls_one": cls_cond_cov_cls_one,
                    "cls_cond_cov_cls_zero": cls_cond_cov_cls_zero,
                }
            )

    # Final per-method DataFrames indexed by alpha.
    summary: Dict[str, pd.DataFrame] = {}
    for method in methods:
        df = pd.DataFrame(method_rows[method]).set_index("alpha")
        summary[method] = df

    # Cache and return.
    with open(cache_path, "wb") as f:
        pickle.dump(summary, f)
    return summary


def main() -> None:
    """Initialize settings, load dataset, split, and run baseline/CP evaluations."""
    args = parse_args()
    
    # Get CP methods from args
    methods = [m.strip().lower() for m in args.cp_methods]
    ensure_directory(args.results_dir)
    # exp_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
    # ensure_directory(exp_dir)

    results_dict_test_path = os.path.join(args.results_dir, "uni_eval_results", "uni_results_dict.pkl")
    results_dict_test = load_results_dict(results_dict_test_path)

    dataset_csv_path = os.path.join(args.results_dir, "tumor_idh_mutation_status.csv")
    dataset_test_df = load_dataset(dataset_csv_path, list(results_dict_test.keys()))

    test_size = args.test_prop / (args.test_prop + args.calib_prop)
    split_cache_path = os.path.join(
        args.results_dir, f"calib_test_splits_n_{args.n_splits}.pkl"
    )
    split_results = load_or_create_splits(
        dataset_test_df, test_size, args.n_splits, args.random_state, split_cache_path
    )

    split_to_baseline: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_vanilla_cp: Dict[int, Dict[str, pd.DataFrame]] = {}
    split_to_stratcp: Dict[int, Dict[str, pd.DataFrame]] = {}


    # When n_splits set to 500; pre-load existing results if available.

    for split_idx, split_info in split_results.items():
        print("-" * 80)
        print(f"Processing split {split_idx + 1}/{args.n_splits}")
        print("-" * 80)

        calib_probs, calib_labels, test_probs, test_labels = extract_split_arrays(
            split_info, dataset_test_df, results_dict_test
        )

        baseline_cache_path = os.path.join(
            args.results_dir, "stratcp_eval_results",
            BASELINE_CACHE_TEMPLATE.format(split_idx=split_idx, n_splits=args.n_splits),
        )
        vanilla_cache_path = os.path.join(
            args.results_dir, "stratcp_eval_results",
            VANILLA_CP_CACHE_TEMPLATE.format(split_idx=split_idx, n_splits=args.n_splits),
        )
        stratcp_cache_path = os.path.join(
            args.results_dir, "stratcp_eval_results",
            STRATCP_CACHE_TEMPLATE.format(split_idx=split_idx, n_splits=args.n_splits),
        )

        split_to_baseline[split_idx] = compute_baselines_for_split(
            split_idx,
            args.n_splits,
            ALPHA_GRID,
            test_probs,
            test_labels,
            baseline_cache_path,
        )

        split_to_vanilla_cp[split_idx] = run_vanilla_cp_for_split(
            split_idx,
            args.n_splits,
            ALPHA_GRID,
            calib_probs,
            calib_labels,
            test_probs,
            test_labels,
            vanilla_cache_path,
            methods
        )

        split_to_stratcp[split_idx] = run_stratified_cp_for_split(
            split_idx,
            args.n_splits,
            ALPHA_GRID,
            calib_probs,
            calib_labels,
            test_probs,
            test_labels,
            stratcp_cache_path,
            methods
        )


    # Persist aggregated dictionaries for quick reuse.
    with open(os.path.join(args.results_dir, "stratcp_eval_results", GLOBAL_BASELINE_CACHE), "wb") as f:
        pickle.dump(split_to_baseline, f)

    with open(os.path.join(args.results_dir, "stratcp_eval_results", GLOBAL_VANILLA_CP_CACHE), "wb") as f:
        pickle.dump(split_to_vanilla_cp, f)

    with open(os.path.join(args.results_dir, "stratcp_eval_results", GLOBAL_STRATCP_CACHE), "wb") as f:
        pickle.dump(split_to_stratcp, f)

    print("Saved all split-level results to disk.")

    # Aggregate and display overall summaries
    alpha_range = (0.025, 0.3)
    aggr_results_baseline, se_results_baseline = aggregate_conformal_results(
        split_to_baseline, method='mean', alpha_range=alpha_range)

    aggr_results_vanilla_cp, se_results_vanilla_cp = aggregate_conformal_results(
        split_to_vanilla_cp, method='mean', alpha_range=alpha_range)

    aggr_results_strat_cp, se_results_strat_cp = aggregate_conformal_results(
        split_to_stratcp, method='mean', alpha_range=alpha_range)

    summary_sources = [
        ("baseline", aggr_results_baseline, se_results_baseline),
        ("vanilla_cp", aggr_results_vanilla_cp, se_results_vanilla_cp),
        ("stratified_cp", aggr_results_strat_cp, se_results_strat_cp),
    ]

    metrics = (
        "mgn_cov",
        "mgn_size",
        "coverage_cls_one_sel",
        "coverage_cls_zero_sel",
        "num_sel_cls_one",
        "num_sel_cls_zero",
        "unselected_coverage",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    )
    summary_df = summarize_methods_at_alpha(
        summary_sources=summary_sources,
        alpha=args.alpha_fixed,
        metrics=metrics,
        include_se=True,     # set False if you don't want *_se columns
        nearest=True,        # set False to require exact alpha match
        atol=5e-3,           # tolerance for nearest-match (e.g., 0.005 around 0.1)
    )

    # Print summary tables
    for idx, row in summary_df.iterrows():
        print(f"\n=== {row['source']:<12} | {row['method']} ===")
        vals = row.drop(["source","method","alpha_requested","alpha_selected"])
        print(vals.to_frame(name="value"))

    return


if __name__ == "__main__":
    main()