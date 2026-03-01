"""
he_diagnosis_in_atdg.py

Reproduction analysis script for Stratified Conformal Prediction (StratCP) on the
H&E diagnosis-in-ATDG (adult-type diffuse glioma; ATDG) intraoperative triage task
using cached pathology model outputs.

This module evaluates and summarizes error-controlled decision-making pipelines using
the StratCP framework for morphology-triaged CNS diffuse glioma workups. Given cached
per-slide model predictions (class probabilities and labels) and slide-level metadata,
the script constructs or reuses stratified *case-level* calibration/test splits and
performs split-wise conformal evaluation over an alpha grid.

Clinical / pathology context: morphology-triaged ATDG work-up
-------------------------------------------------------------
In routine ATDG evaluation, H&E morphology guides downstream IHC/molecular testing:
  - If microvascular proliferation (MVP) or necrosis is present → high-grade branch.
  - If mitotic activity is present (without MVP/necrosis) → anaplastic branch.
  - If neither MVP/necrosis nor mitotic activity is present → lower-grade branch.

Ancillary markers (e.g., IDH mutation status and 1p/19q codeletion) then resolve the
final integrated WHO-aligned entity. StratCP operationalizes an “H&E-only selection
under a pre-specified error budget”: a subset of slides can be finalized from H&E-only
predictions (selected), while the remainder are deferred for molecular/IHC work-up.

Task modes and their morphology branches / label spaces
-------------------------------------------------------
This script supports three task modes that correspond to the morphology branches above:

  1) --task_mode mvp_3_subtypes   (MVP or necrosis branch)
     Intended differential after incorporating IDH and 1p/19q:
       - Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeletion
       - Glioblastoma, IDH-mutant
       - Glioblastoma, IDH-wildtype

  2) --task_mode miotic_3_subtypes  (Mitotic activity branch)
     Intended differential after incorporating IDH and 1p/19q:
       - Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeletion
       - Anaplastic astrocytoma, IDH-mutant
       - Anaplastic astrocytoma, IDH-wildtype

  3) --task_mode neither_2_subtypes (No MVP/necrosis/mitotic activity branch)
     Intended differential after incorporating IDH and 1p/19q:
       - Oligodendroglioma, IDH-mutant and 1p/19q codeletion
       - Diffuse astrocytoma, IDH-mutant

Inputs
------
1) Dataset metadata CSV (under --results_dir):
       he_diagnosis_in_atdg_metadata_{task_mode}.csv

   Required columns (used by this pipeline):
       - slide_id: unique slide identifier (must match pickle keys)
       - label: integer class label in [0, K-1] for the selected task_mode
       - case_id: case identifier used for stratified case-level splitting

   Additional columns (often present, not strictly required here):
       - pat_id, diagnosis, etc.

   Note:
       This script performs *case-level* stratification using `case_id` via
       `load_or_create_splits(..., patient_id_col="case_id")`, consistent with the
       downstream evaluation loop that uses split_info["test_cases"] /
       split_info["calib_cases"].

2) Cached model outputs pickle (under --results_dir/uni_eval_results):
       uni_results_dict_{task_mode}.pkl

   Expected structure:
       results_dict_test[slide_id] = {
           "prob": ndarray of shape (1, K) or (K,),    # class probabilities
           "label": int,                                # true label in [0, K-1]
           ...                                          # optional extras ignored
       }

Pipeline overview
-----------------
For each split, the script:
  1) Loads/creates stratified case-level calibration/test splits and caches them:
         {results_dir}/calib_test_splits_n_{n_splits}.pkl
     Use `--overwrite_split_cache` to force regeneration.

  2) Constructs calibration and test arrays by assigning slides to the split cohorts
     via `case_id` membership.

  3) For each alpha in the evaluation grid (linspace over [--alpha_min, --alpha_max]):
       a) StratCP:
           Runs StratifiedCP with eligibility="per_class" and extracts per-class
           selected cohorts plus the deferred (unselected) cohort.
           For each predicted class g, the script reports:
               - number selected as g,
               - false positive rate among selected-as-g,
               - total true cases in class g, and
               - deferred indices within the predicted stratum (argmax == g).

       b) Vanilla conformal prediction baseline (TPS / APS / RAPS):
           Builds conformal prediction sets using `compute_score_{tps,aps,raps}` and
           `conformal()`. The “selected” subset is defined as samples with singleton
           prediction sets (|S(x)| == 1). The implied predicted label is the unique
           member of S(x), enabling per-class selected counts and selected-set FPR.

       c) Random group-matched baseline:
           For each predicted class g (top-1 argmax), randomly selects the same number
           of samples as StratCP selected in that class from the pool of samples with
           argmax == g, and computes analogous selected-set statistics.

Outputs and summaries
---------------------
Across splits, the script aggregates metrics (mean and standard error) and constructs:
  - a long-format summary table (alpha × group × method),
  - an optional side-by-side comparison table at a target alpha, and
  - nested alpha→group comparison tables used for printing/reporting.

Eligibility and selected-side interpretation
-------------------------------------------
This pipeline uses:
    eligibility="per_class"

Meaning:
  - Eligibility/selection is defined separately for each predicted class.
  - StratCP returns K per-class selected cohorts plus a residual deferred cohort.
  - Selected-side metrics (e.g., false positive rate among selected-as-class-g) are
    computed per predicted class.

Vanilla CP method selection
---------------------------
The vanilla CP score method is configured by:
    --cp_method {aps, raps, tps}

This controls the score builder used for `conformal()` and affects prediction-set
sizes and singleton-selection behavior.

Typical use case
----------------
- Reproduce and compare StratCP vs Vanilla CP vs random group-matched baselines on
  morphology-triaged ATDG subtyping tasks using cached model probabilities.
- Analyze how H&E-only selection rates and selected-subset error metrics vary across
  alpha, interpreting deferred cases as those requiring additional IHC/NGS work-up.

Example executions
------------------
# APS as the Vanilla CP score method
python he_diagnosis_in_atdg.py \
  --results_dir ../../data/uni_pathology_tasks/he_diagnosis_in_atdg \
  --task_mode mvp_3_subtypes \
  --cp_method aps \
  --n_splits 10

# Force regeneration of case-level split cache
python he_diagnosis_in_atdg.py \
  --results_dir ../../data/uni_pathology_tasks/he_diagnosis_in_atdg \
  --task_mode miotic_3_subtypes \
  --cp_method raps \
  --n_splits 10 \
  --overwrite_split_cache
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from stratcp.stratified import StratifiedCP
from stratcp.conformal.core import conformal
from stratcp.conformal.scores import (
    compute_score_aps, compute_score_raps, compute_score_tps
)
from stratcp.eval_utils import (
    load_or_create_splits
)


def ensure_directory(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# I/O helpers
def load_pickle(path: str | os.PathLike) -> Any:
    """
    Load a pickle file from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | os.PathLike) -> None:
    """
    Save a Python object to disk as a pickle.

    Args:
        obj: Object to serialize.
        path: Output path.

    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# Summary-table helpers (unchanged)
def build_alpha_group_comparison_dict(
    summary_df: pd.DataFrame,
    *,
    include_group_count_se: bool = True,
    alpha_round_decimals: int | None = 6,
) -> Dict[float, Dict[str, pd.DataFrame]]:
    """
    Convert `summary_df` into nested dict alpha -> group_name -> comparison table.

    Args:
        summary_df: Aggregated table from build_selection_summary_across_splits().
        include_group_count_se: Whether to include SE of group count in output.
        alpha_round_decimals: Optional rounding to stabilize alpha keys.

    Returns:
        Dict[alpha][group_name] -> DataFrame comparison table.
    """
    required_cols = {
        "alpha", "group_id", "group_name", "method",
        "n_total_group_mean", "n_total_group_se",
        "n_selected_mean", "n_selected_se",
        "false_positive_rate_mean", "false_positive_rate_se",
    }
    missing = required_cols - set(summary_df.columns)
    if missing:
        raise ValueError(f"summary_df is missing required columns: {sorted(missing)}")

    method_name_map = {
        "StratCP": "StratCP",
        "VanillaCP_singleton": "Vanilla CP",
        "Random_group_matched": "Random Baseline",
    }
    method_col_order = ["StratCP", "Vanilla CP", "Random Baseline"]

    df = summary_df.copy()
    df["alpha_key"] = df["alpha"].round(alpha_round_decimals) if alpha_round_decimals is not None else df["alpha"]

    out: Dict[float, Dict[str, pd.DataFrame]] = {}

    for (alpha_key, group_name), subdf in df.groupby(["alpha_key", "group_name"], sort=True):
        row_labels = ["fpr_mean", "fpr_se", "num_selected_mean", "num_selected_se", "n_total_true_cases_mean"]
        if include_group_count_se:
            row_labels.append("n_total_true_cases_se")

        table = pd.DataFrame(np.nan, index=row_labels, columns=method_col_order, dtype=float)

        for _, row in subdf.iterrows():
            raw_method = row["method"]
            if raw_method not in method_name_map:
                continue
            col = method_name_map[raw_method]
            table.loc["fpr_mean", col] = row["false_positive_rate_mean"]
            table.loc["fpr_se", col] = row["false_positive_rate_se"]
            table.loc["num_selected_mean", col] = row["n_selected_mean"]
            table.loc["num_selected_se", col] = row["n_selected_se"]

        n_total_vals = subdf["n_total_group_mean"].dropna().unique()
        n_total_se_vals = subdf["n_total_group_se"].dropna().unique()

        n_total_mean = float(n_total_vals[0]) if len(n_total_vals) > 0 else np.nan
        n_total_se = float(n_total_se_vals[0]) if len(n_total_se_vals) > 0 else np.nan

        table.loc["n_total_true_cases_mean", :] = n_total_mean
        if include_group_count_se:
            table.loc["n_total_true_cases_se", :] = n_total_se

        out.setdefault(float(alpha_key), {})[str(group_name)] = table

    return out


def _mean_se(x: np.ndarray, *, skip_nan: bool = True) -> Tuple[float, float, int]:
    """
    Compute mean and standard error.

    Args:
        x: Array-like numeric values.
        skip_nan: Whether to drop NaNs before computing statistics.

    Returns:
        (mean, se, n_used)
    """
    x = np.asarray(x, dtype=float)
    if skip_nan:
        x = x[~np.isnan(x)]

    n = int(x.size)
    if n == 0:
        return np.nan, np.nan, 0

    mean = float(np.mean(x))
    if n < 2:
        return mean, np.nan, n

    se = float(np.std(x, ddof=1) / np.sqrt(n))
    return mean, se, n


def build_selection_summary_across_splits(
    split_to_sel_test_info: Dict[int, Dict[float, Dict[str, Any]]],
    alphas: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build raw and aggregated summaries across splits.

    Args:
        split_to_sel_test_info: Nested dict split_idx -> alpha -> selection info payload.
        alphas: Optional filter of alphas to include.

    Returns:
        (raw_df, summary_df)
    """
    method_extractors = {
        "StratCP": lambda alpha_info: alpha_info["per_group"],
        "VanillaCP_singleton": lambda alpha_info: alpha_info["baselines"]["vanilla_cp_singleton"]["per_group"],
        "Random_group_matched": lambda alpha_info: alpha_info["baselines"]["random_group_matched"]["per_group"],
    }

    rows: List[Dict[str, Any]] = []

    for split_idx, alpha_to_info in split_to_sel_test_info.items():
        if not isinstance(alpha_to_info, dict) or len(alpha_to_info) == 0:
            continue

        alpha_keys = list(alpha_to_info.keys())
        if alphas is not None:
            alpha_keys = [a for a in alpha_keys if any(np.isclose(float(a), float(t)) for t in alphas)]
        alpha_keys = sorted(alpha_keys, key=float)

        for alpha in alpha_keys:
            alpha_info = alpha_to_info[alpha]
            group_name_to_total_case = alpha_info.get("group_name_to_total_case", {})

            for method_name, get_per_group in method_extractors.items():
                per_group = get_per_group(alpha_info)

                for g, g_stats in per_group.items():
                    group_name = g_stats.get("group_name", str(g))
                    n_total_group = g_stats.get("n_total_group", group_name_to_total_case.get(group_name, np.nan))

                    rows.append(
                        {
                            "split_idx": int(split_idx),
                            "alpha": float(alpha),
                            "group_id": int(g),
                            "group_name": group_name,
                            "method": method_name,
                            "n_total_group": float(n_total_group) if n_total_group is not None else np.nan,
                            "n_selected": float(g_stats.get("n_selected", np.nan)),
                            "false_positive_rate": float(g_stats.get("false_positive_rate", np.nan))
                            if g_stats.get("false_positive_rate", np.nan) is not None
                            else np.nan,
                        }
                    )

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        raise ValueError("No rows were collected. Check split_to_sel_test_info structure and keys.")

    summary_rows: List[Dict[str, Any]] = []
    group_cols = ["alpha", "group_id", "group_name", "method"]

    for keys, df_sub in raw_df.groupby(group_cols, dropna=False):
        alpha, group_id, group_name, method = keys

        n_total_mean, n_total_se, n_total_n = _mean_se(df_sub["n_total_group"].values, skip_nan=True)
        n_sel_mean, n_sel_se, n_sel_n = _mean_se(df_sub["n_selected"].values, skip_nan=True)
        fpr_mean, fpr_se, fpr_n = _mean_se(df_sub["false_positive_rate"].values, skip_nan=True)

        summary_rows.append(
            {
                "alpha": float(alpha),
                "group_id": int(group_id),
                "group_name": group_name,
                "method": method,
                "n_total_group_mean": n_total_mean,
                "n_total_group_se": n_total_se,
                "n_total_group_n_splits": n_total_n,
                "n_selected_mean": n_sel_mean,
                "n_selected_se": n_sel_se,
                "n_selected_n_splits": n_sel_n,
                "false_positive_rate_mean": fpr_mean,
                "false_positive_rate_se": fpr_se,
                "false_positive_rate_n_splits_non_nan": fpr_n,
            }
        )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["alpha", "group_id", "method"])
        .reset_index(drop=True)
    )
    return raw_df, summary_df


def make_comparison_table_for_alpha(summary_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Create a side-by-side comparison table for a single alpha.

    Args:
        summary_df: Aggregated summary table.
        alpha: Target alpha to filter on.

    Returns:
        Wide-format comparison table.
    """
    df_a = summary_df[np.isclose(summary_df["alpha"].values, float(alpha))].copy()
    if df_a.empty:
        raise ValueError(f"No rows found for alpha={alpha}")

    base_cols = ["alpha", "group_id", "group_name", "n_total_group_mean", "n_total_group_se"]
    metrics = ["n_selected_mean", "n_selected_se", "false_positive_rate_mean", "false_positive_rate_se"]

    wide_parts = []
    for metric in metrics:
        tmp = df_a.pivot_table(index=base_cols, columns="method", values=metric, aggfunc="first")
        tmp.columns = [f"{metric}__{c}" for c in tmp.columns]
        wide_parts.append(tmp)

    out = pd.concat(wide_parts, axis=1).reset_index()
    return out.sort_values(["group_id"]).reset_index(drop=True)


# CP helpers
def get_group_label_dict(task_mode: str) -> Dict[int, str]:
    """
    Return a human-readable mapping from class index -> class name.

    Args:
        task_mode: Task mode string.

    Returns:
        Dict mapping integer class id to string label.
    """
    if task_mode == "mvp_3_subtypes":
        return {
            2: "Glioblastoma, IDH-wildtype",
            0: "Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted",
            1: "Glioblastoma, IDH-mutant",
        }
    if task_mode == "miotic_3_subtypes":
        return {
            2: "Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted",
            0: "Anaplastic astrocytoma, IDH-mutant",
            1: "Anaplastic astrocytoma, IDH-wildtype",
        }
    if task_mode == "neither_2_subtypes":
        return {
            1: "Oligodendroglioma, IDH-mutant and 1p/19q codeleted",
            0: "Diffuse astrocytoma, IDH-mutant",
        }
    raise ValueError(f"Unsupported task_mode: {task_mode}")


def to_index_array(selection: Any, *, n: int) -> np.ndarray:
    """
    Convert a selection representation into an index array.

    Args:
        selection: Either a boolean mask of shape (n,) or a 1D array/list of indices.
        n: Expected number of test samples (for validating boolean mask shape).

    Returns:
        1D integer numpy array of selected indices.
    """
    arr = np.asarray(selection)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError(f"Boolean selection mask must have shape (n,), got {arr.shape}, n={n}")
        return np.flatnonzero(arr)
    if arr.ndim == 1:
        return arr.astype(int)
    raise ValueError(f"Unsupported selection format: dtype={arr.dtype}, shape={arr.shape}")


def conformal_set_mat(*args: Any, **kwargs: Any) -> np.ndarray:
    """
    Call conformal(...) and return the set matrix robustly.

    Args:
        *args: Positional args forwarded to conformal().
        **kwargs: Keyword args forwarded to conformal().

    Returns:
        set_mat: A boolean/{0,1} matrix of shape (m_test, n_classes).
    """
    out = conformal(*args, **kwargs)
    # Some implementations return set_mat only; others return tuples.
    if isinstance(out, tuple):
        return np.asarray(out[0])
    return np.asarray(out)


def prepare_vanilla_cp_artifacts(
    calib_probs: np.ndarray,
    test_probs: np.ndarray,
    calib_labels: np.ndarray,
    *,
    methods: List[str],
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, List[np.ndarray]]:
    """
    Prepare Vanilla CP artifacts that don't depend on alpha.

    Args:
        calib_probs: Calibration probabilities, shape (n_calib, K).
        test_probs: Test probabilities, shape (m_test, K).
        calib_labels: Calibration labels, shape (n_calib,).
        methods: List of method names in {"aps","raps","tps"}.

    Returns:
        scores_by_method: dict method -> (calib_scores, test_scores)
        pred_top1: argmax label for each test sample, shape (m_test,)
        ones_ref: list of K arrays, each shape (m_test, n_calib), filled with ones
    """
    m_test = int(test_probs.shape[0])
    n_classes = int(test_probs.shape[1])
    n_calib = int(calib_labels.shape[0])

    pred_top1 = np.argmax(test_probs, axis=1)

    # conformal() expects a list of length K; each element is (m_test, n_calib)
    ones_ref = [np.ones((m_test, n_calib), dtype=float) for _ in range(n_classes)]

    def _scores_for_method(method_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if method_name == "raps":
            return compute_score_raps(calib_probs, test_probs, calib_labels)
        if method_name == "aps":
            return compute_score_aps(calib_probs, test_probs, calib_labels)
        if method_name == "tps":
            return compute_score_tps(calib_probs, test_probs, calib_labels)
        raise RuntimeError(f"Unexpected method name: {method_name}")

    scores_by_method = {meth: _scores_for_method(meth) for meth in methods}
    return scores_by_method, pred_top1, ones_ref


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    Args:
        None

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(description="StratCP evaluation over multiple splits/alphas")

    # Experiment bookkeeping
    parser.add_argument(
        "--results_dir",
        default="../../data/uni_pathology_tasks/he_diagnosis_in_atdg",
        help="Root directory for predictions and where evaluation outputs are saved.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproducibility")
    parser.add_argument("--random_state", type=int, default=42, help="Random state offset for split generation")

    # Data split configuration (within the held-out set you loaded from pickle)
    parser.add_argument("--calib_prop", type=float, default=0.20, help="Calibration proportion within (calib+test)")
    parser.add_argument("--test_prop", type=float, default=0.15, help="Test proportion within (calib+test)")
    parser.add_argument("--n_splits", type=int, default=10, help="Number of calib/test splits")
    parser.add_argument(
        "--overwrite_split_cache",
        action="store_true",
        default=False,
        help="If set, delete cached splits and re-create them.",
    )

    # Task configuration
    parser.add_argument(
        "--task_mode",
        type=str,
        choices=["mvp_3_subtypes", "miotic_3_subtypes", "neither_2_subtypes"],
        default="mvp_3_subtypes",
        help="Task mode controls which pickle file + group label mapping is used",
    )
    parser.add_argument(
        "--cp_method",
        default="aps",
        choices=["aps", "raps", "tps"],
        help="Vanilla CP score method and (optionally) StratCP score_fn if your StratifiedCP accepts these strings.",
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
    parser.add_argument(
        "--alpha_to_display",
        type=float,
        default=0.05,
        help="Example alpha to display detailed comparison table for.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Directories
    eval_results_dir = os.path.join(args.results_dir, "stratcp_eval_results")
    os.makedirs(eval_results_dir, exist_ok=True)

    # Load dataset CSV
    # Expected under: {results_dir}/he_diagnosis_in_atdg_metadata_{task_mode}.csv
    csv_path = os.path.join(args.results_dir, f"he_diagnosis_in_atdg_metadata_{args.task_mode}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at expected path: {csv_path}")

    dataset_df = pd.read_csv(csv_path)

    # Load model outputs on held-out set
    results_dict_test_filename = os.path.join(
        args.results_dir,
        "uni_eval_results",
        f"uni_results_dict_{args.task_mode}.pkl",
    )
    if not os.path.exists(results_dict_test_filename):
        raise FileNotFoundError(f"Result file not found: {results_dict_test_filename}")

    results_dict_test = load_pickle(results_dict_test_filename)
    print(f"Loaded results_dict_test with {len(results_dict_test)} slides from:\n  {results_dict_test_filename}")

    # Restrict dataset_df to slides present in results_dict_test
    test_slide_ids = list(results_dict_test.keys())
    dataset_test_df = dataset_df.loc[dataset_df["slide_id"].isin(test_slide_ids)].copy()

    if dataset_test_df.empty:
        raise ValueError("After filtering to test_slide_ids, dataset_test_df is empty. Check slide_id keys.")

    # Map slide_id -> pat_id for patient-level splitting
    slide_id_to_pat_id = dataset_test_df.set_index("slide_id")["case_id"].to_dict()

    # 1) Split creation / loading (case-level stratification by pat_id)
    test_size = args.test_prop / (args.test_prop + args.calib_prop)
    splits_path = os.path.join(args.results_dir, f"calib_test_splits_n_{args.n_splits}_task_{args.task_mode}.pkl")

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
        patient_id_col="case_id",
    )
    print(f"Loaded/created {len(split_results)} splits at:\n  {splits_path}")

    # Group-name mapping for readable reporting
    group_label_dict = get_group_label_dict(args.task_mode)

    # Compute selection info across all splits
    split_to_sel_test_info: Dict[int, Dict[float, Dict[str, Any]]] = {}

    # Alphas to evaluate
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points).tolist()

    for split_idx, split_info in split_results.items():
        print("-" * 80)
        print(f"Running CP for split {split_idx + 1}/{args.n_splits}")
        print("-" * 80)

        test_pat_ids = split_info["test_cases"]
        calib_pat_ids = split_info["calib_cases"]

        test_pat_set = set(test_pat_ids.values.tolist())
        calib_pat_set = set(calib_pat_ids.values.tolist())
        # Collect probs/labels by (calib/test) membership (patient-based)
        calib_probs_list: List[np.ndarray] = []
        calib_labels_list: List[int] = []
        test_probs_list: List[np.ndarray] = []
        test_labels_list: List[int] = []

        for slide_id, payload in results_dict_test.items():
            if slide_id not in slide_id_to_pat_id:
                # If a slide id exists in pickle but not in CSV, fail loudly.
                raise KeyError(f"slide_id '{slide_id}' present in results_dict_test but missing in dataset CSV.")

            pat_id = slide_id_to_pat_id[slide_id]

            # Decide membership by patient id
            if pat_id in test_pat_set:
                test_probs_list.append(np.asarray(payload["prob"]))
                test_labels_list.append(int(payload["label"]))
            elif pat_id in calib_pat_set:
                calib_probs_list.append(np.asarray(payload["prob"]))
                calib_labels_list.append(int(payload["label"]))
            else:
                # This can happen if dataset_test_df has patients with no slides in pickle, or vice-versa.
                # Given we filtered dataset_test_df to pickle keys, this should be rare.
                continue

        # Normalize shapes: expect (n_samples, n_classes)
        calib_probs = np.asarray(calib_probs_list).squeeze(axis=1)
        test_probs = np.asarray(test_probs_list).squeeze(axis=1)
        calib_labels = np.asarray(calib_labels_list, dtype=int)
        test_labels = np.asarray(test_labels_list, dtype=int)

        if calib_probs.ndim != 2 or test_probs.ndim != 2:
            raise ValueError(
                f"Expected probs to be 2D after squeeze; got calib_probs.shape={calib_probs.shape}, test_probs.shape={test_probs.shape}"
            )

        n_classes = int(test_probs.shape[1])
        n_test = int(test_probs.shape[0])

        # Prepare Vanilla CP artifacts (scores_by_method, pred_top1, ones_ref)
        methods = [args.cp_method]
        scores_by_method, pred_top1, ones_ref = prepare_vanilla_cp_artifacts(
            calib_probs, test_probs, calib_labels, methods=methods
        )

        alpha_to_sel_test_info: Dict[float, Dict[str, Any]] = {}

        for alpha_i, alpha in enumerate(tqdm(alphas, desc=f"Split {split_idx:02d}: StratCP+baselines")):
            # Deterministic RNG per (split, alpha) for random baseline
            rng = np.random.default_rng(int(args.seed) + 10007 * int(split_idx) + int(alpha_i))

            # 2) StratCP via StratifiedCP
            scp = StratifiedCP(
                score_fn=args.cp_method,
                alpha_sel=float(alpha),
                alpha_cp=float(alpha),
                eligibility="per_class",
            ).fit(calib_probs, calib_labels)

            res = scp.predict(test_probs, test_labels)

            all_selected = res["all_selected"]  # expected length K+1 (per-class + unselected)
            tau_list = np.asarray(res.get("thresholds", []), dtype=float) if "thresholds" in res else None

            # Convert unselected to indices
            if len(all_selected) < n_classes + 1:
                raise ValueError(
                    f"Expected all_selected length >= n_classes+1 ({n_classes+1}), got {len(all_selected)}"
                )

            global_unselected_idx = to_index_array(all_selected[n_classes], n=n_test)

            # Build union-selected mask (fallback if unselected empty)
            selected_any_mask = np.zeros(n_test, dtype=bool)
            for g in range(n_classes):
                sel_idx_g = to_index_array(all_selected[g], n=n_test)
                selected_any_mask[sel_idx_g] = True

            if global_unselected_idx.size == 0:
                global_unselected_idx = np.flatnonzero(~selected_any_mask)

            test_pred_group = np.argmax(test_probs, axis=1)

            # Build per-group stats
            per_group_stats: Dict[int, Dict[str, Any]] = {}
            for g in range(n_classes):
                sel_global_g = to_index_array(all_selected[g], n=n_test)
                tau_g = float(tau_list[g]) if (tau_list is not None and len(tau_list) > g) else np.nan

                if sel_global_g.size > 0:
                    cover_sum_g = int(np.sum(test_labels[sel_global_g] == g))
                    fpr_g = float(np.mean(test_labels[sel_global_g] != g))
                else:
                    cover_sum_g, fpr_g = 0, np.nan

                test_idx_g = np.flatnonzero(test_pred_group == g)
                unselected_global_g = np.intersect1d(test_idx_g, global_unselected_idx, assume_unique=False)

                per_group_stats[g] = {
                    "group_name": group_label_dict.get(g, str(g)),
                    "tau": tau_g,
                    "n_selected": int(sel_global_g.size),
                    "cover_sum": cover_sum_g,
                    "false_positive_rate": fpr_g,
                    "sel_idx": sel_global_g,
                    "unselected_idx": unselected_global_g,
                    "n_total_group": int(np.sum(test_labels == g)),
                }

            # 3) Vanilla CP baseline (singleton-selection)
            calib_scores, test_scores = scores_by_method[args.cp_method]

            set_mat = conformal_set_mat(
                calib_scores,
                test_scores,
                calib_labels,
                float(alpha),
                nonempty=True,
                test_max_id=pred_top1,
                if_in_ref=ones_ref,
                class_conditional=False,
            )

            # Prediction set sizes for each test sample
            size = np.sum(set_mat, axis=1).astype(int)

            # Indices selected as SINGLETON sets
            van_sel_union = np.flatnonzero(size == 1)

            # Decode singleton label from set membership
            van_pred_labels = np.empty(van_sel_union.size, dtype=int)
            for j, i in enumerate(van_sel_union):
                idxs = np.flatnonzero(set_mat[i].astype(bool))
                if idxs.size != 1:
                    raise ValueError("size==1 but membership decoding found !=1 element; check conformal() output.")
                van_pred_labels[j] = int(idxs[0])

            van_per_group_stats: Dict[int, Dict[str, Any]] = {}
            for g in range(n_classes):
                g_idx = van_sel_union[van_pred_labels == g]

                if g_idx.size > 0:
                    cover_sum_g = int(np.sum(test_labels[g_idx] == g))
                    fpr_g = float(np.mean(test_labels[g_idx] != g))
                else:
                    cover_sum_g, fpr_g = 0, np.nan

                van_per_group_stats[g] = {
                    "group_name": group_label_dict.get(g, str(g)),
                    "n_selected": int(g_idx.size),
                    "cover_sum": cover_sum_g,
                    "false_positive_rate": fpr_g,
                    "sel_idx": g_idx,
                    "n_total_group": int(np.sum(test_labels == g)),
                }

            van_baseline = {"per_group": van_per_group_stats}

            # 4) Random baseline: group-matched by predicted group
            idx_by_group = [np.where(test_pred_group == g)[0] for g in range(n_classes)]

            rand_per_group_stats: Dict[int, Dict[str, Any]] = {}
            for g in range(n_classes):
                n_target = int(per_group_stats[g]["n_selected"])
                pool = idx_by_group[g]
                pool_size = int(pool.size)

                if n_target <= 0 or pool_size == 0:
                    rand_per_group_stats[g] = {
                        "group_name": group_label_dict.get(g, str(g)),
                        "tau": np.nan,
                        "n_selected": 0,
                        "cover_sum": 0,
                        "false_positive_rate": np.nan,
                        "sel_idx": np.array([], dtype=int),
                        "n_target": n_target,
                        "n_pool": pool_size,
                        "n_total_group": int(np.sum(test_labels == g)),
                    }
                    continue

                n_sample = min(n_target, pool_size)
                rand_sel_g = rng.choice(pool, size=n_sample, replace=False)

                cover_sum_g = int(np.sum(test_labels[rand_sel_g] == g))
                fpr_g = float(np.mean(test_labels[rand_sel_g] != g))

                rand_per_group_stats[g] = {
                    "group_name": group_label_dict.get(g, str(g)),
                    "tau": np.nan,
                    "n_selected": int(n_sample),
                    "cover_sum": cover_sum_g,
                    "false_positive_rate": fpr_g,
                    "sel_idx": rand_sel_g,
                    "n_target": n_target,
                    "n_pool": pool_size,
                    "n_total_group": int(np.sum(test_labels == g)),
                }

            rand_baseline_group_matched = {"per_group": rand_per_group_stats}

            # Group totals keyed by readable group name
            group_name_to_total_case = {
                group_label_dict.get(g, str(g)): int(np.sum(test_labels == g)) for g in range(n_classes)
            }

            alpha_to_sel_test_info[float(alpha)] = {
                "per_group": per_group_stats,
                "baselines": {
                    "vanilla_cp_singleton": van_baseline,
                    "random_group_matched": rand_baseline_group_matched,
                },
                "group_name_to_total_case": group_name_to_total_case,
            }

        split_to_sel_test_info[int(split_idx)] = alpha_to_sel_test_info

    # Build summaries and show quick diagnostics
    raw_df, summary_df = build_selection_summary_across_splits(split_to_sel_test_info)
    print("\n[Summary df head]")
    print(summary_df.head(20))

    # Comparison table at alpha=0.10 (if present)
    try:
        cmp_alpha_010 = make_comparison_table_for_alpha(summary_df, alpha=0.10)
        print("\n[Comparison table alpha=0.10]")
        print(cmp_alpha_010)
    except ValueError as e:
        print(f"[WARN] Could not build alpha=0.10 comparison table: {e}")

    # Nested dict of per-alpha per-group comparison tables
    alpha_to_group_comparison = build_alpha_group_comparison_dict(summary_df)
    group_example = list(alpha_to_group_comparison[args.alpha_to_display].keys())[0]
    if args.task_mode == "mvp_3_subtypes":
        group_example = "Glioblastoma, IDH-wildtype"
    elif args.task_mode == "miotic_3_subtypes":
        group_example = "Anaplastic oligodendroglioma, IDH-mutant and 1p/19q codeleted"
    elif args.task_mode == "neither_2_subtypes":
        group_example = "Oligodendroglioma, IDH-mutant and 1p/19q codeleted"

    print(f"\nExample comparison table for alpha={args.alpha_to_display}, group='{group_example}':")
    print(alpha_to_group_comparison[args.alpha_to_display][group_example])


if __name__ == "__main__":
    main()