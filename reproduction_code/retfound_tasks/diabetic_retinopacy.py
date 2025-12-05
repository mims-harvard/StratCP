"""
Diabetic retinopathy (RETFound) reproduction with Stratified Conformal Prediction.

This script mirrors the structure of ``cns_tumor_subtype.py`` while following
the original RETFound experimental pipeline:

1) Baselines: Top-1 and naive cumulative thresholding.
2) Vanilla CP (TPS/APS/RAPS) on the full test set.
3) Stratified CP with per-class selection (JOMI sets for unselected cases).
4) Aggregation across repeated stratified splits and a summary at a fixed α.

Expected inputs under ``--results_dir`` (defaults shown):
  - ``predicted_probabilities.npy`` : shape (n_samples, n_classes)
  - ``true_labels.npy``             : shape (n_samples,)

Outputs are cached in ``{results_dir}/stratcp_eval_results_diabetic/``:
  - Per-split caches for baselines, vanilla CP, StratCP.
  - Aggregated dictionaries across splits.
  - Printed summary table at ``--alpha_fixed``.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stratcp import StratifiedCP
from stratcp.conformal.core import conformal
from stratcp.conformal.scores import compute_score_aps, compute_score_raps, compute_score_tps
from stratcp.eval_utils import (
    aggregate_conformal_results,
    compute_baselines_for_split,
    run_stratified_cp_for_split,
    run_vanilla_cp_for_split,
    summarize_methods_at_alpha,
)

# Cache templates (per split and aggregated)
BASELINE_CACHE_TEMPLATE = "top1_thresh_results_split_{split_idx}_of_{n_splits}.pkl"
VANILLA_CP_CACHE_TEMPLATE = "cp_vanilla_results_split_{split_idx}_of_{n_splits}.pkl"
STRATCP_CACHE_TEMPLATE = "stratcp_results_split_{split_idx}_of_{n_splits}.pkl"

GLOBAL_BASELINE_CACHE = "split_to_baseline_top1_thresh.pkl"
GLOBAL_VANILLA_CP_CACHE = "split_to_cp_vanilla.pkl"
GLOBAL_STRATCP_CACHE = "split_to_stratcp.pkl"
GLOBAL_COND_CACHE = "split_to_conditional.pkl"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the diabetic retinopathy reproduction."""
    parser = argparse.ArgumentParser(description="Retinopathy reproduction with StratCP.")

    # I/O
    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/retfound_tasks/diabetic_retinopathy",
        help="Directory containing predictions/labels and where outputs are written.",
    )
    parser.add_argument(
        "--preds_file",
        type=str,
        default="predicted_probabilities.npy",
        help="Filename (within results_dir) for test probabilities.",
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        default="true_labels.npy",
        help="Filename (within results_dir) for test labels.",
    )

    # Repetitions
    parser.add_argument(
        "--n_runs",
        type=int,
        default=500,
        help="Number of repeated stratified splits (calibration/test).",
    )
    parser.add_argument(
        "--calib_frac",
        type=float,
        default=0.5,
        help="Fraction of samples allocated to calibration (train_size in train_test_split).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Base seed; each split uses random_state + split_idx.",
    )

    # CP configuration
    parser.add_argument(
        "--cp_methods",
        nargs="+",
        default=["aps"],
        help="CP methods to run (subset of tps, aps, raps).",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.025, 0.05, 0.1, 0.2],
        help="Alpha grid for evaluation.",
    )
    parser.add_argument(
        "--alpha_fixed",
        type=float,
        default=0.05,
        help="Alpha at which to print the comparison table.",
    )
    parser.add_argument(
        "--alpha_aggr_min",
        type=float,
        default=None,
        help="Lower bound (inclusive) for alpha aggregation; defaults to min(alphas).",
    )
    parser.add_argument(
        "--alpha_aggr_max",
        type=float,
        default=None,
        help="Upper bound (inclusive) for alpha aggregation; defaults to max(alphas).",
    )

    # StratCP options
    parser.add_argument(
        "--eligibility",
        type=str,
        default="per_class",
        help="Eligibility for StratCP (use 'per_class' to mimic original experiment).",
    )
    parser.add_argument(
        "--return_per_class_metrics",
        default=True,
        action="store_true",
        help="If set, include per-class singleton coverage/counts in outputs.",
    )

    # Caching toggles
    parser.add_argument(
        "--overwrite_split_cache",
        action="store_true",
        help="Recreate splits even if a cached file exists.",
    )
    parser.add_argument(
        "--overwrite_eval_cache",
        action="store_true",
        help="Recompute per-split results even if caches exist.",
    )

    return parser.parse_args()


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    os.makedirs(path, exist_ok=True)


def load_arrays(results_dir: str, preds_file: str, labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load probabilities and labels from disk."""
    preds_path = os.path.join(results_dir, preds_file)
    labels_path = os.path.join(results_dir, labels_file)

    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing predictions at {preds_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels at {labels_path}")

    probs = np.load(preds_path)
    labels = np.load(labels_path)

    if probs.shape[0] != labels.shape[0]:
        raise ValueError("predicted_probabilities and true_labels have mismatched lengths.")
    if probs.ndim != 2:
        raise ValueError("predicted_probabilities must have shape (n_samples, n_classes).")

    return probs, labels.astype(int)


def load_or_create_splits(
    labels: np.ndarray,
    n_splits: int,
    calib_frac: float,
    base_seed: int,
    splits_path: str,
    overwrite: bool,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Create or load stratified calibration/test splits."""
    if (not overwrite) and os.path.exists(splits_path):
        with open(splits_path, "rb") as f:
            return pickle.load(f)

    splits: Dict[int, Dict[str, np.ndarray]] = {}
    all_indices = np.arange(labels.shape[0])

    for split_idx in range(n_splits):
        seed = base_seed + split_idx
        calib_idx, test_idx = train_test_split(
            all_indices,
            train_size=calib_frac,
            stratify=labels,
            random_state=seed,
        )
        splits[split_idx] = {"calib_idx": calib_idx, "test_idx": test_idx}

    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)

    return splits


def eval_dec_cond_diabetic(cp_set: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
    """Decision-conditional coverage/counts for diabetic retinopathy (5 classes)."""
    if_only_normal = (np.sum(cp_set, axis=1) == 1) & (cp_set[:, 0] == 1)
    if_only_mild = (np.sum(cp_set, axis=1) == 1) & (cp_set[:, 1] == 1)
    if_only_modr = (np.sum(cp_set, axis=1) == 1) & (cp_set[:, 2] == 1)
    if_only_sever = (np.sum(cp_set, axis=1) == 1) & (cp_set[:, 3] == 1)
    if_only_prol = (np.sum(cp_set, axis=1) == 1) & (cp_set[:, 4] == 1)
    if_other = ~(if_only_normal | if_only_mild | if_only_modr | if_only_sever | if_only_prol)

    def _safe_mean(mask: np.ndarray, cls: int | None) -> float:
        if mask.sum() == 0:
            return 1.0
        return float(np.mean(test_labels[mask] == cls)) if cls is not None else float(np.mean(cp_set[mask, test_labels[mask]]))

    return dict(
        cov_normal=_safe_mean(if_only_normal, 0),
        cov_mild=_safe_mean(if_only_mild, 1),
        cov_modr=_safe_mean(if_only_modr, 2),
        cov_sever=_safe_mean(if_only_sever, 3),
        cov_prol=_safe_mean(if_only_prol, 4),
        cov_other=_safe_mean(if_other, None),
        num_normal=int(if_only_normal.sum()),
        num_mild=int(if_only_mild.sum()),
        num_modr=int(if_only_modr.sum()),
        num_sever=int(if_only_sever.sum()),
        num_prol=int(if_only_prol.sum()),
        num_other=int(if_other.sum()),
    )


def eval_dec_cond_diabetic_sel(sel_list: list[np.ndarray], test_labels: np.ndarray) -> Dict[str, float]:
    """Coverage/counts for selected singleton predictions per class (DR)."""

    def _safe_sel(arr: np.ndarray, cls: int) -> float:
        if len(arr) == 0:
            return 1.0
        return float(np.mean(test_labels[arr] == cls))

    return dict(
        cov_normal=_safe_sel(sel_list[0], 0),
        cov_mild=_safe_sel(sel_list[1], 1),
        cov_modr=_safe_sel(sel_list[2], 2),
        cov_sever=_safe_sel(sel_list[3], 3),
        cov_prol=_safe_sel(sel_list[4], 4),
        num_normal=len(sel_list[0]),
        num_mild=len(sel_list[1]),
        num_modr=len(sel_list[2]),
        num_sever=len(sel_list[3]),
        num_prol=len(sel_list[4]),
    )


def _make_naive_set(probs: np.ndarray, alpha: float) -> np.ndarray:
    """Naive cumulative prediction set until cumulative prob >= 1-alpha."""
    m, _ = probs.shape
    val_pi = probs.argsort(axis=1)[:, ::-1]
    val_srt = np.take_along_axis(probs, val_pi, axis=1).cumsum(axis=1)
    naive_set = np.zeros_like(probs, dtype=np.uint8)
    for j in range(m):
        size_naive = int(np.searchsorted(val_srt[j], 1 - alpha))
        naive_set[j, val_pi[j, : size_naive + 1]] = 1
    return naive_set


def _vanilla_scores(method: str, calib_probs: np.ndarray, test_probs: np.ndarray, calib_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute calibration/test scores for vanilla CP."""
    if method == "aps":
        return compute_score_aps(calib_probs, test_probs, calib_labels)
    if method == "tps":
        return compute_score_tps(calib_probs, test_probs, calib_labels)
    if method == "raps":
        return compute_score_raps(calib_probs, test_probs, calib_labels)
    raise ValueError(f"Unknown method {method}")


def _build_stratcp_pred_set(
    method: str,
    alpha: float,
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    eligibility: str,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Run StratifiedCP and assemble full prediction sets + selection metrics."""
    scp = StratifiedCP(
        score_fn=method,
        alpha_sel=alpha,
        alpha_cp=alpha,
        eligibility=eligibility,
    )
    results = scp.fit_predict(calib_probs, calib_labels, test_probs, test_labels)

    m, k = test_probs.shape
    pred_full = np.zeros((m, k), dtype=bool)

    if eligibility == "per_class":
        all_sel = results["all_selected"]
        unsel_idx = all_sel[k].astype(int)
        for cls in range(k):
            sel_idx = all_sel[cls].astype(int)
            pred_full[sel_idx, np.argmax(test_probs[sel_idx], axis=1)] = True
        pred_full[unsel_idx, :] = results["prediction_sets"]
        sel_list = [arr.astype(int) for arr in all_sel[:k]]
    else:
        sel_idx = results["selected_idx"].astype(int)
        unsel_idx = results["unselected_idx"].astype(int)
        pred_full[sel_idx, np.argmax(test_probs[sel_idx], axis=1)] = True
        pred_full[unsel_idx, :] = results["prediction_sets"]
        sel_list = [sel_idx]

    sel_metrics = eval_dec_cond_diabetic_sel(sel_list, test_labels) if eligibility == "per_class" else {}
    return pred_full, sel_metrics


def main() -> None:
    args = parse_args()

    alphas = np.array(sorted(set(float(a) for a in args.alphas)))
    alpha_range = (
        float(args.alpha_aggr_min) if args.alpha_aggr_min is not None else float(alphas.min()),
        float(args.alpha_aggr_max) if args.alpha_aggr_max is not None else float(alphas.max()),
    )

    # Directories
    ensure_dir(args.results_dir)
    eval_dir = os.path.join(args.results_dir, "stratcp_eval_results_diabetic")
    ensure_dir(eval_dir)

    # Load data
    probs, labels = load_arrays(args.results_dir, args.preds_file, args.labels_file)

    split_to_baseline: Dict[int, Dict[str, np.ndarray]] = {}
    split_to_vanilla: Dict[int, Dict[str, np.ndarray]] = {}
    split_to_stratcp: Dict[int, Dict[str, np.ndarray]] = {}
    split_to_cond: Dict[int, pd.DataFrame] = {}

    for run_idx in range(args.n_runs):
        print("-" * 80)
        print(f"Run {run_idx + 1}/{args.n_runs}")
        print("-" * 80)

        calib_idx, test_idx = train_test_split(
            np.arange(labels.shape[0]),
            train_size=args.calib_frac,
            stratify=labels,
            random_state=args.random_state + run_idx,
        )
        calib_probs = probs[calib_idx]
        calib_labels = labels[calib_idx]
        test_probs = probs[test_idx]
        test_labels = labels[test_idx]

        # Baselines
        baseline_results = compute_baselines_for_split(
            alphas=alphas,
            test_probs=test_probs,
            test_labels=test_labels,
            return_per_class_metrics=bool(args.return_per_class_metrics),
            pbar_desc="Baselines",
        )
        split_to_baseline[run_idx] = baseline_results

        # Vanilla CP
        vanilla_results = run_vanilla_cp_for_split(
            alphas=alphas,
            calib_probs=calib_probs,
            calib_labels=calib_labels,
            test_probs=test_probs,
            test_labels=test_labels,
            methods=args.cp_methods,
            return_per_class_metrics=bool(args.return_per_class_metrics),
            pbar_desc="Vanilla CP",
        )
        split_to_vanilla[run_idx] = vanilla_results

        # Stratified CP (per-class eligibility by default)
        stratcp_results = run_stratified_cp_for_split(
            alphas=alphas,
            calib_probs=calib_probs,
            calib_labels=calib_labels,
            test_probs=test_probs,
            test_labels=test_labels,
            methods=args.cp_methods,
            eligibility=args.eligibility,
            return_per_class_metrics=bool(args.return_per_class_metrics),
            grade_consist_set=False,
            grade_map=None,
            size_bins=None,
            pbar_desc="Stratified CP",
        )
        split_to_stratcp[run_idx] = stratcp_results

        # Decision-category evaluation (diabetic-specific)
        cond_rows = []

        # Top-1 baseline (alpha=1 marker)
        top1_set = np.zeros_like(test_probs, dtype=bool)
        top1_idx = np.argmax(test_probs, axis=1)
        top1_set[np.arange(len(test_probs)), top1_idx] = True
        top1_cond = eval_dec_cond_diabetic(top1_set, test_labels)
        top1_cond.update({"method": "top1", "source": "baseline", "alpha": 1.0, "run": run_idx})
        cond_rows.append(top1_cond)

        # Naive thresholding per alpha
        for a in alphas:
            naive_set = _make_naive_set(test_probs, a)
            naive_cond = eval_dec_cond_diabetic(naive_set, test_labels)
            naive_cond.update({"method": "raw_cut", "source": "baseline", "alpha": float(a), "run": run_idx})
            cond_rows.append(naive_cond)

        # Vanilla CP per method/alpha
        for meth in args.cp_methods:
            cal_scores, test_scores = _vanilla_scores(meth, calib_probs, test_probs, calib_labels)
            for a in alphas:
                pred_sets = conformal(
                    cal_scores,
                    test_scores,
                    calib_labels,
                    alpha=a,
                    nonempty=True,
                    test_max_id=np.argmax(test_probs, axis=1),
                    if_in_ref=[np.ones((test_probs.shape[0], calib_labels.shape[0])) for _ in range(test_probs.shape[1])],
                )
                vanilla_cond = eval_dec_cond_diabetic(pred_sets, test_labels)
                vanilla_cond.update({"method": meth, "source": "vanilla_cp", "alpha": float(a), "run": run_idx})
                cond_rows.append(vanilla_cond)

        # Stratified CP per method/alpha
        for meth in args.cp_methods:
            for a in alphas:
                pred_sets_full, sel_metrics = _build_stratcp_pred_set(
                    method=meth,
                    alpha=a,
                    calib_probs=calib_probs,
                    calib_labels=calib_labels,
                    test_probs=test_probs,
                    test_labels=test_labels,
                    eligibility=args.eligibility,
                )
                strat_cond = eval_dec_cond_diabetic(pred_sets_full, test_labels)
                strat_cond.update(sel_metrics)
                strat_cond.update({"method": meth, "source": "stratified_cp", "alpha": float(a), "run": run_idx})
                cond_rows.append(strat_cond)

        cond_df = pd.DataFrame(cond_rows)
        split_to_cond[run_idx] = cond_df

    # Save evaluated results (single file per source)
    # Flatten baseline/vanilla/stratcp to long DataFrames with run id
    def _flatten_results(split_dict: Dict[int, Dict[str, pd.DataFrame]], source: str) -> pd.DataFrame:
        rows = []
        for ridx, group_dict in split_dict.items():
            for method_name, df in group_dict.items():
                tmp = df.copy()
                tmp = tmp.reset_index().rename(columns={"index": "alpha"})
                tmp["method"] = method_name
                tmp["source"] = source
                tmp["run"] = ridx
                rows.append(tmp)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    baseline_df = _flatten_results(split_to_baseline, "baseline")
    vanilla_df = _flatten_results(split_to_vanilla, "vanilla_cp")
    stratcp_df = _flatten_results(split_to_stratcp, "stratified_cp")
    cond_all_df = pd.concat(split_to_cond.values(), ignore_index=True) if split_to_cond else pd.DataFrame()

    baseline_df.to_csv(os.path.join(eval_dir, "baseline_eval.csv"), index=False)
    vanilla_df.to_csv(os.path.join(eval_dir, "vanilla_eval.csv"), index=False)
    stratcp_df.to_csv(os.path.join(eval_dir, "stratcp_eval.csv"), index=False)
    cond_all_df.to_csv(os.path.join(eval_dir, "conditional_eval.csv"), index=False)
    print("Saved evaluated results to CSV files.")

    # Aggregate across splits
    aggr_baseline, se_baseline = aggregate_conformal_results(
        split_to_baseline, method="mean", alpha_range=alpha_range
    )
    aggr_vanilla, se_vanilla = aggregate_conformal_results(
        split_to_vanilla, method="mean", alpha_range=alpha_range
    )
    aggr_stratcp, se_stratcp = aggregate_conformal_results(
        split_to_stratcp, method="mean", alpha_range=alpha_range
    )

    summary_sources = [
        ("baseline", aggr_baseline, se_baseline),
        ("vanilla_cp", aggr_vanilla, se_vanilla),
        ("stratified_cp", aggr_stratcp, se_stratcp),
    ]

    metrics = (
        "mgn_cov",
        "mgn_size",
        "selected_coverage",
        "unselected_coverage",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    )

    summary_df = summarize_methods_at_alpha(
        summary_sources=summary_sources,
        alpha=float(args.alpha_fixed),
        metrics=metrics,
        include_se=True,
        nearest=True,
        atol=5e-3,
    )

    print(f"===== Summary at alpha={args.alpha_fixed:.3f} (nearest on grid) =====")
    for _, row in summary_df.iterrows():
        print(f"=== {row['source']:<12} | {row['method']} ===")
        vals = row.drop(
            ["source", "method", "alpha_requested", "alpha_selected"],
            errors="ignore",
        )
        print(vals.to_frame(name="value"))

    # Aggregate decision-conditional metrics (mean over splits)
    cond_all = []
    for split_idx, df in split_to_cond.items():
        df = df.copy()
        df["split"] = split_idx
        cond_all.append(df)
    cond_all_df = pd.concat(cond_all, ignore_index=True)
    cond_summary = cond_all_df.groupby(["source", "method", "alpha"]).mean(numeric_only=True).reset_index()

    print(f"\n===== Decision-category summary at alpha={args.alpha_fixed:.3f} (mean over splits) =====")
    cond_alpha = cond_summary[np.isclose(cond_summary["alpha"], args.alpha_fixed)]
    if not cond_alpha.empty:
        print(cond_alpha)
    else:
        print("No rows matched alpha_fixed; inspect cond_summary for full table.")

    # Persist summaries for easy inspection
    summary_df.to_csv(os.path.join(eval_dir, "summary_df.csv"), index=False)
    cond_summary.to_csv(os.path.join(eval_dir, "cond_summary.csv"), index=False)
    print(f"\nSaved summary_df.csv and cond_summary.csv to {eval_dir}")


if __name__ == "__main__":
    main()
