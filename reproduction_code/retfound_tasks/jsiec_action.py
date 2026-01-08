"""
JSIEC reproduction with action-based similarity (utility-aware) and overall eligibility.

- Methods: TPS, APS, RAPS, expand_greedy (utility greedy), expand_weighted (utility weighted)
- Vanilla CP and Stratified CP (overall selection on max prob)
- Similarity metrics on prediction sets using a provided similarity matrix

Outputs in `{results_dir}/stratcp_eval_results_jsiec_action/`:
  - stratified_action_eval.csv   (metrics across runs, vanilla + stratified)
  - stratified_size_sim.csv      (size vs similarity aggregates)
  - summary_df.csv               (marginal summary at alpha_fixed)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from stratcp.conformal.core import conformal
from stratcp.conformal.scores import compute_score_aps
from stratcp.conformal.utility import compute_score_utility, eval_similarity
from stratcp.selection.single import get_reference_sel_single, get_sel_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSIEC action-based StratCP reproduction.")

    parser.add_argument(
        "--results_dir",
        type=str,
        default="data/retfound_tasks/jsiec",
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
    parser.add_argument(
        "--sim_file",
        type=str,
        required=True,
        help="Path to similarity matrix (.npy) for action overlap (shape n_classes x n_classes).",
    )

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
        help="Base seed; each run uses random_state + run_idx.",
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
        "--utility_methods",
        nargs="+",
        default=["expand_greedy", "expand_weighted"],
        help="Utility-aware methods to include (subset of expand_greedy, expand_weighted).",
    )

    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_arrays(results_dir: str, preds_file: str, labels_file: str) -> Tuple[np.ndarray, np.ndarray]:
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


def eval_singleton_cover(pred_set: np.ndarray, test_labels: np.ndarray) -> Tuple[float, float]:
    set_size = np.sum(pred_set, axis=1)
    if np.any(set_size == 1):
        scov = float(np.mean(pred_set[set_size == 1, test_labels[set_size == 1]]))
    else:
        scov = 1.0
    if np.any(set_size != 1):
        nscov = float(np.mean(pred_set[set_size != 1, test_labels[set_size != 1]]))
    else:
        nscov = 1.0
    return scov, nscov


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


def main() -> None:
    args = parse_args()

    alphas = np.array(sorted(set(float(a) for a in args.alphas)))

    ensure_dir(args.results_dir)
    eval_dir = os.path.join(args.results_dir, "stratcp_eval_results_jsiec_action")
    ensure_dir(eval_dir)

    probs, labels = load_arrays(args.results_dir, args.preds_file, args.labels_file)
    sim_mat = np.load(args.sim_file)

    all_res = []
    all_size_sim_rows = []
    strat_size_sim_unsel = []

    m_total, n_classes = probs.shape

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
        n = calib_probs.shape[0]
        m = test_probs.shape[0]

        # Selection preparation (overall eligibility)
        cal_argmax = np.argmax(calib_probs, axis=1)
        cal_label_for_sel = (cal_argmax == calib_labels).astype(int)
        cal_scores_for_sel = np.max(calib_probs, axis=1)
        cal_elig_for_sel = np.ones(n)
        test_argmax = np.argmax(test_probs, axis=1)
        test_scores_for_sel = np.max(test_probs, axis=1)
        test_elig_for_sel = np.ones(m)

        test_imputed_labels_for_ref = np.zeros((m, n_classes))
        for y in range(n_classes):
            test_imputed_labels_for_ref[:, y] = (test_argmax == y).astype(int)

        # Score builders (APS + optional utility methods)
        scores_std: Dict[str, tuple[np.ndarray, np.ndarray]] = {
            "aps": compute_score_aps(calib_probs, test_probs, calib_labels),
        }
    if "expand_greedy" in args.utility_methods:
        cal_scores_greedy, test_scores_greedy = compute_score_utility(
            calib_probs, test_probs, calib_labels, sim_mat, method="greedy", nonempty=True, null_lab=0
        )
        scores_std["expand_greedy"] = (cal_scores_greedy, test_scores_greedy)
        if "expand_weighted" in args.utility_methods:
            cal_scores_weighted, test_scores_weighted = compute_score_utility(
                calib_probs, test_probs, calib_labels, sim_mat, method="weighted", nonempty=True, null_lab=0
            )
            scores_std["expand_weighted"] = (cal_scores_weighted, test_scores_weighted)

        # Top-1 baseline (alpha=1 marker)
        top1_set = np.zeros_like(test_probs, dtype=bool)
        top1_set[np.arange(m), test_argmax] = True
        top1_cov = top1_set[np.arange(m), test_labels]
        top1_scov, top1_nscov = eval_singleton_cover(top1_set, test_labels)
        all_res.append(
            dict(
                cov=float(np.mean(top1_cov)),
                scov=top1_scov,
                nscov=top1_nscov,
                n_sel=int(m),
                size=1.0,
                sim_avg=1.0,
                method="top1",
                conformal="baseline",
                alpha=1.0,
                run=run_idx,
            )
        )
        all_size_sim_rows.append(
            pd.DataFrame({
                "size": [1],
                "method": ["top1"],
                "average_sim": [1.0],
                "count": [m],
                "conformal": "baseline",
                "alpha": 1.0,
                "run": run_idx,
                "subset": "all",
            })
        )

        for alpha in alphas:
            ones_ref = [np.ones((m, n), dtype=float) for _ in range(n_classes)]

            # Naive threshold baseline
            naive_set = _make_naive_set(test_probs, alpha)
            naive_cov = naive_set[np.arange(m), test_labels]
            naive_scov, naive_nscov = eval_singleton_cover(naive_set, test_labels)
            _, naive_sim = eval_similarity(naive_set, sim_mat, null_lab=None)
            all_res.append(
                dict(
                    cov=float(np.mean(naive_cov)),
                    scov=naive_scov,
                    nscov=naive_nscov,
                    n_sel=int(np.sum(np.sum(naive_set, axis=1) == 1)),
                    size=float(np.mean(np.sum(naive_set, axis=1))),
                    sim_avg=float(naive_sim),
                    method="raw_cut",
                    conformal="baseline",
                    alpha=float(alpha),
                    run=run_idx,
                )
            )
            avg_sim_per_size = (
                pd.DataFrame({
                    "size": np.sum(naive_set, axis=1),
                    "sim": eval_similarity(naive_set, sim_mat, null_lab=None)[0],
                    "method": "raw_cut",
                })
                .groupby(["size", "method"], as_index=False)
                .agg(average_sim=("sim", "mean"), count=("sim", "size"))
                .assign(conformal="baseline", alpha=float(alpha), run=run_idx, subset="all")
            )
            all_size_sim_rows.append(avg_sim_per_size)

            # Vanilla CP
            for meth, (cal_scores, test_scores) in scores_std.items():
                pred_sets = conformal(
                    cal_scores if cal_scores.ndim == 1 else cal_scores,
                    test_scores,
                    calib_labels,
                    alpha=alpha,
                    nonempty=True,
                    test_max_id=np.argmax(test_probs, axis=1),
                    if_in_ref=ones_ref,
                )
                size = np.sum(pred_sets, axis=1)
                cov = pred_sets[np.arange(m), test_labels]
                scov, nscov = eval_singleton_cover(pred_sets, test_labels)
                _, sim_avg = eval_similarity(pred_sets, sim_mat, null_lab=None)

                all_res.append(
                    dict(
                        cov=float(np.mean(cov)),
                        scov=scov,
                        nscov=nscov,
                        n_sel=int(np.sum(size == 1)),
                        size=float(np.mean(size)),
                        sim_avg=float(sim_avg),
                        method=meth,
                        conformal="vanilla",
                        alpha=float(alpha),
                        run=run_idx,
                    )
                )

                # Size vs similarity for vanilla
                avg_sim_per_size = (
                    pd.DataFrame({
                        "size": size,
                        "sim": eval_similarity(pred_sets, sim_mat, null_lab=None)[0],
                        "method": meth,
                    })
                    .groupby(["size", "method"], as_index=False)
                    .agg(average_sim=("sim", "mean"), count=("sim", "size"))
                )
                avg_sim_per_size = avg_sim_per_size.assign(
                    conformal="vanilla", alpha=float(alpha), run=run_idx, subset="all"
                )
                all_size_sim_rows.append(avg_sim_per_size)

            # Stratified CP (overall eligibility)
            sel_idx, unsel_idx, _ = get_sel_single(
                cal_scores=cal_scores_for_sel,
                cal_labels=cal_label_for_sel,
                test_scores=test_scores_for_sel,
                alpha=alpha,
                cal_eligs=cal_elig_for_sel,
                test_eligs=test_elig_for_sel,
            )

            # Selected prediction set (singleton argmax)
            scp_set = np.zeros((m, n_classes), dtype=bool)
            scp_set[np.arange(m), test_argmax] = True
            scp_scov, _ = (
                eval_singleton_cover(scp_set[sel_idx], test_labels[sel_idx]) if len(sel_idx) > 0 else (np.nan, np.nan)
            )

            # Reference sets for unselected
            ref_mat_list = None
            if len(unsel_idx) > 0:
                ref_mat_list = get_reference_sel_single(
                    unsel_idx=unsel_idx,
                    cal_conf_labels=cal_label_for_sel,
                    cal_conf_scores=cal_scores_for_sel,
                    test_conf_scores=test_scores_for_sel,
                    test_imputed_conf_labels=test_imputed_labels_for_ref,
                    alpha=alpha,
                    cal_eligs=cal_elig_for_sel,
                    test_eligs=test_elig_for_sel,
                )
                for s in range(len(ref_mat_list)):
                    ref_mat_list[s] = ref_mat_list[s][unsel_idx, :]

            for meth, (cal_scores, test_scores) in scores_std.items():
                if len(unsel_idx) > 0:
                    pred_unsel = conformal(
                        cal_scores,
                        test_scores[unsel_idx],
                        calib_labels,
                        alpha=alpha,
                        nonempty=True,
                        test_max_id=np.argmax(test_probs[unsel_idx], axis=1),
                        if_in_ref=ref_mat_list,
                    )
                    cov_unsel = pred_unsel[np.arange(len(unsel_idx)), test_labels[unsel_idx]]
                    size_unsel = np.sum(pred_unsel, axis=1)
                    scov_unsel, nscov_unsel = eval_singleton_cover(pred_unsel, test_labels[unsel_idx])
                    avg_sim_unsel_all, avg_sim_unsel = eval_similarity(pred_unsel, sim_mat, null_lab=None)

                    m_sel = m - len(unsel_idx)
                    cov_combined = (
                        (scp_scov * m_sel + np.sum(cov_unsel)) / m
                        if not np.isnan(scp_scov)
                        else float(np.mean(cov_unsel))
                    )
                    size_combined = (m_sel + np.sum(size_unsel)) / m
                    sim_combined = (m_sel + avg_sim_unsel * len(unsel_idx)) / m
                    nscov = float(np.mean(cov_unsel))
                    n_sel = m_sel

                    all_res.append(
                        dict(
                            cov=float(cov_combined),
                            scov=scp_scov,
                            nscov=nscov,
                            n_sel=int(n_sel),
                            size=float(size_combined),
                            sim_avg=float(sim_combined),
                            method=meth,
                            conformal="stratified",
                            alpha=float(alpha),
                            run=run_idx,
                        )
                    )

                    # Size vs similarity for unselected
                    avg_sim_per_size = (
                        pd.DataFrame({"size": size_unsel, "sim": avg_sim_unsel_all, "method": meth})
                        .groupby(["size", "method"], as_index=False)
                        .agg(average_sim=("sim", "mean"), count=("sim", "size"))
                    )
                    avg_sim_per_size = avg_sim_per_size.assign(
                        conformal="stratified", alpha=float(alpha), run=run_idx, subset="unselected"
                    )
                    all_size_sim_rows.append(avg_sim_per_size)
                    strat_size_sim_unsel.append(avg_sim_per_size)
                else:
                    all_res.append(
                        dict(
                            cov=float(scp_scov),
                            scov=scp_scov,
                            nscov=np.nan,
                            n_sel=int(m),
                            size=1.0,
                            sim_avg=1.0,
                            method=meth,
                            conformal="stratified",
                            alpha=float(alpha),
                            run=run_idx,
                        )
                    )

    # Save evaluated results
    all_res_df = pd.DataFrame(all_res)
    all_size_sim_df = pd.concat(all_size_sim_rows, ignore_index=True) if all_size_sim_rows else pd.DataFrame()
    strat_size_sim_unsel_df = (
        pd.concat(strat_size_sim_unsel, ignore_index=True) if strat_size_sim_unsel else pd.DataFrame()
    )

    # Duplicate Top-1 baseline across all alphas for consistent summaries
    if not all_res_df.empty:
        top1_rows = all_res_df[all_res_df["method"] == "top1"].copy()
        if not top1_rows.empty:
            dup_rows = []
            for a in alphas:
                t = top1_rows.copy()
                t["alpha"] = float(a)
                dup_rows.append(t)
            all_res_df = pd.concat([all_res_df] + dup_rows, ignore_index=True)

    if not all_size_sim_df.empty:
        top1_sim = all_size_sim_df[all_size_sim_df["method"] == "top1"].copy()
        if not top1_sim.empty:
            dup_sim = []
            for a in alphas:
                t = top1_sim.copy()
                t["alpha"] = float(a)
                dup_sim.append(t)
            all_size_sim_df = pd.concat([all_size_sim_df] + dup_sim, ignore_index=True)

    all_res_df.to_csv(os.path.join(eval_dir, "stratified_action_eval.csv"), index=False)
    all_size_sim_df.to_csv(os.path.join(eval_dir, "stratified_size_sim.csv"), index=False)
    strat_size_sim_unsel_df.to_csv(os.path.join(eval_dir, "stratified_size_sim_unselected.csv"), index=False)

    # Aggregate summary at alpha_fixed
    summary_df = (
        all_res_df.groupby(["conformal", "method", "alpha"], as_index=False)
        .mean(numeric_only=True)
        .pipe(lambda df: df[df["alpha"] == args.alpha_fixed])
    )
    summary_df.to_csv(os.path.join(eval_dir, "summary_df.csv"), index=False)

    print("Saved stratified_action_eval.csv, stratified_size_sim.csv, and summary_df.csv")


if __name__ == "__main__":
    main()
