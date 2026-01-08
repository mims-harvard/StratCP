"""Utility functions for evaluating (Stratified) Conformal Prediction on WSIs.

This module provides helper functions to:

- Build simple Top-1 and naive cumulative baselines.
- Run vanilla conformal prediction (APS/TPS/RAPS).
- Run StratifiedCP (overall/per-class eligibility, optional grade consistency).
- Create stratified case-level splits and extract split-wise arrays.
- Aggregate and summarize results across splits and α values.

The intent is to keep the *logic* for scoring and aggregation in one place,
while experiment scripts (e.g., IDH mutation status prediction) handle I/O and
orchestration.

Note:
    This file intentionally contains both legacy and current versions of some
    helpers (e.g., an older binary-only `evaluate_top1` and a newer multiclass
    version). The latter definitions shadow the former at import time, but the
    legacy code is kept for reference.
"""

import os
import pickle
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split

from stratcp.conformal.core import conformal
from stratcp.conformal.scores import (
    compute_score_aps,
    compute_score_raps,
    compute_score_tps,
)
from stratcp.stratified import StratifiedCP


def evaluate_top1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute baseline Top-1 metrics for binary classification (legacy).

    This older helper assumes **binary** labels {0, 1} and computes:
    - Overall accuracy (marginal coverage).
    - Class-conditional coverage, conditional on predicted class being 0 or 1.
    - Counts of predictions for each class.
    - Dummy fields for "unselected" metrics (not applicable for Top-1).

    Args:
        preds:
            Predicted binary labels of shape ``(n_samples,)``.
        labels:
            Ground-truth binary labels of shape ``(n_samples,)``.

    Returns:
        Dict[str, float]:
            {
                "mgn_cov": float,
                "mgn_size": 1.0,
                "coverage_cls_one_sel": float,
                "coverage_cls_zero_sel": float,
                "num_sel_cls_one": int,
                "num_sel_cls_zero": int,
                "unselected_coverage": np.nan,
                "unselected_set_size": np.nan,
                "num_unsel": np.nan,
            }

    Notes:
        This function is superseded by the later, multiclass-aware `evaluate_top1`
        but is retained here for backward compatibility and reference.
    """
    # Marginal metrics (overall accuracy and singleton set size).
    mgn_cov = float(np.mean(preds == labels))
    mgn_size = 1.0  # Top-1 implies singleton sets

    # Conditional metrics for class 1 predictions.
    mask_one = preds == 1
    n_one = int(mask_one.sum())
    cls_cond_cov_one = float(np.mean(labels[mask_one] == 1)) if n_one > 0 else 1.0

    # Conditional metrics for class 0 predictions.
    mask_zero = preds == 0
    n_zero = int(mask_zero.sum())
    cls_cond_cov_zero = float(np.mean(labels[mask_zero] == 0)) if n_zero > 0 else 1.0

    # Aggregate results; non-applicable fields set to NaN for consistency.
    return dict(
        mgn_cov=mgn_cov,
        mgn_size=mgn_size,
        coverage_cls_one_sel=cls_cond_cov_one,
        coverage_cls_zero_sel=cls_cond_cov_zero,
        num_sel_cls_one=n_one,
        num_sel_cls_zero=n_zero,
        unselected_coverage=np.nan,
        unselected_set_size=np.nan,
        num_unsel=np.nan,
    )


def evaluate_naive_cumulative(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    *,
    return_per_class_metrics: bool = False,
    classes: Optional[Iterable[int]] = None,
    empty_policy: str = "nan",
) -> Dict[str, Any]:
    """Build naive cumulative prediction sets and compute multiclass metrics.

    For each sample, classes are sorted by descending probability and included
    in the prediction set until the cumulative probability first exceeds
    ``1 - alpha``.

    Args:
        probs:
            Array of shape ``(n_samples, n_classes)`` with class probabilities
            (or logits).
        labels:
            Array of shape ``(n_samples,)`` with integer ground-truth class indices.
        alpha:
            Miscoverage level in ``[0, 1]``; target coverage is approximately
            ``1 - alpha``.
        return_per_class_metrics:
            If ``True``, return:
                - aggregate metrics (mgn_cov, mgn_size), and
                - per-class singleton metrics:
                    * coverage_by_pred_class: P(y=k | singleton pred=k)
                    * num_sel_by_class: # singleton predictions with class k.
            If ``False`` (default), return aggregate + “unselected” metrics:
                - unselected_coverage, unselected_set_size, num_unsel.
        classes:
            Optional iterable of class IDs to include in per-class metrics. If
            ``None``, the union of *singleton* predicted classes and all labels
            is used.
        empty_policy:
            Value for per-class coverage when no singleton predictions are made
            for a class. One of:
                * ``"one"``  → 1.0 (vacuous truth),
                * ``"nan"``  → ``np.nan``,
                * ``"zero"`` → 0.0.

    Returns:
        Dict[str, Any]:

        If ``return_per_class_metrics=True``:
            {
                "mgn_cov": float,
                "mgn_size": float,
                "coverage_by_pred_class": Dict[int, float],
                "num_sel_by_class": Dict[int, int],
            }

        Otherwise:
            {
                "mgn_cov": float,
                "mgn_size": float,
                "unselected_coverage": float | np.nan,
                "unselected_set_size": float | np.nan,
                "num_unsel": int,
            }

    Raises:
        ValueError:
            If shapes are inconsistent, alpha is out of range, or
            ``empty_policy`` is invalid.
    """
    # Validate inputs
    probs = np.asarray(probs)
    labels = np.asarray(labels).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("probs must be 2D with shape (n_samples, n_classes).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("labels must be 1D and match probs.shape[0].")

    n, K = probs.shape
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    thr = 1.0 - float(alpha)

    # Sort classes per sample and compute cumulative sums (descending prob order)
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]  # (n, K)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)  # (n, K)
    cum = np.cumsum(sorted_probs, axis=1)  # (n, K)

    # Determine minimal cut position k_i such that cum[i, k_i] >= 1 - alpha
    # Use searchsorted on each row’s cumulative sums (vectorized with broadcasting)
    # k_pos[i] = smallest j with cum[i, j] >= thr
    k_pos = np.sum(cum < thr, axis=1)  # (n,)

    # Build binary prediction-set matrix: include all sorted positions <= k_pos[i]
    # mask_sorted[i, j] = 1{ j <= k_pos[i] }
    mask_sorted = np.arange(K)[None, :] <= k_pos[:, None]
    pred_set = np.zeros_like(probs, dtype=np.uint8)  # (n, K)
    pred_set[np.arange(n)[:, None], sorted_idx] = mask_sorted  # scatter mask back to class space

    # Coverage and set sizes
    row_cov = pred_set[np.arange(n), labels].astype(float)  # 1 if y_i in set_i
    row_size = pred_set.sum(axis=1).astype(float)  # |set_i|

    mgn_cov = float(row_cov.mean())
    mgn_size = float(row_size.mean())

    # If per-class metrics requested: compute among singleton prediction sets
    if return_per_class_metrics:
        # Classes with singleton sets only
        singleton_mask = row_size == 1
        if singleton_mask.any():
            preds_single = np.argmax(pred_set[singleton_mask], axis=1)  # predicted class for singleton sets
            labels_single = labels[singleton_mask]
        else:
            preds_single = np.array([], dtype=int)
            labels_single = np.array([], dtype=int)

        # Class inventory to report
        if classes is None:
            if preds_single.size > 0 or labels.size > 0:
                classes_arr = np.unique(np.concatenate([preds_single, labels]))
            else:
                classes_arr = np.array([], dtype=int)
        else:
            classes_arr = np.array(list(classes), dtype=int)

        # Empty-policy resolver
        if empty_policy == "one":
            empty_val = 1.0
        elif empty_policy == "nan":
            empty_val = np.nan
        elif empty_policy == "zero":
            empty_val = 0.0
        else:
            raise ValueError("empty_policy must be one of {'one','nan','zero'}")

        coverage_by_pred_class: Dict[int, float] = {}
        num_sel_by_class: Dict[int, int] = {}

        for k in classes_arr:
            mask_k = preds_single == k
            n_k = int(mask_k.sum())
            num_sel_by_class[int(k)] = n_k
            if n_k > 0:
                coverage_by_pred_class[int(k)] = float(np.mean(labels_single[mask_k] == k))
            else:
                coverage_by_pred_class[int(k)] = float(empty_val) if not np.isnan(empty_val) else np.nan

        return dict(
            mgn_cov=mgn_cov,
            mgn_size=mgn_size,
            coverage_by_pred_class=coverage_by_pred_class,
            num_sel_by_class=num_sel_by_class,
        )

    # Otherwise, return baseline-oriented “unselected” metrics (non-singletons)
    unselected_mask = row_size > 1
    if unselected_mask.any():
        unselected_coverage = float(row_cov[unselected_mask].mean())
        unselected_set_size = float(row_size[unselected_mask].mean())
        num_unsel = int(unselected_mask.sum())
    else:
        unselected_coverage = np.nan
        unselected_set_size = np.nan
        num_unsel = 0

    return dict(
        mgn_cov=mgn_cov,
        mgn_size=mgn_size,
        unselected_coverage=unselected_coverage,
        unselected_set_size=unselected_set_size,
        num_unsel=num_unsel,
    )


def stratified_split_return_case_ids(
    data: pd.DataFrame,
    test_ratio: float,
    random_state: int = 42,
    patient_id_col: str = "case_id",
    label_col: str = "label",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create stratified case-level calibration/test splits.

    De-duplicates by case ID so each case appears once with its label, then
    performs a stratified split of case IDs into test and calibration sets.

    Args:
        data:
            DataFrame containing at least ``patient_id_col`` and ``label_col``.
        test_ratio:
            Proportion of unique cases to assign to the **test** split in
            the interval ``(0, 1]``.
        random_state:
            Seed for reproducibility in the stratified split.
        patient_id_col:
            Column name for patient/case IDs in ``data``.
        label_col:
            Column name for labels in ``data``.

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            (test_cases, calib_cases, test_labels, calib_labels), where each
            series is aligned to its respective case series.

    Raises:
        ValueError:
            If ``test_ratio`` is outside ``(0, 1]``.
    """
    # Drop duplicates so each case appears once with its label.
    unique_cases = data[[patient_id_col, label_col]].drop_duplicates()
    cases = unique_cases[patient_id_col]
    labels = unique_cases[label_col]

    # Stratified split at the patient/case level.
    test_cases, calib_cases, test_labels, calib_labels = train_test_split(
        cases,
        labels,
        train_size=test_ratio,
        stratify=labels,
        random_state=random_state,
    )

    return test_cases, calib_cases, test_labels, calib_labels


def aggregate_conformal_results(
    split_to_conformal_results: dict,
    method: str = "mean",
    splits_to_include: list | None = None,
    alpha_range: tuple | None = None,
) -> Tuple[dict, dict | None]:
    """Aggregate conformal-prediction results across splits.

    Expects a nested mapping:
        {split_id: {group: {method_name: DataFrame}}}
    where each DataFrame is indexed by alpha (α) and contains metric columns.

    Args:
        split_to_conformal_results:
            Nested results per split in the form
            ``{split_id: {group: {method_name: DataFrame}}}``.
        method:
            Aggregation statistic across splits: one of ``"mean"`` or ``"median"``.
        splits_to_include:
            Optional subset of split IDs to aggregate. If ``None``, all keys of
            ``split_to_conformal_results`` are used.
        alpha_range:
            Optional ``(min_alpha, max_alpha)`` to filter rows by α before
            aggregation (inclusive on both ends).

    Returns:
        Tuple[dict, dict | None]:
            (agg_dict, se_dict), where:

            * agg_dict:
                Same nesting as input but with a single aggregated DataFrame per
                (group, method_name).
            * se_dict:
                Same nesting with standard-error DataFrames when ``method="mean"``,
                otherwise ``None``.

    Raises:
        ValueError:
            If ``method`` is not one of ``{"mean", "median"}``.
    """
    if splits_to_include is None:
        splits_to_include = list(split_to_conformal_results.keys())

    # Helper to slice an α-range.
    def _clip(df: pd.DataFrame) -> pd.DataFrame:
        if alpha_range is None:
            return df
        lo, hi = alpha_range
        return df[(df.index >= lo) & (df.index <= hi)]

    agg_dict, se_dict = {}, {} if method == "mean" else None

    # We assume every split has the same groups/methods structure.
    template_split = splits_to_include[0]
    for group in split_to_conformal_results[template_split]:
        agg_dict[group] = {}
        if method == "mean":
            se_dict[group] = {}

        for method_name in split_to_conformal_results[template_split][group]:
            # Collect DataFrames for the requested splits.
            dfs = [_clip(split_to_conformal_results[split][group][method_name]) for split in splits_to_include]
            cat = pd.concat(dfs)  # Stack rows; α remains the index.

            if method == "mean":
                # Mean across splits at each α.
                agg_df = cat.groupby(level=0).mean()
                # SE = sample std / sqrt(n_splits) with unbiased std (ddof=1).
                se_df = cat.groupby(level=0).apply(lambda x: x.std(ddof=1) / np.sqrt(len(dfs)))
                agg_dict[group][method_name] = agg_df
                se_dict[group][method_name] = se_df

            elif method == "median":
                agg_df = cat.groupby(level=0).median()
                agg_dict[group][method_name] = agg_df

            else:
                raise ValueError(f"Unsupported aggregation method: {method}")

    return agg_dict, se_dict


def _ensure_df(obj: pd.Series | pd.DataFrame, default_metric: str) -> pd.DataFrame:
    """Return a DataFrame regardless of input being Series or DataFrame.

    Args:
        obj:
            A pandas Series (single metric over α) or DataFrame (multi-metric).
        default_metric:
            Column name to use if ``obj`` is a Series.

    Returns:
        pd.DataFrame:
            A DataFrame view of the input, with a single column named
            ``default_metric`` when the input is a Series.

    Raises:
        TypeError:
            If ``obj`` is neither a Series nor a DataFrame.
    """
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=default_metric)
    if isinstance(obj, pd.DataFrame):
        return obj
    raise TypeError(f"Expected Series or DataFrame, got {type(obj)}")


def _pick_alpha_row(
    df: pd.DataFrame,
    alpha: float,
    nearest: bool,
    atol: float,
) -> pd.Series | None:
    """Select the row at a given alpha from a DataFrame indexed by alpha.

    Args:
        df:
            DataFrame whose index consists of alpha values (floats).
        alpha:
            Target alpha value.
        nearest:
            If ``True``, select the nearest alpha within ``atol`` when an exact
            match is not found.
        atol:
            Absolute tolerance used when ``nearest=True``.

    Returns:
        pd.Series | None:
            The selected row as a Series, or ``None`` if no suitable row exists.
    """
    if df.empty:
        return None

    idx_vals = df.index.values.astype(float)

    # Exact match if available.
    try:
        if float(alpha) in idx_vals:
            return df.loc[float(alpha)]
    except Exception:
        pass

    # If exact match not required, pick nearest within tolerance.
    if not nearest:
        return None

    i = int(np.argmin(np.abs(idx_vals - float(alpha))))
    chosen_alpha = float(idx_vals[i])
    if abs(chosen_alpha - alpha) <= atol:
        return df.iloc[i]
    return None


def summarize_methods_at_alpha(
    summary_sources: Iterable[
        Tuple[
            str,
            Dict[str, Dict[str, pd.Series | pd.DataFrame]],
            Dict[str, Dict[str, pd.Series | pd.DataFrame]] | None,
        ]
    ],
    alpha: float,
    metrics: Iterable[str],
    methods: Iterable[str] | None = None,
    include_se: bool = True,
    nearest: bool = True,
    atol: float = 5e-3,
) -> pd.DataFrame:
    """Summarize specified metrics at a fixed alpha for each (source, method).

    Each source is given as:
        (source_label, aggr_results, se_results)

    where:
        * aggr_results[method_name][metric_name] is a Series/DataFrame indexed by α
        * se_results has the same structure, containing standard-error estimates.

    Args:
        summary_sources:
            Iterable of ``(source_label, aggr_results, se_results)`` tuples,
            typically for:
                - baselines
                - vanilla CP
                - Stratified CP.
        alpha:
            Target alpha at which to extract metrics.
        metrics:
            Iterable of metric names to extract (e.g., ``"mgn_cov"``,
            ``"mgn_size"``, ``"num_sel_cls_1"``, etc.).
        methods:
            Optional subset of methods to include. If ``None``, methods are
            inferred per source from its ``aggr_results``.
        include_se:
            If ``True``, append columns with suffix ``"_se"`` when SE data
            are available.
        nearest:
            If ``True``, select the nearest alpha within ``atol`` if an exact
            alpha is not present in the index.
        atol:
            Absolute tolerance used when ``nearest=True``.

    Returns:
        pd.DataFrame:
            A tidy DataFrame with one row per (source, method), containing:

            - identifier columns:
                * ``source``
                * ``method``
                * ``alpha_requested``
                * ``alpha_selected``
            - one column per requested metric (if found)
            - optional ``<metric>_se`` columns when ``include_se`` is True.

    Notes:
        - Rows are included only when at least one requested metric was found
          for the (source, method) pair.
        - If ``"num_total"`` is missing but components
          (``num_sel_cls_one``, ``num_sel_cls_zero``, ``num_unsel``) exist,
          ``num_total`` is derived as their sum.
    """
    rows: list[Dict[str, Any]] = []

    for source_label, aggr_results, se_results in summary_sources:
        # Determine which methods to use for this specific source.
        source_methods = list(methods) if methods is not None else list(aggr_results.keys())

        for mname in source_methods:
            # Skip methods not present in this source.
            if mname not in aggr_results:
                continue

            rec: Dict[str, Any] = {
                "source": source_label,
                "method": mname,
                "alpha_requested": float(alpha),
                "alpha_selected": np.nan,
            }

            alpha_selected_set = False
            found_any_metric = False

            for metric in metrics:
                val = np.nan
                val_se = np.nan

                # Main metric value at/near alpha.
                obj = aggr_results[mname].get(metric, None)
                if obj is not None:
                    df_main = _ensure_df(obj, default_metric=metric)
                    row = _pick_alpha_row(df_main, alpha, nearest=nearest, atol=atol)
                    if row is not None:
                        found_any_metric = True
                        # Prefer named column if present; otherwise take first column.
                        val = row[metric] if metric in row.index else row.iloc[0]
                        if not alpha_selected_set:
                            rec["alpha_selected"] = float(row.name)
                            alpha_selected_set = True
                rec[metric] = val

                # Standard error (optional).
                if include_se and se_results is not None and mname in se_results:
                    obj_se = se_results[mname].get(metric, None)
                    if obj_se is not None:
                        df_se = _ensure_df(obj_se, default_metric=f"{metric}_se")
                        # Prefer exact 'alpha_selected' once it's set.
                        se_row = None
                        if alpha_selected_set and not pd.isna(rec["alpha_selected"]):
                            se_row = _pick_alpha_row(df_se, float(rec["alpha_selected"]), nearest=False, atol=0.0)
                        if se_row is None:
                            se_row = _pick_alpha_row(df_se, alpha, nearest=nearest, atol=atol)
                        if se_row is not None:
                            if f"{metric}_se" in se_row.index:
                                val_se = se_row[f"{metric}_se"]
                            elif metric in se_row.index:
                                val_se = se_row[metric]
                            else:
                                val_se = se_row.iloc[0]
                if include_se:
                    rec[f"{metric}_se"] = val_se

            # Derive num_total if missing and components exist.
            if pd.isna(rec.get("num_total", np.nan)):
                parts = [
                    rec.get("num_sel_cls_one", np.nan),
                    rec.get("num_sel_cls_zero", np.nan),
                    rec.get("num_unsel", np.nan),
                ]
                if not any(pd.isna(p) for p in parts):
                    rec["num_total"] = float(parts[0]) + float(parts[1]) + float(parts[2])

            # Append only if at least one metric was found for this (source, method).
            if found_any_metric:
                rows.append(rec)

    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        # Order columns: identifiers, then metrics with their _se right after each.
        ordered = ["source", "method", "alpha_requested", "alpha_selected"]
        for metric in metrics:
            if metric in out.columns:
                ordered.append(metric)
            se_col = f"{metric}_se"
            if se_col in out.columns:
                ordered.append(se_col)
        leftover = [c for c in out.columns if c not in ordered]
        out = out[ordered + leftover].sort_values(["method", "source"]).reset_index(drop=True)
    return out


def load_or_create_splits(
    dataset_df: pd.DataFrame,
    test_size: float,
    n_splits: int,
    random_state: int,
    cache_path: str,
    patient_id_col: str = "case_id",
    label_col: str = "label",
) -> Dict[int, Dict[str, Any]]:
    """Load cached splits or create new stratified splits at the **case** level.

    Uses :func:`stratified_split_return_case_ids` to generate calibration/test
    splits and caches the resulting case/label mappings to disk.

    Args:
        dataset_df:
            DataFrame with columns at least ``patient_id_col`` and ``label_col``.
        test_size:
            Proportion of unique cases to assign to the **test** split ``(0, 1]``.
        n_splits:
            Number of independent stratified splits to generate.
        random_state:
            Base RNG seed; each split uses ``random_state + split_idx``.
        cache_path:
            File path from which to load / to which to save splits (pickle).
        patient_id_col:
            Column name for patient/case IDs in ``dataset_df``.
        label_col:
            Column name for labels in ``dataset_df``.

    Returns:
        Dict[int, Dict[str, Any]]:
            Dictionary indexed by split index (0..n_splits-1) with keys:
                - "test_cases":  pd.Series of case_ids in test split.
                - "calib_cases": pd.Series of case_ids in calibration split.
                - "test_labels": pd.Series of labels aligned to ``test_cases``.
                - "calib_labels": pd.Series of labels aligned to ``calib_cases``.

    Notes:
        If ``cache_path`` exists, its content is returned without recomputing.
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
            dataset_df,
            test_size,
            random_state=random_state + split_idx,
            patient_id_col=patient_id_col,
            label_col=label_col,
        )
        splits[split_idx] = {
            "test_cases": test_cases,
            "calib_cases": calib_cases,
            "test_labels": test_labels,
            "calib_labels": calib_labels,
        }
        print(f"Split {split_idx + 1}/{n_splits}: calib={len(calib_cases)}, test={len(test_cases)}")

    # Persist to cache for later re-use.
    with open(cache_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {n_splits} split results to {cache_path}")
    return splits


def extract_split_arrays(
    split_info: Dict[str, Any],
    dataset_df: pd.DataFrame,
    results_dict: Dict[str, Dict[str, Any]],
    patient_id_col: str = "case_id",
    slide_id_col: str = "slide_id",
    prob_key: str = "prob",
    label_key: str = "label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble calibration/test arrays (probs and labels) for a single split.

    Maps slide-level outputs to case-level splits, assigning each slide to
    calibration or test based on its associated case ID.

    Args:
        split_info:
            One split entry from :func:`load_or_create_splits` containing
            ``"calib_cases"`` and ``"test_cases"`` (Series) and their labels.
        dataset_df:
            DataFrame with columns including ``slide_id_col`` and
            ``patient_id_col`` used to map slides to cases.
        results_dict:
            Mapping ``slide_id -> {prob_key: array_like, label_key: int}``.
        patient_id_col:
            Column name for patient/case IDs in ``dataset_df``.
        slide_id_col:
            Column name for slide IDs in ``dataset_df``.
        prob_key:
            Key in each ``results_dict[slide_id]`` giving per-slide probabilities.
        label_key:
            Key in each ``results_dict[slide_id]`` giving per-slide labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (calib_probs, calib_labels, test_probs, test_labels).

    Raises:
        ValueError:
            If a slide’s case ID is not found in either calibration or test sets.

    Notes:
        If probability arrays have shape ``(n, 1, C)``, the singleton middle
        dimension is squeezed (to keep shapes consistent).
    """
    # Map each slide to its case id.
    slide_to_case = dataset_df.set_index(slide_id_col)[patient_id_col].to_dict()

    # Collectors for each partition.
    calib_probs, test_probs = [], []
    calib_labels, test_labels = [], []

    # Case sets for fast membership testing.
    calib_case_set = set(split_info["calib_cases"].values)
    test_case_set = set(split_info["test_cases"].values)

    # Route each slide to calibration or test by its case_id.
    for slide_id, payload in results_dict.items():
        case_id = slide_to_case[slide_id]
        prob = np.asarray(payload[prob_key])
        label = payload[label_key]

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
    calib_probs_arr = (
        np.squeeze(np.array(calib_probs), axis=1) if np.array(calib_probs).ndim == 3 else np.array(calib_probs)
    )
    test_probs_arr = (
        np.squeeze(np.array(test_probs), axis=1) if np.array(test_probs).ndim == 3 else np.array(test_probs)
    )

    return (
        calib_probs_arr,
        np.asarray(calib_labels),
        test_probs_arr,
        np.asarray(test_labels).flatten(),
    )


def evaluate_top1(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: Optional[Iterable[int]] = None,
    empty_policy: str = "nan",
    return_per_class_metrics: bool = False,
) -> Dict[str, Any]:
    """Compute Top-1 multiclass mecompute_baselines_for_splittrics.

    This is the **multiclass** version of Top-1 evaluation and shadows the
    earlier binary-only function with the same name.

    It supports two modes controlled by ``return_per_class_metrics``:

    - If ``True``:
        * overall Top-1 accuracy (mgn_cov),
        * average set size (always 1.0),
        * per-class Top-1 precision (coverage_by_pred_class),
        * per-class selection counts (num_sel_by_class).

    - If ``False``:
        * only overall Top-1 metrics (mgn_cov, mgn_size).

    Args:
        preds:
            Either:
                * (n,) array of integer predicted class IDs, or
                * (n, K) array of class probabilities/logits (argmax is used).
        labels:
            (n,) array of integer ground-truth labels.
        classes:
            Optional iterable of class IDs to report. If ``None``, uses the
            union of predicted and true labels.
        empty_policy:
            How to score coverage when no samples are predicted as a class:
                * ``"one"``  → 1.0 (vacuous truth),
                * ``"nan"``  → ``np.nan``,
                * ``"zero"`` → 0.0.
        return_per_class_metrics:
            If ``True``, include per-class precision/count dictionaries in the
            returned dict; otherwise only global metrics.

    Returns:
        Dict[str, Any]:
            If ``return_per_class_metrics=True``:
                {
                    "mgn_cov": float,
                    "mgn_size": float,
                    "coverage_by_pred_class": Dict[int, float],
                    "num_sel_by_class": Dict[int, int],
                }
            else:
                {
                    "mgn_cov": float,
                    "mgn_size": float,
                }

    Raises:
        ValueError:
            If input shapes are inconsistent or ``empty_policy`` is invalid.
    """
    # Coerce inputs and validate
    preds = np.asarray(preds)
    labels = np.asarray(labels).reshape(-1)

    # Convert (n, K) probabilities/logits -> (n,) Top-1 class indices.
    if preds.ndim == 2:
        pred_idx = np.argmax(preds, axis=1)
    elif preds.ndim == 1:
        pred_idx = preds.astype(int)
    else:
        raise ValueError("preds must be either (n,) predicted class indices or (n, K) probabilities/logits.")

    if pred_idx.shape[0] != labels.shape[0]:
        raise ValueError("preds and labels must have the same number of samples.")

    n = labels.shape[0]

    # Determine class inventory to report
    if classes is None:
        classes_arr = np.unique(np.concatenate([pred_idx, labels]))
    else:
        classes_arr = np.array(list(classes), dtype=int)

    # Global Top-1 metrics
    mgn_cov = float(np.mean(pred_idx == labels))
    mgn_size = 1.0

    # Policy for classes with no predicted samples
    if empty_policy == "one":
        empty_val = 1.0
    elif empty_policy == "nan":
        empty_val = np.nan
    elif empty_policy == "zero":
        empty_val = 0.0
    else:
        raise ValueError("empty_policy must be one of {'one','nan','zero'}")

    # Per-class aggregates
    coverage_by_pred_class: Dict[int, float] = {}
    num_sel_by_class: Dict[int, int] = {}

    for k in classes_arr:
        mask_k = pred_idx == k
        n_k = int(mask_k.sum())
        num_sel_by_class[int(k)] = n_k

        if n_k > 0:
            coverage_by_pred_class[int(k)] = float(np.mean(labels[mask_k] == k))
        else:
            coverage_by_pred_class[int(k)] = float(empty_val) if not np.isnan(empty_val) else np.nan

    if return_per_class_metrics:
        return_dict = dict(
            mgn_cov=mgn_cov,
            mgn_size=mgn_size,
            coverage_by_pred_class=coverage_by_pred_class,
            num_sel_by_class=num_sel_by_class,
        )
    else:
        return_dict = dict(
            mgn_cov=mgn_cov,
            mgn_size=mgn_size,
        )

    return return_dict


def compute_baselines_for_split(
    alphas: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    *,
    return_per_class_metrics: bool = False,
    pbar_desc: str = "Baselines (Top1 and Thresh)",
) -> Dict[str, pd.DataFrame]:
    """Compute Top-1 and naive-cumulative baselines for a single split.

    This constructs:

    - A Top-1 baseline that is α-independent (same row repeated at each α).
    - A naive cumulative baseline that depends on α (cumulative thresholding).

    Args:
        alphas:
            1D array of α values used for the naive-cumulative baseline.
        test_probs:
            (n_test, n_classes) array of class probabilities (or logits).
        test_labels:
            (n_test,) array of integer ground-truth class IDs.
        return_per_class_metrics:
            If ``True``, expand per-class columns:
                * coverage_cls_<k>_sel
                * num_sel_cls_<k>
            for every class k. If ``False``, these columns are omitted.
        pbar_desc:
            Description string for the tqdm progress-bar over α.

    Returns:
        Dict[str, pd.DataFrame]:
            {
                "top1":   Top-1 baseline (α-independent row repeated at each α),
                "thresh": Naive cumulative baseline (computed per α),
            }

        The columns always include:
            - mgn_cov
            - mgn_size
            - unselected_coverage
            - unselected_set_size
            - num_unsel
            - num_total

        If ``return_per_class_metrics=True``, for each class k:
            - coverage_cls_<k>_sel
            - num_sel_cls_<k>
    """
    # Validation
    probs = np.asarray(test_probs)
    labels = np.asarray(test_labels).reshape(-1)
    if probs.ndim != 2:
        raise ValueError("test_probs must be 2D with shape (n_test, n_classes).")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("test_labels must be 1D and match test_probs length.")
    alphas = np.asarray(alphas, dtype=float)

    n_test, n_classes = probs.shape

    def _expand_per_class_cols(
        coverage_by_pred_class: Dict[int, float] | None,
        num_sel_by_class: Dict[int, int] | None,
        n_classes: int,
    ) -> Dict[str, float | int]:
        """Expand per-class dicts into flat columns for all classes."""
        out: Dict[str, float | int] = {}
        # Descending order to match your example (two, one, zero for n=3)
        for k in range(n_classes - 1, -1, -1):
            cov = np.nan if (coverage_by_pred_class is None) else float(coverage_by_pred_class.get(k, np.nan))
            cnt = 0 if (num_sel_by_class is None) else int(num_sel_by_class.get(k, 0))
            out[f"coverage_cls_{k}_sel"] = cov
            out[f"num_sel_cls_{k}"] = cnt
        return out

    # Top-1 (α-independent)
    pred_top1 = np.argmax(probs, axis=1)
    # Expected output from evaluate_top1 when return_per_class_metrics=True:
    # {
    #   'mgn_cov': float,
    #   'mgn_size': 1.0,
    #   'coverage_by_pred_class': {k: cov_k, ...},
    #   'num_sel_by_class': {k: n_k, ...}
    # }
    top1 = evaluate_top1(pred_top1, labels, return_per_class_metrics=return_per_class_metrics)

    def _row_top1(num_total: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "mgn_cov": float(top1.get("mgn_cov", np.nan)),
            "mgn_size": float(top1.get("mgn_size", np.nan)),
            "unselected_coverage": np.nan,  # Not applicable for Top-1
            "unselected_set_size": np.nan,  # Not applicable for Top-1
            "num_unsel": np.nan,  # Not applicable for Top-1
            "num_total": int(num_total),
        }
        if return_per_class_metrics:
            row.update(
                _expand_per_class_cols(
                    top1.get("coverage_by_pred_class"),
                    top1.get("num_sel_by_class"),
                    n_classes,
                )
            )
        return row

    # Naive cumulative (α-dependent)
    def _row_thresh(alpha_val: float, num_total: int) -> Dict[str, Any]:
        # Evaluate once with the desired per-class flag so the dict has everything we need
        agg = evaluate_naive_cumulative(
            probs, labels, float(alpha_val), return_per_class_metrics=return_per_class_metrics
        )
        row: Dict[str, Any] = {
            "mgn_cov": float(agg.get("mgn_cov", np.nan)),
            "mgn_size": float(agg.get("mgn_size", np.nan)),
            "unselected_coverage": float(agg.get("unselected_coverage", np.nan)),
            "unselected_set_size": float(agg.get("unselected_set_size", np.nan)),
            "num_unsel": float(agg.get("num_unsel", np.nan)),
            "num_total": int(num_total),
        }
        if return_per_class_metrics:
            row.update(
                _expand_per_class_cols(
                    agg.get("coverage_by_pred_class"),
                    agg.get("num_sel_by_class"),
                    n_classes,
                )
            )
        return row

    # Column order
    base_cols: List[str] = [
        "mgn_cov",
        "mgn_size",
        "unselected_coverage",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    ]
    if return_per_class_metrics:
        # Generate per-class columns (descending order)
        per_class_cols: List[str] = []
        for k in range(n_classes - 1, -1, -1):
            per_class_cols.append(f"coverage_cls_{k}_sel")
        for k in range(n_classes - 1, -1, -1):
            per_class_cols.append(f"num_sel_cls_{k}")
        col_order = base_cols + per_class_cols
    else:
        col_order = base_cols

    # Assemble dataframes
    # Top-1: repeat the α-independent row at each α index (keeps shapes aligned)
    top1_rows = [_row_top1(n_test) for _ in range(len(alphas))]
    top1_df = pd.DataFrame(top1_rows, index=alphas)[col_order]

    # Naive-cumulative: compute per α
    thresh_rows = []
    for a in tqdm.tqdm(alphas, desc=pbar_desc):
        thresh_rows.append(_row_thresh(a, n_test))
    thresh_df = pd.DataFrame(thresh_rows, index=alphas)[col_order]

    return {"top1": top1_df, "thresh": thresh_df}


def run_vanilla_cp_for_split(
    alphas: np.ndarray,
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    methods: Sequence[str],
    return_per_class_metrics: bool = False,
    pbar_desc: str = "Vanilla CP",
) -> Dict[str, pd.DataFrame]:
    """Run vanilla conformal prediction (APS/TPS/RAPS) for one split.

    For each requested method, this:
      1. Computes nonconformity scores once (on calibration + test).
      2. For each α, runs JOMI conformal prediction with a uniform reference
         (all-ones) to emulate standard "vanilla" CP.
      3. Extracts:
           - marginal coverage and size,
           - coverage and average set size of *unselected* (non-singleton) samples,
           - coverage and count of *selected* (singleton) samples,
           - optional per-class singleton metrics.

    Args:
        alphas:
            1D array of α values to evaluate.
        calib_probs:
            Calibration probabilities, shape ``(n_calib, n_classes)``.
        calib_labels:
            Calibration labels, shape ``(n_calib,)``.
        test_probs:
            Test probabilities, shape ``(n_test, n_classes)``.
        test_labels:
            Test labels, shape ``(n_test,)``.
        methods:
            Iterable subset of ``{"tps", "aps", "raps"}`` specifying which
            nonconformity scores to use.
        return_per_class_metrics:
            If ``True``, add per-class singleton coverage/count columns:
                * coverage_cls_<k>_sel
                * num_sel_cls_<k>.
        pbar_desc:
            Description string used by tqdm for the α-loop.

    Returns:
        Dict[str, pd.DataFrame]:
            Mapping ``method_name -> DataFrame``, each indexed by α, with
            columns:

            Always:
                - mgn_cov
                - mgn_size
                - selected_coverage
                - unselected_coverage
                - unselected_set_size
                - num_sel
                - num_unsel
                - num_total

            Additionally, if ``return_per_class_metrics=True``:
                - coverage_cls_<k>_sel
                - num_sel_cls_<k>   (for each class k)

    Raises:
        ValueError:
            If any method is not in ``{"tps", "aps", "raps"}``, or shapes
            are inconsistent.
    """
    # Validate and normalize methods
    methods = tuple(m.lower() for m in methods)
    allowed = {"tps", "aps", "raps"}
    invalid = set(methods) - allowed
    if invalid:
        raise ValueError(f"Unknown method(s): {sorted(invalid)}. Allowed: {sorted(allowed)}")

    # Basic dimensions and convenience variables
    test_probs = np.asarray(test_probs)
    test_labels = np.asarray(test_labels).astype(int)
    calib_probs = np.asarray(calib_probs)
    calib_labels = np.asarray(calib_labels).astype(int)

    m = int(test_labels.shape[0])
    n_classes = int(test_probs.shape[1])
    if m == 0:
        raise ValueError("test_labels is empty.")
    if calib_labels.shape[0] == 0:
        raise ValueError("calib_labels is empty.")
    if test_probs.ndim != 2:
        raise ValueError("test_probs must be 2D (n_test, n_classes).")

    # Reference matrices for vanilla CP (all ones)
    # conformal() expects a list of length K; each element is (m, n_calib)
    ones_ref = [np.ones((m, calib_labels.shape[0]), dtype=float) for _ in range(n_classes)]

    # Prepare score builders once per method (scores do NOT depend on alpha)
    def _scores_for_method(method_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if method_name == "raps":
            return compute_score_raps(calib_probs, test_probs, calib_labels)
        if method_name == "aps":
            return compute_score_aps(calib_probs, test_probs, calib_labels)
        if method_name == "tps":
            return compute_score_tps(calib_probs, test_probs, calib_labels)
        raise RuntimeError("Unexpected method name")

    scores_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]] = {meth: _scores_for_method(meth) for meth in methods}

    # Helper: construct the final column order (base + per-class)
    base_cols = [
        "mgn_cov",
        "mgn_size",
        "selected_coverage",
        "unselected_coverage",
        "unselected_set_size",
        "num_sel",
        "num_unsel",
        "num_total",
    ]
    if return_per_class_metrics:
        per_class_cov_cols = [f"coverage_cls_{c}_sel" for c in range(n_classes - 1, -1, -1)]
        per_class_num_cols = [f"num_sel_cls_{c}" for c in range(n_classes - 1, -1, -1)]
        col_order = base_cols + per_class_cov_cols + per_class_num_cols
    else:
        col_order = base_cols

    # Storage for results per method across α
    summary: Dict[str, pd.DataFrame] = {}

    # Precompute most-likely class for each test sample (used by conformal)
    pred_top1 = np.argmax(test_probs, axis=1)

    # For each method, accumulate rows for each α
    for method_name in methods:
        calib_scores, test_scores = scores_by_method[method_name]
        rows, idx = [], []

        for alpha in tqdm.tqdm(alphas, desc=pbar_desc):
            # Build prediction sets for this α
            set_mat = conformal(
                calib_scores,
                test_scores,
                calib_labels,
                float(alpha),
                nonempty=True,
                test_max_id=pred_top1,
                if_in_ref=ones_ref,
                class_conditional=False,
            )
            # Coverage indicator for the true label, and set sizes
            cov = set_mat[np.arange(m), test_labels].astype(float)
            size = np.sum(set_mat, axis=1).astype(int)

            # Singleton vs unselected (non-singleton)
            singleton_mask = size == 1
            unsel_mask = ~singleton_mask

            # Marginal coverage over all samples
            mgn_cov = float(np.mean(cov)) if m > 0 else np.nan

            # Marginal size: mean of set sizes
            mgn_size = float(np.mean(size)) if m > 0 else np.nan

            # Selected coverage = coverage among singleton sets
            if np.any(singleton_mask):
                selected_coverage = float(np.mean(cov[singleton_mask]))
                num_sel = int(np.sum(singleton_mask))
            else:
                selected_coverage = np.nan
                num_sel = 0

            # Unselected summaries (non-singletons)
            if np.any(unsel_mask):
                unselected_coverage = float(np.mean(cov[unsel_mask]))
                unselected_set_size = float(np.mean(size[unsel_mask]))
                num_unsel = int(np.sum(unsel_mask))
            else:
                unselected_coverage = np.nan
                unselected_set_size = np.nan
                num_unsel = 0

            # Base row fields
            row: Dict[str, Any] = {
                "mgn_cov": mgn_cov,
                "mgn_size": mgn_size,
                "selected_coverage": selected_coverage,
                "unselected_coverage": unselected_coverage,
                "unselected_set_size": unselected_set_size,
                "num_sel": num_sel,
                "num_unsel": num_unsel,
                "num_total": m,
            }

            # Optional per-class singleton metrics (scales to K classes)
            if return_per_class_metrics:
                if np.any(singleton_mask):
                    # For singleton rows, argmax over columns gives the predicted class
                    pred_single = np.argmax(set_mat[singleton_mask], axis=1)
                    cov_single = cov[singleton_mask]

                    # For each class, compute coverage among singletons predicted as that class
                    for c in range(n_classes):
                        # token = _class_token(c)
                        mask_c = pred_single == c
                        n_c = int(np.sum(mask_c))
                        # By convention, if there are no singletons predicted as class c,
                        # set coverage to 1.0 (consistent with earlier baselines).
                        cov_c = float(np.mean(cov_single[mask_c])) if n_c > 0 else 1.0
                        row[f"coverage_cls_{c}_sel"] = cov_c
                        row[f"num_sel_cls_{c}"] = n_c
                else:
                    # No singletons at all: coverage defaults to 1.0, counts 0
                    for c in range(n_classes):
                        # token = _class_token(c)
                        row[f"coverage_cls_{c}_sel"] = 1.0
                        row[f"num_sel_cls_{c}"] = 0

            rows.append(row)
            idx.append(float(alpha))

        # Assemble DataFrame (α-indexed) with consistent column order
        df = pd.DataFrame(rows, index=idx)
        # Ensure all expected columns exist (use NaN/0 defaults if missing)
        for col in col_order:
            if col not in df.columns:
                df[col] = 0 if col.startswith("num_sel_cls_") else np.nan
        summary[method_name] = df[col_order]
    return summary


def run_stratified_cp_for_split(
    alphas: np.ndarray,
    calib_probs: np.ndarray,
    calib_labels: np.ndarray,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
    methods: Sequence[str],
    *,
    eligibility: str = "overall",
    return_per_class_metrics: bool = False,
    grade_consist_set: bool = False,
    grade_map: Dict[Any, List[int]] | None = None,
    size_bins: List[Tuple[int, int]] | None = None,
    pbar_desc: str = "Stratified CP",
) -> Dict[str, pd.DataFrame]:
    """Run StratifiedCP across α and aggregate metrics; optionally add grade diagnostics.

    On each α, this function:

      1. Fits a :class:`StratifiedCP` model using the requested score function.
      2. Performs FDR-controlled selection to split test samples into
         selected/unselected sets.
      3. Applies JOMI conformal prediction on the unselected cohort.
      4. Aggregates baseline metrics:
           - marginal coverage/size,
           - coverage / size of the selected and unselected cohorts.
      5. Optionally aggregates per-class selection metrics.
      6. Optionally computes grade-range consistency diagnostics over set sizes.

    Args:
        alphas:
            1D array of α values (used for both selection FDR and CP miscoverage).
        calib_probs:
            (n_calib, K) calibration probabilities.
        calib_labels:
            (n_calib,) integer labels in ``[0, K-1]`` for the calibration set.
        test_probs:
            (n_test, K) test probabilities.
        test_labels:
            (n_test,) integer labels in ``[0, K-1]`` for the test set.
        methods:
            Iterable of score function names in
            ``{"aps", "tps", "raps", "utility"}``.
        eligibility:
            Either:
                * ``"overall"``  → one global selection threshold, or
                * ``"per_class"`` → K per-class thresholds.
        return_per_class_metrics:
            If ``True``, add per-class metrics:
                - coverage_cls_<i>_sel
                - num_sel_cls_<i>
                and, in per_class mode, decision_tau_cls_<i>.
        grade_consist_set:
            If ``True`` and a ``grade_map`` is supplied, APS is routed through a
            utility-aware score (score_fn="utility") with a block similarity
            matrix to encourage grade-consistent expansions.
        grade_map:
            Mapping grade → list[int] of class IDs in that grade, used when
            ``grade_consist_set=True``.
        size_bins:
            List of ``(low, high)`` inclusive bounds for set-size bins used in
            grade-range diagnostics. If ``None``, defaults to:
                [(2, 4), (5, 7), (8, 10), (11, 50), (2, 50)].
        pbar_desc:
            Description string used by tqdm for the α-loop.

    Returns:
        Dict[str, pd.DataFrame]:
            Mapping ``method -> DataFrame indexed by α`` with base columns:

                - alpha
                - selection_threshold       (scalar in overall mode; NaN in per_class mode)
                - num_sel
                - num_unsel
                - num_total
                - mgn_cov
                - mgn_size
                - selected_coverage
                - unselected_coverage
                - unselected_set_size

            If ``return_per_class_metrics=True``, also includes for each class i:

                - coverage_cls_<i>_sel
                - num_sel_cls_<i>
                - decision_tau_cls_<i>     (per-class thresholds; per_class mode only)

            If ``grade_consist_set=True`` and ``grade_map`` is provided, an
            additional column is added:

                - grade_range_consistency  (dict keyed by size_bin tuple → score)

    Raises:
        ValueError:
            If an unknown method or invalid eligibility mode is supplied, or if
            input shapes are inconsistent.

    Notes:
        - “Selected” samples do not receive CP sets; they are treated as top-1.
        - “Unselected” samples receive CP sets; their coverage and size feed
          unselected/marginal metrics.
        - Grade-consistent APS is implemented by switching APS → score_fn="utility"
          with a block similarity matrix (S[i, i]=1 and S[i, j]=1 when i, j share
          a grade; 0 otherwise) and using greedy utility expansion.
    """
    # Validation & setup
    methods = tuple(m.lower() for m in methods)
    allowed = {"tps", "aps", "raps", "utility"}
    bad = set(methods) - allowed
    if bad:
        raise ValueError(f"Unknown method(s): {sorted(bad)}. Allowed: {sorted(allowed)}")
    if eligibility not in {"overall", "per_class"}:
        raise ValueError("eligibility must be 'overall' or 'per_class'.")

    # Standardize inputs
    alphas = np.asarray(alphas, dtype=float)
    calib_probs = np.asarray(calib_probs)
    test_probs = np.asarray(test_probs)
    calib_labels = np.asarray(calib_labels, dtype=int)
    test_labels = np.asarray(test_labels, dtype=int)

    if calib_probs.ndim != 2 or test_probs.ndim != 2:
        raise ValueError("calib_probs and test_probs must be 2D (n, K).")
    if calib_probs.shape[1] != test_probs.shape[1]:
        raise ValueError("calib_probs and test_probs must have same #classes.")
    if calib_labels.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("calib_labels and test_labels must be 1D.")
    if calib_probs.shape[0] != calib_labels.shape[0]:
        raise ValueError("calib_probs and calib_labels length mismatch.")
    if test_probs.shape[0] != test_labels.shape[0]:
        raise ValueError("test_probs and test_labels length mismatch.")

    n_test = int(test_probs.shape[0])
    n_classes = int(test_probs.shape[1])

    # Top-1 predictions are reused multiple times (selected coverage, etc.)
    pred_top1 = np.argmax(test_probs, axis=1)

    # Provide a default binning for grade-range diagnostics if not supplied
    if size_bins is None:
        size_bins = [(2, 4), (5, 7), (8, 10), (11, 50), (2, 50)]

    # Helper functions
    def _safe_mean(x: np.ndarray) -> float:
        """Return mean(x) or NaN if x is empty."""
        return float(np.mean(x)) if x.size > 0 else np.nan

    def _build_block_similarity(nc: int, gmap: Dict[Any, List[int]] | None) -> np.ndarray:
        """
        Build a block similarity matrix S ∈ [0,1]^{K×K}:
          S[i,i]=1; if i and j share a grade (via gmap), S[i,j]=S[j,i]=1; else 0.
        This makes utility-based expansion (greedy) prefer within-grade labels.
        """
        S = np.zeros((nc, nc), dtype=float)
        np.fill_diagonal(S, 1.0)
        if gmap is not None:
            for _, ids in gmap.items():
                ids = np.asarray(ids, dtype=int)
                if ids.size:
                    S[np.ix_(ids, ids)] = 1.0
        return S

    def _resolve_scp_args_for_method(mname: str) -> Dict[str, Any]:
        """
        Decide StratifiedCP arguments for the given method:
          - If grade_consist_set and APS: route to score_fn='utility' with block similarity.
          - Otherwise: keep the literal method name.
        You can extend TPS/RAPS similarly if you desire grade-consistent variants.
        """
        args: Dict[str, Any] = {"score_fn": mname, "similarity_matrix": None, "utility_method": "greedy"}
        if grade_consist_set and (grade_map is not None):
            if mname == "aps":
                args["score_fn"] = "utility"
                args["similarity_matrix"] = _build_block_similarity(n_classes, grade_map)
                args["utility_method"] = "greedy"  # strict grade-first expansion
        return args

    def _assemble_base_row(
        alpha_val: float,
        selection_threshold: float | None,
        selected_mask: np.ndarray,
        unselected_mask: np.ndarray,
        pred_sets_unsel: np.ndarray,
        sizes_unsel: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Create the baseline metric row for this α, independent of per-class additions.

        selected_mask, unselected_mask: boolean masks of length n_test on full test set.
        pred_sets_unsel, sizes_unsel: arrays restricted to order of unselected rows.
        """
        row: Dict[str, Any] = {
            "alpha": float(alpha_val),
            "selection_threshold": (float(selection_threshold) if selection_threshold is not None else np.nan),
            "num_sel": int(np.sum(selected_mask)),
            "num_unsel": int(np.sum(unselected_mask)),
            "num_total": n_test,
        }

        # Selected cohort accuracy (over the union of selected samples)
        row["selected_coverage"] = (
            _safe_mean((pred_top1[selected_mask] == test_labels[selected_mask]).astype(float))
            if np.any(selected_mask)
            else np.nan
        )

        if pred_sets_unsel.shape[0] > 0:
            # Coverage and size among unselected (returned in unselected ordering)
            unsel_true = test_labels[unselected_mask]
            unsel_cov = pred_sets_unsel[np.arange(unsel_true.shape[0]), unsel_true].astype(float)
            row["unselected_coverage"] = _safe_mean(unsel_cov)
            row["unselected_set_size"] = _safe_mean(sizes_unsel)

            # Marginal coverage/size across *all* test rows
            covered_total = float(np.sum(unsel_cov)) + float(
                np.sum(pred_top1[selected_mask] == test_labels[selected_mask])
            )
            row["mgn_cov"] = covered_total / float(n_test)
            row["mgn_size"] = (float(np.sum(sizes_unsel)) + float(np.sum(selected_mask))) / float(n_test)
        else:
            # Everyone selected → unselected fields are not applicable
            row["unselected_coverage"] = np.nan
            row["unselected_set_size"] = np.nan
            if np.any(selected_mask):
                row["mgn_cov"] = float(np.mean(pred_top1[selected_mask] == test_labels[selected_mask]))
                row["mgn_size"] = 1.0
            else:
                row["mgn_cov"] = np.nan
                row["mgn_size"] = np.nan

        return row

    def _add_per_class_from_selected_partition(
        row: Dict[str, Any],
        selected_mask: np.ndarray,
    ) -> None:
        """
        OVERALL mode: derive per-class metrics by partitioning the selected union
        according to *predicted* top-1 class:
          coverage_cls_i_sel = accuracy among selected with pred==i
          num_sel_cls_i      = count of selected with pred==i
        """
        if not return_per_class_metrics:
            return
        if np.any(selected_mask):
            sel_pred = pred_top1[selected_mask]
            sel_true = test_labels[selected_mask]
            for i in range(n_classes):
                m_i = sel_pred == i
                n_i = int(np.sum(m_i))
                cov_i = float(np.mean((sel_true[m_i] == i).astype(float))) if n_i > 0 else 1.0
                row[f"coverage_cls_{i}_sel"] = cov_i
                row[f"num_sel_cls_{i}"] = n_i
        else:
            for i in range(n_classes):
                row[f"coverage_cls_{i}_sel"] = 1.0
                row[f"num_sel_cls_{i}"] = 0

    def _add_per_class_from_per_class_selection(
        row: Dict[str, Any], all_selected: List[np.ndarray], tau_list: np.ndarray | None, n_test: int
    ) -> None:
        """
        PER_CLASS mode: use class-specific selection masks returned by StratifiedCP.

        all_selected is length K+1:
          - all_selected[i] is a boolean mask (n_test,) for “selected for class i”
          - all_selected[K] is the unselected mask (not selected by any class)
        We report (for each class i):
          coverage_cls_i_sel, num_sel_cls_i, decision_tau_cls_i
        """
        if not return_per_class_metrics:
            return
        for i in range(n_classes):
            sel_mask_i = np.zeros(n_test, dtype=bool)
            selected_array = all_selected[i]
            if selected_array.size:
                sel_mask_i[selected_array] = True
            # sel_mask_i = np.asarray(all_selected[i]).astype(bool)
            n_i = int(np.sum(sel_mask_i))
            cov_i = float(np.mean((test_labels[sel_mask_i] == i).astype(float))) if n_i > 0 else 1.0
            row[f"coverage_cls_{i}_sel"] = cov_i
            row[f"num_sel_cls_{i}"] = n_i
            row[f"decision_tau_cls_{i}"] = (
                float(tau_list[i]) if (tau_list is not None and tau_list.size > i) else np.nan
            )

    def _attach_grade_consistency(
        row: Dict[str, Any],
        pred_sets_unsel: np.ndarray,
        unselected_mask: np.ndarray,
    ) -> None:
        """
        If grade-consistent analysis is enabled, compute grade-range consistency
        on the *unselected* cohort and attach it to the row as a dict under
        'grade_range_consistency'. Keys are size-bin tuples; values are the
        corresponding consistency scores.
        """
        if not grade_consist_set or grade_map is None:
            return
        if pred_sets_unsel.shape[0] == 0:
            row["grade_range_consistency"] = {}
            return

        # Map unselected mask to row indices in test_probs
        unsel_idx = np.where(unselected_mask)[0]

        gr = check_grade_consistency(
            pred_sets_unsel,  # CP sets for unselected
            test_probs[unsel_idx, :],  # probs for the same unselected rows
            grade_map,  # grade → [class ids]
            size_bins=size_bins,
        )
        row["grade_range_consistency"] = gr

    # Main loop
    out: Dict[str, pd.DataFrame] = {}

    for method in methods:
        method_rows: List[Dict[str, Any]] = []
        scp_args = _resolve_scp_args_for_method(method)

        for alpha in tqdm.tqdm(alphas, desc=pbar_desc):
            # Initialize StratifiedCP for this α (same α for selection & CP)
            scp = StratifiedCP(
                score_fn=scp_args["score_fn"],
                alpha_sel=float(alpha),
                alpha_cp=float(alpha),
                eligibility=eligibility,
                nonempty=True,
                rand=True,
                similarity_matrix=scp_args.get("similarity_matrix"),
                utility_method=scp_args.get("utility_method", "greedy"),
            ).fit(calib_probs, calib_labels)

            if eligibility == "overall":
                # Overall eligibility (single threshold)
                res = scp.predict(test_probs, test_labels)

                sel_idx = np.asarray(res["selected_idx"], dtype=int)
                unsel_idx = np.asarray(res["unselected_idx"], dtype=int)
                tau = float(res["threshold"])

                # Unselected cohort: CP sets and sizes
                pred_sets_unsel = np.asarray(res["prediction_sets"], dtype=bool)
                sizes_unsel = np.asarray(res["set_sizes"])

                # Build boolean masks over full test set
                selected_mask = np.zeros(n_test, dtype=bool)
                if sel_idx.size > 0:
                    selected_mask[sel_idx] = True
                unselected_mask = np.zeros(n_test, dtype=bool)
                if unsel_idx.size > 0:
                    unselected_mask[unsel_idx] = True

                # Baseline metrics
                row = _assemble_base_row(alpha, tau, selected_mask, unselected_mask, pred_sets_unsel, sizes_unsel)
                # Optional per-class (partition selected union by predicted class)
                _add_per_class_from_selected_partition(row, selected_mask)
                # Optional grade diagnostics (unselected only)
                _attach_grade_consistency(row, pred_sets_unsel, unselected_mask)
                # breakpoint()
            else:
                # Per-class eligibility (K thresholds + residual unselected)
                res = scp.predict(test_probs, test_labels)
                all_selected = res["all_selected"]  # length K+1 list of boolean masks
                tau_list = np.asarray(res.get("thresholds", []), dtype=float) if "thresholds" in res else None

                # Unselected = not selected by any class (index K)
                unselected_mask = np.zeros(n_test, dtype=bool)
                unselected_array = all_selected[n_classes]
                if unselected_array.size:
                    unselected_mask[unselected_array] = True
                # unselected_mask = np.asarray(all_selected[n_classes]).astype(bool)

                # Union of per-class selected cohorts
                selected_mask = np.zeros(n_test, dtype=bool)
                for i in range(n_classes):
                    processed_mask = np.zeros(n_test, dtype=bool)
                    selected_array = all_selected[i]
                    if selected_array.size:
                        processed_mask[selected_array] = True
                    # breakpoint()
                    selected_mask |= processed_mask

                # CP outputs for unselected only
                pred_sets_unsel = np.asarray(res["prediction_sets"], dtype=bool)
                sizes_unsel = np.asarray(res["set_sizes"])

                # Baseline metrics (no single scalar threshold in per_class mode)
                row = _assemble_base_row(alpha, None, selected_mask, unselected_mask, pred_sets_unsel, sizes_unsel)
                # Optional per-class fields (true class per selected-for-class-i)
                _add_per_class_from_per_class_selection(row, all_selected, tau_list, n_test)
                # Optional grade diagnostics (unselected only)
                _attach_grade_consistency(row, pred_sets_unsel, unselected_mask)

            method_rows.append(row)

        out[method] = pd.DataFrame(method_rows).set_index("alpha")

    return out


def check_grade_consistency(
    prediction_sets: np.ndarray,
    test_probs: np.ndarray,
    grade_map: Mapping[Any, Sequence[int]],
    size_bins: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute grade-range consistency of conformal prediction sets, optionally
    stratified by set-size bins.

    This is a simplified version tailored to the use case:

        check_grade_consistency(
            pred_sets_unsel,          # (m, n_classes) CP sets for unselected samples
            test_probs[unsel_idx, :], # (m, n_classes) probabilities for same rows
            all_labels_vec,           # all class IDs (kept for API symmetry; unused)
            grade_map,                # dict: grade -> list[int] of class IDs
            check_for_grade_range=True,
            size_bins=size_bins,
        )

    Behavior (range-based consistency):
        • For each sample i:
            - Let top_idx = argmax_j test_probs[i, j] (Top-1 prediction).
            - Map top_idx to its grade via label_to_grade(top_idx, grade_map).
            - Let included_labels = {j : prediction_sets[i, j] == True}.
            - Map included_labels → grades, drop labels without a grade, deduplicate, sort.
            - Convert grades to numeric via roman_to_int (e.g., 'II' → 2, 'III' → 3).
            - The set is grade-consistent iff:
                * top_grade is among the included grades, AND
                * max(numeric_grades) - min(numeric_grades) < 2
              (i.e., no skipping across more than one grade).

        • If `size_bins` is provided (e.g., [(2, 4), (5, 7)]), each set contributes
          to all bins (low, high) such that low <= |set| <= high.
        • If `size_bins` is None, a single “marginal” bin (0, inf) is used.

    Args:
        prediction_sets:
            Binary array of shape (m, n_classes). Entry [i, j] is True/1 if
            class j is included in the CP prediction set for sample i.
        test_probs:
            Array of shape (m, n_classes) with predicted probabilities (or
            scores) per class for the same m samples.
        grade_map:
            Mapping from grade identifier (e.g., 'II', 'III') to a sequence of
            class IDs belonging to that grade. Used by `label_to_grade`.
        size_bins:
            Optional sequence of (low, high) tuples specifying set-size bins.
            For example: [(2, 4), (5, 7), (8, 10)]. Each prediction set with
            size S contributes to every bin (l, h) where l <= S <= h.
            If None, a single bin (0, inf) is used.

    Returns:
        Dict[Tuple[int, int], float]:
            Dictionary mapping each bin (low, high) → proportion of
            grade-consistent sets among all sets whose size falls in that bin.
            If a bin has zero eligible sets, its value is np.nan.

    Notes:
        • This function assumes the existence of two helpers in scope:
              label_to_grade(label_id, grade_map) -> grade or None
              roman_to_int(grade_str) -> int
          where `grade_str` is something like 'II', 'III', etc.
        • Samples with no included labels in `prediction_sets` or no mapped
          grades are skipped entirely.
    """
    prediction_sets = np.asarray(prediction_sets)
    test_probs = np.asarray(test_probs)

    if prediction_sets.shape != test_probs.shape:
        raise ValueError(
            f"prediction_sets and test_probs must have the same shape; "
            f"got {prediction_sets.shape} vs {test_probs.shape}."
        )

    m, n_classes = prediction_sets.shape

    # Storage: per-bin counts for (consistent, total)
    bin_counts: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: {"consistent": 0, "total": 0})

    # If no bins are provided, use a single “marginal” bin covering all sizes.
    if size_bins is None:
        size_bins = [(0, float("inf"))]  # type: ignore[list-item]

    for i in range(m):
        # Top-1 predicted class and its grade
        top_idx = int(np.argmax(test_probs[i, :]))
        top_grade = label_to_grade(top_idx, grade_map)

        # Indices of labels included in the CP set for this sample
        included_labels = np.flatnonzero(prediction_sets[i, :])
        set_size = int(included_labels.size)

        # Map included labels to grades (drop labels that have no grade)
        included_grades = [label_to_grade(lbl, grade_map) for lbl in included_labels]
        included_grades = sorted({g for g in included_grades if g is not None})

        # If no included grades, skip this sample entirely
        if not included_grades:
            continue

        # Determine which bins this set size belongs to
        matched_bins: List[Tuple[int, int]] = []
        for low, high in size_bins:
            if low <= set_size <= high:
                matched_bins.append((low, high))

        # If the set does not fall into any bin, skip (no contribution)
        if not matched_bins:
            continue

        # ----- Grade-range consistency check -----
        # Convert grades (e.g., 'II', 'III') to integers
        numeric_grades = sorted(roman_to_int(g) for g in included_grades)

        # Condition:
        #   - top_grade must be in included_grades, and
        #   - no grade skipping: max - min < 2
        is_consistent = False
        if top_grade in included_grades:
            if (max(numeric_grades) - min(numeric_grades)) < 2:
                is_consistent = True

        # Update counts for every matched bin
        for b in matched_bins:
            bin_counts[b]["total"] += 1
            if is_consistent:
                bin_counts[b]["consistent"] += 1

    # Convert counts to proportions per bin
    proportions: Dict[Tuple[int, int], float] = {}
    for bin_range, counts in bin_counts.items():
        total = counts["total"]
        if total > 0:
            proportions[bin_range] = counts["consistent"] / total
        else:
            proportions[bin_range] = np.nan

    return proportions


def roman_to_int(roman):
    """
    Converts a Roman numeral to an integer.

    Args:
        roman (str): The Roman numeral to convert.

    Returns:
        int: The integer representation of the Roman numeral.
    """
    roman_numerals = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
    return roman_numerals.get(roman, 0)


def label_to_grade(lbl, grade_map):
    """
    Maps a given label to its corresponding grade.

    Args:
        lbl (str): The label to be mapped to a grade.
        grade_map (dict): A dictionary mapping grade names to lists of labels.

    Returns:
        str: The grade associated with the given label, or None if not found.
    """
    for grade, labels in grade_map.items():
        if lbl in labels:
            return grade  # Return the grade name if the label is found
    return None  # Return None if the label is not found in any grade
