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
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import norm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from stratcp.conformal.core import conformal
from stratcp.conformal.scores import (
    compute_score_aps,
    compute_score_raps,
    compute_score_tps,
)
from stratcp.stratified import StratifiedCP


def evaluate_naive_cumulative(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    return_per_class_metrics: bool = False,
    classes: Optional[Iterable[int]] = None,
    empty_policy: str = "nan",
) -> Dict[str, Any]:
    """Evaluate naive cumulative multiclass prediction sets and compute coverage/size metrics.

    This baseline constructs a prediction set S_i for each sample i by sorting classes in
    descending probability and taking the smallest prefix whose cumulative mass reaches
    at least (1 - alpha).

    Metric definitions (exact computations):
        Let S_i ⊆ {0, ..., K-1} be the prediction set for sample i, and y_i be its label.

        - row_cov[i] = 1{ y_i ∈ S_i }.
        - row_size[i] = |S_i|.

        - mgn_cov = (1/n) * Σ_i row_cov[i]
            = empirical marginal coverage over all samples.

        - mgn_size = (1/n) * Σ_i row_size[i]
            = average prediction set size over all samples.

        Selected / unselected definitions (used when return_per_class_metrics=False):
            selected   := { i : |S_i| = 1 }   (singleton prediction sets)
            unselected := { i : |S_i| > 1 }   (non-singleton prediction sets)

        - selected_coverage = mean(row_cov[i] for i with |S_i|=1)
            = (1/|Sel|) * Σ_{i∈Sel} 1{ y_i ∈ S_i }.

        - selected_set_size = mean(row_size[i] for i with |S_i|=1)
            = 1.0 when at least one singleton exists; otherwise nan.

        - unselected_coverage = mean(row_cov[i] for i with |S_i|>1)
            = (1/|Unsel|) * Σ_{i∈Unsel} 1{ y_i ∈ S_i }.

        - unselected_set_size = mean(row_size[i] for i with |S_i|>1)
            = (1/|Unsel|) * Σ_{i∈Unsel} |S_i|.

        - num_unsel = |Unsel|
            = number of non-singleton prediction sets.

        Per-class singleton precision metrics (when return_per_class_metrics=True):
            For selected samples (|S_i|=1), define p_i as the unique class in S_i.

        - num_sel_by_class[k] = #{ i : |S_i|=1 and p_i = k }.

        - coverage_by_pred_class[k] = P(y=k | singleton pred=k) estimated by
              mean( 1{ y_i = k } over i with |S_i|=1 and p_i=k )
            If num_sel_by_class[k] == 0, the value is set by empty_policy.

    Args:
        probs:
            Array of shape (n_samples, n_classes) with class probabilities (or logits).
            This function treats the values as scores for ranking; if they are logits,
            behavior corresponds to ranking by logit and cum-summing logits (which is
            usually not meaningful). In practice, pass probabilities.
        labels:
            Array of shape (n_samples,) with integer ground-truth class indices.
        alpha:
            Miscoverage level in [0, 1]; target coverage is approximately 1 - alpha.
        return_per_class_metrics:
            If True, return per-class singleton precision + singleton counts.
            If False, return aggregate metrics plus selected/unselected metrics.
        classes:
            Optional iterable of class IDs to include in per-class metrics. If None, uses
            the union of singleton predicted classes and all labels.
        empty_policy:
            Value for per-class coverage when no singleton predictions are made for a class:
                - "one"  -> 1.0 (vacuous truth)
                - "nan"  -> np.nan
                - "zero" -> 0.0

    Returns:
        Dict[str, Any]:
            If return_per_class_metrics=True:
                {
                    "mgn_cov": float,
                    "mgn_size": float,
                    "coverage_by_pred_class": Dict[int, float],
                    "num_sel_by_class": Dict[int, int],
                    "unselected_coverage": float | np.nan,
                    "unselected_set_size": float | np.nan,
                    "num_unsel": int,
                }
            Otherwise:
                {
                    "mgn_cov": float,
                    "mgn_size": float,
                    "selected_coverage": float | np.nan,
                    "selected_set_size": float | np.nan,
                    "unselected_coverage": float | np.nan,
                    "unselected_set_size": float | np.nan,
                    "num_unsel": int,
                }

    Raises:
        ValueError:
            If shapes are inconsistent, alpha is out of range, or empty_policy is invalid.
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

    # Build naive cumulative prediction sets
    # Sort classes by descending probability per sample.
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]  # (n, K)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)  # (n, K)

    # Cumulative probability mass in the sorted order.
    cum = np.cumsum(sorted_probs, axis=1)  # (n, K)

    # Minimal cut position k_i such that cum[i, k_i] >= (1 - alpha).
    # k_pos[i] equals the count of entries strictly below thr, i.e., the first index
    # where cum >= thr.
    k_pos = np.sum(cum < thr, axis=1)  # (n,)

    # Include all positions <= k_pos[i] in the sorted order, then scatter back.
    mask_sorted = np.arange(K)[None, :] <= k_pos[:, None]  # (n, K) boolean
    pred_set = np.zeros_like(probs, dtype=np.uint8)  # (n, K)
    pred_set[np.arange(n)[:, None], sorted_idx] = mask_sorted.astype(np.uint8)

    # Compute per-sample coverage and set size
    row_cov = pred_set[np.arange(n), labels].astype(float)  # 1{ y_i ∈ S_i }
    row_size = pred_set.sum(axis=1).astype(float)           # |S_i|

    # Marginal (across-all-samples) metrics
    mgn_cov = float(row_cov.mean())
    mgn_size = float(row_size.mean())

    # Selected := singleton prediction sets; Unselected := non-singletons
    selected_mask = row_size == 1
    unselected_mask = row_size > 1

    # Per-class singleton precision metrics (if requested)
    if return_per_class_metrics:
        if selected_mask.any():
            # For singleton sets, argmax over the binary set indicator gives the unique class.
            preds_single = np.argmax(pred_set[selected_mask], axis=1).astype(int)  # p_i
            labels_single = labels[selected_mask].astype(int)                      # y_i
        else:
            preds_single = np.array([], dtype=int)
            labels_single = np.array([], dtype=int)

        # Class inventory to report
        if classes is None:
            if preds_single.size > 0 or labels.size > 0:
                classes_arr = np.unique(np.concatenate([preds_single, labels.astype(int)]))
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
            mask_k = preds_single == int(k)
            n_k = int(mask_k.sum())
            num_sel_by_class[int(k)] = n_k

            # Precision among singleton predictions of class k:
            # coverage_by_pred_class[k] = mean(1{y_i=k} | p_i=k)
            if n_k > 0:
                coverage_by_pred_class[int(k)] = float(np.mean(labels_single[mask_k] == int(k)))
            else:
                coverage_by_pred_class[int(k)] = float(empty_val) if not np.isnan(empty_val) else np.nan

        # Report unselected metrics in this branch
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
            coverage_by_pred_class=coverage_by_pred_class,
            num_sel_by_class=num_sel_by_class,
        )

    # Selected / unselected metrics (when return_per_class_metrics=False)
    if selected_mask.any():
        selected_coverage = float(row_cov[selected_mask].mean())
        selected_set_size = float(row_size[selected_mask].mean())  # should be 1.0
    else:
        selected_coverage = np.nan
        selected_set_size = np.nan

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
        selected_coverage=selected_coverage,
        selected_set_size=selected_set_size,
        unselected_coverage=unselected_coverage,
        unselected_set_size=unselected_set_size,
        num_unsel=num_unsel,
    )


def evaluate_top1(
    preds: np.ndarray,
    labels: np.ndarray,
    classes: Optional[Iterable[int]] = None,
    empty_policy: str = "nan",
    return_per_class_metrics: bool = False,
) -> Dict[str, Any]:
    r"""Evaluate Top-1 multiclass predictions and compute accuracy (and optional per-class precision).

    Metric definitions (exact computations):
        Let \hat{y}_i be the Top-1 prediction (argmax over classes) and y_i the true label.

        - mgn_cov = (1/n) * Σ_i 1{ \hat{y}_i = y_i }
            = empirical Top-1 accuracy across all samples.

        - mgn_size = 1.0
            = Top-1 prediction sets are always singletons.

        Selection rule (for "selected_*" metrics when return_per_class_metrics=False):
            selected := all samples

        - selected_coverage = mean( 1{ \hat{y}_i = y_i } over all samples )
            = mgn_cov (identical by definition).

        - selected_set_size = 1.0

        - num_sel = n

        Per-class precision metrics (when return_per_class_metrics=True):
        For each class k:

        - num_sel_by_class[k] = #{ i : \hat{y}_i = k }.

        - coverage_by_pred_class[k] = P(y=k | \hat{y}=k) estimated by
              mean( 1{ y_i = k } over i with \hat{y}_i = k )
            If num_sel_by_class[k] == 0, the value is set by empty_policy.

    Args:
        preds:
            Either:
                - (n,) array of integer predicted class IDs, or
                - (n, K) array of class probabilities/logits (argmax is used).
        labels:
            (n,) array of integer ground-truth labels.
        classes:
            Optional iterable of class IDs to report. If None, uses the union of predicted
            and true labels.
        empty_policy:
            How to score per-class precision when no samples are predicted as a class:
                - "one"  -> 1.0 (vacuous truth)
                - "nan"  -> np.nan
                - "zero" -> 0.0
        return_per_class_metrics:
            If True, include per-class precision/count dictionaries.
            If False, also return selected_coverage (selected := all samples).

    Returns:
        Dict[str, Any]:
            If return_per_class_metrics=True:
                {
                    "mgn_cov": float,                         # Top-1 accuracy
                    "mgn_size": float,                        # 1.0
                    "coverage_by_pred_class": Dict[int, float],  # P(y=k | pred=k)
                    "num_sel_by_class": Dict[int, int],          # # predicted as k
                }
            Otherwise:
                {
                    "mgn_cov": float,                         # Top-1 accuracy
                    "mgn_size": float,                        # 1.0
                    "selected_coverage": float,               # equals mgn_cov
                    "selected_set_size": float,               # 1.0
                    "num_sel": int,                           # n
                }

    Raises:
        ValueError:
            If input shapes are inconsistent or empty_policy is invalid.
    """
    # Coerce inputs and validate
    preds = np.asarray(preds)
    labels = np.asarray(labels).reshape(-1)

    # Convert (n, K) probabilities/logits -> (n,) predicted class indices
    if preds.ndim == 2:
        pred_idx = np.argmax(preds, axis=1).astype(int)
    elif preds.ndim == 1:
        pred_idx = preds.astype(int)
    else:
        raise ValueError("preds must be either (n,) predicted class indices or (n, K) probabilities/logits.")

    if pred_idx.shape[0] != labels.shape[0]:
        raise ValueError("preds and labels must have the same number of samples.")

    n = int(labels.shape[0])

    # Global Top-1 metrics
    # mgn_cov := mean(1{pred_i == y_i})
    mgn_cov = float(np.mean(pred_idx == labels))

    # mgn_size := 1 always for Top-1
    mgn_size = 1.0

    # Selected := all instances for Top-1
    selected_coverage = mgn_cov
    num_sel = n

    # Per-class metrics (precision by predicted class)
    if classes is None:
        classes_arr = np.unique(np.concatenate([pred_idx, labels.astype(int)]))
    else:
        classes_arr = np.array(list(classes), dtype=int)

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
        mask_k = pred_idx == int(k)
        n_k = int(mask_k.sum())
        num_sel_by_class[int(k)] = n_k

        # precision for class k: mean(1{y_i=k} among i with pred_i=k)
        if n_k > 0:
            coverage_by_pred_class[int(k)] = float(np.mean(labels[mask_k] == int(k)))
        else:
            coverage_by_pred_class[int(k)] = float(empty_val) if not np.isnan(empty_val) else np.nan

    if return_per_class_metrics:
        return dict(
            mgn_cov=mgn_cov,
            mgn_size=mgn_size,
            coverage_by_pred_class=coverage_by_pred_class,
            num_sel_by_class=num_sel_by_class,
        )

    return dict(
        mgn_cov=mgn_cov,
        mgn_size=mgn_size,
        selected_coverage=selected_coverage,
        num_sel=num_sel,
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

    Supports two input nestings:

    (A) One-level:
        {split_id: {method_name: DataFrame}}

    (B) Two-level:
        {split_id: {group: {method_name: DataFrame}}}

    Each DataFrame is indexed by alpha (α) and contains metric columns.
    Some columns (e.g., ``grade_range_consistency``) may be dict-valued (object dtype).
    These dict-valued columns are aggregated per α by aggregating values per dict key.

    Args:
        split_to_conformal_results:
            Results per split.
        method:
            Aggregation statistic across splits: ``"mean"`` or ``"median"``.
        splits_to_include:
            Optional subset of split IDs to aggregate. If ``None``, aggregate all splits.
        alpha_range:
            Optional ``(min_alpha, max_alpha)`` to filter rows by α before aggregation
            (inclusive on both ends).

    Returns:
        (agg_dict, se_dict):
            agg_dict has same nesting as input, but with one aggregated DataFrame per
            (method_name) or (group, method_name).

            se_dict is returned only when method == "mean"; it mirrors agg_dict and
            contains standard-error DataFrames (including dict-valued SE dicts for dict columns).

    Raises:
        ValueError:
            If ``method`` is not one of ``{"mean", "median"}``.
    """
    if method not in {"mean", "median"}:
        raise ValueError(f"Unsupported aggregation method: {method}")

    if splits_to_include is None:
        splits_to_include = list(split_to_conformal_results.keys())
    if len(splits_to_include) == 0:
        return {}, {} if method == "mean" else None

    # Helper to slice an α-range.
    def _clip(df: pd.DataFrame) -> pd.DataFrame:
        if alpha_range is None:
            return df
        lo, hi = alpha_range
        return df[(df.index >= lo) & (df.index <= hi)]

    def _is_dict_col(s: pd.Series) -> bool:
        # Look at a small sample of non-null values to detect dict payloads.
        non_null = s.dropna()
        if non_null.empty:
            return False
        sample = non_null.head(25)
        return any(isinstance(x, dict) for x in sample)

    def _agg_dicts(dicts: list[dict], stat: str) -> dict:
        """Aggregate a list of dicts by key -> stat(values)."""
        if len(dicts) == 0:
            return {}

        keys = set()
        for d in dicts:
            keys.update(d.keys())

        out: dict = {}
        for k in keys:
            vals = []
            for d in dicts:
                if k in d and d[k] is not None and not (isinstance(d[k], float) and np.isnan(d[k])):
                    vals.append(float(d[k]))
            if len(vals) == 0:
                out[k] = np.nan
            else:
                out[k] = float(np.mean(vals)) if stat == "mean" else float(np.median(vals))
        return out

    def _se_dicts(dicts: list[dict]) -> dict:
        """SE per key: std(ddof=1)/sqrt(n) across dict values."""
        if len(dicts) == 0:
            return {}

        keys = set()
        for d in dicts:
            keys.update(d.keys())

        out: dict = {}
        for k in keys:
            vals = []
            for d in dicts:
                if k in d and d[k] is not None and not (isinstance(d[k], float) and np.isnan(d[k])):
                    vals.append(float(d[k]))
            n = len(vals)
            if n < 2:
                out[k] = np.nan
            else:
                out[k] = float(np.std(vals, ddof=1) / np.sqrt(n))
        return out

    def _aggregate_df_list(dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Aggregate a list of conformal metric DataFrames indexed by α."""
        # Clip and outer-align columns.
        dfs = [_clip(df.copy()) for df in dfs]
        cat = pd.concat(dfs, axis=0, sort=True)  # stacks rows; α stays the index

        # Identify dict-valued columns.
        dict_cols = [c for c in cat.columns if _is_dict_col(cat[c])]

        # Numeric columns: everything else; coerce to numeric where possible.
        numeric_cols = [c for c in cat.columns if c not in dict_cols]

        # Pre-coerce numeric part to numeric (object -> float) safely.
        cat_num = (
            cat[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=cat.index)
        )

        # Aggregate per α.
        alphas = sorted(cat.index.unique())
        agg_rows: list[dict[str, Any]] = []
        se_rows: list[dict[str, Any]] = []

        for a in alphas:
            row_num = cat_num.loc[a]
            if isinstance(row_num, pd.Series):
                row_num = row_num.to_frame().T  # single row

            # mean/median over rows at this α (across splits)
            if method == "mean":
                agg_num = row_num.mean(axis=0, skipna=True)
            else:
                agg_num = row_num.median(axis=0, skipna=True)

            out_row: dict[str, Any] = {col: agg_num.get(col, np.nan) for col in numeric_cols}

            # Dict-valued aggregation
            for col in dict_cols:
                sub = cat.loc[a]
                if isinstance(sub, pd.Series):
                    # If only one row at this α, cat.loc[a] might be a Series of columns
                    # (this happens when there's only one split). Normalize.
                    sub = sub.to_frame().T

                dict_list = [x for x in sub[col].dropna().tolist() if isinstance(x, dict)]
                out_row[col] = _agg_dicts(dict_list, stat=method)

            agg_rows.append(out_row)

            # Standard errors only for mean aggregation
            if method == "mean":
                se_row: dict[str, Any] = {}
                # numeric SE per column: std/sqrt(n) using available (non-nan) entries
                for col in numeric_cols:
                    vals = row_num[col].dropna().values
                    n = len(vals)
                    if n < 2:
                        se_row[col] = np.nan
                    else:
                        se_row[col] = float(np.std(vals, ddof=1) / np.sqrt(n))

                # dict SE per key
                for col in dict_cols:
                    sub = cat.loc[a]
                    if isinstance(sub, pd.Series):
                        sub = sub.to_frame().T
                    dict_list = [x for x in sub[col].dropna().tolist() if isinstance(x, dict)]
                    se_row[col] = _se_dicts(dict_list)

                se_rows.append(se_row)

        agg_df = pd.DataFrame(agg_rows, index=pd.Index(alphas, name=cat.index.name))
        if method == "mean":
            se_df = pd.DataFrame(se_rows, index=pd.Index(alphas, name=cat.index.name))
        else:
            se_df = None

        return agg_df, se_df

    # Detect whether input is one-level or two-level nesting
    template_split = splits_to_include[0]
    template_obj = split_to_conformal_results[template_split]

    is_one_level = isinstance(template_obj, dict) and all(isinstance(v, pd.DataFrame) for v in template_obj.values())

    agg_dict: dict = {}
    se_dict: dict | None = {} if method == "mean" else None

    if is_one_level:
        # {split: {method_name: DataFrame}}
        for method_name in template_obj.keys():
            dfs = [
                split_to_conformal_results[split][method_name]
                for split in splits_to_include
                if method_name in split_to_conformal_results[split]
            ]
            if len(dfs) == 0:
                continue
            agg_df, se_df = _aggregate_df_list(dfs)
            agg_dict[method_name] = agg_df
            if method == "mean" and se_dict is not None:
                se_dict[method_name] = se_df

    else:
        # {split: {group: {method_name: DataFrame}}}
        for group in split_to_conformal_results[template_split]:
            agg_dict[group] = {}
            if method == "mean" and se_dict is not None:
                se_dict[group] = {}

            for method_name in split_to_conformal_results[template_split][group]:
                dfs = [
                    split_to_conformal_results[split][group][method_name]
                    for split in splits_to_include
                    if group in split_to_conformal_results[split]
                    and method_name in split_to_conformal_results[split][group]
                ]
                if len(dfs) == 0:
                    continue
                agg_df, se_df = _aggregate_df_list(dfs)
                agg_dict[group][method_name] = agg_df
                if method == "mean" and se_dict is not None:
                    se_dict[group][method_name] = se_df

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

    Special handling:
        If ``"grade_range_consistency"`` is requested in ``metrics``, this
        function expects ``aggr_results[method]["grade_range_consistency"]`` to be a
        dict-valued Series/DataFrame indexed by α (as produced by your updated
        ``aggregate_conformal_results``).

        It will expand the dict at the selected α into one column per bin:
            ``grade_range_consistency_<lo>_<hi>``
        e.g., ``grade_range_consistency_2_4``.

        If ``include_se=True`` and SE data are available, it will also expand
        the SE dict into:
            ``grade_range_consistency_<lo>_<hi>_se``

        IMPORTANT: This function does *not* include a scalar ``grade_range_consistency``
        column, and it does *not* include ``grade_range_consistency_overall``.

    Args:
        summary_sources:
            Iterable of ``(source_label, aggr_results, se_results)`` tuples.
        alpha:
            Target alpha at which to extract metrics.
        metrics:
            Iterable of metric names to extract.
        methods:
            Optional subset of methods to include. If ``None``, methods are inferred
            per source from its ``aggr_results``.
        include_se:
            If ``True``, append columns with suffix ``"_se"`` when SE data
            are available (including bin-wise SE for grade_range_consistency).
        nearest:
            If ``True``, select the nearest alpha within ``atol`` if an exact
            alpha is not present in the index.
        atol:
            Absolute tolerance used when ``nearest=True``.

    Returns:
        pd.DataFrame:
            One row per (source, method), with identifier columns plus metric columns.

    Notes:
        - Rows are included only when at least one requested metric was found
          for the (source, method) pair.
        - If ``"num_total"`` is missing but components
          (``num_sel_cls_one``, ``num_sel_cls_zero``, ``num_unsel``) exist,
          ``num_total`` is derived as their sum.
    """

    def _as_float_or_nan(x: Any) -> float:
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _extract_grade_consistency_dict(row: pd.Series) -> Dict[tuple, float] | None:
        """Given a picked alpha row (Series), extract the dict payload if present."""
        payload = None
        if "grade_range_consistency" in row.index:
            payload = row["grade_range_consistency"]
        else:
            payload = row.iloc[0] if len(row) > 0 else None

        return payload if isinstance(payload, dict) else None

    rows: list[Dict[str, Any]] = []

    for source_label, aggr_results, se_results in summary_sources:
        source_methods = list(methods) if methods is not None else list(aggr_results.keys())

        for mname in source_methods:
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
                # Special case: grade_range_consistency -> expand bins only
                if metric == "grade_range_consistency":
                    d_main: Dict[tuple, float] | None = None

                    obj = aggr_results[mname].get("grade_range_consistency", None)
                    if obj is not None:
                        df_main = _ensure_df(obj, default_metric="grade_range_consistency")
                        row = _pick_alpha_row(df_main, alpha, nearest=nearest, atol=atol)
                        if row is not None:
                            found_any_metric = True
                            if not alpha_selected_set:
                                rec["alpha_selected"] = float(row.name)
                                alpha_selected_set = True

                            d_main = _extract_grade_consistency_dict(row)
                            if d_main is not None:
                                for k, v in d_main.items():
                                    if isinstance(k, tuple) and len(k) == 2:
                                        lo, hi = k
                                        col = f"grade_range_consistency_{lo}_{hi}"
                                        rec[col] = _as_float_or_nan(v)

                    # Bin-wise SE expansion (if available)
                    if include_se and se_results is not None and mname in se_results:
                        obj_se = se_results[mname].get("grade_range_consistency", None)
                        if obj_se is not None:
                            df_se = _ensure_df(obj_se, default_metric="grade_range_consistency")

                            se_row = None
                            # Prefer exact alpha_selected if we already picked it from main.
                            if alpha_selected_set and not pd.isna(rec["alpha_selected"]):
                                se_row = _pick_alpha_row(df_se, float(rec["alpha_selected"]), nearest=False, atol=0.0)
                            if se_row is None:
                                se_row = _pick_alpha_row(df_se, alpha, nearest=nearest, atol=atol)

                            if se_row is not None:
                                d_se = _extract_grade_consistency_dict(se_row)
                                if d_se is not None:
                                    for k, v in d_se.items():
                                        if isinstance(k, tuple) and len(k) == 2:
                                            lo, hi = k
                                            col = f"grade_range_consistency_{lo}_{hi}"
                                            rec[f"{col}_se"] = _as_float_or_nan(v)

                    # Ensure SE columns exist (NaN) for bins present in main but missing in se.
                    if include_se and d_main is not None:
                        for k in d_main.keys():
                            if isinstance(k, tuple) and len(k) == 2:
                                lo, hi = k
                                col = f"grade_range_consistency_{lo}_{hi}"
                                rec.setdefault(f"{col}_se", np.nan)

                    # IMPORTANT: do NOT set rec["grade_range_consistency"] or rec["grade_range_consistency_se"]
                    continue

                # Default behavior for scalar/Series/DataFrame metrics
                val = np.nan
                val_se = np.nan

                obj = aggr_results[mname].get(metric, None)
                if obj is not None:
                    df_main = _ensure_df(obj, default_metric=metric)
                    row = _pick_alpha_row(df_main, alpha, nearest=nearest, atol=atol)
                    if row is not None:
                        found_any_metric = True
                        val = row[metric] if metric in row.index else row.iloc[0]
                        if not alpha_selected_set:
                            rec["alpha_selected"] = float(row.name)
                            alpha_selected_set = True
                rec[metric] = val

                if include_se and se_results is not None and mname in se_results:
                    obj_se = se_results[mname].get(metric, None)
                    if obj_se is not None:
                        df_se = _ensure_df(obj_se, default_metric=f"{metric}_se")

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

            if found_any_metric:
                rows.append(rec)

    out = pd.DataFrame.from_records(rows)
    if out.empty:
        return out

    # Column ordering
    ordered = ["source", "method", "alpha_requested", "alpha_selected"]

    def _bin_key(c: str) -> tuple[int, int]:
        # c like "grade_range_consistency_2_4"
        parts = c.split("_")
        try:
            lo = int(parts[-2])
            hi = int(parts[-1])
            return (lo, hi)
        except Exception:
            return (10**9, 10**9)

    for metric in metrics:
        if metric == "grade_range_consistency":
            # Include bin columns and their SE columns, but not the raw metric name.
            bin_cols = [c for c in out.columns if c.startswith("grade_range_consistency_") and not c.endswith("_se")]
            bin_cols_sorted = sorted(bin_cols, key=_bin_key)
            for c in bin_cols_sorted:
                if c not in ordered:
                    ordered.append(c)
                se_c = f"{c}_se"
                if include_se and se_c in out.columns and se_c not in ordered:
                    ordered.append(se_c)
            continue

        if metric in out.columns:
            ordered.append(metric)
        se_col = f"{metric}_se"
        if include_se and se_col in out.columns:
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
        task_type:
            Type of task, e.g., "classification" or "time_to_event_regression".

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


def extract_split_arrays_tte(
    split_info: Dict[str, Any],
    dataset_df: pd.DataFrame,
    model_preds: Dict[str, Dict[str, Any]],
    patient_id_col: str = "case_id",
    slide_id_col: str = "slide_id",
    loc_key: str = "mu_pred",
    scale_key: str = "log_sigma",
    label_key: str = "time",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract calibration/test TTE prediction and label arrays from slide-level results using case-level split membership.

    Args:
        split_info: Split metadata containing calibration and test case IDs (e.g., in "calib_cases" and "test_cases").
        dataset_df: DataFrame with slide-to-patient mapping and (optionally) labels.
        model_preds: Slide-level model outputs keyed by slide ID; each payload contains prediction/location/scale/label entries.
        patient_id_col: Column name in dataset_df for patient/case ID.
        slide_id_col: Column name in dataset_df for slide ID.
        loc_key: Key in each model_preds payload for the predicted location/mean (mu).
        scale_key: Key in each model_preds payload for the predicted scale (default assumes log sigma if "log_sigma").
        label_key: Key in each model_preds payload for the target survival label.

    Returns:
        A dictionary with NumPy arrays:
            - "calib_mu_pred", "test_mu_pred"
            - "calib_sigma_hat", "test_sigma_hat"
            - "calib_labels", "test_labels"
            - "calib_case_ids", "test_case_ids"
    """
    # Map each slide to its case id.
    slide_to_case = dataset_df.set_index(slide_id_col)[patient_id_col].to_dict()

    # Collectors for each partition.
    calib_mu_pred, test_mu_pred = [], []
    calib_sigma_hat, test_sigma_hat = [], []
    calib_labels, test_labels = [], []
    calib_case_ids, test_case_ids = [], []

    # Case sets for fast membership testing.
    calib_case_set = set(split_info["calib_cases"].values)
    test_case_set = set(split_info["test_cases"].values)

    # Route each slide to calibration or test by its case_id.
    for slide_id, payload in model_preds.items():
        case_id = slide_to_case[slide_id]
        loc = np.asarray(payload[loc_key])
        log_sigma = np.asarray(payload[scale_key])
        sigma = np.exp(log_sigma)  # Convert log sigma to sigma for interpretability.
        label = payload[label_key]

        if case_id in test_case_set:
            test_mu_pred.append(loc)
            test_sigma_hat.append(sigma)
            test_labels.append(label)
            test_case_ids.append(case_id)
        elif case_id in calib_case_set:
            calib_mu_pred.append(loc)
            calib_sigma_hat.append(sigma)
            calib_labels.append(label)
            calib_case_ids.append(case_id)
        else:
            # Defensive: split definitions must cover all cases in model_preds.
            raise ValueError(f"Case ID {case_id} not assigned to calibration or test split.")

    return {
        "calib_mu_pred": np.asarray(calib_mu_pred),
        "test_mu_pred": np.asarray(test_mu_pred),
        "calib_sigma_hat": np.asarray(calib_sigma_hat),
        "test_sigma_hat": np.asarray(test_sigma_hat),
        "calib_labels": np.asarray(calib_labels),
        "test_labels": np.asarray(test_labels),
        "calib_case_ids": np.asarray(calib_case_ids),
        "test_case_ids": np.asarray(test_case_ids),
    }


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
            "selected_coverage": float(top1.get("selected_coverage", np.nan)),
            "selected_set_size": float(top1.get("selected_set_size", np.nan)),
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
            "selected_coverage": float(agg.get("selected_coverage", np.nan)),
            "selected_set_size": float(agg.get("selected_set_size", np.nan)),
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
        "selected_coverage",
        "selected_set_size",
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
    grade_consist_eval: bool = False,
    grade_map: Dict[Any, List[int]] | None = None,
    size_bins: List[Tuple[int, int]] | None = None,
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
        grade_consist_eval:
            If ``True``, compute grade-range consistency on the unselected
            cohort and attach it to each row as a dict under
            'grade_range_consistency'.
        grade_map:
            Mapping ``grade -> [class ids]`` used for grade-range consistency
            evaluation. Required if ``grade_consist_eval=True``.
        size_bins:
            List of ``(min_size, max_size)`` tuples defining size bins for
            grade-range consistency evaluation. If ``None``, no binning is done.
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
    if grade_consist_eval:
        base_cols.append("grade_range_consistency")

    if return_per_class_metrics:
        per_class_cov_cols = [f"coverage_cls_{c}_sel" for c in range(n_classes - 1, -1, -1)]
        per_class_num_cols = [f"num_sel_cls_{c}" for c in range(n_classes - 1, -1, -1)]
        col_order = base_cols + per_class_cov_cols + per_class_num_cols
    else:
        col_order = base_cols

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
        if not grade_consist_eval or grade_map is None:
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
            pred_sets_unsel = set_mat[unsel_mask]
            _attach_grade_consistency(row, pred_sets_unsel, unsel_mask)
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
    grade_consist_eval: bool = False,
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
        grade_consist_eval:
            If ``True`` and a ``grade_map`` is supplied, compute grade-range
            consistency diagnostics on the unselected cohort and attach them to
            each row as a dict under 'grade_range_consistency'.
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
        if not (grade_consist_set or grade_consist_eval) or grade_map is None:
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
                if grade_consist_set or grade_consist_eval:
                    _attach_grade_consistency(row, pred_sets_unsel, unselected_mask)
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
                    selected_mask |= processed_mask

                # CP outputs for unselected only
                pred_sets_unsel = np.asarray(res["prediction_sets"], dtype=bool)
                sizes_unsel = np.asarray(res["set_sizes"])

                # Baseline metrics (no single scalar threshold in per_class mode)
                row = _assemble_base_row(alpha, None, selected_mask, unselected_mask, pred_sets_unsel, sizes_unsel)
                # Optional per-class fields (true class per selected-for-class-i)
                _add_per_class_from_per_class_selection(row, all_selected, tau_list, n_test)
                # Optional grade diagnostics (unselected only)
                if grade_consist_set or grade_consist_eval:
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


def apply_administrative_censoring(
    df: pd.DataFrame,
    study_end_time: float,
    duration_col: str = "survival_time",
    event_col: str = "event",
) -> pd.DataFrame:
    """Apply administrative censoring at `study_end_time` and normalize survival time to [0, 1].

    Args:
        df: Input DataFrame containing survival duration and event indicator columns.
        study_end_time: Administrative study cutoff time. Durations at or beyond this
            cutoff are censored and clipped to this value before normalization.
        duration_col: Name of the survival duration column.
        event_col: Name of the event indicator column (1 = event observed, 0 = censored).

    Returns:
        A copy of `df` with administrative censoring applied:
            - durations > `study_end_time` are clipped to `study_end_time`
            - rows with duration >= `study_end_time` are marked censored (`event_col = 0`)
            - `duration_col` is normalized by dividing by `study_end_time`
    """
    df = df.copy()

    # Keep the original durations so the administrative-censoring rule is explicit/readable.
    original_duration = df[duration_col].astype(float)

    # Mark samples at/after the study cutoff as administratively censored.
    # (This matches the original behavior: durations >= study_end_time -> event = 0.)
    admin_censored_mask = original_duration >= float(study_end_time)

    # Clip follow-up time at the study end.
    df[duration_col] = np.where(
        original_duration > float(study_end_time),
        float(study_end_time),
        original_duration,
    )

    # Overwrite event indicator for administratively censored rows.
    df[event_col] = np.where(
        admin_censored_mask,
        0,
        df[event_col],
    )

    # Normalize durations to [0, 1] relative to the study horizon.
    df[duration_col] = df[duration_col] / float(study_end_time)

    return df


def get_ipcw_weights(
    df_cal: pd.DataFrame,
    duration_col: str = "survival_time",  # Y_i
    event_col: str = "event",             # Δ_i  (1 = event observed, 0 = censored)
    covar_cols: list = ("age_at_index", "gender_male", "mu_pred"),
    n_folds: int = 10,
    clip_eps: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, CoxPHFitter]:
    """
    Compute cross-fitted inverse-probability-of-censoring weights (IPCW) for calibration samples using a Cox censoring model.

    Args:
        df_cal: Calibration dataframe containing survival duration, event indicator, gender, and covariates used for the censoring model.
        duration_col: Column name for observed follow-up time / survival time \(Y_i\).
        event_col: Column name for event indicator \(\Delta_i\), where 1 means event observed and 0 means censored.
        covar_cols: Covariate column names used to fit the Cox proportional hazards censoring model (after gender_male is created).
        n_folds: Number of folds for cross-fitting the censoring model when estimating out-of-fold IPCW weights.
        clip_eps: Lower bound used to clip \(\hat G_c(Y_i \mid X_i)\) to avoid unstable/extremely large weights.
        random_state: Random seed for shuffled K-fold splits.

    Returns:
        A tuple `(weights, cph_full)` where:
        - `weights` is a NumPy array of shape `(n_samples,)` containing cross-fitted IPCW weights \(w_i = \Delta_i / \hat G_c(Y_i \mid X_i)\).
        - `cph_full` is a `CoxPHFitter` trained on the full calibration set for the censoring process (useful for downstream prediction/inspection).
    """
    n = len(df_cal)
    weights = np.empty(n)

    # Make a copy and one‑hot‑encode gender → gender_male ∈ {0,1}
    df = df_cal.copy()
    df["gender_male"] = (df["gender"].str.lower() == "male").astype(int)

    # Event for *censoring* model: 1 = censored, 0 = death
    df["censor_event"] = 1 - df[event_col]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(np.arange(n)):
        df_train = df.iloc[train_idx]
        df_test  = df.iloc[test_idx]

        # Fit Cox PH for censoring
        cph = CoxPHFitter()
        cph.fit(
            df_train[[duration_col, "censor_event", *covar_cols]],
            duration_col=duration_col,
            event_col="censor_event",
            show_progress=False,
        )

        # Predict survival probability of *remaining uncensored* at own Y_i
        # lifelines returns a survival curve; take value at each Y_i
        surv_funcs = cph.predict_survival_function(
            df_test[covar_cols], times=df_test[duration_col]
        )

        # surv_funcs is shape (len(times), len(test_idx)); diagonal contains Ĝ_c(Y_i|X_i)
        G_hat = np.diag(surv_funcs.values)

        # clip to avoid huge weights
        G_hat = np.maximum(G_hat, clip_eps)

        weights[test_idx] = df_test[event_col].values / G_hat

        # Print concordance index for predicting censoring for the held-out test set
        try:
            cindex = concordance_index(
                df_test[duration_col].values,
                -1*cph.predict_partial_hazard(df_test[covar_cols]).values,
                df_test["censor_event"].values
            )
            # print(f"CoxPH censoring model concordance index (fold): {cindex:.4f}")
        except Exception as e:
            print(f"Error computing concordance index: {e}")
        

    # Fit Cox PH model on the entire calibration data for censoring
    cph_full = CoxPHFitter()
    cph_full.fit(
        df[[duration_col, "censor_event", *covar_cols]],
        duration_col=duration_col,
        event_col="censor_event",
        show_progress=False,
    )


    return weights, cph_full


def compute_baselines_for_split_tte(
    alphas: np.ndarray,
    df_test: pd.DataFrame,
    mu_pred: np.ndarray,
    sigma_hat: np.ndarray,
    favorable_thresh_norm: float,
    censor_model,  # e.g., lifelines.CoxPHFitter fit on calibration with event = 1-Δ
    covar_cols: List[str],
    clip_eps: float = 0.05,
    study_end_time_year: float = 5.0,
    pbar_desc: str = "Baselines (Top1 and LNQ)",
) -> Dict[str, pd.DataFrame]:
    """Compute Top1 and LNQ TTE baseline metrics across alphas using IPCW-based coverage estimates.

    Args:
        alphas: 1D array of miscoverage levels in (0, 1). Each alpha produces one LNQ row;
            Top1 is alpha-independent and repeated across all alpha rows for convenience.
        df_test: Test DataFrame containing at least:
            - `survival_time` (observed/censored follow-up time)
            - `event` (1 if event observed, 0 if censored)
            - any covariates listed in `covar_cols`
            - `gender` or `gender_male` if required by downstream utilities.
        mu_pred: Predicted location parameter (e.g., log-time mean) for each test sample;
            must have length `len(df_test)`.
        sigma_hat: Predicted scale parameter for each test sample, either scalar or length
            `len(df_test)`. Must be strictly positive.
        favorable_thresh_norm: Normalized clinical/favorable horizon `c` (in [0, 1]) used for selected-side
            thresholding and Top1/LNQ selection via `S_hat(c | x)`.
        censor_model: Fitted censoring model used by IPCW helper functions to estimate
            `G_hat(Y | X)`.
        covar_cols: Covariate column names used as inputs to the censoring model.
        clip_eps: Lower clipping value for `G_hat` to stabilize IPCW weights.
        study_end_time_year: Administrative study endpoint (in years) used for set-size
            calculations.
        pbar_desc: Progress-bar description for the alpha loop (LNQ baseline).

    Returns:
        A dictionary with two DataFrames:
            - `"top1"`: Top1 baseline metrics (alpha-independent row repeated for each alpha)
            - `"lnq"`: LNQ baseline metrics (alpha-dependent)
        Both DataFrames are indexed by `alphas` and contain columns:
            [
                "selected_coverage_ipcw",
                "mgn_cov_ipcw",
                "mgn_size",
                "unselected_coverage_ipcw",
                "unselected_set_size",
                "num_unsel",
                "num_total",
            ]

    Raises:
        ValueError: If inputs are malformed (e.g., invalid alpha values, size mismatches,
            non-positive sigma, or invalid horizon values).
    """
    # ------------------------------------------------------------------
    # 0) Validate and normalize inputs
    # ------------------------------------------------------------------
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if alphas.size == 0:
        raise ValueError("alphas must be non-empty.")
    if np.any(~np.isfinite(alphas)):
        raise ValueError("alphas contains non-finite values.")
    if np.any((alphas <= 0.0) | (alphas >= 1.0)):
        raise ValueError(
            "alphas must be strictly between 0 and 1 for norm.ppf(alpha) to be finite."
        )

    if not np.isfinite(favorable_thresh_norm) or favorable_thresh_norm <= 0 or favorable_thresh_norm > 1:
        raise ValueError("favorable_thresh_norm must be in (0, 1].")
    if not np.isfinite(study_end_time_year) or study_end_time_year <= 0:
        raise ValueError("study_end_time_year must be > 0.")

    m = len(df_test)

    mu = np.asarray(mu_pred, dtype=float).reshape(-1)
    sig = np.asarray(sigma_hat, dtype=float).reshape(-1)

    if mu.shape[0] != m:
        raise ValueError(f"mu_pred must have length {m}, got {mu.shape[0]}.")

    # Allow sigma_hat to be scalar and broadcast to all test samples.
    if sig.size == 1:
        sig = np.full(m, float(sig.item()), dtype=float)
    if sig.shape[0] != m:
        raise ValueError(f"sigma_hat must be scalar or length {m}, got {sig.shape[0]}.")
    if np.any(~np.isfinite(sig)) or np.any(sig <= 0):
        raise ValueError("sigma_hat must be finite and > 0 everywhere.")

    # 1) Compute S_hat(c | x) once (used by both Top1 and LNQ selectors)
    # Assuming a log-normal style survival model:
    #   T | x ~ LogNormal(mu, sigma^2), so
    #   S_hat(c|x) = P(T > c | x) = 1 - Phi((log c - mu)/sigma)
    z_val = (np.log(float(favorable_thresh_norm)) - mu) / sig
    S_hat_c = 1.0 - norm.cdf(z_val)  # shape (m,)

    base_cols: List[str] = [
        "selected_coverage_ipcw",
        "mgn_cov_ipcw",
        "mgn_size",
        "unselected_coverage_ipcw",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    ]

    # 2) Top1 baseline (alpha-independent): select if S_hat(c|x) >= 0.5
    #    We compute once and repeat the row across all alpha indices.
    sel_mask_top1 = S_hat_c >= 0.5
    sel_idx_top1 = np.flatnonzero(sel_mask_top1)
    unsel_idx_top1 = np.flatnonzero(~sel_mask_top1)

    # Selected-side IPCW coverage at horizon c.
    sel_cov_ipcw_top1 = test_fdp_triplet(
        df_test=df_test,
        sel_idx=sel_idx_top1,
        horizon_c=favorable_thresh_norm,
        cph_censor=censor_model,
        covar_cols=covar_cols,
        clip_eps=clip_eps,
    )

    # Unselected Top1 baseline uses an uninformative LPB = 0 => prediction set [0, inf).
    lpb_top1_unsel = np.zeros(len(unsel_idx_top1), dtype=float)

    if len(unsel_idx_top1) > 0:
        unsel_cov_ipcw_top1 = coverage_lpb_tte(
            df_test=df_test,
            unsel_idx=unsel_idx_top1,
            hat_LPB=lpb_top1_unsel,
            cph_censor=censor_model,
            covar_cols=covar_cols,
            clip_eps=clip_eps,
        )
    else:
        unsel_cov_ipcw_top1 = float("nan")

    # Marginal coverage across all rows:
    #   selected -> threshold c
    #   unselected -> threshold LPB (= 0 for Top1 baseline)
    mgn_cov_ipcw_top1 = marginal_coverage_tte(
        df_test=df_test,
        sel_idx=sel_idx_top1,
        unsel_idx=unsel_idx_top1,
        hat_LPB=lpb_top1_unsel,
        val_sel_threshold=np.full(
            len(sel_idx_top1), favorable_thresh_norm, dtype=float
        ),
        censor_model=censor_model,
        covar_cols=tuple(covar_cols),
        clip_eps=clip_eps,
    )

    # Mean set sizes (times are already in years here, so use scaling factor = 1.0).
    mgn_sz_top1, unsel_sz_top1 = mean_set_sizes(
        m=m,
        sel_idx=sel_idx_top1,
        unsel_idx=unsel_idx_top1,
        horizon_c=favorable_thresh_norm,
        lpb_unsel=lpb_top1_unsel,
        study_end_time_year=float(study_end_time_year),
    )

    top1_row = {
        "selected_coverage_ipcw": float(sel_cov_ipcw_top1),
        "mgn_cov_ipcw": float(mgn_cov_ipcw_top1),
        "mgn_size": float(mgn_sz_top1),
        "unselected_coverage_ipcw": float(unsel_cov_ipcw_top1),
        "unselected_set_size": float(unsel_sz_top1),
        "num_unsel": int(len(unsel_idx_top1)),
        "num_total": int(m),
    }

    # Repeat the same Top1 row for each alpha so output shapes align with LNQ output.
    top1_df = pd.DataFrame([top1_row for _ in range(len(alphas))], index=alphas)[base_cols]

    # 3) LNQ baseline (alpha-dependent)
    #    - Select if S_hat(c|x) >= 1 - alpha
    #    - For unselected rows, use LPB = exp(mu + sigma * z_alpha), z_alpha = Phi^{-1}(alpha)
    lnq_rows: List[Dict[str, Any]] = []

    for a in tqdm.tqdm(alphas, desc=pbar_desc):
        a = float(a)
        z_alpha = float(norm.ppf(a))

        # Alpha-dependent selection rule.
        sel_mask = S_hat_c >= (1.0 - a)
        sel_idx = np.flatnonzero(sel_mask)
        unsel_idx = np.flatnonzero(~sel_mask)

        # Selected-side IPCW coverage at horizon c.
        sel_cov_ipcw = test_fdp_triplet(
            df_test=df_test,
            sel_idx=sel_idx,
            horizon_c=favorable_thresh_norm,
            cph_censor=censor_model,
            covar_cols=covar_cols,
            clip_eps=clip_eps,
        )

        # LNQ unselected LPB (computed for all rows, then subset to unselected rows).
        lpb_all = np.exp(mu + sig * z_alpha)
        lpb_unsel = lpb_all[unsel_idx]  # aligned with unsel_idx

        # Unselected-side IPCW coverage for LNQ LPBs.
        if len(unsel_idx) > 0:
            unsel_cov_ipcw = coverage_lpb_tte(
                df_test=df_test,
                unsel_idx=unsel_idx,
                hat_LPB=lpb_unsel,
                cph_censor=censor_model,
                covar_cols=covar_cols,
                clip_eps=clip_eps,
            )
        else:
            unsel_cov_ipcw = float("nan")

        # Marginal coverage across all rows:
        #   selected -> threshold c
        #   unselected -> threshold LPB(alpha)
        mgn_cov_ipcw = marginal_coverage_tte(
            df_test=df_test,
            sel_idx=sel_idx,
            unsel_idx=unsel_idx,
            hat_LPB=lpb_unsel,
            val_sel_threshold=np.full(len(sel_idx), favorable_thresh_norm, dtype=float),
            censor_model=censor_model,
            covar_cols=tuple(covar_cols),
            clip_eps=clip_eps,
        )

        # Mean set sizes (times are already in years here, so use scaling factor = 1.0).
        mgn_sz, unsel_sz = mean_set_sizes(
            m=m,
            sel_idx=sel_idx,
            unsel_idx=unsel_idx,
            horizon_c=favorable_thresh_norm,
            lpb_unsel=lpb_unsel,
            study_end_time_year=float(study_end_time_year),  # no extra scaling since thresholds are in years
        )

        lnq_rows.append(
            {
                "selected_coverage_ipcw": float(sel_cov_ipcw),
                "mgn_cov_ipcw": float(mgn_cov_ipcw),
                "mgn_size": float(mgn_sz),
                "unselected_coverage_ipcw": float(unsel_cov_ipcw),
                "unselected_set_size": float(unsel_sz),
                "num_unsel": int(len(unsel_idx)),
                "num_total": int(m),
            }
        )

    lnq_df = pd.DataFrame(lnq_rows, index=alphas)[base_cols]

    return {"top1": top1_df, "lnq": lnq_df}


def _ensure_gender_male(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a binary `gender_male` column exists in the DataFrame.

    Args:
        df: Input DataFrame that must contain either:
            - a precomputed `gender_male` column, or
            - a `gender` column from which `gender_male` can be derived.

    Returns:
        A DataFrame containing `gender_male` (0/1). If `gender_male` already exists,
        the original DataFrame is returned unchanged. Otherwise, a copy is returned
        with `gender_male` added.

    Raises:
        KeyError: If neither `gender_male` nor `gender` exists in `df`.
    """
    # If already present, do not overwrite (preserves existing preprocessing).
    if "gender_male" not in df.columns:
        if "gender" not in df.columns:
            raise KeyError("df must contain 'gender' or already have 'gender_male'.")

        # Copy only when we need to add a new column.
        df = df.copy()

        # Normalize to string/lowercase so values like "Male", "male", etc. map correctly.
        df["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(int)

    return df


def _predict_G_hat_at_times(
    censor_model,
    X: pd.DataFrame,
    times: np.ndarray,
    clip_eps: float,
) -> np.ndarray:
    """Predict censoring survival probabilities G_hat(t_i | x_i) for each row-specific time.

    Args:
        censor_model: Fitted censoring survival model (e.g., lifelines CoxPHFitter trained
            for censoring), expected to implement `predict_survival_function(X, times=...)`.
        X: Covariate DataFrame of shape (n, p), one row per individual.
        times: Array-like of shape (n,) containing the evaluation time for each row.
            The i-th output is G_hat(times[i] | X.iloc[i]).
        clip_eps: Lower bound used to clip predicted survival probabilities away from zero
            for numerical stability.

    Returns:
        A NumPy array of shape (n,) containing clipped censoring survival probabilities
        `G_hat(times[i] | x_i)`.

    Raises:
        ValueError: If `times` is not 1D or if its length does not match `len(X)`.
    """
    # Convert to a flat float array for consistent downstream indexing.
    times = np.asarray(times, dtype=float).reshape(-1)

    if times.ndim != 1:
        raise ValueError("times must be a 1D array.")
    if len(times) != len(X):
        raise ValueError("times length must match number of rows in X.")

    # Evaluate only at unique times for efficiency and stable indexing.
    uniq_times = np.unique(times)

    # lifelines convention:
    #   index   -> evaluation times
    #   columns -> individuals in the same order as rows in X
    #   shape   -> (n_unique_times, n_individuals)
    surv_df = censor_model.predict_survival_function(X, times=uniq_times)

    # Build a mapping from time value -> row index in surv_df.
    # This assumes surv_df.index matches the requested uniq_times (as lifelines typically does).
    time_to_row = {
        float(t): i for i, t in enumerate(np.asarray(surv_df.index, dtype=float))
    }

    # For each original row i, find which row in surv_df corresponds to times[i].
    row_idx = np.fromiter(
        (time_to_row[float(t)] for t in times),
        dtype=int,
        count=len(times),
    )

    # Column index is just the sample index (same order as X rows).
    col_idx = np.arange(len(X), dtype=int)

    # Gather diagonal-like entries: one (time_i, subject_i) value per sample.
    vals = surv_df.values
    G_hat = vals[row_idx, col_idx]

    # Clip away from 0 to prevent extreme/unstable IPCW weights.
    G_hat = np.maximum(G_hat, float(clip_eps))
    return G_hat


def test_fdp_triplet(
    df_test: pd.DataFrame,
    sel_idx: np.ndarray,
    horizon_c: float,
    cph_censor,  # CoxPHFitter (fit on censoring: event = 1 - Δ)
    covar_cols: List[str],
    clip_eps: float = 0.05,
) -> float:
    """Compute IPCW-adjusted selected-set coverage at a fixed clinical horizon.

    Args:
        df_test: Test DataFrame containing at least `survival_time`, `event`, and covariate
            columns used by the censoring model (plus `gender` or `gender_male` if needed).
        sel_idx: Integer indices of selected samples within `df_test`.
        horizon_c: Clinical time horizon `c` used to define the selected-set threshold.
        cph_censor: Fitted censoring model used to estimate `G_hat(Y | X)`.
        covar_cols: Covariate column names (must match the censoring model inputs).
        clip_eps: Minimum value used to clip `G_hat` for IPCW stability.

    Returns:
        Selected-set IPCW coverage estimate:
            `1 - mean(Δ * 1{Y <= c} / G_hat(Y|X))`
        over selected samples, or `np.nan` if `sel_idx` is empty.
    """
    sel_idx = np.asarray(sel_idx, dtype=int)
    if sel_idx.size == 0:
        return float("nan")

    # Restrict to selected rows.
    sub = df_test.iloc[sel_idx].copy()
    sub = _ensure_gender_male(sub)

    # Observed follow-up time Y and event indicator Δ.
    Y = sub["survival_time"].to_numpy(dtype=float)
    D = sub["event"].to_numpy(dtype=int)

    # Covariates for censoring survival prediction G_hat(Y | X).
    X = sub[covar_cols]
    G_hat = _predict_G_hat_at_times(
        censor_model=cph_censor,
        X=X,
        times=Y,
        clip_eps=clip_eps,
    )

    # IPCW error proxy for "event occurred by horizon c" among selected.
    # Coverage = 1 - error.
    err_ipcw = (D * (Y <= float(horizon_c))) / G_hat
    fdp_ipcw = float(np.mean(err_ipcw))
    return float(1.0 - fdp_ipcw)


def coverage_lpb_tte(
    df_test: pd.DataFrame,
    unsel_idx: np.ndarray,
    hat_LPB: np.ndarray,  # aligned with unsel_idx
    cph_censor,
    covar_cols: List[str],
    clip_eps: float = 0.05,
) -> float:
    """Compute IPCW-adjusted coverage for unselected samples using per-sample LPB thresholds.

    Args:
        df_test: Test DataFrame containing at least `survival_time`, `event`, and covariate
            columns used by the censoring model (plus `gender` or `gender_male` if needed).
        unsel_idx: Integer indices of unselected (deferred) samples within `df_test`.
        hat_LPB: Array of lower predictive bounds (or analogous thresholds) aligned with
            `unsel_idx` (same length and same order).
        cph_censor: Fitted censoring model used to estimate `G_hat(Y | X)`.
        covar_cols: Covariate column names (must match the censoring model inputs).
        clip_eps: Minimum value used to clip `G_hat` for IPCW stability.

    Returns:
        Unselected-set IPCW coverage estimate:
            `1 - mean(Δ * 1{Y < L} / G_hat(Y|X))`
        over unselected samples, or `np.nan` if `unsel_idx` is empty.

    Raises:
        ValueError: If `hat_LPB` length does not match `unsel_idx`.
    """
    unsel_idx = np.asarray(unsel_idx, dtype=int)
    if unsel_idx.size == 0:
        return float("nan")

    # Restrict to unselected rows.
    sub = df_test.iloc[unsel_idx].copy()
    sub = _ensure_gender_male(sub)

    # Validate and align LPB thresholds to unselected rows.
    L = np.asarray(hat_LPB, dtype=float).reshape(-1)
    if L.shape[0] != len(unsel_idx):
        raise ValueError("hat_LPB must have the same length as unsel_idx.")

    Y = sub["survival_time"].to_numpy(dtype=float)
    D = sub["event"].to_numpy(dtype=int)

    X = sub[covar_cols]
    G_hat = _predict_G_hat_at_times(
        censor_model=cph_censor,
        X=X,
        times=Y,
        clip_eps=clip_eps,
    )

    # IPCW error proxy for the event "true time T < L".
    # Coverage = 1 - error.
    err_ipcw = (D * (Y < L)) / G_hat
    err_rate = float(np.mean(err_ipcw))
    return float(1.0 - err_rate)


def marginal_coverage_tte(
    df_test: pd.DataFrame,
    sel_idx: np.ndarray,
    unsel_idx: np.ndarray,
    hat_LPB: np.ndarray,
    val_sel_threshold: np.ndarray,
    censor_model,
    covar_cols: Sequence[str],
    clip_eps: float = 0.05,
) -> float:
    """Compute IPCW-adjusted marginal coverage across all test samples.

    Args:
        df_test: Test DataFrame containing at least `survival_time`, `event`, and covariate
            columns used by the censoring model (plus `gender` or `gender_male` if needed).
        sel_idx: Integer indices of selected samples.
        unsel_idx: Integer indices of unselected (deferred) samples.
        hat_LPB: Thresholds for unselected samples (e.g., lower predictive bounds),
            aligned with `unsel_idx`.
        val_sel_threshold: Thresholds for selected samples, aligned with `sel_idx`
            (often a constant horizon `c` repeated).
        censor_model: Fitted censoring model used to estimate `G_hat(Y | X)`.
        covar_cols: Covariate column names (must match the censoring model inputs).
        clip_eps: Minimum value used to clip `G_hat` for IPCW stability.

    Returns:
        IPCW marginal coverage across all test rows:
            `1 - mean(Δ * 1{Y <= theta} / G_hat(Y|X))`,
        where `theta` is per-row threshold assembled from selected/unselected assignments.

    Raises:
        ValueError: If threshold arrays do not align with their corresponding index arrays,
            or if `sel_idx` and `unsel_idx` do not jointly cover all test rows.
    """
    m = len(df_test)
    sel_idx = np.asarray(sel_idx, dtype=int)
    unsel_idx = np.asarray(unsel_idx, dtype=int)

    # Build a per-row threshold vector theta_j:
    #   - selected rows use val_sel_threshold (typically c)
    #   - unselected rows use hat_LPB
    theta = np.full(m, np.nan, dtype=float)

    if sel_idx.size > 0:
        v = np.asarray(val_sel_threshold, dtype=float).reshape(-1)
        if v.shape[0] != sel_idx.size:
            raise ValueError("val_sel_threshold must align with sel_idx.")
        theta[sel_idx] = v

    if unsel_idx.size > 0:
        L = np.asarray(hat_LPB, dtype=float).reshape(-1)
        if L.shape[0] != unsel_idx.size:
            raise ValueError("hat_LPB must align with unsel_idx.")
        theta[unsel_idx] = L

    # Ensure every test row has a threshold assignment.
    if np.any(np.isnan(theta)):
        raise ValueError(
            "theta contains NaNs; sel_idx and unsel_idx must cover all test rows."
        )

    df = df_test.copy()
    df = _ensure_gender_male(df)

    Y = df["survival_time"].to_numpy(dtype=float)
    D = df["event"].to_numpy(dtype=int)

    X = df[list(covar_cols)]
    G_hat = _predict_G_hat_at_times(
        censor_model=censor_model,
        X=X,
        times=Y,
        clip_eps=clip_eps,
    )

    # IPCW error proxy for per-row thresholded event; coverage is one minus mean error.
    err_ipcw = (D * (Y <= theta)) / G_hat
    mc = float(1.0 - np.mean(err_ipcw))
    return mc


def mean_set_sizes(
    m: int,
    sel_idx: np.ndarray,
    unsel_idx: np.ndarray,
    horizon_c: float,
    lpb_unsel: np.ndarray,  # aligned with unsel_idx
    study_end_time_year: float = 5.0,
) -> Tuple[float, float]:
    """Compute marginal and unselected mean set sizes from per-sample lower thresholds.

    Args:
        m: Total number of test samples.
        sel_idx: Integer indices of selected samples (used implicitly via default threshold
            assignment to `horizon_c`).
        unsel_idx: Integer indices of unselected samples.
        horizon_c: Threshold assigned to selected samples (e.g., fixed clinical horizon).
        lpb_unsel: Per-unselected lower predictive bounds (LPBs), aligned with `unsel_idx`.
        study_end_time_year: Scaling factor applied to set sizes (e.g., convert normalized
            horizon units to years). Defaults to 5.0.

    Returns:
        A tuple `(mgn_size, unselected_set_size)` where:
            - `mgn_size` is the mean set size across all `m` samples.
            - `unselected_set_size` is the mean set size among unselected samples only,
              or `np.nan` if there are no unselected samples.

    Raises:
        ValueError: If `lpb_unsel` length does not match `unsel_idx`.
    """
    sel_idx = np.asarray(sel_idx, dtype=int)  # kept for interface consistency / validation context
    unsel_idx = np.asarray(unsel_idx, dtype=int)

    # Initialize all thresholds as horizon_c (selected default).
    theta_all = np.full(int(m), float(horizon_c), dtype=float)

    # Replace thresholds for unselected rows with their LPBs.
    if unsel_idx.size > 0:
        L = np.asarray(lpb_unsel, dtype=float).reshape(-1)
        if L.shape[0] != unsel_idx.size:
            raise ValueError("lpb_unsel must align with unsel_idx.")
        theta_all[unsel_idx] = L

    # Set size for threshold theta is interval length [theta, study_end_time], truncated at 0.
    sizes_all = np.maximum(0.0, 1.0 - theta_all)
    mgn_size = float(np.mean(sizes_all) * float(study_end_time_year))

    if unsel_idx.size > 0:
        sizes_unsel = np.maximum(
            0.0,
            1.0 - np.asarray(lpb_unsel, dtype=float),
        )
        unsel_size = float(np.mean(sizes_unsel) * float(study_end_time_year))
    else:
        unsel_size = float("nan")

    return mgn_size, unsel_size


def run_vanilla_cp_for_split_tte(
    alphas: np.ndarray,
    cal_labels: np.ndarray,       
    cal_mu_pred: np.ndarray,      
    cal_sigma_hat: np.ndarray,    
    df_test: pd.DataFrame,
    test_mu_pred: np.ndarray,     
    test_sigma_hat: np.ndarray,  
    favorable_thresh_norm: float,    
    censor_model: CoxPHFitter,               
    covar_cols: List[str],
    clip_eps: float = 0.05,
    clip_ppf: float = 1e-6,
    eps_u: float = 1e-12,
    study_end_time_year: float = 5.0,
    pbar_desc: str = "Vanilla CP",
) -> pd.DataFrame:
    """Run vanilla conformal prediction for TTE and return IPCW-based evaluation metrics across alphas.

    Args:
        alphas: 1D array of miscoverage levels in (0, 1). One row is produced per alpha.
        cal_labels: Calibration observed labels (e.g., transformed event/censor times used for
            conformalization), expected to be strictly positive because `log(cal_labels)` is used.
        cal_mu_pred: Predicted location parameter (e.g., log-time mean) for calibration samples.
            Must align with `cal_labels`.
        cal_sigma_hat: Predicted scale parameter(s) for calibration samples. Can be scalar or
            an array aligned with `cal_mu_pred`. Must be strictly positive.
        df_test: Test DataFrame used for evaluation; must contain at least `survival_time`,
            `event`, and the covariates used by `censor_model` (plus `gender` or `gender_male`
            if required by helper utilities).
        test_mu_pred: Predicted location parameter for test samples; must have length `len(df_test)`.
        test_sigma_hat: Predicted scale parameter(s) for test samples; can be scalar or length
            `len(df_test)`. Must be strictly positive.
        favorable_thresh_norm: The favorable threshold in normalized time (e.g., 0.4 for 40% of the study horizon).
        censor_model: Fitted censoring model used by IPCW helper functions to estimate `G_hat(Y|X)`.
        covar_cols: Covariate column names required by `censor_model`.
        clip_eps: Lower clipping value for `G_hat` in IPCW computations for numerical stability.
        clip_ppf: Clipping level for probabilities passed into `norm.ppf` to avoid infinities.
        eps_u: Clipping level for calibration PIT-style values `U_cal` to keep them in `(0,1)`.
        study_end_time_year: Scaling factor used by `mean_set_sizes` to convert set sizes to years
            (e.g., 5.0 if times are normalized to `[0, 1]` over a 5-year horizon; use 1.0 if
            times are already in years).
        pbar_desc: Progress-bar description shown during the alpha loop.
        censor_model: Fitted censoring model used by IPCW helper functions to estimate `G_hat(Y|X)`.
        covar_cols: Covariate column names required by `censor_model`.
        clip_eps: Lower clipping value for `G_hat` in IPCW computations for numerical stability.
        clip_ppf: Clipping level for probabilities passed into `norm.ppf` to avoid infinities.
        eps_u: Clipping level for calibration PIT-style values `U_cal` to keep them in `(0,1)`.
        study_end_time_year: Scaling factor used by `mean_set_sizes` to convert set sizes to years
            (e.g., 5.0 if times are normalized to `[0, 1]` over a 5-year horizon; use 1.0 if
            times are already in years).
        pbar_desc: Progress-bar description shown during the alpha loop.

    Returns:
        A DataFrame indexed by `alphas` with columns:
            - `selected_coverage_ipcw`
            - `mgn_cov_ipcw`
            - `mgn_size`
            - `unselected_coverage_ipcw`
            - `unselected_set_size`
            - `num_unsel`
            - `num_total`

    Raises:
        ValueError: If input shapes/values are invalid (e.g., alpha out of range, length mismatch,
            non-positive sigma, non-positive labels for log-transform, etc.).
    """
    # ------------------------------------------------------------------
    # 0) Validate and normalize inputs
    # ------------------------------------------------------------------
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if alphas.size == 0:
        raise ValueError("alphas must be non-empty.")
    if np.any(~np.isfinite(alphas)):
        raise ValueError("alphas contains non-finite values.")
    if np.any((alphas <= 0.0) | (alphas >= 1.0)):
        raise ValueError("alphas must be strictly between 0 and 1 (norm.ppf(alpha) finite).")

    # Calibration arrays (used to compute the vanilla conformal offset per alpha).
    cal_labels = np.asarray(cal_labels, dtype=float).reshape(-1)
    cal_mu = np.asarray(cal_mu_pred, dtype=float).reshape(-1)
    cal_sig = np.asarray(cal_sigma_hat, dtype=float).reshape(-1)

    if cal_labels.shape[0] != cal_mu.shape[0]:
        raise ValueError("cal_labels and cal_mu_pred must have the same length.")

    # Allow scalar sigma and broadcast across all calibration samples.
    if cal_sig.size == 1:
        cal_sig = np.full_like(cal_mu, float(cal_sig.item()), dtype=float)
    if cal_sig.shape[0] != cal_mu.shape[0]:
        raise ValueError("cal_sigma_hat must be scalar or the same length as cal_mu_pred.")

    if np.any(~np.isfinite(cal_labels)) or np.any(cal_labels <= 0):
        raise ValueError("cal_labels must be finite and > 0 (used in log).")
    if np.any(~np.isfinite(cal_mu)):
        raise ValueError("cal_mu_pred contains non-finite values.")
    if np.any(~np.isfinite(cal_sig)) or np.any(cal_sig <= 0):
        raise ValueError("cal_sigma_hat must be finite and > 0 everywhere.")

    # Test arrays (used to construct LPBs and evaluate metrics on df_test).
    m = len(df_test)
    test_mu = np.asarray(test_mu_pred, dtype=float).reshape(-1)
    test_sig = np.asarray(test_sigma_hat, dtype=float).reshape(-1)

    if test_mu.shape[0] != m:
        raise ValueError(f"test_mu_pred must have length {m}, got {test_mu.shape[0]}.")

    # Allow scalar sigma and broadcast across all test samples.
    if test_sig.size == 1:
        test_sig = np.full(m, float(test_sig.item()), dtype=float)
    if test_sig.shape[0] != m:
        raise ValueError(f"test_sigma_hat must be scalar or length {m}, got {test_sig.shape[0]}.")

    if np.any(~np.isfinite(test_mu)):
        raise ValueError("test_mu_pred contains non-finite values.")
    if np.any(~np.isfinite(test_sig)) or np.any(test_sig <= 0):
        raise ValueError("test_sigma_hat must be finite and > 0 everywhere.")

    if not np.isfinite(favorable_thresh_norm) or favorable_thresh_norm <= 0:
        raise ValueError("favorable_thresh_norm must be finite and > 0.")
    if not np.isfinite(study_end_time_year) or study_end_time_year <= 0:
        raise ValueError("study_end_time_year must be finite and > 0.")
    if not np.isfinite(clip_eps) or clip_eps <= 0:
        raise ValueError("clip_eps must be finite and > 0.")
    if not np.isfinite(clip_ppf) or not (0 < clip_ppf < 0.5):
        raise ValueError("clip_ppf must satisfy 0 < clip_ppf < 0.5.")
    if not np.isfinite(eps_u) or not (0 < eps_u < 0.5):
        raise ValueError("eps_u must satisfy 0 < eps_u < 0.5.")

    # Normalize the selection threshold(s) to either:
    #   - scalar float, or
    #   - 1D array of length m
    if np.isscalar(favorable_thresh_norm):
        vt: float | np.ndarray = float(favorable_thresh_norm)
        if not np.isfinite(vt):
            raise ValueError("favorable_thresh_norm (scalar) must be finite.")
    else:
        vt = np.asarray(favorable_thresh_norm, dtype=float).reshape(-1)
        if vt.shape[0] != m:
            raise ValueError(f"favorable_thresh_norm must be scalar or length {m}, got {vt.shape[0]}.")
        if np.any(~np.isfinite(vt)):
            raise ValueError("favorable_thresh_norm contains non-finite values.")

    base_cols = [
        "selected_coverage_ipcw",
        "mgn_cov_ipcw",
        "mgn_size",
        "unselected_coverage_ipcw",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    ]

    # ------------------------------------------------------------------
    # 1) Compute calibration U-values once (alpha-independent)
    #    U_cal = Phi((log(T_cal) - mu_cal) / sigma_cal)
    # ------------------------------------------------------------------
    U_cal = norm.cdf((np.log(cal_labels) - cal_mu) / cal_sig)
    U_cal = np.clip(U_cal, float(eps_u), 1.0 - float(eps_u))

    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 2) Loop over alphas: construct vanilla CP LPB on test and evaluate metrics
    # ------------------------------------------------------------------
    for a in tqdm.tqdm(alphas, desc=pbar_desc):
        a = float(a)

        # Vanilla CP offset construction:
        #   V_i = alpha - U_i
        #   eta = quantile_{1-alpha}(V) using the "higher" quantile rule
        #   p_target = alpha - eta
        #   LPB(x) = exp(mu(x) + sigma(x) * Phi^{-1}(p_target))
        V = a - U_cal

        eta = float(np.quantile(V, 1.0 - a, method="higher"))       
        p_target = float(a - eta)

        # Clip probability before ppf to avoid +/- inf numerical issues.
        p_target = float(np.clip(p_target, float(clip_ppf), 1.0 - float(clip_ppf)))
        z_pt = float(norm.ppf(p_target))

        # LPB on all test points (same units as the model output time scale).
        LPB_test = np.exp(test_mu + test_sig * z_pt)  # shape (m,)

        # Selection rule: selected if LPB_test >= threshold c_j
        if np.isscalar(vt):
            sel_mask = LPB_test >= float(vt)
            # Threshold values for selected rows (needed by marginal coverage helper).
            val_sel_threshold = np.full(int(np.sum(sel_mask)), float(vt), dtype=float)
        else:
            sel_mask = LPB_test >= vt
            val_sel_threshold = vt[np.flatnonzero(sel_mask)]

        sel_idx = np.flatnonzero(sel_mask)
        unsel_idx = np.flatnonzero(~sel_mask)

        # Selected-side coverage at the fixed horizon (FDP-style IPCW metric)
        selected_cov = test_fdp_triplet(
            df_test=df_test,
            sel_idx=sel_idx,
            horizon_c=favorable_thresh_norm,
            cph_censor=censor_model,
            covar_cols=covar_cols,
            clip_eps=clip_eps,
        )

        # Unselected-side coverage using vanilla CP LPBs
        hat_LPB_unsel = LPB_test[unsel_idx]  # aligned with unsel_idx

        unselected_cov = coverage_lpb_tte(
            df_test=df_test,
            unsel_idx=unsel_idx,
            hat_LPB=hat_LPB_unsel,
            cph_censor=censor_model,
            covar_cols=covar_cols,
            clip_eps=clip_eps,
        )

        # Marginal coverage across all test samples
        #   selected   -> threshold = val_sel_threshold (typically c)
        #   unselected -> threshold = hat_LPB_unsel
        mgn_cov = marginal_coverage_tte(
            df_test=df_test,
            sel_idx=sel_idx,
            unsel_idx=unsel_idx,
            hat_LPB=hat_LPB_unsel,
            val_sel_threshold=val_sel_threshold,
            censor_model=censor_model,
            covar_cols=tuple(covar_cols),
            clip_eps=clip_eps,
        )

        # Mean set sizes (all + unselected only)
        # NOTE:
        #   - `study_end_time_year` is the scaling factor to convert sizes to years
        mgn_sz, unsel_sz = mean_set_sizes(
            m=m,
            sel_idx=sel_idx,
            unsel_idx=unsel_idx,
            horizon_c=favorable_thresh_norm,
            lpb_unsel=hat_LPB_unsel,
            study_end_time_year=float(study_end_time_year),
        )

        rows.append(
            {
                "selected_coverage_ipcw": float(selected_cov),
                "mgn_cov_ipcw": float(mgn_cov),
                "mgn_size": float(mgn_sz),
                "unselected_coverage_ipcw": float(unselected_cov),
                "unselected_set_size": float(unsel_sz),
                "num_unsel": int(len(unsel_idx)),
                "num_total": int(m),
            }
        )

    vanilla_df = pd.DataFrame(rows, index=alphas)[base_cols]
    return vanilla_df


def compute_stratcp_survival_for_split(
    alphas: np.ndarray,
    cal_labels: np.ndarray,
    cal_mu_pred: np.ndarray,
    cal_sigma_hat: np.ndarray,
    df_test: pd.DataFrame,
    test_mu_pred: np.ndarray,
    test_sigma_hat: np.ndarray,
    favorable_thresh_norm: float,
    censor_model,
    covar_cols: List[str],
    w_ipcw: Optional[np.ndarray] = None,
    clip_eps: float = 0.05,
    clip_ppf: float = 1e-12,
    study_end_time_year: float = 5.0,
    pbar_desc: str = "StratCP (two-stage survival)",
) -> pd.DataFrame:
    """Run two-stage StratCP for survival on one split and return IPCW-based metrics across alphas.

    Args:
        alphas: 1D array of miscoverage levels in (0, 1). One output row is produced per alpha.
        cal_labels: Calibration labels used by StratCP (must be strictly positive if the
            underlying survival model assumes log-time transforms).
        cal_mu_pred: Calibration predicted location parameters (e.g., log-time means), aligned
            with `cal_labels`.
        cal_sigma_hat: Calibration predicted scale parameters; may be scalar or length `n_cal`.
            Must be strictly positive.
        df_test: Test DataFrame used for IPCW evaluation. Must contain at least:
            - `survival_time`
            - `event`
            - covariates in `covar_cols`
            - `gender` or `gender_male` (if required by downstream helpers)
        test_mu_pred: Test predicted location parameters, length `len(df_test)`.
        test_sigma_hat: Test predicted scale parameters; may be scalar or length `len(df_test)`.
            Must be strictly positive.
        favorable_thresh_norm: Real-world favorable threshold (in normalized time) used for StratCP
            selection/JOMI thresholding. Internally converted to model time domain.
        censor_model: Fitted censoring model used by IPCW helper functions to estimate
            `G_hat(Y | X)`.
        covar_cols: Covariate column names required by `censor_model`.
        w_ipcw: Optional IPCW weights for StratCP calibration (passed into `StratifiedCP`).
        clip_eps: Lower clipping value for `G_hat` in IPCW computations.
        clip_ppf: Probability clipping passed to StratCP prediction to avoid `norm.ppf`
            numerical issues in the survival path.
        study_end_time_year: Study endpoint in the real-world time domain (e.g., 5.0 years).
            Used for set-size calculations and threshold conversion.
        pbar_desc: Progress-bar description for the alpha loop.

    Returns:
        A DataFrame indexed by `alphas` with columns:
            - `selected_coverage_ipcw`
            - `mgn_cov_ipcw`
            - `mgn_size`
            - `unselected_coverage_ipcw`
            - `unselected_set_size`
            - `num_unsel`
            - `num_total`

    Raises:
        ValueError: If inputs are malformed (e.g., invalid alpha range, non-positive sigma,
            shape mismatches, invalid time conversion parameters, or misaligned StratCP output).
    """
    # 0) Validate scalar inputs and alpha grid
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if alphas.size == 0:
        raise ValueError("alphas must be non-empty.")
    if np.any(~np.isfinite(alphas)):
        raise ValueError("alphas contains non-finite values.")
    if np.any((alphas <= 0.0) | (alphas >= 1.0)):
        raise ValueError("alphas must be strictly between 0 and 1.")

    if not np.isfinite(favorable_thresh_norm) or favorable_thresh_norm <= 0:
        raise ValueError("favorable_thresh_norm must be a finite positive number (normalized time).")
    if not np.isfinite(clip_eps) or clip_eps <= 0:
        raise ValueError("clip_eps must be finite and > 0.")
    if not np.isfinite(clip_ppf) or not (0 < clip_ppf < 0.5):
        raise ValueError("clip_ppf must satisfy 0 < clip_ppf < 0.5.")
    if not np.isfinite(study_end_time_year) or study_end_time_year <= 0:
        raise ValueError("study_end_time_year must be a finite positive number (years).")

    # 1) Normalize calibration and test arrays
    cal_labels = np.asarray(cal_labels, dtype=float).reshape(-1)
    cal_mu_pred = np.asarray(cal_mu_pred, dtype=float).reshape(-1)
    cal_sigma_hat = np.asarray(cal_sigma_hat, dtype=float).reshape(-1)

    n_cal = cal_labels.shape[0]
    if cal_mu_pred.shape[0] != n_cal:
        raise ValueError("cal_labels and cal_mu_pred must have the same length.")
    if np.any(~np.isfinite(cal_labels)):
        raise ValueError("cal_labels contains non-finite values.")
    if np.any(~np.isfinite(cal_mu_pred)):
        raise ValueError("cal_mu_pred contains non-finite values.")

    # Broadcast calibration sigma if scalar.
    if cal_sigma_hat.size == 1:
        cal_sigma_hat = np.full(n_cal, float(cal_sigma_hat.item()), dtype=float)
    if cal_sigma_hat.shape[0] != n_cal:
        raise ValueError("cal_sigma_hat must be scalar or the same length as cal_mu_pred.")
    if np.any(~np.isfinite(cal_sigma_hat)) or np.any(cal_sigma_hat <= 0):
        raise ValueError("cal_sigma_hat must be finite and > 0 everywhere.")

    # If your survival path is log-normal based, positive labels are required.
    if np.any(cal_labels <= 0):
        raise ValueError("cal_labels must be > 0 for the survival conformalization path.")

    m = len(df_test)
    test_mu_pred = np.asarray(test_mu_pred, dtype=float).reshape(-1)
    test_sigma_hat = np.asarray(test_sigma_hat, dtype=float).reshape(-1)

    if test_mu_pred.shape[0] != m:
        raise ValueError(f"test_mu_pred must have length {m}, got {test_mu_pred.shape[0]}.")
    if np.any(~np.isfinite(test_mu_pred)):
        raise ValueError("test_mu_pred contains non-finite values.")

    # Broadcast test sigma if scalar.
    if test_sigma_hat.size == 1:
        test_sigma_hat = np.full(m, float(test_sigma_hat.item()), dtype=float)
    if test_sigma_hat.shape[0] != m:
        raise ValueError("test_sigma_hat must be scalar or the same length as df_test.")
    if np.any(~np.isfinite(test_sigma_hat)) or np.any(test_sigma_hat <= 0):
        raise ValueError("test_sigma_hat must be finite and > 0 everywhere.")

    # Optional StratCP calibration weights (if provided) should align to calibration rows.
    if w_ipcw is not None:
        w_ipcw = np.asarray(w_ipcw, dtype=float).reshape(-1)
        if w_ipcw.shape[0] != n_cal:
            raise ValueError("w_ipcw must have the same length as calibration data.")
        if np.any(~np.isfinite(w_ipcw)):
            raise ValueError("w_ipcw contains non-finite values.")
        if np.any(w_ipcw < 0):
            raise ValueError("w_ipcw must be non-negative.")

    # Broadcast explicit threshold vectors for calibration and test.
    cal_threshold_arr = np.full(n_cal, favorable_thresh_norm, dtype=float)
    test_threshold_arr = np.full(m, favorable_thresh_norm, dtype=float)

    # This is the selected-side threshold used in marginal coverage / set-size helpers.
    base_cols = [
        "selected_coverage_ipcw",
        "mgn_cov_ipcw",
        "mgn_size",
        "unselected_coverage_ipcw",
        "unselected_set_size",
        "num_unsel",
        "num_total",
    ]
    rows: List[Dict[str, Any]] = []

    # 3) Build and fit StratifiedCP engine once on calibration data
    # Current StratifiedCP API may still require `cal_probs` / `test_probs` even in survival mode.
    # We pass dummy placeholders for compatibility; the survival path should ignore them.
    dummy_cal_probs = np.zeros((n_cal, 1), dtype=float)

    cp = StratifiedCP(
        task_type="time_to_event_regression",
        alpha_sel=float(alphas[0]),  # placeholder; overwritten inside loop
        w_ipcw=w_ipcw,
    )
    cp.fit(
        cal_probs=dummy_cal_probs,
        cal_labels=cal_labels,
        cal_loc_hat=cal_mu_pred,
        cal_scale_hat=cal_sigma_hat,
        cal_threshold=cal_threshold_arr,
    )

    dummy_test_probs = np.zeros((m, 1), dtype=float)

    # 4) Loop over alphas: run StratCP (selection + JOMI LPB), then evaluate
    for a in tqdm.tqdm(alphas, desc=pbar_desc):
        a = float(a)

        # In this two-stage setup, alpha_sel controls both:
        #   (i) selection
        #   (ii) JOMI LPB construction for unselected samples
        cp.alpha_sel = a

        out = cp.predict(
            test_probs=dummy_test_probs,   # compatibility placeholder for current API
            test_loc_hat=test_mu_pred,
            test_scale_hat=test_sigma_hat,
            test_threshold=test_threshold_arr,
            surv_model_family="log_normal",
            clip_ppf=clip_ppf,
        )
        # StratCP should return a partition of test indices.
        sel_idx = np.asarray(out["selected_idx"], dtype=int)
        unsel_idx = np.asarray(out["unselected_idx"], dtype=int)

        # Support either old key name ("lcb_unsel") or newer ("lpb_unsel"), but use LPB terminology locally.
        hat_LPB_unsel = np.asarray(out["lcb_unsel"], dtype=float).reshape(-1)

        # Optional retained threshold output (useful for debugging / parity checks if needed).
        _tau_hat = out.get("threshold", None)

        # Ensure LPBs align exactly with the unselected partition.
        if hat_LPB_unsel.shape[0] != unsel_idx.size:
            raise ValueError("StratifiedCP returned LPBs not aligned with unselected_idx.")

        # Selected-side IPCW coverage at horizon_c
        # NOTE: `test_fdp_triplet` returns coverage directly (not an FDP triplet tuple).
        selected_cov = test_fdp_triplet(
            df_test=df_test,
            sel_idx=sel_idx,
            horizon_c=favorable_thresh_norm,
            cph_censor=censor_model,
            covar_cols=covar_cols,
            clip_eps=clip_eps,
        )

        # Unselected-side IPCW coverage using StratCP LPBs
        unselected_cov = coverage_lpb_tte(
            df_test=df_test,
            unsel_idx=unsel_idx,
            hat_LPB=hat_LPB_unsel,
            cph_censor=censor_model,
            covar_cols=covar_cols,
            clip_eps=clip_eps,
        )

        # Marginal IPCW coverage across all test samples
        #   selected   -> threshold = c_model
        #   unselected -> threshold = LPB_j
        val_sel_threshold = np.full(sel_idx.size, favorable_thresh_norm, dtype=float)
        mgn_cov = marginal_coverage_tte(
            df_test=df_test,
            sel_idx=sel_idx,
            unsel_idx=unsel_idx,
            hat_LPB=hat_LPB_unsel,
            val_sel_threshold=val_sel_threshold,
            censor_model=censor_model,
            covar_cols=tuple(covar_cols),
            clip_eps=clip_eps,
        )

        # Set sizes (model-domain sizes scaled to years by `scaling_factor`)
        mgn_sz, unsel_sz = mean_set_sizes(
            m=m,
            sel_idx=sel_idx,
            unsel_idx=unsel_idx,
            horizon_c=favorable_thresh_norm,
            lpb_unsel=hat_LPB_unsel,
            study_end_time_year=study_end_time_year,
        )

        rows.append(
            {
                "selected_coverage_ipcw": float(selected_cov),
                "mgn_cov_ipcw": float(mgn_cov),
                "mgn_size": float(mgn_sz),
                "unselected_coverage_ipcw": float(unselected_cov),
                "unselected_set_size": float(unsel_sz),
                "num_unsel": int(unsel_idx.size),
                "num_total": int(m),
            }
        )

    stratcp_df = pd.DataFrame(rows, index=alphas)[base_cols]
    return stratcp_df