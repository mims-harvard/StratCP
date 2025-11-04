from collections.abc import Iterable
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def evaluate_top1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute baseline Top-1 metrics for binary classification.

    Computes marginal accuracy and class-conditional coverage for Top-1
    (singleton) predictions when the predicted class is 0 or 1.

    Args:
        preds: Predicted binary labels of shape (n_samples,).
        labels: Ground-truth binary labels of shape (n_samples,).

    Returns:
        A dictionary with:
            mgn_cov: Overall accuracy (float).
            mgn_size: Average set size (always 1.0 for Top-1).
            cls_cond_cov_cls_one: P(y=1 | pred=1) among predicted class-1 samples.
            cls_cond_cov_cls_zero: P(y=0 | pred=0) among predicted class-0 samples.
            num_sel_for_cls_one: Number of samples with pred=1 (int).
            num_sel_for_cls_zero: Number of samples with pred=0 (int).
            unselected_coverage: NaN (not applicable to Top-1).
            unselected_set_size: NaN (not applicable to Top-1).
            num_unsel: NaN (not applicable to Top-1).

    Notes:
        - When no samples are predicted as a class, the corresponding
          class-conditional coverage defaults to 1.0.
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
        cls_cond_cov_cls_one=cls_cond_cov_one,
        cls_cond_cov_cls_zero=cls_cond_cov_zero,
        num_sel_for_cls_one=n_one,
        num_sel_for_cls_zero=n_zero,
        unselected_coverage=np.nan,
        unselected_set_size=np.nan,
        num_unsel=np.nan,
    )


def evaluate_naive_cumulative(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> Dict[str, float]:
    """Compute naive cumulative prediction sets and associated metrics.

    For each sample, classes are sorted by descending probability and included
    until the cumulative probability exceeds ``1 - alpha``. Metrics include
    marginal coverage/size, class-conditional coverage for singleton sets, and
    averages for the "unselected" group (non-singleton sets or singleton with
    predicted class ≠ 1).

    Args:
        probs: Class probability matrix of shape (n_samples, n_classes).
        labels: Ground-truth class indices of shape (n_samples,).
        alpha: Miscoverage level in [0, 1]; target coverage ≈ ``1 - alpha``.

    Returns:
        A dictionary with:
            mgn_cov: Mean per-sample coverage of the constructed sets.
            mgn_size: Mean prediction-set size.
            cls_cond_cov_cls_one: P(y=1 | singleton pred=1).
            cls_cond_cov_cls_zero: P(y=0 | singleton pred=0).
            num_sel_for_cls_one: Number of singleton predictions equal to class 1.
            num_sel_for_cls_zero: Number of singleton predictions equal to class 0.
            unselected_coverage: Mean coverage within the unselected group, or NaN.
            unselected_set_size: Mean size within the unselected group, or NaN.
            num_unsel: Count of samples in the unselected group (int).

    Notes:
        - If there are no singleton predictions, class-conditional coverages
          default to 1.0 and counts are 0.
    """
    m, _ = probs.shape

    # Sort probabilities per sample in descending order and compute cumulative sums.
    val_pi = probs.argsort(axis=1)[:, ::-1]
    val_cum = np.take_along_axis(probs, val_pi, axis=1).cumsum(axis=1)

    # Build 0/1 prediction-set matrix based on cumulative threshold crossing.
    naive_set = np.zeros_like(probs, dtype=np.uint8)
    for i in range(m):
        # First index where cumulative probability exceeds 1 - alpha.
        k = int(np.searchsorted(val_cum[i], 1.0 - alpha))
        naive_set[i, val_pi[i, : k + 1]] = 1

    # Per-sample coverage (whether true label is in set) and set size.
    row_cov = naive_set[np.arange(m), labels].astype(float)
    row_size = naive_set.sum(axis=1).astype(float)

    # Marginal metrics.
    mgn_cov = float(row_cov.mean())
    mgn_size = float(row_size.mean())

    # Conditional metrics restricted to singleton prediction sets.
    singleton_mask = row_size == 1
    preds_single = np.argmax(naive_set[singleton_mask], axis=1) if np.any(singleton_mask) else np.array([])
    labels_single = labels[singleton_mask] if np.any(singleton_mask) else np.array([])

    if preds_single.size > 0:
        # Coverage when the singleton predicted class is 1
        mask_one = preds_single == 1
        n_one = int(mask_one.sum())
        cls_cond_cov_one = float(np.mean(labels_single[mask_one] == 1)) if n_one > 0 else 1.0

        # Coverage when the singleton predicted class is 0
        mask_zero = preds_single == 0
        n_zero = int(mask_zero.sum())
        cls_cond_cov_zero = float(np.mean(labels_single[mask_zero] == 0)) if n_zero > 0 else 1.0
    else:
        n_one = 0
        n_zero = 0
        cls_cond_cov_one = 1.0
        cls_cond_cov_zero = 1.0

    # Treat non-singleton sets (or singleton ≠ 1) as "unselected".
    not_singleton_mask = (row_size != 1) | (np.argmax(naive_set, axis=1) != 1)
    unselected_cov = row_cov[not_singleton_mask]
    unselected_size = row_size[not_singleton_mask]

    if unselected_cov.size > 0:
        unselected_coverage = float(unselected_cov.mean())
        unselected_set_size = float(unselected_size.mean())
    else:
        unselected_coverage = np.nan
        unselected_set_size = np.nan

    # Aggregate metrics.
    return dict(
        mgn_cov=mgn_cov,
        mgn_size=mgn_size,
        cls_cond_cov_cls_one=cls_cond_cov_one,
        cls_cond_cov_cls_zero=cls_cond_cov_zero,
        num_sel_for_cls_one=n_one,
        num_sel_for_cls_zero=n_zero,
        unselected_coverage=unselected_coverage,
        unselected_set_size=unselected_set_size,
        num_unsel=int(not_singleton_mask.sum()),
    )


def stratified_split_return_case_ids(
    data: pd.DataFrame,
    test_ratio: float,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create stratified case-level calibration/test splits.

    De-duplicates by ``case_id`` to ensure one label per case and performs a
    stratified split of case IDs into test and calibration partitions.

    Args:
        data: DataFrame containing at least the columns ``'case_id'`` and ``'label'``.
        test_ratio: Proportion of cases to assign to the **test** split in (0, 1].
        random_state: Seed for reproducibility in the stratified split.

    Returns:
        A 4-tuple:
            test_cases: Case IDs assigned to the test split.
            calib_cases: Case IDs assigned to the calibration split.
            test_labels: Labels for ``test_cases`` (aligned with the order of test_cases).
            calib_labels: Labels for ``calib_cases`` (aligned with the order of calib_cases).

    Raises:
        ValueError: If ``test_ratio`` is outside (0, 1].
    """
    # Drop duplicates so each case appears once with its label.
    unique_cases = data[["case_id", "label"]].drop_duplicates()
    cases = unique_cases["case_id"]
    labels = unique_cases["label"]

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

    Expects a nested mapping
    ``{split_id: {group: {method_name: DataFrame}}}`` where each DataFrame is
    indexed by alpha (α) and contains metric columns. Aggregates across the
    selected splits using a mean or median at each α. When ``method='mean'``,
    also returns a parallel structure of standard errors (SE) computed as
    sample std / sqrt(n_splits) with ``ddof=1``.

    Args:
        split_to_conformal_results: Nested results per split in the form
            ``{split_id: {group: {method_name: DataFrame}}}``.
        method: Aggregation statistic across splits. One of ``'mean'`` or ``'median'``.
        splits_to_include: Optional subset of split IDs to aggregate. If ``None``,
            all keys of ``split_to_conformal_results`` are used.
        alpha_range: Optional ``(min_alpha, max_alpha)`` to filter rows by α before
            aggregation (inclusive on both ends).

    Returns:
        A 2-tuple ``(agg_dict, se_dict)`` where:
            agg_dict: Same nesting as input but with one DataFrame per
                (group, method_name) holding the aggregated statistic across splits.
            se_dict: Same nesting with standard-error DataFrames when
                ``method='mean'``; otherwise ``None``.

    Raises:
        ValueError: If ``method`` is not one of ``'mean'`` or ``'median'``.
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
        obj: A pandas Series (single metric over α) or DataFrame (multi-metric).
        default_metric: Column name to use if ``obj`` is a Series.

    Returns:
        A DataFrame view of the input, with a single column named ``default_metric``
        when the input is a Series.

    Raises:
        TypeError: If ``obj`` is neither a Series nor a DataFrame.
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
        df: DataFrame whose index consists of alpha values (floats).
        alpha: Target alpha value.
        nearest: If True, select the nearest alpha within ``atol`` when
            an exact match is not found.
        atol: Absolute tolerance used when ``nearest=True``.

    Returns:
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
    methods: Iterable[str] | None = None,  # If None, infer per-source.
    include_se: bool = True,
    nearest: bool = True,
    atol: float = 5e-3,
) -> pd.DataFrame:
    """Summarize specified metrics at a fixed alpha for each (source, method).

    Expects input dictionaries with the following structure:
    - ``aggr_results``: {method_name: {metric_name: Series/DataFrame indexed by alpha}}
    - ``se_results``:   {method_name: {metric_name: Series/DataFrame indexed by alpha}}

    Args:
        summary_sources: Iterable of (source_label, aggr_results, se_results) tuples.
        alpha: Target alpha at which to extract metrics.
        metrics: Iterable of metric names to extract (e.g., 'mgn_cov', 'mgn_size', ...).
        methods: Optional subset of methods to include. If None, methods are
            inferred independently for each source from its ``aggr_results``.
        include_se: If True, append columns with suffix ``_se`` when available.
        nearest: If True, select the nearest alpha within ``atol`` if exact alpha
            is not present in an index.
        atol: Absolute tolerance used when ``nearest=True``.

    Returns:
        A tidy DataFrame with one row per (source, method), containing:
            - identifier columns: 'source', 'method', 'alpha_requested', 'alpha_selected'
            - one column per requested metric
            - optional ``<metric>_se`` columns when ``include_se`` and data available.

    Notes:
        - Rows are only included when at least one requested metric was found
          for the (source, method) pair.
        - If 'num_total' is not present, it is derived as
          (num_sel_cls_one + num_sel_cls_zero + num_unsel) when those components exist.
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
