import numpy as np
from typing import Dict

def format_rich(value: str, markup: str) -> str:
    """Format string with rich markup.

    Args:
        value: The string to format.
        markup: The rich markup to apply.

    Returns:
        The formatted string.
    """
    return f"[{markup}]{value}[/{markup}]"


def evaluate_top1(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute baseline Top-1 (argmax) metrics for binary classification.

    Args:
        preds (np.ndarray): Predicted class indices with shape (m,), values in {0, 1}.
        labels (np.ndarray): True class indices with shape (m,), values in {0, 1}.

    Returns:
        Dict[str, float]: Dictionary with:
            - mgn_cov: Marginal coverage (Top-1 accuracy).
            - mgn_size: Average set size (always 1.0 for Top-1).
            - cls_cond_cov_cls_one: Coverage among predictions assigned to class 1.
            - cls_cond_cov_cls_zero: Coverage among predictions assigned to class 0.
            - num_sel_for_cls_one: Number of predictions assigned to class 1.
            - num_sel_for_cls_zero: Number of predictions assigned to class 0.
            - unselected_coverage: Not applicable for Top-1 (NaN).
            - unselected_set_size: Not applicable for Top-1 (NaN).
            - num_unsel: Not applicable for Top-1 (NaN).
    """
    # Marginal metrics
    mgn_cov = float(np.mean(preds == labels))  # accuracy
    mgn_size = 1.0                             # singleton set

    # Conditional metrics for class = 1
    mask_one = preds == 1
    n_one = int(mask_one.sum())
    if n_one > 0:
        cls_cond_cov_one = float(np.mean(labels[mask_one] == 1))
    else:
        cls_cond_cov_one = 1.0  # convention when there are no predictions for the class

    # Conditional metrics for class = 0
    mask_zero = preds == 0
    n_zero = int(mask_zero.sum())
    if n_zero > 0:
        cls_cond_cov_zero = float(np.mean(labels[mask_zero] == 0))
    else:
        cls_cond_cov_zero = 1.0

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


def evaluate_naive_cumulative(probs: np.ndarray, labels: np.ndarray, alpha: float) -> Dict[str, float]:
    """Build naive cumulative prediction sets and compute metrics.

    For each sample i, form S_i = {top-k labels} such that sum_{j<=k} p_i(j) >= 1 - alpha.

    Args:
        probs (np.ndarray): Class probabilities with shape (m, n_class).
        labels (np.ndarray): True class indices with shape (m,), values in {0, 1}.
        alpha (float): Miscoverage target in [0, 1].

    Returns:
        Dict[str, float]: Dictionary with:
            - mgn_cov: Marginal coverage (mean 1{y in S}).
            - mgn_size: Average set size |S|.
            - cls_cond_cov_cls_one: Coverage among singleton predictions with class 1.
            - cls_cond_cov_cls_zero: Coverage among singleton predictions with class 0.
            - num_sel_for_cls_one: Count of singleton predictions with class 1.
            - num_sel_for_cls_zero: Count of singleton predictions with class 0.
            - unselected_coverage: Coverage among non-singleton (or singleton not class-1) predictions.
            - unselected_set_size: Average set size among the same unselected subset.
            - num_unsel: Number of unselected samples in that subset.
    """
    m, _ = probs.shape

    # Sort probs descending per-row and take cumulative sums
    val_pi = probs.argsort(axis=1)[:, ::-1]                         # sort order (desc)
    val_cum = np.take_along_axis(probs, val_pi, axis=1).cumsum(1)   # cumulative sums

    # Build prediction set matrix (m, n_class) as 0/1
    naive_set = np.zeros_like(probs, dtype=np.uint8)
    for i in range(m):
        # Smallest k with cumulative >= 1 - alpha
        k = int(np.searchsorted(val_cum[i], 1.0 - alpha))
        naive_set[i, val_pi[i, : k + 1]] = 1

    # Per-sample coverage and size
    row_cov = naive_set[np.arange(m), labels].astype(float)  # 1 if true label in S_i
    row_size = naive_set.sum(axis=1).astype(float)           # |S_i|

    # Marginal metrics
    mgn_cov = float(row_cov.mean())
    mgn_size = float(row_size.mean())

    # Conditional-on-singleton metrics
    singleton_mask = row_size == 1
    preds_single = np.argmax(naive_set[singleton_mask], axis=1) if np.any(singleton_mask) else np.array([])
    labels_single = labels[singleton_mask] if np.any(singleton_mask) else np.array([])

    # Class = 1
    if preds_single.size > 0:
        mask_one = preds_single == 1
        n_one = int(mask_one.sum())
        cls_cond_cov_one = float(np.mean(labels_single[mask_one] == 1)) if n_one > 0 else 1.0
    else:
        n_one = 0
        cls_cond_cov_one = 1.0

    # Class = 0
    if preds_single.size > 0:
        mask_zero = preds_single == 0
        n_zero = int(mask_zero.sum())
        cls_cond_cov_zero = float(np.mean(labels_single[mask_zero] == 0)) if n_zero > 0 else 1.0
    else:
        n_zero = 0
        cls_cond_cov_zero = 1.0

    # Unselected subset: non-singleton or singleton with predicted label != 1
    not_singleton_mask = (row_size != 1) | (np.argmax(naive_set, axis=1) != 1)
    unselected_cov = row_cov[not_singleton_mask]
    unselected_size = row_size[not_singleton_mask]

    if unselected_cov.size > 0:
        unselected_coverage = float(unselected_cov.mean())
        unselected_set_size = float(unselected_size.mean())
    else:
        unselected_coverage = np.nan
        unselected_set_size = np.nan

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