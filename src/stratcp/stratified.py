"""
High-level API for stratified conformal prediction.

This module provides a simple, end-to-end interface for stratified conformal
prediction with FDR-controlled selection.
"""

import numpy as np
from typing import Literal, Optional

from stratcp.conformal import (
    compute_score_aps,
    compute_score_raps,
    compute_score_tps,
    compute_score_utility,
    conformal,
)
from stratcp.selection import (
    get_reference_sel_single,
    get_sel_single,
    get_reference_sel_multiple,
    get_sel_multiple,
)


class StratifiedCP:
    """
    End-to-end stratified conformal prediction for multi-class classification.

    This class provides a simple interface for:
    1. FDR-controlled selection of high-confidence predictions, separate for every class or together 
    2. JOMI conformal prediction with coverage for unselected samples

    Parameters
    ----------
    score_fn : {'raps', 'aps', 'tps', 'utility'}, default='raps'
        Nonconformity score function to use.
        Use 'utility' for utility-aware CP with similarity matrix.
    alpha_sel : float, default=0.1
        FDR level for selection (e.g., 0.1 for 10% FDR)
    alpha_cp : float, default=0.1
        Miscoverage level for conformal prediction (e.g., 0.1 for 90% coverage)
    eligibility : {'per_class', 'overall'}, default='overall'
        Eligibility criterion for selection
    nonempty : bool, default=True
        Whether to force prediction sets to be non-empty
    rand : bool, default=True
        Whether to randomize p-values for exact finite-sample coverage
    lam_reg : float, default=0.01, optional
        Regularization parameter for RAPS
    similarity_matrix : np.ndarray, optional
        Similarity matrix between classes (n_classes, n_classes).
        Required when score_fn='utility'. Higher values = more similar.
    utility_method : {'weighted', 'greedy'}, default='greedy'
        Method for utility-aware expansion (only used when score_fn='utility')

    Attributes
    ----------
    selected_indices_ : np.ndarray
        Indices of selected test samples (after calling predict)
    unselected_indices_ : np.ndarray
        Indices of unselected test samples (after calling predict)
    selection_threshold_ : float
        Selection threshold on scores
    prediction_sets_ : dict
        Dictionary with 'selected' and 'unselected' prediction sets
    coverage_ : dict
        Dictionary with 'selected' and 'unselected' coverage indicators
    set_sizes_ : dict
        Dictionary with 'selected' and 'unselected' set sizes

    Examples
    --------
    >>> from stratcp import StratifiedCP
    >>> import numpy as np
    >>>
    >>> # Fit on calibration data
    >>> scp = StratifiedCP(score_fn='raps', alpha_sel=0.1, alpha_cp=0.1)
    >>> scp.fit(cal_probs, cal_labels)
    >>>
    >>> # Predict on test data
    >>> results = scp.predict(test_probs, test_labels)
    >>>
    >>> # Access results
    >>> print(f"Selected: {len(results['selected_idx'])} samples")
    >>> print(f"Avg set size (selected): {results['set_sizes']['selected'].mean():.2f}")
    >>> print(f"Avg set size (unselected): {results['set_sizes']['unselected'].mean():.2f}")
    """

    def __init__(
        self,
        score_fn: Literal["aps", "tps", "raps", "utility"] = "aps",
        alpha_sel: float = 0.1,
        alpha_cp: float = 0.1,
        eligibility: Literal["per_class", "overall"] = "overall",
        nonempty: bool = True,
        rand: bool = True,
        lam_reg: float = 0.01,
        similarity_matrix: Optional[np.ndarray] = None,
        utility_method: Literal["weighted", "greedy"] = "greedy",
    ):
        self.score_fn = score_fn
        self.alpha_sel = alpha_sel
        self.alpha_cp = alpha_cp
        self.eligibility = eligibility
        self.nonempty = nonempty
        self.rand = rand
        self.lam_reg = lam_reg
        self.similarity_matrix = similarity_matrix
        self.utility_method = utility_method

        # Validate utility-aware requirements
        if score_fn == "utility" and similarity_matrix is None:
            raise ValueError("similarity_matrix must be provided when score_fn='utility'")

        # Attributes set after fit
        self.cal_probs_ = None
        self.cal_labels_ = None
        self.cal_scores_ = None
        self.n_classes_ = None

        # Attributes set after predict
        self.selected_indices_ = None  # For overall mode
        self.unselected_indices_ = None  # For overall mode
        self.all_selected_ = None  # For per_class mode (list of K boolean arrays)
        self.selection_threshold_ = None  # For overall mode (float)
        self.selection_thresholds_ = None  # For per_class mode (array of K thresholds)
        self.prediction_sets_ = None
        self.coverage_ = None
        self.set_sizes_ = None

    def fit(self, cal_probs: np.ndarray, cal_labels: np.ndarray,
            cal_elig: Optional[np.ndarray] = None,
            cal_confs: Optional[np.ndarray] = None) -> "StratifiedCP":
        """
        Fit the stratified CP model on calibration data.

        Parameters
        ----------
        cal_probs : np.ndarray
            Predicted class probabilities for calibration data (n, n_classes)
        cal_labels : np.ndarray
            True labels for calibration data (n,)

        Returns
        -------
        self : StratifiedCP
            Fitted estimator
        """
        self.cal_probs_ = cal_probs
        self.cal_labels_ = cal_labels
        self.n_classes_ = cal_probs.shape[1]

        if cal_elig is not None:
            self.cal_elig = cal_elig
        if cal_confs is not None:
            self.cal_confs = cal_confs

        # Compute nonconformity scores
        if self.score_fn == "raps":
            self.cal_scores_, _ = compute_score_raps(
                cal_probs, cal_probs[:1], cal_labels, lam_reg=self.lam_reg, nonempty=self.nonempty
            )
        elif self.score_fn == "aps":
            self.cal_scores_, _ = compute_score_aps(
                cal_probs, cal_probs[:1], cal_labels, nonempty=self.nonempty
            )
        elif self.score_fn == "tps":
            self.cal_scores_, _ = compute_score_tps(
                cal_probs, cal_probs[:1], cal_labels, nonempty=self.nonempty
            )
        elif self.score_fn == "utility":
            self.cal_scores_, _ = compute_score_utility(
                cal_probs,
                None,
                cal_labels, 
                self.similarity_matrix,
                method=self.utility_method,
                nonempty=self.nonempty,
            )
        else:
            raise ValueError(f"Unknown score function: {self.score_fn}")

        return self

    def predict(
        self, 
        test_probs: np.ndarray, 
        test_labels: Optional[np.ndarray] = None, 
        test_eligs: Optional[np.ndarray] = None, 
        eligibility: Optional[str] = None,
        cal_eligs: Optional[np.ndarray] = None,  # optionally update calibration eligs
        cal_conf_labels: Optional[np.ndarray] = None  # optionally update calibration confidence labels
    ) -> dict:
        """
        Make stratified conformal predictions on test data.

        This method performs:
        1. FDR-controlled selection based on prediction confidence
        2. JOMI CP for unselected samples

        Parameters
        ----------
        test_probs : np.ndarray
            Predicted class probabilities for test data (m, n_classes)
        test_labels : np.ndarray, optional
            True labels for test data (m,). Required for computing coverage.
        test_eligs : np.ndarray, optional
            Eligibility indicators for test data.
            - If eligibility='overall': (m,) array or None (default: all ones)
            - If eligibility='per_class': (m, n_classes) array or None (default: based on argmax)

        Returns
        -------
        results : dict
            Dictionary containing:
            For eligibility='overall':
            - 'selected_idx': Selected sample indices
            - 'unselected_idx': Unselected sample indices
            - 'threshold': Selection threshold
            - 'prediction_sets': Dict with 'selected' and 'unselected' prediction sets
            - 'coverage': Dict with 'selected' and 'unselected' coverage (if labels provided)
            - 'set_sizes': Dict with 'selected' and 'unselected' set sizes

            For eligibility='per_class':
            - 'all_selected': List of K boolean arrays (one per class)
            - 'thresholds': Array of K thresholds
            - 'prediction_sets': Array of prediction sets for unselected samples
            - 'coverage': Coverage for unselected samples (if labels provided)
            - 'set_sizes': Set sizes for unselected samples

        Examples
        --------
        >>> scp = StratifiedCP(eligibility='overall')
        >>> scp.fit(cal_probs, cal_labels)
        >>> results = scp.predict(test_probs, test_labels)
        >>> print(f"Selected {len(results['selected_idx'])} high-confidence predictions")
        """
        if self.cal_probs_ is None:
            raise ValueError("Model not fitted. Call fit() before predict().")
        
        if eligibility is not None:
            self.eligibility = eligibility

        if self.eligibility == "overall":
            return self._predict_overall(test_probs, test_labels, test_eligs, cal_eligs, cal_conf_labels)
        else:  # per_class
            return self._predict_per_class(test_probs, test_labels, test_eligs, cal_eligs, cal_conf_labels)

    def _predict_overall(
        self, test_probs: np.ndarray, 
        test_labels: Optional[np.ndarray] = None, 
        test_eligs: Optional[np.ndarray] = None,
        cal_eligs: Optional[np.ndarray] = None, # optionally update cal eligibility
        cal_conf_labels: Optional[np.ndarray] = None # optionally update cal conf labels
    ) -> dict:
        """Predict with overall eligibility (original behavior)."""
        m = test_probs.shape[0]

        # Default: all samples eligible
        if test_eligs is None:
            test_eligs = np.ones(m)
        
        cal_eligs = np.ones(len(self.cal_labels_)) if cal_eligs is None else cal_eligs

        # Use dummy labels if not provided (for prediction only)
        if test_labels is None:
            test_labels = np.zeros(m, dtype=int) 

        # Step 1: Compute selection scores (max probability = confidence)
        cal_sel_scores = np.max(self.cal_probs_, axis=1)
        test_sel_scores = np.max(test_probs, axis=1)

        # Selection labels: whether prediction is correct
        cal_conf_labels = (self.cal_labels_ == np.argmax(self.cal_probs_, axis=1)).astype(int) if cal_conf_labels is None else cal_conf_labels

        # Step 2: FDR-controlled selection
        sel_idx, unsel_idx, tau = get_sel_single(
            cal_sel_scores, cal_conf_labels, test_sel_scores, self.alpha_sel, cal_eligs, test_eligs 
        )

        # Step 3: Compute nonconformity scores for test data
        if self.score_fn == "raps":
            _, test_scores = compute_score_raps(
                self.cal_probs_, test_probs, self.cal_labels_,  lam_reg=self.lam_reg, nonempty=self.nonempty
            )
        elif self.score_fn == "aps":
            _, test_scores = compute_score_aps(
                self.cal_probs_, test_probs, self.cal_labels_,  nonempty=self.nonempty
            )
        elif self.score_fn == "tps":
            _, test_scores = compute_score_tps(
                self.cal_probs_, test_probs, self.cal_labels_,  nonempty=self.nonempty
            )
        elif self.score_fn == "utility":
            _, test_scores = compute_score_utility(
                self.cal_probs_,
                test_probs,
                self.cal_labels_, 
                self.similarity_matrix,
                method=self.utility_method,
                nonempty=self.nonempty,
            )

        # Initialize result containers 
        pred_sets_unsel = np.zeros((0, self.n_classes_), dtype=bool)  

        # Step 4: JOMI conformal prediction for unselected samples
        if len(unsel_idx) > 0:
            ref_mats = [np.ones((m, self.cal_probs_.shape[0])) for _ in range(self.n_classes_)]

            if len(sel_idx) > 0: # unsel_idx != [m]
    
                # Compute reference sets
                test_imputed_labels = np.zeros(test_probs.shape)  
                # Impute test samples Conf(X, y) for every y class
                for k in range(self.n_classes_):
                    test_imputed_labels[:, k] = (np.argmax(test_probs, axis=1)==k).astype(int)

                ref_mats = get_reference_sel_single(
                    unsel_idx,
                    cal_conf_labels,
                    cal_sel_scores,
                    test_sel_scores,
                    test_imputed_labels,
                    self.alpha_sel,
                    cal_eligs,
                    test_eligs,
                )

            for k in range(self.n_classes_):
                ref_mats[k] = ref_mats[k][unsel_idx, :]

            # JOMI conformal prediction
            pred_sets_unsel = conformal(
                self.cal_scores_,
                test_scores[unsel_idx],
                self.cal_labels_, 
                alpha=self.alpha_cp,
                nonempty=self.nonempty,
                if_in_ref=ref_mats,
                test_max_id=np.argmax(test_probs[unsel_idx], axis=1),
                rand=self.rand,
            )

            # Store attributes
            self.selected_indices_ = sel_idx
            self.unselected_indices_ = unsel_idx
            self.selection_threshold_ = tau
            self.prediction_sets_unsel_ = pred_sets_unsel
            self.set_sizes_ = np.sum(pred_sets_unsel, axis=1)

        else:
            # No unselected samples
            self.selected_indices_ = sel_idx
            self.unselected_indices_ = unsel_idx
            self.selection_threshold_ = tau
            self.prediction_sets_unsel_ = np.zeros((0, self.n_classes_), dtype=bool) 
            self.set_sizes_ = np.array([])
 

        # Build results dictionary
        results = {
            "selected_idx": sel_idx,
            "unselected_idx": unsel_idx,
            "threshold": tau,
            "prediction_sets": self.prediction_sets_unsel_,
            "set_sizes": self.set_sizes_,
        } 

        return results

    def _predict_per_class(
        self, test_probs: np.ndarray, test_labels: Optional[np.ndarray] = None, test_eligs: Optional[np.ndarray] = None, # optionally specified test eligibility
        cal_eligs: Optional[np.ndarray] = None, # optionally update cal eligibility
        cal_conf_labels: Optional[np.ndarray] = None # optionally update cal conf labels
    ) -> dict:
        """Predict with per-class eligibility."""
        m = test_probs.shape[0]
        n = len(self.cal_labels_)

        # Use dummy labels if not provided
        if test_labels is None:
            test_labels = np.zeros(m, dtype=int) 

        # Step 1: Setup eligibility matrices
        # Calibration eligibility: eligible if argmax == k if not provided
        cal_argmax = np.argmax(self.cal_probs_, axis=1)
        if cal_eligs is None: 
            cal_eligs = np.zeros((n, self.n_classes_))
            for k in range(self.n_classes_):
                cal_eligs[:, k] = (cal_argmax == k).astype(int)

        # Calibration labels: correct if true label == k == argmax 
        if cal_conf_labels is None:
            cal_conf_labels = np.zeros((n, self.n_classes_)) 
            for k in range(self.n_classes_):
                cal_conf_labels[:, k] = ( self.cal_labels_ == k ).astype(int)

        # Test eligibility
        test_argmax = np.argmax(test_probs, axis=1)
        if test_eligs is None:
            # Default: eligible if argmax == k
            test_eligs = np.zeros((m, self.n_classes_))
            for k in range(self.n_classes_):
                test_eligs[:, k] = (test_argmax == k).astype(int)

        # Step 2: FDR-controlled selection per class
        all_sel, tau_list = get_sel_multiple(
            self.cal_probs_, cal_eligs, cal_conf_labels, test_probs, test_eligs, self.alpha_sel
        )

        # Step 3: Compute nonconformity scores for test data
        if self.score_fn == "raps":
            _, test_scores = compute_score_raps(
                self.cal_probs_, test_probs, self.cal_labels_, lam_reg=self.lam_reg, nonempty=self.nonempty
            )
        elif self.score_fn == "aps":
            _, test_scores = compute_score_aps(
                self.cal_probs_, test_probs, self.cal_labels_, nonempty=self.nonempty
            )
        elif self.score_fn == "tps":
            _, test_scores = compute_score_tps(
                self.cal_probs_, test_probs, self.cal_labels_, nonempty=self.nonempty
            )
        elif self.score_fn == "utility":
            _, test_scores = compute_score_utility(
                self.cal_probs_,
                test_probs,
                self.cal_labels_, 
                self.similarity_matrix,
                method=self.utility_method,
                nonempty=self.nonempty,
            )

        # Step 4: JOMI conformal prediction for unselected samples (all_sel[K])
        # Unselected: samples not selected by any class
        unsel_idx = all_sel[self.n_classes_]          # indices
        unsel_mask = np.zeros(m, dtype=bool)
        if unsel_idx.size > 0:
            unsel_mask[unsel_idx] = True
        ref_mats = [np.ones((m,n)) for _ in range(self.n_classes_)]

        if unsel_mask.sum() > 0:
            if unsel_mask.sum() < m: 
                # Prepare imputed labels for reference set computation
                val_imputed_confs = []
                for k in range(self.n_classes_):
                    conf_k = np.zeros((m, self.n_classes_))
                    conf_k[:, k] = 1
                    val_imputed_confs.append(conf_k)

                # Compute reference sets
                ref_mats = get_reference_sel_multiple(
                    unsel_idx,
                    cal_conf_labels,
                    cal_eligs,
                    self.cal_probs_,
                    test_eligs,
                    test_probs,
                    val_imputed_confs,
                    self.alpha_sel,
                )
            
            for k in range(self.n_classes_):
                ref_mats[k] = ref_mats[k][unsel_mask,:]

            # JOMI conformal prediction for unselected
            pred_sets_unsel = conformal(
                self.cal_scores_,
                test_scores[unsel_mask],
                self.cal_labels_, 
                alpha=self.alpha_cp,
                nonempty=self.nonempty,
                if_in_ref=ref_mats,
                test_max_id=np.argmax(test_probs[unsel_mask], axis=1),
                rand=self.rand,
            )
            sizes_unsel = np.sum(pred_sets_unsel, axis = 1)
        else:
            pred_sets_unsel = np.zeros((0, self.n_classes_), dtype=bool) 
            sizes_unsel = np.array([])

        # Store attributes
        self.all_selected_ = all_sel
        self.selection_thresholds_ = tau_list
        self.prediction_sets_ = pred_sets_unsel 
        self.set_sizes_ = sizes_unsel 
 

        # Build results dictionary
        results = {
            "all_selected": all_sel,
            "thresholds": tau_list,
            "prediction_sets": pred_sets_unsel,
            "set_sizes": sizes_unsel,
        } 

        return results

    def fit_predict(
        self, cal_probs: np.ndarray, cal_labels: np.ndarray, test_probs: np.ndarray, 
        test_labels: Optional[np.ndarray] = None,
        test_eligs: Optional[np.ndarray] = None,
        cal_eligs: Optional[np.ndarray] = None, # optionally updated calibration eligibility
        cal_conf_labels: Optional[np.ndarray] = None #optionaly updated cal conf labels
    ) -> dict:
        """
        Fit and predict in one step.

        Parameters
        ----------
        cal_probs : np.ndarray
            Predicted class probabilities for calibration data (n, n_classes)
        cal_labels : np.ndarray
            True labels for calibration data (n,)
        test_probs : np.ndarray
            Predicted class probabilities for test data (m, n_classes)
        test_labels : np.ndarray, optional
            True labels for test data (m,)

        Returns
        -------
        results : dict
            Prediction results (same as predict method)

        Examples
        --------
        >>> scp = StratifiedCP(alpha_sel=0.1, alpha_cp=0.1)
        >>> results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)
        """
        return self.fit(cal_probs, cal_labels).predict(test_probs, test_labels, test_eligs, cal_eligs, cal_conf_labels)

    def summary(self) -> str:
        """
        Get a summary of the stratified CP results.

        Returns
        -------
        summary : str
            Human-readable summary of results
        """
        if self.eligibility == "overall":
            return self._summary_overall()
        else:  # per_class
            return self._summary_per_class()

    def _summary_overall(self) -> str:
        """Summary for overall eligibility mode."""
        if self.selected_indices_ is None:
            return "Model fitted but no predictions made yet. Call predict() first."

        n_sel = len(self.selected_indices_)
        n_unsel = len(self.unselected_indices_)
        n_total = n_sel + n_unsel

        summary_lines = [
            "=" * 70,
            "Stratified Conformal Prediction Summary (Overall Eligibility)",
            "=" * 70,
            f"Score function: {self.score_fn}",
            f"Selection FDR level (α_sel): {self.alpha_sel}",
            f"CP miscoverage level (α_cp): {self.alpha_cp}",
            "",
            "Selection Results:",
            f"  Total samples: {n_total}",
            f"  Selected (high confidence): {n_sel} ({n_sel/n_total:.1%})",
            f"  Unselected (low confidence): {n_unsel} ({n_unsel/n_total:.1%})",
            f"  Selection threshold: {self.selection_threshold_:.3f}",
            "",
        ]

        # Prediction set sizes
        if 'selected' in self.set_sizes_ and len(self.set_sizes_['selected']) > 0:
            summary_lines.extend([
                "Prediction Set Sizes:",
                f"  Selected - Mean: {self.set_sizes_['selected'].mean():.2f}, "
                f"Median: {np.median(self.set_sizes_['selected']):.0f}",
            ])

        if 'unselected' in self.set_sizes_ and len(self.set_sizes_['unselected']) > 0:
            summary_lines.append(
                f"  Unselected - Mean: {self.set_sizes_['unselected'].mean():.2f}, "
                f"Median: {np.median(self.set_sizes_['unselected']):.0f}"
            )
        elif 'unselected' not in self.set_sizes_:
            # Handle old format where set_sizes_ is just an array
            if isinstance(self.set_sizes_, np.ndarray) and len(self.set_sizes_) > 0:
                summary_lines.extend([
                    "Prediction Set Sizes (unselected):",
                    f"  Mean: {self.set_sizes_.mean():.2f}, "
                    f"Median: {np.median(self.set_sizes_):.0f}"
                ])

        # Coverage if available
        if self.coverage_ is not None:
            summary_lines.append("")
            summary_lines.append("Coverage (empirical):")
            if 'selected' in self.coverage_ and len(self.coverage_['selected']) > 0:
                summary_lines.append(
                    f"  Selected: {self.coverage_['selected'].mean():.2%}"
                )
            if 'unselected' in self.coverage_ and len(self.coverage_['unselected']) > 0:
                summary_lines.append(
                    f"  Unselected: {self.coverage_['unselected'].mean():.2%}"
                )
                summary_lines.append(
                    f"  Overall: {np.concatenate([self.coverage_['selected'], self.coverage_['unselected']]).mean():.2%}"
                )

        summary_lines.append("=" * 70)
        return "\n".join(summary_lines)

    def _summary_per_class(self) -> str:
        """Summary for per-class eligibility mode."""
        if self.all_selected_ is None:
            return "Model fitted but no predictions made yet. Call predict() first."

        n_total = 0
        for k in range(1+self.n_classes_):
            n_total += len(self.all_selected_[k])

        summary_lines = [
            "=" * 70,
            "Stratified Conformal Prediction Summary (Per-Class Eligibility)",
            "=" * 70,
            f"Score function: {self.score_fn}",
            f"Selection FDR level (α_sel): {self.alpha_sel}",
            f"CP miscoverage level (α_cp): {self.alpha_cp}",
            f"Number of classes: {self.n_classes_}",
            "",
            "Selection Results by Class:",
            f"  Total test samples: {n_total}",
            "",
        ]

        # Per-class selection statistics
        for k in range(self.n_classes_):
            n_sel_k = len(self.all_selected_[k])
            threshold_k = self.selection_thresholds_[k] if self.selection_thresholds_ is not None else None

            summary_lines.append(
                f"  Class {k}: {n_sel_k} selected ({n_sel_k/n_total:.1%})"
            )
            if threshold_k is not None:
                summary_lines.append(f"    Threshold: {threshold_k:.3f}")

        # Unselected samples
        n_unsel = len(self.all_selected_[self.n_classes_])
        summary_lines.extend([
            "",
            f"  Unselected (not selected by any class): {n_unsel} ({n_unsel/n_total:.1%})",
            "",
        ])

        # Prediction set sizes for unselected
        if 'unselected' in self.set_sizes_ and len(self.set_sizes_['unselected']) > 0:
            summary_lines.extend([
                "Prediction Set Sizes (unselected only):",
                f"  Mean: {self.set_sizes_.mean():.2f}, "
                f"Median: {np.median(self.set_sizes_):.0f}",
            ])
        elif isinstance(self.set_sizes_, np.ndarray) and len(self.set_sizes_) > 0:
            # Handle case where set_sizes_ is just an array
            summary_lines.extend([
                "Prediction Set Sizes (unselected only):",
                f"  Mean: {self.set_sizes_.mean():.2f}, "
                f"Median: {np.median(self.set_sizes_):.0f}",
            ])

        # Coverage for unselected if available
        if self.coverage_ is not None:
            summary_lines.append("")
            if 'unselected' in self.coverage_ and len(self.coverage_['unselected']) > 0:
                summary_lines.extend([
                    "Coverage (empirical, unselected only):",
                    f"  {self.coverage_['unselected'].mean():.2%}",
                ])
            elif isinstance(self.coverage_, np.ndarray) and len(self.coverage_) > 0:
                summary_lines.extend([
                    "Coverage (empirical, unselected only):",
                    f"  {self.coverage_.mean():.2%}",
                ])

        summary_lines.append("=" * 70)
        return "\n".join(summary_lines)
