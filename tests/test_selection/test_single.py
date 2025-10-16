"""
Unit tests for single selection with FDR control.
"""

import numpy as np
import pytest

from stratcp.selection.single import get_reference_sel_single, get_sel_single


class TestGetSelSingle:
    """Tests for get_sel_single function."""

    def test_basic_selection(self):
        """Test basic selection functionality."""
        np.random.seed(42)
        n, m = 100, 50

        # Generate synthetic data
        cal_scores = np.random.rand(n)
        cal_eligs = np.ones(n)
        cal_labels = np.random.binomial(1, 0.7, n)

        val_scores = np.random.rand(m)
        val_eligs = np.ones(m)

        alpha = 0.1

        sel_idx, unsel_idx, hat_tau = get_sel_single(
            cal_scores, cal_eligs, cal_labels, val_scores, val_eligs, alpha
        )

        # Basic sanity checks
        assert len(sel_idx) + len(unsel_idx) == m
        assert len(np.intersect1d(sel_idx, unsel_idx)) == 0
        assert 0 <= hat_tau <= 1

    def test_no_selection_strict_alpha(self):
        """Test that very strict alpha leads to no selection."""
        np.random.seed(42)
        n, m = 100, 50

        cal_scores = np.random.rand(n)
        cal_eligs = np.ones(n)
        cal_labels = np.zeros(n)  # All failures

        val_scores = np.random.rand(m)
        val_eligs = np.ones(m)

        alpha = 0.01  # Very strict

        sel_idx, unsel_idx, hat_tau = get_sel_single(
            cal_scores, cal_eligs, cal_labels, val_scores, val_eligs, alpha
        )

        # With all failures and strict alpha, should select nothing
        assert len(sel_idx) == 0
        assert len(unsel_idx) == m

    def test_eligibility_filtering(self):
        """Test that only eligible samples can be selected."""
        np.random.seed(42)
        n, m = 100, 50

        cal_scores = np.random.rand(n)
        cal_eligs = np.ones(n)
        cal_labels = np.ones(n)  # All successes

        val_scores = np.random.rand(m)
        val_eligs = np.zeros(m)  # No one eligible

        alpha = 0.1

        sel_idx, unsel_idx, hat_tau = get_sel_single(
            cal_scores, cal_eligs, cal_labels, val_scores, val_eligs, alpha
        )

        # No one should be selected since no one is eligible
        assert len(sel_idx) == 0
        assert len(unsel_idx) == m

    def test_output_types(self):
        """Test that outputs are of correct types."""
        np.random.seed(42)
        n, m = 50, 30

        cal_scores = np.random.rand(n)
        cal_eligs = np.ones(n)
        cal_labels = np.random.binomial(1, 0.5, n)

        val_scores = np.random.rand(m)
        val_eligs = np.ones(m)

        alpha = 0.1

        sel_idx, unsel_idx, hat_tau = get_sel_single(
            cal_scores, cal_eligs, cal_labels, val_scores, val_eligs, alpha
        )

        assert isinstance(sel_idx, np.ndarray)
        assert isinstance(unsel_idx, np.ndarray)
        assert isinstance(hat_tau, (float, np.floating))


class TestGetReferenceSelSingle:
    """Tests for get_reference_sel_single function."""

    def test_basic_reference_computation(self):
        """Test basic reference set computation."""
        np.random.seed(42)
        n, m, nclass = 50, 30, 3

        cal_scores = np.random.rand(n)
        cal_eligs = np.ones(n)
        cal_labels = np.random.binomial(1, 0.5, n)

        val_scores = np.random.rand(m)
        val_eligs = np.ones(m)
        val_imputed_labels = np.random.binomial(1, 0.5, (m, nclass))

        # Get unselected samples
        sel_idx, unsel_idx, _ = get_sel_single(
            cal_scores, cal_eligs, cal_labels, val_scores, val_eligs, alpha=0.1
        )

        if len(unsel_idx) > 0:
            ref_mats = get_reference_sel_single(
                unsel_idx,
                cal_labels,
                cal_eligs,
                cal_scores,
                val_eligs,
                val_scores,
                val_imputed_labels,
                alpha=0.1,
            )

            # Check output structure
            assert len(ref_mats) == nclass
            assert all(mat.shape == (m, n) for mat in ref_mats)
            assert all(np.all((mat == 0) | (mat == 1)) for mat in ref_mats)

    def test_all_selected(self):
        """Test reference sets when all samples are selected."""
        np.random.seed(42)
        n, m, nclass = 50, 30, 3

        cal_scores = np.random.rand(n)
        cal_eligs = np.ones(n)
        cal_labels = np.ones(n)  # All successes

        val_scores = np.ones(m) * 0.9  # High scores
        val_eligs = np.ones(m)
        val_imputed_labels = np.random.binomial(1, 0.5, (m, nclass))

        # Force all selections
        sel_idx = np.arange(m)
        unsel_idx = np.array([])

        ref_mats = get_reference_sel_single(
            unsel_idx,
            cal_labels,
            cal_eligs,
            cal_scores,
            val_eligs,
            val_scores,
            val_imputed_labels,
            alpha=0.5,
        )

        # Should return all ones when no unselected samples
        assert len(ref_mats) == nclass
        assert all(np.all(mat == 1) for mat in ref_mats)
