"""
Unit tests for the high-level StratifiedCP API.
"""

import numpy as np
import pytest

from stratcp import StratifiedCP


class TestStratifiedCP:
    """Tests for StratifiedCP class."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict workflow."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 200, 100, 5

        # Generate synthetic data
        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.array([np.random.choice(n_classes, p=p / p.sum()) for p in cal_probs])

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.array([np.random.choice(n_classes, p=p / p.sum()) for p in test_probs])

        # Fit and predict
        scp = StratifiedCP(alpha_sel=0.1, alpha_cp=0.1)
        results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        # Check result structure
        assert "selected_idx" in results
        assert "unselected_idx" in results
        assert "threshold" in results
        assert "prediction_sets" in results
        assert "coverage" in results
        assert "set_sizes" in results

        # Check basic properties
        assert len(results["selected_idx"]) + len(results["unselected_idx"]) == n_test
        assert 0 <= results["threshold"] <= 1

    def test_separate_fit_predict(self):
        """Test separate fit() and predict() calls."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 100, 50, 3

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        # Fit
        scp = StratifiedCP()
        scp.fit(cal_probs, cal_labels)

        # Predict
        results = scp.predict(test_probs, test_labels)

        assert len(results["selected_idx"]) + len(results["unselected_idx"]) == n_test

    def test_predict_without_labels(self):
        """Test prediction without providing test labels."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 100, 50, 3

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)

        # Fit and predict without labels
        scp = StratifiedCP()
        scp.fit(cal_probs, cal_labels)
        results = scp.predict(test_probs)  # No labels

        # Should still work but no coverage
        assert "coverage" not in results
        assert "prediction_sets" in results
        assert "set_sizes" in results

    def test_different_score_functions(self):
        """Test different score functions."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 100, 50, 5

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        for score_fn in ["tps", "aps", "raps"]:
            scp = StratifiedCP(score_fn=score_fn)
            results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

            assert len(results["selected_idx"]) + len(results["unselected_idx"]) == n_test

    def test_attributes_set_correctly(self):
        """Test that attributes are set correctly after fit/predict."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 100, 50, 3

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        scp = StratifiedCP()

        # Before fit
        assert scp.cal_probs_ is None
        assert scp.selected_indices_ is None

        # After fit
        scp.fit(cal_probs, cal_labels)
        assert scp.cal_probs_ is not None
        assert scp.cal_labels_ is not None
        assert scp.n_classes_ == n_classes

        # After predict
        scp.predict(test_probs, test_labels)
        assert scp.selected_indices_ is not None
        assert scp.unselected_indices_ is not None
        assert scp.selection_threshold_ is not None
        assert scp.prediction_sets_ is not None

    def test_summary_method(self):
        """Test summary method."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 100, 50, 3

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        scp = StratifiedCP()
        scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        summary = scp.summary()

        # Check that summary contains expected information
        assert "Stratified Conformal Prediction" in summary
        assert "Selected" in summary
        assert "Unselected" in summary
        assert "Coverage" in summary
        assert isinstance(summary, str)

    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises an error."""
        np.random.seed(42)
        n_test, n_classes = 50, 3

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)

        scp = StratifiedCP()

        with pytest.raises(ValueError, match="Model not fitted"):
            scp.predict(test_probs)

    def test_coverage_guarantees(self):
        """Test that coverage is approximately at the desired level."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 500, 200, 5

        # Generate data with good predictions
        cal_probs = np.random.dirichlet(np.ones(n_classes) * 3, n_cal)
        cal_labels = np.argmax(cal_probs, axis=1)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 3, n_test)
        test_labels = np.argmax(test_probs, axis=1)

        alpha_cp = 0.1
        scp = StratifiedCP(alpha_cp=alpha_cp, alpha_sel=0.1)
        results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        # Check overall coverage is close to 1 - alpha_cp
        all_coverage = np.concatenate([results["coverage"]["selected"], results["coverage"]["unselected"]])

        # Coverage should be at least 1 - alpha (with some slack for finite samples)
        assert all_coverage.mean() >= 1 - alpha_cp - 0.1


class TestUtilityAwareCP:
    """Tests for utility-aware conformal prediction."""

    @pytest.fixture
    def similarity_matrix(self):
        """Generate a test similarity matrix."""
        # 5 classes with hierarchical structure
        return np.array([
            [1.0, 0.9, 0.3, 0.3, 0.1],
            [0.9, 1.0, 0.3, 0.3, 0.1],
            [0.3, 0.3, 1.0, 0.9, 0.1],
            [0.3, 0.3, 0.9, 1.0, 0.1],
            [0.1, 0.1, 0.1, 0.1, 1.0],
        ])

    def test_utility_aware_basic(self, similarity_matrix):
        """Test basic utility-aware CP."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 200, 100, 5

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        # Use utility-aware CP
        scp = StratifiedCP(score_fn="utility", similarity_matrix=similarity_matrix, alpha_sel=0.1, alpha_cp=0.1)
        results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        # Check result structure
        assert "selected_idx" in results
        assert "unselected_idx" in results
        assert "prediction_sets" in results
        assert len(results["selected_idx"]) + len(results["unselected_idx"]) == n_test

    def test_utility_methods(self, similarity_matrix):
        """Test both weighted and greedy methods."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 150, 75, 5

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        for method in ["weighted", "greedy"]:
            scp = StratifiedCP(
                score_fn="utility", similarity_matrix=similarity_matrix, utility_method=method, alpha_sel=0.1, alpha_cp=0.1
            )
            results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

            assert len(results["selected_idx"]) + len(results["unselected_idx"]) == n_test
            assert "prediction_sets" in results

    def test_utility_requires_similarity_matrix(self):
        """Test that utility score requires similarity matrix."""
        with pytest.raises(ValueError, match="similarity_matrix must be provided"):
            StratifiedCP(score_fn="utility", similarity_matrix=None)

    def test_utility_prediction_coherence(self, similarity_matrix):
        """Test that utility-aware CP produces coherent prediction sets."""
        from stratcp.conformal import eval_similarity

        np.random.seed(42)
        n_cal, n_test, n_classes = 200, 100, 5

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        # Standard RAPS
        scp_standard = StratifiedCP(score_fn="raps", alpha_sel=0.1, alpha_cp=0.1)
        results_standard = scp_standard.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        # Utility-aware
        scp_utility = StratifiedCP(
            score_fn="utility", similarity_matrix=similarity_matrix, utility_method="weighted", alpha_sel=0.1, alpha_cp=0.1
        )
        results_utility = scp_utility.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        # Evaluate coherence for unselected samples with size > 1
        if len(results_standard["unselected_idx"]) > 0:
            pred_sets_standard = results_standard["prediction_sets"]["unselected"]
            mask_standard = np.sum(pred_sets_standard, axis=1) > 1

            if mask_standard.sum() > 0:
                _, sim_standard = eval_similarity(pred_sets_standard[mask_standard], similarity_matrix)
            else:
                sim_standard = np.nan

        if len(results_utility["unselected_idx"]) > 0:
            pred_sets_utility = results_utility["prediction_sets"]["unselected"]
            mask_utility = np.sum(pred_sets_utility, axis=1) > 1

            if mask_utility.sum() > 0:
                _, sim_utility = eval_similarity(pred_sets_utility[mask_utility], similarity_matrix)
            else:
                sim_utility = np.nan

        # Both should produce valid similarity scores (or NaN if no multi-class sets)
        assert np.isnan(sim_standard) or (0 <= sim_standard <= 1)
        assert np.isnan(sim_utility) or (0 <= sim_utility <= 1)

    def test_utility_coverage_maintained(self, similarity_matrix):
        """Test that utility-aware CP maintains coverage guarantees."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 500, 200, 5

        # Generate data
        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        alpha_cp = 0.1
        scp = StratifiedCP(
            score_fn="utility", similarity_matrix=similarity_matrix, utility_method="weighted", alpha_sel=0.1, alpha_cp=alpha_cp
        )
        results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        # Check coverage
        all_coverage = np.concatenate([results["coverage"]["selected"], results["coverage"]["unselected"]])

        # Coverage should be at least 1 - alpha (with some slack)
        assert all_coverage.mean() >= 1 - alpha_cp - 0.15

    def test_utility_separate_fit_predict(self, similarity_matrix):
        """Test utility-aware CP with separate fit and predict."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 150, 75, 5

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        # Fit
        scp = StratifiedCP(score_fn="utility", similarity_matrix=similarity_matrix)
        scp.fit(cal_probs, cal_labels)

        # Predict
        results = scp.predict(test_probs, test_labels)

        assert len(results["selected_idx"]) + len(results["unselected_idx"]) == n_test

    def test_utility_summary(self, similarity_matrix):
        """Test summary method with utility-aware CP."""
        np.random.seed(42)
        n_cal, n_test, n_classes = 150, 75, 5

        cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
        cal_labels = np.random.randint(0, n_classes, n_cal)

        test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
        test_labels = np.random.randint(0, n_classes, n_test)

        scp = StratifiedCP(score_fn="utility", similarity_matrix=similarity_matrix)
        scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

        summary = scp.summary()

        # Check that summary contains expected information
        assert "utility" in summary
        assert "Stratified Conformal Prediction" in summary
        assert isinstance(summary, str)
