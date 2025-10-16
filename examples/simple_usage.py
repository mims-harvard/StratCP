"""
Simple example demonstrating the 1-2 line StratifiedCP API.

This shows how to use StratCP with minimal code for end-to-end
stratified conformal prediction.
"""

import numpy as np
from stratcp import StratifiedCP

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Generate synthetic data (replace with your model predictions)
# ============================================================================

# Calibration data
n_cal = 1000
n_classes = 5
cal_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_cal)
cal_labels = np.array([np.random.choice(n_classes, p=p / p.sum()) for p in cal_probs])

# Test data
n_test = 500
test_probs = np.random.dirichlet(np.ones(n_classes) * 2, n_test)
test_labels = np.array([np.random.choice(n_classes, p=p / p.sum()) for p in test_probs])

print(f"Calibration: {n_cal} samples, {n_classes} classes")
print(f"Test: {n_test} samples")
print()

# ============================================================================
# 2. ONE-LINE USAGE: Fit and predict with default settings
# ============================================================================

scp = StratifiedCP()
results = scp.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Print summary
print(scp.summary())
print()

# ============================================================================
# 3. Access results
# ============================================================================

print("Detailed Results:")
print(f"Selected indices: {results['selected_idx'][:10]}...")  # First 10
print(f"Unselected indices: {results['unselected_idx'][:10]}...")  # First 10
print()

# Prediction sets for selected samples
if len(results["selected_idx"]) > 0:
    print(f"First selected sample prediction set:")
    print(f"  Classes: {np.where(results['prediction_sets']['selected'][0])[0]}")
    print(f"  Set size: {results['set_sizes']['selected'][0]:.0f}")
    print()

# Prediction sets for unselected samples
if len(results["unselected_idx"]) > 0:
    print(f"First unselected sample prediction set:")
    print(f"  Classes: {np.where(results['prediction_sets']['unselected'][0])[0]}")
    print(f"  Set size: {results['set_sizes']['unselected'][0]:.0f}")
    print()

# ============================================================================
# 4. ALTERNATIVE: More control with separate fit/predict
# ============================================================================

print("\n" + "=" * 60)
print("Alternative: Separate fit() and predict()")
print("=" * 60)

# Fit once on calibration data
scp2 = StratifiedCP(score_fn="raps", alpha_sel=0.15, alpha_cp=0.1)
scp2.fit(cal_probs, cal_labels)

# Predict on multiple test sets
results2 = scp2.predict(test_probs, test_labels)
print(f"\nTest set 1: {len(results2['selected_idx'])} selected")

# Can predict on another test set without refitting
test_probs2 = np.random.dirichlet(np.ones(n_classes) * 2, 200)
test_labels2 = np.array([np.random.choice(n_classes, p=p / p.sum()) for p in test_probs2])
results3 = scp2.predict(test_probs2, test_labels2)
print(f"Test set 2: {len(results3['selected_idx'])} selected")

# ============================================================================
# 5. Different score functions
# ============================================================================

print("\n" + "=" * 60)
print("Comparing different score functions")
print("=" * 60)

for score_fn in ["tps", "aps", "raps"]:
    scp_test = StratifiedCP(score_fn=score_fn, alpha_sel=0.1, alpha_cp=0.1)
    res = scp_test.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

    avg_size_sel = res["set_sizes"]["selected"].mean() if len(res["selected_idx"]) > 0 else 0
    avg_size_unsel = res["set_sizes"]["unselected"].mean() if len(res["unselected_idx"]) > 0 else 0

    print(f"\n{score_fn.upper()}:")
    print(f"  Selected: {len(res['selected_idx'])} samples, avg set size: {avg_size_sel:.2f}")
    print(f"  Unselected: {len(res['unselected_idx'])} samples, avg set size: {avg_size_unsel:.2f}")
