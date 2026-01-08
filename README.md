# StratCP: Error controlled decisions for safe use of medical foundation models

[![Release](https://img.shields.io/github/v/release/mims-harvard/stratcp)](https://img.shields.io/github/v/release/mims-harvard/stratcp)
[![License](https://img.shields.io/github/license/mims-harvard/stratcp)](https://img.shields.io/github/license/mims-harvard/stratcp)

Foundation models show promise in medicine, but clinical use requires outputs that clinicians can act on under pre-specified error budgets, such as a cap on false-positive clinical calls. Without error control, strong average accuracy can still lead to harmful errors among the very cases labeled confident and to inefficient use of follow-up testing.

Here we introduce StratCP, a stratified conformal framework that turns foundation-model predictions into decision-ready outputs by combining selective action with calibrated deferral. StratCP first selects a subset of patients for immediate clinical calls while controlling the false discovery rate among those calls at a user-chosen level. It then returns calibrated prediction sets for deferred patients that meet the target error rate and guide confirmatory evaluation. The procedure is model agnostic and can be applied to pretrained foundation models without retraining.

We validate StratCP in ophthalmology and neuro-oncology across diagnostic classification and time-to-event prognosis. Across tasks, StratCP maintains false discovery rate control on selected patients and produces coherent prediction sets for deferred patients. In neuro-oncology, it supports diagnosis from H&E whole-slide images under a fixed error budget, reducing the need for reflex molecular assays and lowering laboratory cost and turnaround time. StratCP lays the groundwork for safe use of medical foundation models by converting predictions into error-controlled actions when evidence is strong and calibrated uncertainty otherwise.

## Highlights

- 🎯 **FDR-controlled selection** - Identify high-confidence predictions with false discovery rate control
- 📊 **Post-selection inference** - Valid conformal prediction after selection for low-confidence predictions (JOMI)
- 🔧 **Multiple score functions** - TPS, APS, RAPS, and utility-aware for different prediction scenarios
- 🧬 **Utility-aware CP** - Leverage label similarity for more coherent prediction sets
- 🏥 **Medical applications** - Designed for clinical decision support and medical diagnosis
- 📈 **Ordinal labels** - Support for ordered categories (e.g., disease severity)
- ⚡ **Fast and scalable** - Efficient implementations for large datasets

## Installation

```bash
# Install from PyPI (coming soon)
pip install stratcp

# Or install from source
git clone https://github.com/mims-harvard/stratcp.git
cd stratcp
make install
```

## Quick Start

### 🚀 Simple 1-2 Line Usage

Below we show a use case in multi-class classification where we:
- FDR-controlled selection of high-confidence predictions, whose argmax equals unknown true label
- Conformal prediction sets for unselected, low-confident cases with 90% coverage


```python
import numpy as np
import stratcp as scp

# Labeled (calibration) and unlabeled (test) data
n, m, K = 1000, 500, 5
cal_probs = np.random.dirichlet(np.ones(K), size=n)
cal_labels = np.array([np.random.choice(K, p=cal_probs[i,:]) for i in range(n)])
test_probs = np.random.dirichlet(np.ones(K), size=m)
test_labels = np.array([np.random.choice(K, p=test_probs[i]) for i in range(m)])

# One-line usage: fit and predict
# Goal: select confident predictions with FDR <= alpha_sel, 
#       for unselected units, construct prediction sets with coverage >= 1-alpha_cp
model = scp.StratifiedCP(alpha_sel=0.1, alpha_cp=0.1)
results = model.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Access results
print(f"Selected: {len(results['selected_idx'])} high-confidence predictions")
print(f"Avg set size (unselected): {results['set_sizes'].mean():.2f}")

# Print detailed summary
print(model.summary())
```



You can also perform per-class selection where we:
- FDR-controlled selection of high-confidence predictions for every class of argmax prediction, whose argmax equals unknown true label
- Conformal prediction sets for unselected, low-confident cases with 90% coverage

by simply adding the argument `eligibility = 'per_class'`.

```python
import stratcp as scp

# One-line usage: fit and predict
model = scp.StratifiedCP(alpha_sel=0.1, alpha_cp=0.1, eligibility='per_class')
results = model.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Print detailed summary
print(model.summary())
```

The `results` object contains:
- `all_selected`, a list that contains confident indices selected for each class, followed by indices unselected by any class (low-confidence).
- `thresholds`, a list of per-class selection thresholds.
- `prediction_sets`, boolean array of shape `(n_unselected, n_class)` for unselected samples only (those in `all_selected[K]`).
- `set_sizes`, sizes of each prediction set in `prediction_sets`.

### 🧬 Utility-Aware CP with Label Similarity

When you have semantic relationships between labels (e.g., medical diagnoses, biological taxonomies), use utility-aware CP for more coherent and interpretable prediction sets:

```python
import stratcp as scp
import numpy as np

# Define similarity matrix between classes
# Higher values = more similar (range [0, 1])
similarity_matrix = np.array([
    [1.0, 0.9, 0.3, 0.3, 0.1],  # Class 0: very similar to 1
    [0.9, 1.0, 0.3, 0.3, 0.1],  # Class 1: very similar to 0
    [0.3, 0.3, 1.0, 0.9, 0.1],  # Class 2: very similar to 3
    [0.3, 0.3, 0.9, 1.0, 0.1],  # Class 3: very similar to 2
    [0.1, 0.1, 0.1, 0.1, 1.0],  # Class 4: dissimilar to all
])

# Use utility-aware CP
model = scp.StratifiedCP(
    score_fn='utility',
    similarity_matrix=similarity_matrix,
    utility_method='weighted',  # or 'greedy'
    alpha_sel=0.1,
    alpha_cp=0.1
)
results = model.fit_predict(cal_probs, cal_labels, test_probs, test_labels)

# Evaluate prediction set coherence
avg_sim, overall_sim = scp.eval_similarity(
    results['prediction_sets']['unselected'],
    similarity_matrix
)
print(f"Average pairwise similarity: {overall_sim:.3f}")
```

**Benefits:**
- More coherent prediction sets (similar classes grouped together)
- Better interpretability for domain experts
- Maintains valid coverage guarantees
<!--
### 📦 Import Patterns

All key functions are available from the top-level `stratcp` package:

```python
# Option 1: Import specific functions
from stratcp import (
    StratifiedCP,
    get_sel_single,
    get_sel_multiple,
    compute_score_raps,
    conformal,
)

# Option 2: Import with namespace (recommended for cleaner code)
import stratcp as scp

scp.StratifiedCP(...)
scp.get_sel_single(...)
scp.compute_score_raps(...)
```

You can also import from submodules if needed:
```python
from stratcp.selection import get_sel_single, get_sel_multiple
from stratcp.conformal import compute_score_raps, conformal
from stratcp.metrics import size_cond_cov, label_cond_cov
``` -->

### 📊 Advanced Usage (Lower-Level API)

You can also use lower-level functions for more bespoke use cases. Let's say you want to select confident predictions according to K criteria, where
- $I_k(x,y)=1$ means the desired $k$-th criterion (confident prediction) is satisfied
- $G_k(x)$ means the sample is eligble to be selected for $k$-th criterion (optional)
- $f_k(x,y)$ is a predicted score for the $k$-th criterion

You have $n$ labeled calibration data and $m$ unlabeled test data awaiting decisions.

```python
import numpy as np
import stratcp as scp

# Step 1: FDR-controlled selection
sel_idx, unsel_idx, tau = scp.get_sel_single(
    cal_conf_scores=cal_confidence,      # Calibration confidence scores (n,)
    cal_conf_labels=cal_conf_labels,     # Binary labels (n,) for correctness/confidence
    test_conf_scores=test_confidence,    # Test confidence scores (m,)
    alpha=0.1                       # FDR level (10%)
)

# Step 2: Compute (your own) nonconformity scores
cal_conformal_scores, test_conformal_scores = scp.compute_score_raps(
    cal_probs, test_probs, cal_labels
)

# Step 3: JOMI conformal prediction for unselected samples
# This reference mats can be plugged into your own score functions
ref_mats = scp.get_reference_sel_single(
    unsel_idx,
    cal_conf_labels = cal_conf_labels, # Binary labels (n,) for correctness/confidence
    cal_conf_scores = cal_confidence,
    test_conf_scores = test_confidence,
    test_imputed_conf_labels = test_imputed_labels, # Imputed test confidence labels L(X_n+j, y) for all labels y (m, nclass)
    alpha=0.1
)

# obtain conformal prediction sets
pred_sets_unsel = scp.conformal(
    cal_scores = cal_conformal_scores, # conformity score s(X,Y) (n,)
    test_scores = test_conformal_scores[unsel_idx], # conformity score s(X,y) for all y (m, class)
    cal_labels = cal_y,  # calibration labels Y (n, )
    alpha=0.1,
    if_in_ref=[ref_mats[k][unsel_idx] for k in range(len(ref_mats))],  # Use reference sets for unselected indices
)

print(f"Selected: {len(sel_idx)} samples with avg set size {sizes_sel.mean():.2f}")
print(f"Unselected: {len(unsel_idx)} samples with avg set size {sizes_unsel.mean():.2f}")
```

## Key Features

#### Selection Methods

**Single Property Selection** - Select samples where a binary property (confidence) holds with FDR control:
```python
import stratcp as scp

sel_idx, unsel_idx, tau = scp.get_sel_single(
    cal_conf_scores=cal_confidence,      # Calibration confidence scores (n,)
    cal_conf_labels=cal_conf_labels,     # Binary labels (n,) for correctness/confidence
    test_conf_scores=test_confidence,    # Test confidence scores (m,)
    alpha=0.1                            # FDR level
)
```

**Multiple Property Selection** - Simultaneously select across multiple selection rules:
```python
import stratcp as scp

all_sel, tau_list = scp.get_sel_multiple(
    cal_scores=cal_confidence,      # Calibration confidence scores (n,) for K criteria
    cal_labels=cal_conf_labels,     # Binary labels (n,K) for correctness/confidence I_k
    test_scores=test_confidence,    # Test confidence scores (m,K) for K criteria
    cal_eligs=cal_eligible,         # Eligibility indicators (n,K) for K criteria
    test_eligs=test_eligible,       # Test eligibility indicators (m,K) for K criteria
    alpha=0.1                       # FDR level (10%)
)
```

**Survival Analysis** - Select long-term survivors with FDR control:
```python
import stratcp as scp

sel_idx, unsel_idx, tau = scp.get_sel_survival(
    cal_survival_times, cal_predictions, cal_thresholds,
    sigma, test_predictions, test_thresholds, alpha=0.1
)
```

#### Conformal Prediction

**Score Functions** - Multiple nonconformity scores available:
```python
import stratcp as scp

# Standard scores
cal_scores, test_scores = scp.compute_score_aps(cal_probs, test_probs, cal_labels)
```

**Utility-Aware Scores** - Leverage label similarity for coherent sets:
```python
import stratcp as scp

# Compute utility-aware scores
cal_scores, test_scores = scp.compute_score_utility(
    cal_probs, test_probs, cal_labels, similarity_matrix
)
# Evaluate prediction set coherence
avg_sim, overall_sim = scp.eval_similarity(prediction_sets, similarity_matrix)
```

**Post-Selection Inference (JOMI)** - Valid conformal prediction for unselected samples:
```python
import stratcp as scp

# Without selection (vanilla CP)
pred_sets = scp.conformal(
    cal_scores, test_scores, cal_labels, alpha=0.1
)

# With selection (JOMI CP)
pred_sets = scp.conformal(
    cal_scores, test_scores, cal_labels,
    alpha=0.1, if_in_ref=reference_sets  # Use JOMI reference sets
)
```
The output is a (m, nclass) matrix where (i, k) indicates whether class k is in $C(X_{n+i})$.

#### Evaluation Metrics

```python
import stratcp as scp

# Coverage by prediction set size
cond_cov, cond_freq = scp.size_cond_cov(pred_set, test_labels)

# Coverage by true label
label_cov, label_freq = scp.label_cond_cov(pred_set, test_labels)
```

## Use Cases

Our framework allows diverse use cases based on the stratCP principle.

**Medical Diagnosis**.  Stratify patients based on model confidence:
- **High confidence**: Make precise diagnoses (argmax = true label)
- **Low confidence**: Provide differential candidate diagnoses (prediction sets with guarantees)

**Quality Control**. Identify items that can be automatically classified v.s. those needing human review:
- **Selected**: Automated decisions with FDR control
- **Unselected**: Flag for manual inspection with uncertainty quantification

**Multi-stage Decision Making**. Implement triaged decision systems:
1. **Stage 1**: Fast, confident decisions for easy cases
2. **Stage 2**: Careful analysis with uncertainty bounds for difficult cases

## Documentation

- **API Reference**: [https://zitniklab.hms.harvard.edu/projects/StratCP](https://zitniklab.hms.harvard.edu/projects/StratCP)
- **Usage Guide**: See [USAGE_SUMMARY.md](USAGE_SUMMARY.md) for comprehensive examples
- **Example Scripts**:
  - `examples/simple_usage.py` - Basic usage patterns
  - `examples/utility_aware_cp.py` - Utility-aware CP with similarity matrices

## Reproduction scripts

Scripts for reproducing the results in the paper are in `reproduction_code/` with wrappers in `run_*.sh`. Each expects paths to files

- `predicted_probabilities.npy` (n_samples x n_classes)
- `true_labels.npy` (n_samples)

We store these files in the `data/` folder. The summarized results will be saved in the same folder by default.

### Retinal disease diagnosis tasks

Across all ophthalmology tasks, we follow the RetFound foundation model [[Zhou et al., 2023](https://www.nature.com/articles/s41586-023-06555-x)] using the provided model checkpoints and data splits available [here](https://github.com/rmaphoh/RETFound/blob/main/BENCHMARK.md). Given model predictions (per-class probabilities for classification tasks), we apply **StratCP** to the task-specific scores:
1. **Action arm.** Select a confident subset under an expert-specified FDR budget (that is, the incorrect predictions among selected).
2. **Deferral arm.** For the remaining (less confident) cases, construct conformal prediction sets with finite-sample coverage guarantees, adjusting for the distribution shift due to the selection in the action arm. 

The experiments in the paper can be reproduced with the following scripts:
- `reproduction_code/retfound_tasks/diabetic_retinpacy.py` for the DR diagnosis task.
- `reproduction_code/retfound_tasks/glaucoma.py` for the Glaucoma diagnosis task.
- `reproduction_code/retfound_tasks/jsiec_action.py` for the eye condition detection task, with utility enhancement. 


### Neuro-oncology tasks

Across all neuro-oncology tasks, we extract patch-level features from H\&E-stained whole-slide images (WSIs) using the UNI pathology foundation model together with the CLAM preprocessing pipeline for tiling and feature extraction ([CLAM GitHub](https://github.com/mahmoodlab/CLAM)). Task-specific slide-level predictors are obtained by fine-tuning these features with attention-based multiple instance learning (ABMIL; [Ilse et al., 2018](https://arxiv.org/abs/1802.04712)) on cohorts curated for each endpoint.

All H\&E model checkpoints for each task are available at

- `data/uni_pathology_tasks/<task_name>/model_checkpoint/`

Given model predictions (per-class probabilities for classification tasks and the mean parameter for the time-to-event regression model), we then apply **StratCP** to the task-specific scores:

1. **Step 1 (selection).** Select a confident subset under an expert-specified error budget using FDR control.
2. **Step 2 (post-selection CP).** For the remaining (less confident) cases, construct conformal prediction sets with finite-sample coverage guarantees.

The main entry points for reproducing neuro-oncology experiments are:

- `idh_mut_status_pred.py` – IDH mutation status prediction.
- `cns_tumor_subtype.py` – central nervous system (CNS) tumor subtype classification.
- `he_time_to_mortaility_pred.py` – H\&E time-to-mortality prediction.
- `he_diagnosis_in_atdg.py` – H\&E-only diagnosis in adult-type diffuse glioma (ATDG).

**Interpreting StratCP outputs (quick guide)**

- **Selection rate**: fraction of test samples selected for immediate action. In `eligibility="overall"`, compute `len(results["selected_idx"]) / m`. In `eligibility="per_class"`, use the per-class counts in `results["all_selected"]` and the unselected set in `results["all_selected"][K]`.
- **FDR (action arm)**: controlled at `alpha_sel` by construction on the selected set. Empirical FDR can be estimated if true labels are available by checking the selected predictions that are incorrect.
- **Coverage (deferral arm)**: prediction sets for unselected samples target `1 - alpha_cp` coverage. With labels, compute the fraction of unselected samples whose true label is inside `results["prediction_sets"]`.
- **Prediction set sizes**: `results["set_sizes"]` summarizes uncertainty for unselected samples; smaller sets indicate higher confidence.

## Datasets and data access for reproduction

The table below summarizes the datasets used in the paper, the corresponding tasks, and how to obtain the data required to reproduce the reported results.

| Task | Dataset | Task type | Underlying FM | Fine-tuning strategy | StratCP guarantee* | Additional data approval required | Download link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diabetic retinopathy diagnosis | Kaggle APTOS-2019 | Classification (5 classes) | RETFound | MLP with cross-entropy loss | Multiple criteria | No | [link](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) |
| Glaucoma diagnosis | Glaucoma Fundus dataset | Classification (3 classes) | RETFound | MLP with cross-entropy loss | Multiple criteria | No | [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1YRRAC) |
| Eye condition diagnosis | JSEIC dataset | Classification (39 classes) | RETFound | MLP with cross-entropy loss | Single criteria | No | [link](https://zenodo.org/records/3477553) |
| IDH mutation status | TCGA–LGG & GBM; EBRAINS | Classification (2 classes) | UNI | ABMIL with cross-entropy loss | Multiple criteria | TCGA: No; EBRAINS: Yes | TCGA: [link](https://portal.gdc.cancer.gov/) <br> EBRAINS: [link](https://search.kg.ebrains.eu/instances/Dataset/8fc108ab-e2b4-406f-8999-60269dc1f994) |
| CNS tumor subtyping | EBRAINS | Classification (30 classes) | UNI | ABMIL with cross-entropy loss | Single criteria | Yes | [link](https://search.kg.ebrains.eu/instances/Dataset/8fc108ab-e2b4-406f-8999-60269dc1f994) |
| H\&E time-to-mortality prediction | TCGA–LGG & GBM | Time-to-event regression | UNI | ABMIL with log-normal AFT loss | Single criteria | No (H\&E WSIs are open access via GDC) | [link](https://portal.gdc.cancer.gov/) |

\* “StratCP guarantee” indicates whether StratCP is applied under multiple or single selection criteria for the task.

For TCGA LGG & GBM H\&E slides, no additional special approval is required beyond TCGA’s standard open-access usage terms; the diagnostic WSIs used here are open access via the GDC Data Portal. EBRAINS access is permissioned and requires a data access request.


## Citation

If you use StratCP in your research, please cite:

```bibtex
@article{stratcp2024,
  title={Stratified Conformal Prediction for Post-Selection Inference},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024}
}
```



## Support
This project is licensed under the [MIT License](LICENSE). For questions and issues, please either open an issue on [GitHub](https://github.com/mims-harvard/stratcp/issues) or contact `yjinstat@wharton.upenn.edu`.
