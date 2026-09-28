# Predictive Analysis Under Distribution Shift

A reproducible machine-learning study of how four classifiers respond when the input distribution changes. Using the Wine dataset, the project compares clean evaluation with a targeted covariate shift and a controlled stress test.

## Key results

| Model | Clean accuracy | Shifted accuracy | Accuracy drop |
|---|---:|---:|---:|
| Logistic Regression | 97.2% | 97.2% | 0.0 pp |
| Gaussian Naive Bayes | 97.2% | 97.2% | 0.0 pp |
| Random Forest | 100.0% | 94.4% | 5.6 pp |
| RBF SVM | 97.2% | 83.3% | 13.9 pp |

Logistic Regression and Gaussian Naive Bayes were the most stable under the observed shift. Random Forest degraded modestly. The RBF SVM was most sensitive, with Class 0 recall falling by 33.3 percentage points as several samples moved into the Class 1 decision region.

A separate controlled stress test amplified the same failure mode: RBF SVM accuracy fell from 97.2% to 61.1% after five features were stretched away from their training-set means.

## Problem setup

The experiment uses 13 continuous chemical measurements from the Wine classification dataset:

- 142 training samples
- 36 clean test samples
- 36 shifted test samples
- 3 target classes

The shifted test set preserves the labels but changes five input features across all test rows: color_intensity, alcohol, proanthocyanins, hue, and nonflavanoid_phenols. This creates a targeted covariate-shift scenario.

## Models

- Logistic Regression with standardization
- Gaussian Naive Bayes
- RBF Support Vector Machine with standardization
- Random Forest with 200 trees

All models are trained only on the training split and evaluated against the same clean and shifted labels. A fixed random seed is used where supported.

## Analysis workflow

1. Validate schema, missing values, duplicate feature rows, and class balance.
2. Identify changed features and values outside the training range.
3. Compare clean and shifted accuracy, precision, recall, F1-score, and confusion matrices.
4. Link class-specific errors to shifted feature values.
5. Stress-test the least robust model with a deterministic perturbation.
6. Discuss deployment safeguards such as drift monitoring, range checks, and retraining.

## Repository contents

| File | Purpose |
|---|---|
| Predictive Analysis Under Distribution Shift.ipynb | Complete analysis, executable code, visualizations, and interpretation |
| X_train.csv and y_train.csv | Training features and labels |
| X_test_clean.csv and y_test.csv | Clean evaluation data and labels |
| X_test_shifted_B.csv | Shifted evaluation features |

## Run locally

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    jupyter lab

On macOS or Linux, activate the environment with source .venv/bin/activate. Open the notebook and run the cells from top to bottom. Keep the CSV files in the repository root because the notebook loads them with relative paths.

## Interpretation

Clean accuracy alone did not predict robustness. Although Random Forest achieved perfect clean accuracy, it lost performance under shift. The RBF SVM showed the largest degradation because the altered features changed distances and local geometry in the standardized feature space. The simpler Logistic Regression and Gaussian Naive Bayes models remained stable for this particular shift; this does not guarantee robustness to other forms or severities of drift.

## Limitations

- The evaluation set contains only 36 samples, so results should be interpreted as a focused case study.
- Only one observed shift and one designed perturbation are evaluated.
- Hyperparameters are intentionally simple rather than exhaustively tuned.
- The stress test includes values that may be unrealistic for some chemical measurements; it is an inductive-bias probe, not a simulation of a specific production process.

## Practical safeguards

For a deployed version, monitor feature distributions, flag inputs outside the training range, track class-wise metrics when labels become available, and retrain when drift is persistent. Predictions far from the training distribution should be reviewed rather than treated as equally reliable.
