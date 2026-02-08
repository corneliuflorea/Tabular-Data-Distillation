# Tabular Data Distillation: Comparative Study

## 1. Purpose

This repository provides a unified experimental framework for **comparing data distillation methods for tabular datasets**.

The goal is to:
- Evaluate different distillation strategies under a common pipeline
- Measure their impact on learning performance, runtime, and data characteristics
- Support reproducible experimentation across multiple datasets and methods

The focus is on **tabular data**, with an emphasis on **dataset reduction**, **runtime efficiency**, and **preservation of statistical and geometric properties**.

---

## 2. Pipeline Overview

The experimental pipeline follows these steps:

1. Dataset acquisition
2. Data cleaning and structuring
3. Application of distillation methods
4. Model training and evaluation
5. Utility analyses (tailness, visualization, duration)

Each step is modular and configurable.

---

## 3. Code Structure

### 3.1 Package Requirements

This project relies on standard Python scientific and machine learning libraries.

Key dependencies include:
- NumPy
- SciPy
- scikit-learn
- PyTorch
- pandas
- matplotlib / seaborn
- umap-learn
- sdv (for generative distillation methods)
- kaggle (for dataset acquisition)

A full list of requirements is provided in `requirements.txt`.

---

### 3.2 Data

#### 3.2.1 Dataset Gathering

Datasets are obtained from public repositories, including:
- Kaggle
- (Optionally) OpenML or other academic sources

Scripts are provided to:
- Download datasets once
- Store them locally
- Load them offline for reproducibility

Each dataset is stored in its own directory.

---

#### 3.2.2 Cleaning and Structuring

All datasets are converted into a unified format:
- Feature matrix stored as `X.npy`
- Labels stored as `Y.npy`

Preprocessing steps may include:
- Handling missing values
- Encoding categorical variables
- Feature scaling or normalization

---

### 3.3 Applying Distillation

The framework supports multiple distillation methods, including:
- Random sampling
- Clustering-based methods (e.g., k-means, k-medoids)
- PCA-based leverage score sampling
- Core-set selection methods
- Generative approaches (e.g., CTGAN, TVAE)
- No-distillation baseline

Each method can be applied at different reduction percentages.

---

### 3.4 Applying Learning

After distillation, standard classifiers are trained on the reduced datasets.

Supported models include:
- Random Forest
- Support Vector Machines (SVM)
- Gradient Boosting / XGBoost (optional)

Performance is evaluated on a held-out test set using metrics such as:
- Accuracy
- F1-score
- Other task-specific metrics

---

### 3.5 Utilities

#### 3.5.1 Tailness Analysis

Statistical measures are provided to characterize dataset tail behavior, including:
- Skewness
- Kurtosis
- Quantile-based tail ratios
- Mahalanobis-distance-based tail metrics

These measures are computed per dataset and aggregated for analysis.

---

#### 3.5.2 Visualization

Dimensionality reduction techniques are used to visualize datasets before and after distillation:
- PCA
- t-SNE
- UMAP

Visualizations support qualitative comparison of:
- Class separability
- Coverage
- Geometric structure preservation

Plots can be saved automatically as image files.

---

#### 3.5.3 Duration Measurement

Runtime measurement utilities are included to:
- Measure distillation execution time
- Log results per dataset, method, and reduction percentage
- Export timing results to CSV files for benchmarking

This enables systematic comparison of computational efficiency.

---

## Notes

This repository is intended as a **research and experimentation framework**.
Users are encouraged to adapt, extend, and refine individual components depending on their use case.

---
