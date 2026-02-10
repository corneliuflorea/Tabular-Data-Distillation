#=======================================================================
# ~ Implements the 4 distillation metrics
#=======================================================================

import os
import numpy as np
import csv
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import mahalanobis
from numpy.linalg import pinv
import pdb


# =========================
# Tailness metrics
# =========================

def univariate_tailness_numpy(X):
    """
    Compute univariate tailness metrics per feature.
    Returns arrays of skewness, kurtosis, tail ratios.
    """
    n_samples, n_features = X.shape

    skewness = np.zeros(n_features)
    kurt = np.zeros(n_features)
    tail_ratio = np.zeros(n_features)

    for j in range(n_features):
        x = X[:, j]
        x = x[~np.isnan(x)]

        if len(x) < 10:
            skewness[j] = np.nan
            kurt[j] = np.nan
            tail_ratio[j] = np.nan
            continue

        q01 = np.quantile(x, 0.01)
        q50 = np.quantile(x, 0.50)
        q99 = np.quantile(x, 0.99)

        skewness[j] = skew(x)
        kurt[j] = kurtosis(x, fisher=True)
        tail_ratio[j] = np.log1p((q99 - q50) / (q50 - q01 + 1e-12))

    return skewness, kurt, tail_ratio


def multivariate_tail_fraction(X, quantile=0.99):
    """
    Fraction of samples in the multivariate tail
    using Mahalanobis distance.
    """
    if X.shape[0] < X.shape[1] + 1:
        return np.nan

    mean = X.mean(axis=0)
    cov_inv = pinv(np.cov(X, rowvar=False))

    dists = np.array([
        mahalanobis(x, mean, cov_inv) for x in X
    ])

    threshold = np.quantile(dists, quantile)
    return np.mean(dists > threshold)


# =========================
# Dataset-level aggregation
# =========================

def compute_dataset_tailness(X):
    skewness, kurt, tail_ratio = univariate_tailness_numpy(X)

    return {
        "mean_abs_skewness": np.nanmean(np.abs(skewness)),
        "mean_excess_kurtosis": np.nanmean(np.maximum(0.0, kurt)),
        "mean_tail_ratio_99_01": np.nanmean(tail_ratio),
        "multivariate_tail_fraction": multivariate_tail_fraction(X)
    }


# =========================
# Fundamental processing function: gets a a root_dir with datasets - returns a set of tailness measure that are written into a .csv file
# =========================

def process_datasets(root_dir, output_prefix="tailness_results_regress"):
    results = []

    for dataset_name in sorted(os.listdir(root_dir)):
        dataset_path = os.path.join(root_dir, dataset_name)
        x_path = os.path.join(dataset_path, "X_train.npy")

        if not os.path.isdir(dataset_path):
            continue
        if not os.path.isfile(x_path):
            continue

        print(f"Processing {dataset_name}")

        X = np.load(x_path)
        metrics = compute_dataset_tailness(X)
        metrics["dataset"] = dataset_name
        results.append(metrics)

    # Convert to structured arrays
    results_np = np.array([
        (
            r["dataset"],
            r["mean_abs_skewness"],
            r["mean_excess_kurtosis"],
            r["mean_tail_ratio_99_01"],
            r["multivariate_tail_fraction"]
        )
        for r in results
    ], dtype=object)

    # Save NumPy
    np.save(f"{output_prefix}.npy", results_np)

    # Save CSV
    csv_path = f"{output_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "mean_abs_skewness",
            "mean_excess_kurtosis",
            "mean_tail_ratio_99_01",
            "multivariate_tail_fraction"
        ])
        for r in results:
            writer.writerow([
                r["dataset"],
                r["mean_abs_skewness"],
                r["mean_excess_kurtosis"],
                r["mean_tail_ratio_99_01"],
                r["multivariate_tail_fraction"]
            ])

    print(f"\nSaved results to:")
    print(f"  {output_prefix}.npy")
    print(f"  {output_prefix}.csv")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    # -- Choose between these
    DATASETS_ROOT = "./datasets/Regression"   
    # DATASETS_ROOT = "./datasets/Classification"   
    process_datasets(DATASETS_ROOT)
