#=======================================================================
# ~ Apply distillation methods on all datasets
# ~ Loops thorugh datasets, percentages, distillation methods.
# ~ Important: uses Multiple Thread - parallel processing (thus more efficient). 
# ~ To be used for:
 # ~ - efficient implementation
 # ~ - requires multi-core CPU
#=======================================================================

# ====================================================
# Apply distillation methods on all folders available
# To set:
#   - the list of distillation methods in DISTIL_METHODS
#   - the list of percentages in the main function
#	- the folder where are the databases in the main function
# ====================================================


import os
import numpy as np
import time
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tabular_distilation_method import distill_kmeans, distill_coreset, distill_ctgan, distill_tvae, distill_Gauss_cop
from tabular_distilation_method import distill_coreset_kmedoids, distill_coreset_leverage_scores, gmm_distill_tabular
import pdb

# ====================================================
# List of distillation methods
# ====================================================

DISTIL_METHODS = {
    "K_means": distill_kmeans,
    "Coreset": distill_coreset, 
    "CTgan": distill_ctgan,
	"TVAE": distill_tvae,
	"Gauss_cop": distill_Gauss_cop,
    "Core_lev_score": distill_coreset_leverage_scores, 
	"GMM": gmm_distill_tabular,
}


# ====================================================
# Logging helper: print to screen + file
# ====================================================

def log(msg, logfile):
    """
    Prints a message to screen and appends it to logfile.
    """
    print(msg)
    with open(logfile, "a") as f:
        f.write(msg + "\n")




# ====================================================
# Dataset loader
# ====================================================

def load_dataset(dataset_path):
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_path, "y_train.npy"))
    X_test  = np.load(os.path.join(dataset_path, "X_test.npy"))
    y_test  = np.load(os.path.join(dataset_path, "y_test.npy"))
    return X_train, y_train, X_test, y_test


# ====================================================
# Distillation task (executed in parallel)
# ====================================================

def distill_single_dataset(args):
    """
    Executed in a separate process.
    Accepts a tuple because ProcessPoolExecutor
    does not support lambda with multiple arguments.
    """
    dataset_path, method_name, percentage = args
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    logfile = os.path.join(os.getcwd(), "distillation_log3.txt")

    log(f"\n=== Processing dataset: {dataset_path} ===", logfile)
    log(f"Method: {method_name}, Percentage: {percentage}%", logfile)

    # Validate method
    if method_name not in DISTIL_METHODS:
        log(f"ERROR: Unknown method {method_name}", logfile)
        return

    # Load data
    X_train, y_train, _, _ = load_dataset(dataset_path)
    log(f"Loaded X_train shape: {X_train.shape}", logfile)
    log(f"Loaded y_train shape: {y_train.shape}", logfile)

    # Apply distillation
    distil_fn = DISTIL_METHODS[method_name]
    X_red, y_red = distil_fn(X_train, y_train, percentage)

    log(f"Reduced X_train shape: {X_red.shape}", logfile)
    log(f"Reduced y_train shape: {y_red.shape}", logfile)

    # Save reduced files
    X_out = os.path.join(dataset_path, f"X_train_{method_name}_{percentage}.npy")
    y_out = os.path.join(dataset_path, f"Y_train_{method_name}_{percentage}.npy")

    np.save(X_out, X_red)
    np.save(y_out, y_red)

    log(f"Saved reduced X: {X_out}", logfile)
    log(f"Saved reduced Y: {y_out}", logfile)
    log(f"=== Completed dataset: {dataset_path} ===\n", logfile)

    return dataset_path  # for tracking progress

# ====================================================
# 3. Run  distillation with time counting. DOes not save output. Just for time measurements
# ====================================================

def distill_single_dataset_duration(args):
    """
    Performs distillation for a dataset, measures duration, logs results,
    and writes timing info to a CSV file. Does NOT save distilled data.
    """
    dataset_path, method_name, percentage = args
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    logfile = os.path.join(os.getcwd(), "distillation_log_timestamp.txt")
    csv_file = os.path.join(os.getcwd(), "distillation_timing2.csv")

    dataset_name = os.path.basename(dataset_path)

    # Start logging session
    log(f"\n=== Processing dataset: {dataset_path} ===", logfile)
    log(f"Method: {method_name}, Percentage: {percentage}%", logfile)

    # Check method
    if method_name not in DISTIL_METHODS:
        log(f"ERROR: Unknown method {method_name}", logfile)
        raise ValueError(f"Unknown distillation method: {method_name}")

    # Load data
    X_train, y_train, _, _ = load_dataset(dataset_path)
    n_original = X_train.shape[0]

    log(f"Loaded X_train shape: {X_train.shape}", logfile)
    log(f"Loaded y_train shape: {y_train.shape}", logfile)

    # Run method with timing
    distil_fn = DISTIL_METHODS[method_name]

    start_time = time.perf_counter()
    X_red, y_red = distil_fn(X_train, y_train, percentage)
    end_time = time.perf_counter()

    duration_sec = end_time - start_time
    n_reduced = X_red.shape[0]

    log(f"Reduced X_train shape: {X_red.shape}", logfile)
    log(f"Reduced y_train shape: {y_red.shape}", logfile)
    log(f"Distillation time: {duration_sec:.6f} seconds", logfile)

    # Write timing results to CSV
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow([
                "dataset",
                "method",
                "percentage",
                "n_original",
                "n_reduced",
                "duration_seconds"
            ])

        writer.writerow([
            dataset_name,
            method_name,
            percentage,
            n_original,
            n_reduced,
            f"{duration_sec:.6f}"
        ])

    log(f"Timing written to {csv_file}", logfile)
    log(f"=== Completed dataset: {dataset_path} ===\n", logfile)

    return dataset_path  # for tracking progress


# ====================================================
# Parallel manager
# ====================================================

def distill_all_datasets_parallel(base_folder, method_name, percentage, num_workers=None):
    """
    Runs dataset distillation on multiple CPU cores in parallel.
    """

    # Collect dataset paths
    dataset_paths = [
        os.path.join(base_folder, d)
        for d in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ]

    print(f"Found {len(dataset_paths)} datasets.")
    print("Starting parallel processing...")

    # Prepare arguments for each task
    tasks = [(p, method_name, percentage) for p in dataset_paths]

    # Use all CPU cores by default
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # ~ futures = {executor.submit(distill_single_dataset, t): t[0] for t in tasks}
        # ~ distill_single_dataset_duration
        futures = {executor.submit(distill_single_dataset_duration, t): t[0] for t in tasks}
        for future in as_completed(futures):
            dataset_name = futures[future]
            try:
                future.result()
                print(f"[DONE] {dataset_name}")
            except Exception as e:
                print(f"[ERROR] {dataset_name} -> {e}")


# ====================================================
# Demo
# ====================================================

if __name__ == "__main__":
    # ~ datasets_root = "datasets/Regression"
    datasets_root = "datasets/Classification"
        
    percentages = [50, 25, 10, 5]
    for method in DISTIL_METHODS:
        for percent in percentages:
            distill_all_datasets_parallel(datasets_root, method, percent)
