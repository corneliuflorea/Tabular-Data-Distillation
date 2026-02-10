#=======================================================================
    # ~ Apply distillation methods on all datasets
    # ~ Loops thorugh datasets, percentages, distillation methods.
    # ~ Important: uses specifically a SingleThread (thus much longer). 
    # ~ To be used:
     # ~ - comparing duration
     # ~ - problems with threading due to operating system
#=======================================================================

import os
import numpy as np
from datetime import datetime
import time
from tabular_distillation_method import distill_kmeans, distill_coreset, distill_ctgan, distill_tvae, distill_Gauss_cop
from tabular_distillation_method import distill_coreset_leverage_scores, gmm_distill_tabular 
import pdb
import csv


DISTIL_METHODS = {
    "K_means": distill_kmeans,
    "Coreset": distill_coreset, #Gonzales algorithm
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
# 2. Dataset loader
# ====================================================

def load_dataset(dataset_path):
    # ~ pdb.set_trace()
    # ~ dataset_path = os.path.join(os.getcwd(), dataset_path)    
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    y_train = np.load(os.path.join(dataset_path, "y_train.npy"))
    X_test  = np.load(os.path.join(dataset_path, "X_test.npy"))
    y_test  = np.load(os.path.join(dataset_path, "y_test.npy"))
    return X_train, y_train, X_test, y_test


# ====================================================
# 2. Apply distillation with logging. Save the ditilled set
# ====================================================

def apply_distillation(dataset_path, method_name, percentage):
    """
    Performs distillation for a dataset, saving reduced files and writing logs.
    """
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    logfile = os.path.join(os.getcwd(), "distillation_log.txt")
    
    # Start logging session
    log(f"\n=== Processing dataset: {dataset_path} ===", logfile)
    log(f"Method: {method_name}, Percentage: {percentage}%", logfile)

    # Check method
    if method_name not in DISTIL_METHODS:
        log(f"ERROR: Unknown method {method_name}", logfile)
        raise ValueError(f"Unknown distillation method: {method_name}")

    # Load data
    X_train, y_train, _, _ = load_dataset(dataset_path)
    log(f"Loaded X_train shape: {X_train.shape}", logfile)
    log(f"Loaded y_train shape: {y_train.shape}", logfile)

    # Run method
    distil_fn = DISTIL_METHODS[method_name]
    X_red, y_red = distil_fn(X_train, y_train, percentage)

    log(f"Reduced X_train shape: {X_red.shape}", logfile)
    log(f"Reduced y_train shape: {y_red.shape}", logfile)

    # Output names
    X_out = os.path.join(dataset_path, f"X_train_{method_name}_{percentage}.npy")
    y_out = os.path.join(dataset_path, f"Y_train_{method_name}_{percentage}.npy")

    # Save reduced datasets
    np.save(X_out, X_red)
    np.save(y_out, y_red)

    log(f"Saved reduced X: {X_out}", logfile)
    log(f"Saved reduced Y: {y_out}", logfile)
    log(f"=== Completed dataset: {dataset_path} ===\n", logfile)





# ====================================================
# 3. Run  distillation with time counting. DOes not save output. Just for time measurements
# ====================================================

def apply_distillation_duration(dataset_path, method_name, percentage):
    """
    Performs distillation for a dataset, measures duration, logs results,
    and writes timing info to a CSV file. Does NOT save distilled data.
    """
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    logfile = os.path.join(os.getcwd(), "distillation_log.txt")
    csv_file = os.path.join(os.getcwd(), "distillation_timing_basic_SingleTh.csv")

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



# ====================================================
# 4. Process all datasets (logging per dataset)
# ====================================================

def distill_all_datasets(base_folder, method_name, percentage):
    """
    Loops through all folders in ./datasets/ and applies distillation.
    """
    i=0
    for dataset_name in os.listdir(base_folder):
        dataset_path = os.path.join(base_folder, dataset_name)
        i = i+1


        if os.path.isdir(dataset_path):
            # ~ apply_distillation(dataset_path, method_name, percentage)
            apply_distillation_duration(dataset_path, method_name, percentage)


# ====================================================
# 5. Example main
# ====================================================

if __name__ == "__main__":
    # ~ datasets_root = "datasets"
    datasets_root = "datasets/Classification"
    percentages = [50, 25, 10, 5]
    for percent in percentages:
        for method in DISTIL_METHODS:
            print(method, percent)
            distill_all_datasets(datasets_root, method, percent)
