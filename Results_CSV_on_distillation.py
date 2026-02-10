import os
import numpy as np
import csv
import pdb

BASE_FOLDER = "./datasets//Classification/"

DISTIL_METHODS = ["K_means", "Coreset", "CTgan", "TVAE", "Gauss_cop", "Core_lev_score", "Core_fac_loc", "GMM"]
PERCENTS = [50, 25, 10, 5]

DISTIL_TAGS = [f"{m}_{p}" for m in DISTIL_METHODS for p in PERCENTS]

CLASSIFIERS = ["RF", "SVM", "XGB"]

# -------------------------------------------------------------
# Helper: extract metric name based on classification/regression
# -------------------------------------------------------------
def get_metric_name(result_block):
    """Determine if metric is accuracy or mse."""
    meta = result_block.get("meta", {})
    is_classif = meta.get("is_classification", True)
    return "acc" if is_classif else "mse"

# ~ def get_avgacc(result_block):
    # ~ """Determine if metric is accuracy or mse."""
    # ~ meta = result_block.get("meta", {})
    # ~ is_classif = meta.get("is_classification", True)
    # ~ pdb.set_trace()
    # ~ per_class = result_block.get('per_class')
    # ~ avgacc = per_class.mean(axis=1).max()
    # ~ return "acc" if is_classif else "mse"

# -------------------------------------------------------------
# Main extraction and CSV writing
# -------------------------------------------------------------
def extract_all_results(base_folder=BASE_FOLDER):
    
    # For each classifier, we collect rows of: dataset_name + metric_per_distil
    table = {clf: [] for clf in CLASSIFIERS}

    for ds_name in os.listdir(base_folder):
        ds_path = os.path.join(base_folder, ds_name)
        if not os.path.isdir(ds_path):
            continue

        # find result file
        result_file = os.path.join(ds_path, f"{ds_name}_results.npy")
        if not os.path.exists(result_file):
            print(f"Missing results file in {ds_path}")
            continue

        try:
            results = np.load(result_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")
            continue

        # results is a vector of objects (each entry = one distillation)
        # Build mapping distil_tag → result_object
        # Each item is a dict with the only key = distillation tag
        distil_map = {}  
        # ~ pdb.set_trace()
        # ~ for entry in results:
            # ~ if not isinstance(entry, dict):
                # ~ continue
            # ~ for tag, block in entry.items():
                # ~ distil_map[tag] = block   # block contains meta + classifiers
        for entry in np.atleast_1d(results):
            # Check if entry is a dictionary
            if isinstance(entry, dict):
                for tag, block in entry.items():
                    distil_map[tag] = block

        # For each classifier we prepare row: [dataset_name, values...]
        # ~ pdb.set_trace()
        for clf in CLASSIFIERS:
            row = [ds_name]
            # ~ pdb.set_trace()
            for tag in DISTIL_TAGS:
                if tag not in distil_map:
                    row.append(np.nan)
                    print("issue line 68")
                    continue

                block = distil_map[tag]

                # metric type
                # ~ metric_name = get_metric_name(block)
                
                

                if clf not in block["classifiers"]:
                    print("issue line 76")
                    row.append(np.nan)
                    continue

                
                clf_block = block["classifiers"][clf]
                # ~ pdb.set_trace()
                best_block = clf_block.get("best", {})
                per_class = clf_block['all']['per_class']
                best_av_acc = per_class.mean(axis=1).max()
                

                # ~ value = best_block.get(metric_name, np.nan)
                row.append(best_av_acc)

            table[clf].append(row)

    # Write CSVs
    for clf in CLASSIFIERS:
        out_file = f"{clf}_results_AvgAcc.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["dataset"] + DISTIL_TAGS
            writer.writerow(header)
            writer.writerows(table[clf])
        print(f"Saved {out_file}")


# -------------------------------------------------------------
# Run
# -------------------------------------------------------------
if __name__ == "__main__":
    extract_all_results()
