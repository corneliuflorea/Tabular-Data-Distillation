import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pdb


# ==========================================================
# Utility: detect classification vs regression
# ==========================================================
def is_classification(y):
    
    if y.dtype.kind == 'O':
        return True
		
    y = np.asarray(y, dtype=float)

    # integer dtype → classification
    if y.dtype.kind == "i":
        return True

    # small number of unique labels → classification
    unique = np.unique(y[~np.isnan(y)])
    if len(unique) <= 20:
        return True

    return False


def yes_no_to_numeric(y):
    y = np.asarray(y, dtype=str)
    encoder = LabelEncoder()
    return encoder.fit_transform(y)

# ==========================================================
# Convert any array to numeric (2D)
# ==========================================================
def convert_array_to_numeric(arr):
    """
    Converts ANY .npy array to numeric-only format:
      - Converts strings to numbers using LabelEncoder
      - Converts None / '' / 'nan' / etc to np.nan
      - Ensures final dtype is float
    """

    # Make a float copy (forces object-to-string conversion below)
    arr = np.array(arr, dtype=object)

    # Convert explicit Python None to string "None"
    arr = np.where(arr == None, np.nan, arr)

    # Standardize all empty strings to NaN
    arr = np.where(arr == "", np.nan, arr)

    # Convert 'nan' or 'NaN' strings to np.nan
    arr = np.where(np.char.lower(arr.astype(str)) == "nan", np.nan, arr)

    # Now convert column-by-column
    arr_clean = np.zeros(arr.shape, dtype=float)

    for col in range(arr.shape[1]):
        col_data = arr[:, col]

        # Try numeric conversion
        try:
            numeric_col = col_data.astype(float)
            # All good → numeric column
            arr_clean[:, col] = numeric_col
            continue
        except:
            pass  # must be categorical or mixed

        # Handle categorical encoding
        str_col = col_data.astype(str)
        nan_mask = np.array([s.lower() == "nan" for s in str_col])
        str_col[nan_mask] = "__MISSING__"

        encoder = LabelEncoder()
        encoded = encoder.fit_transform(str_col)

        # Replace "__MISSING__" label with actual np.nan
        missing_index = np.where(encoder.classes_ == "__MISSING__")[0]
        if len(missing_index) > 0:
            missing_label = missing_index[0]
            encoded = encoded.astype(float)
            encoded[encoded == missing_label] = np.nan

        arr_clean[:, col] = encoded

    # Replace any remaining nan with column mean
    for col in range(arr_clean.shape[1]):
        col_vals = arr_clean[:, col]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            mean_val = np.nanmean(col_vals)
            col_vals[nan_mask] = mean_val
            arr_clean[:, col] = col_vals

    return arr_clean.astype(float)


# ==========================================================
# Convert labels to numeric & repair NaNs
# ==========================================================
def convert_labels_to_numeric(y, is_class):
    if y.dtype.kind == 'O':
        y= yes_no_to_numeric(y)
	
    y = np.array(y, dtype=float)

    nan_mask = np.isnan(y)

    if is_class:
        y[nan_mask] = 0
    else:
        mean_val = np.nanmean(y)
        y[nan_mask] = mean_val

    return y.astype(float)


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def convert_all_npy_to_numeric(datasets_folder="./datasets"):
    print("\n=== Converting Entire Pipeline to Numeric-Only .npy Files ===")

    for dataset_name in os.listdir(datasets_folder):
        dataset_path = os.path.join(datasets_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        print(f"\nDataset: {dataset_name}")

        # -----------------------------------------------------
        # Load X and y files
        # -----------------------------------------------------
        file_paths = {
            #"X_train": os.path.join(dataset_path, "X_train.npy"),
            #"X_test":  os.path.join(dataset_path, "X_test.npy"),
            "y_train": os.path.join(dataset_path, "y_train.npy"),
            "y_test":  os.path.join(dataset_path, "y_test.npy"),
        }

        # Load everything with allow_pickle=True
        data = {}
        for key, path in file_paths.items():
            if os.path.isfile(path):
                print(f"  Loading {key} ...")
                data[key] = np.load(path, allow_pickle=True)
            else:
                print(f"  ⚠ Missing file: {path}")
                data[key] = None

        # Skip incomplete datasets
        if data["y_train"] is None:
            print("  ⚠ Incomplete dataset, skipping.")
            continue

        # -----------------------------------------------------
        # Determine problem type
        # -----------------------------------------------------
        classification = is_classification(data["y_train"])
        print(f"  Type: {'Classification' if classification else 'Regression'}")

        # ~ # -----------------------------------------------------
        # ~ # Convert X arrays to numeric
        # ~ # -----------------------------------------------------
        # ~ print("  Converting X_train to numeric ...")
        # ~ data["X_train"] = convert_array_to_numeric(data["X_train"])

        # ~ print("  Converting X_test to numeric ...")
        # ~ data["X_test"]  = convert_array_to_numeric(data["X_test"])

        # -----------------------------------------------------
        # Convert labels
        # -----------------------------------------------------
        print("  Converting y_train ...")
        data["y_train"] = convert_labels_to_numeric(data["y_train"], classification)

        print("  Converting y_test ...")
        data["y_test"]  = convert_labels_to_numeric(data["y_test"], classification)

        # -----------------------------------------------------
        # Save cleaned arrays back (numeric-only)
        # -----------------------------------------------------
        print("  Saving cleaned numeric arrays ...")
        for key, arr in data.items():
            if arr is None:
                continue
            np.save(file_paths[key], arr.astype(float))

        print("  ✔ Dataset converted successfully")

    print("\n=== ALL DATASETS CLEANED AND NUMERIC-ONLY ===")


# ==========================================================
# Standalone run
# ==========================================================
if __name__ == "__main__":
    convert_all_npy_to_numeric("./datasets")
