import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Path to the datasets folder
BASE_FOLDER = r"local_datasets"

def process_array(arr):
    """
    - Detect categorical columns
    - Encode categorical columns to numbers
    - Replace NaN values with column mean
    """
    arr_clean = arr.copy()

    n_rows, n_cols = arr_clean.shape
    for col in range(n_cols):
        column_data = arr_clean[:, col]

        # Detect if column is categorical (dtype == object OR mixed)
        if column_data.dtype.kind in {"O", "U", "S"}:
            # Convert all to string before encoding
            str_col = column_data.astype(str)

            # Encode string categories
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(str_col)

            arr_clean[:, col] = encoded.astype(float)

        else:
            # Numeric: fix NaNs
            col_vals = column_data.astype(float)
            nan_mask = np.isnan(col_vals)

            if nan_mask.any():
                mean_val = np.nanmean(col_vals)
                col_vals[nan_mask] = mean_val

            arr_clean[:, col] = col_vals

    return arr_clean.astype(float)


def process_dataset(dataset_path):
    """
    Process X_train.npy and X_test.npy in one dataset folder.
    """
    for filename in ["X_train.npy", "X_test.npy"]:
        file_path = os.path.join(dataset_path, filename)
        if not os.path.isfile(file_path):
            print(f"⚠ Missing file: {file_path}")
            continue

        print(f"Processing: {file_path}")

        # Load array
        arr = np.load(file_path, allow_pickle=True)

        # Clean array
        cleaned_arr = process_array(arr)

        # Save back to the same file
        np.save(file_path, cleaned_arr)
        print(f"✔ Saved cleaned file: {file_path}")


def main():
    # Iterate over all dataset folders
    for dataset_name in os.listdir(BASE_FOLDER):
        dataset_path = os.path.join(BASE_FOLDER, dataset_name)

        if os.path.isdir(dataset_path):
            print(f"\n=== Processing dataset: {dataset_name} ===")
            process_dataset(dataset_path)


if __name__ == "__main__":
    main()
