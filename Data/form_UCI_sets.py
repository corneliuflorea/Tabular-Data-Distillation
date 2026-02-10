import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#BASE_FOLDER = r"uci_data_cache"
BASE_FOLDER = r"temp"


def clean_and_encode(X):
    """
    Clean missing numeric data and encode categorical columns (strings/objects).
    - Replace NaN values with column mean.
    - Encode categorical columns using LabelEncoder.
    Returns cleaned X (float matrix).
    """

    X_clean = X.copy()
    n_rows, n_cols = X_clean.shape

    for col in range(n_cols):
        column = X_clean[:, col]

        # Detect categorical (strings / object types)
        if column.dtype.kind in {"O", "U", "S"}:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(column.astype(str))
            X_clean[:, col] = encoded.astype(float)

        else:
            # Numeric column: replace NaN with mean
            col_numeric = column.astype(float)
            mask = np.isnan(col_numeric)
            if mask.any():
                col_mean = np.nanmean(col_numeric)
                col_numeric[mask] = col_mean

            X_clean[:, col] = col_numeric

    return X_clean.astype(float)


def process_dataset(dataset_path):
    """
    Processes one dataset folder:
    - Reads X.npy, Y.npy
    - Cleans X
    - Splits train/test
    - Saves X_train, X_test, Y_train, Y_test, train_idx, test_idx
    """

    X_path = os.path.join(dataset_path, "X.npy")
    Y_path = os.path.join(dataset_path, "Y.npy")

    if not (os.path.isfile(X_path) and os.path.isfile(Y_path)):
        print(f"⚠ Missing X.npy or Y.npy in {dataset_path}")
        return

    print(f"\n=== Processing dataset: {dataset_path} ===")

    # Load data
    X = np.load(X_path, allow_pickle=True)
    Y = np.load(Y_path, allow_pickle=True)

    # Clean feature data
    X_clean = clean_and_encode(X)

    # Generate shuffled indices and split
    indices = np.arange(len(X_clean))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, shuffle=True, random_state=42
    )

    # Split data
    X_train = X_clean[train_idx]
    X_test = X_clean[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # Save outputs
    np.save(os.path.join(dataset_path, "X_train.npy"), X_train)
    np.save(os.path.join(dataset_path, "X_test.npy"), X_test)
    np.save(os.path.join(dataset_path, "Y_train.npy"), Y_train)
    np.save(os.path.join(dataset_path, "Y_test.npy"), Y_test)
    np.save(os.path.join(dataset_path, "train_idx.npy"), train_idx)
    np.save(os.path.join(dataset_path, "test_idx.npy"), test_idx)

    print(f"✔ Saved train/test files in {dataset_path}")


def main():
    """
    Iterates over all dataset folders in uci_data_cache/ and processes them.
    """

    for dataset_name in os.listdir(BASE_FOLDER):
        dataset_path = os.path.join(BASE_FOLDER, dataset_name)
        if os.path.isdir(dataset_path):
            process_dataset(dataset_path)


if __name__ == "__main__":
    main()
