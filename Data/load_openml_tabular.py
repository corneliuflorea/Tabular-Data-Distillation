import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

DATA_DIR = "local_datasets"
META_FILE = os.path.join(DATA_DIR, "datasets_info.json")

os.makedirs(DATA_DIR, exist_ok=True)

###############################################################################
# 1. LIST OF DATASETS (only those that should exist on OpenML)
###############################################################################

# Key: user-friendly name
# Value: dict with OpenML identifier information
DATASETS = {
    "adult": {"openml_name": "adult", "version": 2},
    "bank_marketing": {"openml_name": "BankMarketing", "version": 1},
    # "letter": {"openml_name": "letter", "version": 1}, #not found
    # "spambase": {"openml_name": "spambase", "version": 1}, #not found
    # "magic": {"openml_name": "MagicTelescope", "version": 1}, #not found
    # "credit_default": {"openml_name": "default-of-credit-card-clients", "version": 1}, #not found
    # "landsat": {"openml_name": "satimage", "version": 1}, #not found
    # "coil2000": {"openml_name": "coil2000", "version": 1}, #not found
    # "page_blocks": {"openml_name": "page-blocks", "version": 1}, #not found
    # "segmentation": {"openml_name": "segment", "version": 1}, #not found
    # "optical_digits": {"openml_name": "optical", "version": 1}, #not found
    # "shuttle": {"openml_name": "shuttle", "version": 1}, #not found
    # "promoters": {"openml_name": "promoters", "version": 1}, #not found
    # "sensorless": {"openml_name": "Sensorless", "version": 2}, #not found
    "connect4": {"openml_name": "connect-4", "version": 1},
    # Regression:
    # "bike_sharing_daily": {"openml_name": "Bike_Sharing_Dataset", "version": 2}, #not found
    "wine_quality": {"openml_name": "wine-quality-red", "version": 1}, #not found
    "airfoil": {"openml_name": "AirfoilSelfNoise", "version": 1}, #not found
	
}

###############################################################################
# 2. DOWNLOAD ALL AVAILABLE DATASETS ONCE
###############################################################################

def download_all_datasets():
    """Download all datasets once, store as Parquet offline files."""
    info = {}

    for name, meta in DATASETS.items():
        print(f"=== Processing: {name} ===")

        # Mark datasets known to be unavailable
        if meta.get("available") is False:
            print(f"    -> Marked unavailable on OpenML (skipping): {name}")
            info[name] = {"status": "unavailable"}
            continue

        try:
            print("    Downloading from OpenML ...")
            dataset = fetch_openml(
                name=meta["openml_name"],
                version=meta["version"],
                as_frame=True#,
                #parser="auto"
            )

            X = dataset.data
            y = dataset.target

            # save locally
            save_path = os.path.join(DATA_DIR, f"{name}.parquet")
            full_df = X.copy()
            full_df["__target__"] = y
            full_df.to_parquet(save_path)

            info[name] = {
                "status": "downloaded",
                "path": save_path,
                "n_samples": len(full_df),
                "n_features": full_df.shape[1] - 1
            }

            print(f"    Saved locally to: {save_path}")

        except Exception as e:
            print(f"    ERROR downloading {name}: {e}")
            info[name] = {
                "status": "failed",
                "error": str(e)
            }

    with open(META_FILE, "w") as f:
        json.dump(info, f, indent=4)

    print("\n=== DOWNLOAD FINISHED ===")
    print("Metadata saved to:", META_FILE)


###############################################################################
# 3. LOAD OFFLINE DATASET + TRAIN/TEST + INDICES
###############################################################################

def load_dataset_offline(name, test_size=0.2, random_state=42):
    """
    Load dataset from local storage (offline only).
    Returns:
        X_train, X_test, y_train, y_test, train_idx, test_idx
    """
    meta = json.load(open(META_FILE))

    if name not in meta:
        raise ValueError(f"Dataset '{name}' not found in metadata!")

    entry = meta[name]

    if entry.get("status") != "downloaded":
        raise ValueError(f"Dataset '{name}' is not available offline (status={entry.get('status')}).")

    df = pd.read_parquet(entry["path"])

    X = df.drop(columns=["__target__"])
    y = df["__target__"]

    train_idx, test_idx = train_test_split(
        X.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y if y.nunique() > 1 else None
    )

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    return X_train, X_test, y_train, y_test, train_idx.to_list(), test_idx.to_list()


###############################################################################
# 4. Download everything once
###############################################################################

if __name__ == "__main__":
    download_all_datasets()

    # Example usage offline:
    # X_train, X_test, y_train, y_test, idx_tr, idx_te = load_dataset_offline("adult")
    # print(X_train.shape)
