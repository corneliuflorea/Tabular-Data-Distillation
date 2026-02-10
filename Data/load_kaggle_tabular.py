import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
import pdb


# ---------------------------------------------------------
# 1. DEFINE KAGGLE TABULAR DATASETS TO USE
# ---------------------------------------------------------


DATASETS = {
    # Classification 
    "titanic": ("heptapod/titanic","titanic.csv"),
    "telco_churn": ("blastchar/telco-customer-churn","telco_churn.csv"),
    "heart": ("fedesoriano/heart-failure-prediction","heart.csv"),
    "stroke": ("fedesoriano/stroke-prediction-dataset","stroke.csv"),
    "adult": ("uciml/adult-census-income","adult.csv"),
    "diabetes": ("uciml/pima-indians-diabetes-database","diabetes.csv"),
    "breast_cancer": ("uciml/breast-cancer-wisconsin-data","breast_cancer.csv"),
    "mobile_price": ("iabhishekofficial/mobile-price-classification","mobile_price.csv"),
 	"credit_card_fraud": ( "mlg-ulb/creditcardfraud",  "creditcard.csv" ), # 284k rows (you can subsample to 50k)
    "stroke_prediction": ("fedesoriano/stroke-prediction-dataset", "stroke_prediction.csv"    ),     # 5,110 rows
    "heart_failure": ( "andrewmvd/heart-failure-clinical-data", "heart_failure.csv" ),   # 299 rows (small, optional)
    "hotel_bookings": ("jessemostipak/hotel-booking-demand", "hotel_bookings.csv" ),                    # 119k rows; subsample to 50k
    "crop_recommendation": ("atharvaingle/crop-recommendation-dataset", "Crop_recommendation.csv"),                # 2,200 rows
    

    # Regression 
    "insurance_cost": ("noordeen/insurance-premium-prediction","insurance_cost.csv")
}




DATA_DIR = "local_datasets"


# ---------------------------------------------------------
# 2. DOWNLOAD ALL DATASETS VIA KAGGLE API
# ---------------------------------------------------------

def download_all_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    for name, (kaggle_path, _) in DATASETS.items():
        print(f"Downloading dataset: {name}")
        target_dir = os.path.join(DATA_DIR, name)
        os.makedirs(target_dir, exist_ok=True)
        api.dataset_download_files(kaggle_path, path=target_dir, unzip=True)

    print("\nAll datasets downloaded successfully.")


# ---------------------------------------------------------
# 3. LOAD A SINGLE DATASET FROM LOCAL STORAGE
# ---------------------------------------------------------

def load_raw_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    _, csv_name = DATASETS[name]
    csv_path = os.path.join(DATA_DIR, name, csv_name)

    if not os.path.exists(csv_path):
        # pdb.set_trace()
        raise FileNotFoundError(f"Dataset {name} is not downloaded.")

    df = pd.read_csv(csv_path)

    # Automatic target detection (simple rule)
    # You may customize per dataset
    if "target" in df.columns:
        y = df["target"]
        X = df.drop(columns=["target"])
    elif "Outcome" in df.columns:
        y = df["Outcome"]
        X = df.drop(columns=["Outcome"])
    elif "Survived" in df.columns:
        y = df["Survived"]
        X = df.drop(columns=["Survived"])
    elif "quality" in df.columns:
        y = df["quality"]
        X = df.drop(columns=["quality"])
    elif "cnt" in df.columns:
        y = df["cnt"]
        X = df.drop(columns=["cnt"])
    else:
        # fallback: last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    return X, y


# ---------------------------------------------------------
# 4. SAVE AS NUMPY ARRAYS
# ---------------------------------------------------------

def save_numpy(name, X_train, X_test, y_train, y_test, train_idx, test_idx):
    np_dir = os.path.join(DATA_DIR, name)
    os.makedirs(np_dir, exist_ok=True)

    np.save(os.path.join(np_dir, "X_train.npy"), X_train)
    np.save(os.path.join(np_dir, "X_test.npy"), X_test)
    np.save(os.path.join(np_dir, "y_train.npy"), y_train)
    np.save(os.path.join(np_dir, "y_test.npy"), y_test)
    np.save(os.path.join(np_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(np_dir, "test_idx.npy"), test_idx)


# ---------------------------------------------------------
# 5. LOAD NUMPY ARRAYS (OFFLINE)
# ---------------------------------------------------------

def load_numpy(name):
    np_dir = os.path.join(DATA_DIR, name)

    X_train = np.load(os.path.join(np_dir, "X_train.npy"))
    X_test = np.load(os.path.join(np_dir, "X_test.npy"))
    y_train = np.load(os.path.join(np_dir, "y_train.npy"))
    y_test = np.load(os.path.join(np_dir, "y_test.npy"))
    train_idx = np.load(os.path.join(np_dir, "train_idx.npy"))
    test_idx = np.load(os.path.join(np_dir, "test_idx.npy"))

    return X_train, X_test, y_train, y_test, train_idx, test_idx


# ---------------------------------------------------------
# 6. MAIN FUNCTION: DOWNLOAD MODE vs LOAD MODE
# ---------------------------------------------------------

def get_dataset(name, download=False, test_size=0.2, random_state=42):
    """
    download=True  → downloads dataset + saves numpy version (first run)
    download=False → loads numpy offline (subsequent runs)
    """

    if download:
        print(f"\nDownloading raw dataset: {name}")
        X, y = load_raw_dataset(name)

        # convert to numpy
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # split
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_np,
            y_np,
            np.arange(len(X_np)),
            test_size=test_size,
            random_state=random_state,
            stratify=y_np if len(np.unique(y_np)) < 20 else None
        )

        # save offline copies
        save_numpy(name, X_train, X_test, y_train, y_test, train_idx, test_idx)

        print(f"Dataset {name} saved locally.")
        return X_train, X_test, y_train, y_test, train_idx, test_idx

    else:
        print(f"\nLoading dataset offline: {name}")
        return load_numpy(name)

def get_dataset_first(name, download=False, test_size=0.2, random_state=42):
    """
    download=True  → downloads dataset + saves numpy version (first run)
    download=False → loads numpy offline (subsequent runs)
    """


    print(f"\nDownloading raw dataset: {name}")
    X, y = load_raw_dataset(name)

    # convert to numpy
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # split
    
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_np,
        y_np,
        np.arange(len(X_np)),
        test_size=test_size,
        random_state=random_state,
        stratify=y_np if len(np.unique(y_np)) < 20 else None
    )

    # save offline copies
    save_numpy(name, X_train, X_test, y_train, y_test, train_idx, test_idx)

    print(f"Dataset {name} saved locally.")
    return X_train, X_test, y_train, y_test, train_idx, test_idx



# ---------------------------------------------------------
# 7. STANDALONE MAIN EXECUTION
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download all datasets and save numpy versions.")
    parser.add_argument("--dataset", type=str, default="titanic",
                        choices=DATASETS.keys(),
                        help="Dataset name to load or download.")
    parser.add_argument("--datasets_save", action="store_true", help="All Datasets have been downloaded and now are saved as numpy.")
    args = parser.parse_args()

    if args.download:
        download_all_datasets()
    
    if args.datasets_save:
        for d_name, (kaggle_path, _) in DATASETS.items():
            # pdb.set_trace()
            X_train, X_test, y_train, y_test, train_idx, test_idx = get_dataset_first(d_name)
    
        

    X_train, X_test, y_train, y_test, train_idx, test_idx = get_dataset(
        args.dataset, download=args.download
    )

    print("\nLoaded shapes:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)


if __name__ == "__main__":
    main()
