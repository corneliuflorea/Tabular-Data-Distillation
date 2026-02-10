import pandas as pd
import numpy as np
import os
import json
import requests
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Union, List, Any
from ucimlrepo import fetch_ucirepo 
import pdb

# ~ additional https://github.com/dida-do/eurocropsml

# --- Caching Configuration ---
DATA_CACHE_DIR = "uci_data_cache"

# Dictionary mapping dataset names to their UCI retrieval details
# 'header': 0 if the first row is the header, None if no header
# 'target_column_index': Index of the column containing the target variable (-1 for last)
# 'column_names': List of column names, if header=None. None if header=0.



# ~ bank_marketing = fetch_ucirepo(id=222) 
  
# ~ # data (as pandas dataframes) 
# ~ X = bank_marketing.data.features 
# ~ y = bank_marketing.data.targets 


UCI_DATASET_MAP: Dict[str, int] = {
    # Classification Datasets
    "Bank Marketing": 222, #good
    "mushroom ": 73, #good
    "phishing_websites": 327, #good
    "NATICUSdroid": 722,
    "Online_News_Popularity": 332, #good
    "Connect-4": 26,

 

    # "Regression Datasets
    "Beijing_PM25": 381, 
    "Power_Consumption_Tetouan": 849,
    "Appliances_Energy" : 374,
    "Apartment_for_Rent": 555,
    "Bike_Sharing": 275,
    "Parkinsons_Telemonitoring":189,
    
 
}


def get_cache_paths(dataset_name: str) -> Tuple[Path, Path, Path]:
    """Generates the paths for the cached data files."""
    # Replace spaces with underscores for directory names
    data_dir = Path(DATA_CACHE_DIR) / dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    return (
        data_dir / "X.npy",
        data_dir / "y.npy",
        data_dir / "metadata.json"
    )
def get_cache_paths_dataset(dataset_name: str) -> Tuple[Path, Path, Path]:
    """Generates the paths for the cached data files."""
    # Replace spaces with underscores for directory names
    data_dir = Path(DATA_CACHE_DIR) / dataset_name.replace(" ", "_").replace("(", "").replace(")", "")
    return (
        data_dir / "dataset.npy",
    )

def load_from_cache(dataset_name: str) -> Union[Tuple[pd.DataFrame, pd.Series, str], None]:
    """Attempts to load features, target, and metadata from the local cache."""
    x_path, y_path, meta_path = get_cache_paths(dataset_name)
    
    if x_path.exists() and y_path.exists() and meta_path.exists():
        
        try:
            # 1. Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # 2. Load arrays
            X_array = np.load(x_path, allow_pickle=True)
            y_array = np.load(y_path, allow_pickle=True)
            
            # 3. Reconstruct DataFrame and Series
            X = pd.DataFrame(X_array, columns=metadata['feature_names'])
            y = pd.Series(y_array, name=metadata['target_name'])
            
            
            # Ensure categorical features are correctly typed and stripped of whitespace
            for col in X.columns:
                 if X[col].dtypes.name in ['object', 'category']:
                     # Clean whitespace and convert to category
                     X[col] = X[col].astype(str).str.strip().astype('category')
                     
            print(f"Successfully loaded '{dataset_name}' from local cache.")
            return X, y, metadata['target_name']
        
        except Exception as e:
            print(f"Error loading '{dataset_name}' from cache: {e}. Will attempt re-download.")
            # If cache files are corrupt, delete them to force a fresh download
            if x_path.exists(): os.remove(x_path)
            if y_path.exists(): os.remove(y_path)
            if meta_path.exists(): os.remove(meta_path)
            return None
            
    return None


def save_to_numpy(dataset_name, X_train, X_test, y_train, y_test, train_indices, test_indices):
    """Saves features, target, and metadata to the local cache."""
    x_path = get_cache_paths_dataset(dataset_name)
    data_dir = x_path[0].parent
    
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        
           
        # 2. Save arrays
        np.save(x_path, X_train, X_test, y_train, y_test, train_indices, test_indices)
        
        print(f"Successfully saved '{dataset_name}' to cache at {data_dir}.")

    except Exception as e:
        print(f"Warning: Failed to save '{dataset_name}' to cache. Error: {e}")

def save_to_cache(X: pd.DataFrame, y: pd.Series, dataset_name: str) -> None:
    """Saves features, target, and metadata to the local cache."""
    x_path, y_path, meta_path = get_cache_paths(dataset_name)
    data_dir = x_path.parent
    # pdb.set_trace()
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Prepare metadata
        # Store categorical columns as objects/strings for correct NumPy save/load
        X_save = X.copy()
        for col in X_save.select_dtypes(include=['category']).columns:
             X_save[col] = X_save[col].astype(str)
             
        metadata = {'feature_names': list(X_save.columns)  }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        # 2. Save arrays
        np.save(x_path, X_save.to_numpy())
        np.save(y_path, y.to_numpy())
        
        print(f"Successfully saved '{dataset_name}' to cache at {data_dir}.")

    except Exception as e:
        print(f"Warning: Failed to save '{dataset_name}' to cache. Error: {e}")

def fetch_uci_data(dataset_name: str, dataset_id:int) -> Union[Tuple[pd.DataFrame, pd.Series, str], None]:
	"""Downloads data from UCI, parses it, and separates features (X) from target (y)."""

	dataset = fetch_ucirepo(id=dataset_id) 
	# ~ # data (as pandas dataframes) 
	if dataset_id == 555:
		
		X = dataset.data.features.iloc[:, [0] + [3]+ [4] + [5] + [7] + [8] + [9] + [12]+ [13] + [15] + [16] + [17] +[18] + [19] ]
		y = dataset.data.features.iloc[:, [10] ]
	else:
		X = dataset.data.features 
		y = dataset.data.targets 
	# ~ pdb.set_trace()
            
	return X, y, dataset.variables



def load_and_split_uci_data(
    dataset_name: str,
    dataset_id: int,
    test_size: float = 0.2,
    random_state: int = 42,
    normalize_target: bool = True
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray],
    None
]:
    """
    Loads a specified dataset from UCI (or local cache), preprocesses it, and
    splits it into training and testing sets along with the original indices.
    """
    if dataset_name not in UCI_DATASET_MAP:
        print(f"Error: Dataset '{dataset_name}' is not in the supported UCI list.")
        print("Available datasets:", list(UCI_DATASET_MAP.keys()))
        return None

    # 1. Attempt to load from cache
    loaded_data = load_from_cache(dataset_name)
    if loaded_data:
        X, y, target_name = loaded_data
    else:
        # 2. If cache failed, download from UCI
        result = fetch_uci_data(dataset_name, dataset_id)
        if result is None:
            return None
            
        X, y, target_name = result
        
        # 3. Cache the newly downloaded data (before extensive preprocessing)
        save_to_cache(X, y, dataset_name)


    # --- Preprocessing Steps (Applied to either cached or freshly downloaded data) ---
    
    # 1. Convert object/string columns to categorical type
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
        
    # 2. Handle Categorical Target Conversion (Classification)
    
    is_classification = False
    if (normalize_target and is_classification) and y.any().dtypes.name in ['object', 'category']:
        # Convert categorical targets to numeric codes (0, 1, 2, ...)
        y = y.astype('category').cat.codes
        print(f"Target '{target_name}' converted to numerical codes (dtype: {y.dtype}).")
    
    # 3. Handle missing values
    
    # Drop columns with > 30% missing
    threshold = 0.7 * X.shape[0]
    X.dropna(axis=1, thresh=threshold, inplace=True)
    
    # Simple imputation for remaining numerical NaNs (replace with mean)
    for col in X.select_dtypes(include=np.number).columns:
        X[col].fillna(X[col].mean(), inplace=True)
        
    # Simple imputation for remaining categorical NaNs (replace with most frequent)
    for col in X.select_dtypes(include=['category']).columns:
        X[col].fillna(X[col].mode()[0], inplace=True)
        
    # Final check: remove any rows where target is NaN (should be rare)
    # ~ pdb.set_trace()
    mask = np.isnan(y.to_numpy())
    
    if mask.any():
        print(f"Warning: Dropping {mask.sum()} rows with missing target values.")
        X = X[~mask].reset_index(drop=True)
        y = y[~mask].reset_index(drop=True)
        
    # Create an index array before splitting
    original_indices = X.index.to_numpy() # Use the DataFrame index after cleaning

    # --- Splitting Data ---
    print("Splitting data into train (80%) and test (20%) sets...")
    
    # Determine stratification based on the processed target
    stratify_data =  None
    
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, original_indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_data
    )

    print(f"\nSuccessfully processed '{dataset_name}'.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Target variable name: {target_name}")
    print(f"Train indices array shape: {train_indices.shape}, Test indices array shape: {test_indices.shape}")

    return X_train, X_test, y_train, y_test, train_indices, test_indices


# --- Example Usage ---

if __name__ == '__main__':
    print("--- Caching All Available Datasets ---")
    
    # Attempt to load and cache ALL datasets
    # ~ pdb.set_trace()
    for name in UCI_DATASET_MAP.keys():
        print("="*50)
        # Call the loader function which handles cache check, download, and save
        
        X_train, X_test, y_train, y_test, train_indices, test_indices =	load_and_split_uci_data(name, dataset_id = UCI_DATASET_MAP[name], normalize_target=False)
        
        save_to_numpy(name, X_train, X_test, y_train, y_test, train_indices, test_indices)
    print("="*50)
    print("All datasets have been processed and cached (or loaded from cache).")
    print("\n--- Running Example Scenarios ---")


