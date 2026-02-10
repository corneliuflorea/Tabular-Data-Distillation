"""
Dataset experiments runner for Windows

Folder structure (expected):
./datasets/
    dataset_1/
        X_train.npy
        y_train.npy
        X_test.npy
        y_test.npy
    dataset_2/
        ...

Outputs:
 ./datasets/dataset_name/dataset_name_results.npy   (a dict saved with numpy allow_pickle=True)

Notes:
 - This script assumes X_*.npy are numeric feature matrices and have matching rows with y_*.npy.
 - If your labels are strings, the loader will label-encode them for models that need numeric labels.
"""

import os
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr
from xgboost import XGBClassifier, XGBRegressor
from typing import Tuple, Dict, Any
from joblib import Parallel, delayed


import pdb

BASE_FOLDER = os.path.join(".", "datasets/Classification/")
RANDOM_STATE = 42
log_file  = "basic_clasifiers_distil.log"



Key_string = {#"", 
		"_K_means_5", "_K_means_10" ,"_K_means_25" , "_K_means_50" ,
		"_Coreset_5" , 	"_Coreset_10" ,	"_Coreset_25" ,	"_Coreset_50" ,
		"_CTgan_5", "_CTgan_10" ,"_CTgan_25" , "_CTgan_50" ,
		"_TVAE_5" , 	"_TVAE_10" ,	"_TVAE_25" ,	"_TVAE_50" ,
		"_Gauss_cop_5" , 	"_Gauss_cop_10" ,	"_Gauss_cop_25" ,	"_Gauss_cop_50", 
		 "_Core_lev_score_5", "_Core_lev_score_10", "_Core_lev_score_25", "_Core_lev_score_50",
		 "_GMM_5", "_GMM_10", "_GMM_25", "_GMM_50",
		}


#string used to save the current results in the results.npy file. It should be one of "full", CTgan10
results_dictionary = {
		"" : "full",
		"_K_means_5" : "K_means_5",
		"_K_means_10" : "K_means_10",
		"_K_means_25" : "K_means_25",
		"_K_means_50" : "K_means_50",
		"_Coreset_5" : "Coreset_5",
		"_Coreset_10" : "Coreset_10",
		"_Coreset_25" : "Coreset_25",
		"_Coreset_50" : "Coreset_50",
		"_CTgan_5" : "CTgan_5",
		"_CTgan_10" : "CTgan_10",
		"_CTgan_25" : "CTgan_25",
		"_CTgan_50" : "CTgan_50",
		"_TVAE_5" : "TVAE_5",
		"_TVAE_10" : "TVAE_10",
		"_TVAE_25" : "TVAE_25",
		"_TVAE_50" : "TVAE_50",
		"_Gauss_cop_5" : "Gauss_cop_5",
		"_Gauss_cop_10" : "Gauss_cop_10",
		"_Gauss_cop_25" : "Gauss_cop_25",
		"_Gauss_cop_50" : "Gauss_cop_50",
		"_Core_lev_score_5" : "Core_lev_score_5",
		"_Core_lev_score_10" : "Core_lev_score_10",
		"_Core_lev_score_25" : "Core_lev_score_25",
		"_Core_lev_score_50" : "Core_lev_score_50",
		"_GMM_5" : "GMM_5",
		"_GMM_10" : "GMM_10",
		"_GMM_25" : "GMM_25",
		"_GMM_50" : "GMM_50",
			}

#=======================================================================
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#=======================================================================

# ---------------------------
# Utility / modular loader
# ---------------------------
def load_dataset(dataset_folder: str, key_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load X_train.npy, y_train.npy, X_test.npy, y_test.npy from dataset_folder.
    Returns: X_train, y_train, X_test, y_test, meta
    - If y_* are non-numeric (object/string), they will be label-encoded and the encoder stored in meta['y_encoder'].
    - meta contains original dtype info and number of samples/features.
    """
    def npy_path(name):
        return os.path.join(dataset_folder, name)
        
    X_Tr_file = "X_train" + key_str + ".npy"
    y_Tr_file = "Y_train" + key_str + ".npy"
    X_Ts_file = "X_test.npy" 
    y_Ts_file = "y_test.npy"

    X_tr = np.load(npy_path(X_Tr_file), allow_pickle=True)
    y_tr = np.load(npy_path(y_Tr_file), allow_pickle=True)
    X_te = np.load(npy_path(X_Ts_file), allow_pickle=True)
    y_te = np.load(npy_path(y_Ts_file), allow_pickle=True)

    meta = {
        "X_train_shape": X_tr.shape,
        "X_test_shape": X_te.shape,
        "y_train_shape": y_tr.shape,
        "y_test_shape": y_te.shape,
    }

    # Ensure arrays are 2D/1D oriented correctly
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)
    y_encoder = None
    
    # If labels have the wrong side correct them
    if len(y_tr.shape)>1:
        y_tr = y_tr.squeeze()
		
    if len(y_te.shape)>1:
        y_te = y_te.squeeze()

    # If labels are non-numeric, label-encode them for models that require numeric labels
    if y_tr.dtype.kind in ("U", "S", "O"):  # string/object
        y_encoder = LabelEncoder()
        y_tr_enc = y_encoder.fit_transform(y_tr.astype(str))
        y_te_enc = y_encoder.transform(y_te.astype(str))
        meta["y_encoded_from_strings"] = True
        meta["y_classes"] = list(y_encoder.classes_)
        y_tr = y_tr_enc
        y_te = y_te_enc
        meta["y_encoder"] = y_encoder
    else:
        # still check for float labels that are actually integers (e.g. 0.,1.,2.)
        meta["y_encoded_from_strings"] = False

    return X_tr, y_tr, X_te, y_te, meta
#=======================================================================
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#=======================================================================

# ---------------------------
# Classification vs Regression detector
# ---------------------------
def is_classification_target(y: np.ndarray, unique_threshold: int = 50, ratio_threshold: float = 0.01) -> bool:
    """
    Heuristic to decide whether y corresponds to classification or regression.
    Returns True for classification, False for regression.

    Heuristic:
    - If y dtype is integer-like (np.integer) -> classification if number of unique labels <= unique_threshold
    - If y dtype is float: treat as classification if the number of unique values is small relative to n_samples:
        unique_count <= max(unique_threshold, ratio_threshold * n_samples)
    - If y dtype is object/string -> classification
    - The thresholds can be tuned.

    This returns a boolean (True => classification).
    """
    n = len(y)
    unique_vals = np.unique(y)
    unique_count = len(unique_vals)

    # Object/string => classification
    if y.dtype.kind in ("U", "S", "O"):
        return True

    # Integer dtype
    if np.issubdtype(y.dtype, np.integer) or np.all(np.mod(y, 1) == 0):
        # if integer-like and not too many unique labels => classification
        if unique_count <= unique_threshold:
            return True
        # if many unique integers (near n) => probably regression-type ordinal continuous
        return False

    # Float dtype
    if np.issubdtype(y.dtype, np.floating):
        # if unique values are relatively few -> classification (discrete labels encoded as floats)
        cutoff = max(unique_threshold, int(np.ceil(ratio_threshold * n)))
        return unique_count <= cutoff

    # fallback
    return False
#=======================================================================
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#=======================================================================

# ---------------------------
# Scoring helpers
# ---------------------------
def classification_weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted accuracy = sum_{1/NClacc}* (accuracy_on_class_c) 
    This equals the overall accuracy when using the natural definition, but we compute explicitly per-class
    accuracies and weight them by class prevalence to keep the record of per-class performance.
    """
    labels = np.unique(y_true)
    total = 0.0
    n = len(y_true)
    for c in labels:
        mask = (y_true == c)
        n_c = mask.sum()
        if n_c == 0:
            continue
        acc_c = (y_pred[mask] == y_true[mask]).sum() / n_c
        
        total += acc_c 
    return total/len(labels)

def classification_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[Any, float]:
    labels = np.unique(y_true)
    out = np.zeros( len(labels) )
    i = 0
    for c in labels:
        mask = (y_true == c)
        n_c = mask.sum()
        val = round( float((y_pred[mask] == y_true[mask]).sum() / n_c) ,3)
        out[i] =  val if n_c > 0 else float("nan")
    return out

def regression_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # If constant predictions or truth, pearsonr may fail; handle gracefully
    try:
        r, _ = pearsonr(y_true, y_pred)
        return float(r)
    except Exception:
        return float("nan")

#=======================================================================
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#=======================================================================
def  get_metrics(y_test, y_pred, is_classif, params) : 
	if is_classif:
		acc = round(float(accuracy_score(y_test, y_pred)),4)
		wacc = round(float(classification_weighted_accuracy(y_test, y_pred)),4)
		per_class = classification_per_class_accuracy(y_test, y_pred)
			
		# ~ entry = {"params": params, "accuracy": acc, "weighted_accuracy": wacc, "per_class_accuracy": per_class}
		return acc, wacc, per_class
	else:
		mse = round(float(mean_squared_error(y_test, y_pred)),4)
		pear = round(regression_pearson(y_test, y_pred),4)
		if np.isnan(pear).any():
			pear = 0
		
		# ~ entry = {"params": params, "mse": mse, "pearson": pear}
		return mse, pear
	
	# ~ return entry


# ---------------------------
# Model runners: iterate grid and collect metrics
# ---------------------------


def run_one_param(idx, params, X_train, y_train, X_test, y_test, is_classif, num_labels, RANDOM_STATE):
    """Run a single (C, gamma) experiment."""
    if is_classif:
        model = SVC(kernel="rbf",
                    C=params["C"],
                    gamma=params["gamma"],
                    probability=False,
                    random_state=RANDOM_STATE)
    else:
        model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classif:
        acc, wacc, per_class = get_metrics(y_test, y_pred, is_classif, params)
        return idx, acc, wacc, per_class
    else:
        mse, pear = get_metrics(y_test, y_pred, is_classif, params)
        return idx, mse, pear


def run_svm_grid(X_train, y_train, X_test, y_test, is_classif: bool):
    """
    Parallel SVM with RBF kernel grid search using joblib.
    """
    C_exps = list(range(-5, 11, 4))      # -5, -1, 3, 7, 11
    gamma_exps = list(range(-15, 4, 4))  # -15, -11, -7, -3, 1, 3

    param_grid = [{"C": 2.0 ** ce, "gamma": 2.0 ** ge}
                  for ce in C_exps for ge in gamma_exps]

    num_params = len(param_grid)

    if is_classif:
        num_labels = len(np.unique(y_train))
        # Prepare result holders
        results_acc = np.zeros(num_params)
        results_wacc = np.zeros(num_params)
        results_per_class = np.zeros((num_params, num_labels))
    else:
        results_mse = np.zeros(num_params)
        results_pear = np.zeros(num_params)

    # ---- PARALLEL EXECUTION ----
    out = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_param)(
            idx=i,
            params=param_grid[i],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            is_classif=is_classif,
            num_labels=len(np.unique(y_train)) if is_classif else None,
            RANDOM_STATE=RANDOM_STATE
        )
        for i in range(num_params)
    )

    # ---- Collect results ----
    for item in out:
        if is_classif:
            idx, acc, wacc, per_class = item
            results_acc[idx] = acc
            results_wacc[idx] = wacc
            results_per_class[idx, :] = per_class
        else:
            idx, mse, pear = item
            results_mse[idx] = mse
            results_pear[idx] = pear

    # ---- Best selection ----
    if is_classif:
        results = {
            "acc": results_acc,
            "wacc": results_wacc,
            "per_class": results_per_class
        }
        best = {
            "acc": np.max(results_acc),
            "weight_acc": np.max(results_wacc)
        }
    else:
        results = {
            "mse": results_mse,
            "wacc": results_pear
        }
        best = {
            "mse": np.min(results_mse),
            "pearson": np.max(results_pear)
        }

    return {"all": results, "best": best}



def run_one_rf_param(idx, frac, mf, X_train, y_train, X_test, y_test, is_classif, num_labels, RANDOM_STATE):
    """Run one RF configuration."""
    params = {
        "n_estimators": 50,
        "max_samples": frac,
        "max_features": mf,
        "bootstrap": True
    }

    if is_classif:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            bootstrap=True,
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            bootstrap=True,
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classif:
        acc, wacc, per_class = get_metrics(y_test, y_pred, is_classif, params)
        return idx, acc, wacc, per_class
    else:
        mse, pear = get_metrics(y_test, y_pred, is_classif, params)
        return idx, mse, pear



def run_rf_grid(X_train, y_train, X_test, y_test, is_classif: bool):

    inbag_fracs = [1.0, 0.75, 0.5, 0.2, 0.1]
    max_features_opts = [1.0, 0.5, 0.2, "sqrt"]

    n_frac = len(inbag_fracs)
    n_mf = len(max_features_opts)
    total = n_frac * n_mf

    if is_classif:
        num_labels = len(np.unique(y_train))
        results_acc = np.zeros((n_frac, n_mf))
        results_wacc = np.zeros((n_frac, n_mf))
        results_per_class = np.zeros((total, num_labels))
    else:
        results_mse = np.zeros((n_frac, n_mf))
        results_pear = np.zeros((n_frac, n_mf))

    # Build job list (each with an index 0..total-1)
    job_list = []
    idx = 0
    for i, frac in enumerate(inbag_fracs):
        for j, mf in enumerate(max_features_opts):
            job_list.append((idx, frac, mf, i, j))
            idx += 1

    # ---- PARALLEL EXECUTION ----
    out = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_one_rf_param)(
            idx=k,
            frac=frac,
            mf=mf,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            is_classif=is_classif,
            num_labels=len(np.unique(y_train)) if is_classif else None,
            RANDOM_STATE=RANDOM_STATE
        )
        for (k, frac, mf, i, j) in job_list
    )

    # ---- COLLECT RESULTS ----
    # out items are in arbitrary order → need mapping to grid positions
    for n, (k, frac, mf, i, j) in enumerate(job_list):
        res = out[n]

        if is_classif:
            idx_result, acc, wacc, per_class = res
            results_acc[i, j] = acc
            results_wacc[i, j] = wacc
            results_per_class[idx_result, :] = per_class
        else:
            idx_result, mse, pear = res
            results_mse[i, j] = mse
            results_pear[i, j] = pear

    # ---- PICK BEST ----
    if is_classif:
        results = {
            "acc": results_acc,
            "wacc": results_wacc,
            "per_class": results_per_class
        }
        best = {
            "acc": np.max(results_acc),
            "weight_acc": np.max(results_wacc)
        }
    else:
        results = {
            "mse": results_mse,
            "wacc": results_pear
        }
        best = {
            "mse": np.min(results_mse),
            "pearson": np.max(results_pear)
        }

    return {"all": results, "best": best}


def fix_labels_for_xgb(y_train, y_test):
    # Convert to numpy
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Handle binary {-1, +1}
    if set(np.unique(y_train)).issubset({-1, 1, -1.0, 1.0}):
        y_train_fixed = (y_train == 1).astype(int)
        y_test_fixed  = (y_test  == 1).astype(int)
        return y_train_fixed, y_test_fixed

    # General case: build mapping from both train+test
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    all_labels_sorted = np.sort(all_labels)

    # Create mapping dict original_label → new_label_id; correct gap
    mapping = {label: new_id for new_id, label in enumerate(all_labels_sorted)}

    # Apply mapping
    y_train_fixed = np.vectorize(mapping.get)(y_train)
    y_test_fixed = np.vectorize(mapping.get)(y_test)

    return y_train_fixed, y_test_fixed


def run_xgb_grid(X_train, y_train, X_test, y_test, is_classif: bool):
    """
    XGBoost grid:
      - learning_rate: [0.5, 0.1, 0.05, 0.01]
      - min_child_weight: [1, 3, 5]
      - subsample: [0.6, 0.8, 1.0]
    Use n_estimators=100 (default), set random_state for reproducibility.
    """
    learning_rates = [0.5, 0.1, 0.05, 0.01]
    min_child_weights = [1, 3, 5]
    subsamples = [0.6, 0.8, 1.0]
    grid = list(ParameterGrid({"learning_rate": learning_rates, "min_child_weight": min_child_weights, "subsample": subsamples}))
    
    y_train, y_test = fix_labels_for_xgb(y_train, y_test)

    results = []
    
    if is_classif:
        num_labels = len(np.unique(y_train))
		
        results_acc = np.zeros( (len(grid) ))
        results_wacc = np.zeros( (len(grid) ))
        results_per_class = np.zeros( (len(grid), num_labels))
    else:
        results_mse = np.zeros( (len(grid) ))
        results_pear = np.zeros( (len(grid) ))

    ind = 0 
     
    for params in grid:
        if is_classif:
            model = XGBClassifier(
                objective="multi:softmax" if len(np.unique(y_train)) > 2 else "binary:logistic",
                use_label_encoder=False,
                eval_metric="mlogloss" if len(np.unique(y_train)) > 2 else "logloss",
                learning_rate=params["learning_rate"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                n_estimators=100,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc, wacc, per_class = get_metrics(y_test, y_pred, is_classif, params) 
            results_acc[ind] = acc
            results_wacc[ind] = wacc
            results_per_class[ind,:] = per_class
            
        else:
            model = XGBRegressor(
                learning_rate=params["learning_rate"],
                min_child_weight=params["min_child_weight"],
                subsample=params["subsample"],
                n_estimators=100,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse, pear = get_metrics(y_test, y_pred, is_classif, params)
            results_mse[ind] = mse
            results_pear[ind] = pear
        ind += 0                   



    if is_classif:
        results = {"acc": results_acc, "wacc": results_wacc, "per_class": results_per_class}
        best_acc = max(results_acc.flatten())
        best_wacc = max(results_wacc.flatten())
        best = {"acc": best_acc, "weight_acc": best_wacc}
    else:
        results = {"mse": results_mse, "wacc": results_pear}
        best_mse = min(results_mse.flatten())
        best_pear = max(results_pear.flatten()) 
        best = {"mse": best_mse, "pearson": best_pear}

    return {"all": results, "best": best}


# ---------------------------
# Full per-dataset pipeline
# ---------------------------
def process_one_dataset(dataset_path: str, key_string: str, results_string: str):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    print(f"\n=== Processing dataset: {dataset_name} ===")
    with open(log_file, "a") as f:
        print(f"\n=== Processing dataset: {dataset_name} ===", file=f)

    required_files = ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"]
    for f in required_files:
        if not os.path.exists(os.path.join(dataset_path, f)):
            print(f"  - Missing required file: {f}  -> skipping dataset.")
            return

    X_tr, y_tr, X_te, y_te, meta = load_dataset(dataset_path, key_string)
    meta["dataset_name"] = dataset_name

    # decide task type
    is_classif = is_classification_target(y_tr)
    meta["is_classification"] = bool(is_classif)
    print(f"  - Detected task type: {'Classification' if is_classif else 'Regression'}")
    with open(log_file, "a") as f:
        print(f"  - Detected task type: {'Classification' if is_classif else 'Regression'}", file=f)

    # Save to .npy file (allow_pickle=True)
    out_path = os.path.join(dataset_path, f"{dataset_name}_results.npy")
    if os.path.exists(out_path):
        results = np.load(out_path, allow_pickle=True)
    else:
        results = {}#np.array([])
    
    
    # Results structure with 'full' top-level key
    results = np.append(results, {results_string: {"meta": meta, "classifiers": {}}} )
    
    # SVM / SVR
    print("  - Running SVM grid ...")
    with open(log_file, "a") as f:
        print("  - Running SVM grid ...", file=f)
    
    svm_res = run_svm_grid(X_tr, y_tr, X_te, y_te, is_classif)
    results[len(results)-1][results_string]["classifiers"]["SVM"] = svm_res

    # Random Forest
    print("  - Running Random Forest grid ...")
    with open(log_file, "a") as f:
        print("  - Running Random Forest grid ...", file=f)
    rf_res = run_rf_grid(X_tr, y_tr, X_te, y_te, is_classif)
    results[len(results)-1][results_string]["classifiers"]["RF"] = rf_res

    # XGBoost
    print("  - Running XGBoost grid ...")
    with open(log_file, "a") as f:
        print("  - Running XGBoost grid  ...", file=f)
    
    xgb_res = run_xgb_grid(X_tr, y_tr, X_te, y_te, is_classif)
    results[len(results)-1][results_string]["classifiers"]["XGB"] = xgb_res
    # ~ pdb.set_trace()


    np.save(out_path, results, allow_pickle=True)
    print(f"  -> Saved results to: {out_path}")
   


# ---------------------------
# Main loop
# ---------------------------
def main():
    if not os.path.exists(BASE_FOLDER):
        raise FileNotFoundError(f"Base folder not found: {BASE_FOLDER}")

    dataset_folders = [os.path.join(BASE_FOLDER, d) for d in os.listdir(BASE_FOLDER) if os.path.isdir(os.path.join(BASE_FOLDER, d))]
    if not dataset_folders:
        print("No dataset folders found under", BASE_FOLDER)
        return

    for ks in Key_string:
        print(f"\n============= Distillation: {ks} ==============")
        with open(log_file, "a") as f:
            print(f"\n============= Distillation: {ks} ==============", file=f)
        for ds in dataset_folders:
            try:
               
                res_str = results_dictionary[ks]
                process_one_dataset(ds, ks, res_str)
            except Exception as e:
                # don't crash the whole loop for one dataset error
                print(f"Error while processing {ds}: {e}")


if __name__ == "__main__":
    main()
