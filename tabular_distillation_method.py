#=======================================================================
 # ~ Contains the distillation methods:
 # ~ Input: X, y percentage
 # ~ Output: X_syn, y_syn
	# ~ Distillation methods here:
	# ~ "K_means": distill_kmeans,
    # ~ "Coreset": distill_coreset, #Gonzales algorithm
    # ~ "CTgan": distill_ctgan,
	# ~ "TVAE": distill_tvae,
	# ~ "Gauss_cop": distill_Gauss_cop,
    # ~ "Core_lev_score": distill_coreset_leverage_scores, 
	# ~ "GMM": gmm_distill_tabular,
#=======================================================================

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Generative models (CTGAN, TVAE)
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
import torch

import pdb

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore")

    
def distill_kmeans(X, y, percentage, **kwargs):
    """
    Clustering-based distillation using KMeans.
    Creates n_samples cluster centroids.
    """
    n_samples = int(len(X) * percentage / 100)
    kmeans = KMeans(n_clusters=n_samples, random_state=0)
    kmeans.fit(X)

    X_distilled = kmeans.cluster_centers_  

    # assign labels by nearest centroid
    clf = NearestCentroid()
    clf.fit(X, y)
    y_distilled = clf.predict(X_distilled)

    return X_distilled, pd.Series(y_distilled)

#######################################################################
#-----------------------------------------------------------------------
#######################################################################
    
def distill_coreset(X, y, percentage, **kwargs):
    """
    Core-set selection using greedy K-center.
    Picks real points that maximize coverage.
    """
    n_samples = int(len(X) * percentage / 100)
    # ~ X_np = X.to_numpy()

    # Pick a random start index
    idx = [np.random.randint(0, len(X))]
    dist = np.full(len(X), np.inf)

    for _ in range(n_samples - 1):
        # Update distances to nearest selected center
        new_center = X[idx[-1]]
        dist = np.minimum(dist, np.linalg.norm(X - new_center, axis=1))

        # Choose the farthest point
        idx.append(np.argmax(dist))

    return X[idx].copy(), y[idx].copy()
    
#######################################################################
#-----------------------------------------------------------------------
#######################################################################  
     
def distill_coreset_leverage_scores(X, y, percentage):
    """
    Core-set selection using leverage score via PCA.
    """
    k = int(len(X) * percentage / 100)
    
    X_np = X.to_numpy() if hasattr(X, "to_numpy") else X

    pca = PCA(n_components=min(20, X_np.shape[1]))
    Z = pca.fit_transform(X_np)

    leverage = np.sum(Z ** 2, axis=1)
    probs = leverage / leverage.sum()

    idx = np.random.choice(len(X_np), size=k, replace=False, p=probs)
    return X_np[idx], y[idx], idx

#######################################################################
#-----------------------------------------------------------------------
#######################################################################

def numpy_to_sdv_dataframe(X, y):
    df = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df

def build_metadata(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata



def distill_ctgan(X, y, percentage, model_type="CTGAN", **kwargs):
    """
    Distillation using SDV 1.12+ API (CTGANSynthesizer or TVAESynthesizer).
    """
    n_samples = int(len(X) * percentage / 100)
    
	# Combine into one table with metadata
    df = numpy_to_sdv_dataframe(X, y)
    metadata = build_metadata(df)
    
    model = CTGANSynthesizer(metadata, enforce_min_max_values=True,	epochs=200,	batch_size=512,	pac=8,	cuda=True)
    
    # Fit and sample
    model.fit(df)
    synthetic_data = model.sample(n_samples)

    # Separate again
    X_syn = synthetic_data.drop(columns=['target']).to_numpy(dtype=float)
    y_syn = synthetic_data['target'].to_numpy()

    return X_syn, y_syn


def distill_tvae(X, y, percentage, model_type="CTGAN", **kwargs):
    """
    Distillation using SDV 1.12+ API (CTGANSynthesizer or TVAESynthesizer).
    """
    n_samples = int(len(X) * percentage / 100)
    
    # Combine into one table with metadata
    df = numpy_to_sdv_dataframe(X, y)
    metadata = build_metadata(df)


    model = TVAESynthesizer(metadata)
			
    # Fit and sample
    model.fit(df)
    synthetic_data = model.sample(n_samples)

    # Separate again
    X_syn = synthetic_data.drop(columns=['target']).to_numpy(dtype=float)
    y_syn = synthetic_data['target'].to_numpy()

    return X_syn, y_syn


#######################################################################
#-----------------------------------------------------------------------
#######################################################################


def gmm_distill_tabular(X, y, percentage,  random_state=0):
    """
    Fit a Gaussian Mixture Model to tabular data and generate a reduced/distilled dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    n_components : int
        Number of Gaussian mixture components to fit
    distilled_size : int
        Number of synthetic samples to generate
    random_state : int
        Seed

    Returns
    -------
    X_syn : np.ndarray
        Distilled synthetic features
    y_syn : np.ndarray
        Distilled synthetic labels (sampled from empirical label distribution)
    """
    distilled_size = int(len(X) * percentage / 100)
    n_components=10,
	
    # Scale data to improve GMM stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state
    )

    gmm.fit(X_scaled)

    # Sample synthetic feature vectors
    X_syn_scaled, _ = gmm.sample(distilled_size)

    # Inverse-transform back to original feature space
    X_syn = scaler.inverse_transform(X_syn_scaled)

    # Sample synthetic labels using empirical distribution of y
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    y_syn = np.random.choice(unique, size=distilled_size, p=probs)

    return X_syn, y_syn
