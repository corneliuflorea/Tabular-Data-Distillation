import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


# =========================
# Configuration
# =========================

DATA_DIR = "./datasets/Classification/crop_recommendation/"   # <-- change this
METHOD_NAMES = ["K_means_50", "Coreset_50", "Core_lev_score_50", "Gauss_cop_50", "GMM_50", "CTgan_50", "TVAE_50"]	  # distilled methods
# ~ METHOD_NAMES = ["K_means_25", "Coreset_25", "Core_lev_score_25", "Gauss_cop_25", "GMM_25", "CTgan_25", "TVAE_25"]	  # distilled methods
# ~ METHOD_NAMES = ["K_means_5", "Coreset_5", "Core_lev_score_5", "Gauss_cop_5", "GMM_5", "CTgan_5", "TVAE_5"]	  # distilled methods

N_COMPONENTS = 2
TSNE_PERPLEXITY = 30
RANDOM_STATE = 42

OUTPUT_DIR = DATA_DIR + "/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Utility functions
# =========================

def load_dataset(data_dir, method=None):
    """
    Load X, Y for baseline or a distilled method.
    """
    if method is None:
        X = np.load(os.path.join(data_dir, "X_train.npy"))
        Y = np.load(os.path.join(data_dir, "y_train.npy"))
        name = "baseline"
    else:
        X = np.load(os.path.join(data_dir, f"X_train_{method}.npy"))
        Y = np.load(os.path.join(data_dir, f"Y_train_{method}.npy"))
        name = method

    return X, Y, name


def plot_embedding(Z, Y, title):
    """
    Scatter plot with class-based coloring.
    """
    labels = np.unique(Y)
    cmap = plt.cm.get_cmap("tab10", len(labels))

    for i, lbl in enumerate(labels):
        idx = (Y == lbl)
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=12,
            alpha=0.7,
            color=cmap(i),
            label=str(lbl)
        )

    plt.title(title)
    plt.legend(markerscale=1.5, fontsize=8)
    plt.axis("off")


def plot_embedding_and_save(Z, Y, title, filename):
    labels = np.unique(Y)
    cmap = plt.cm.get_cmap("tab10", len(labels))

    plt.figure(figsize=(5, 5))

    for i, lbl in enumerate(labels):
        idx = (Y == lbl)
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=15,
            alpha=0.7,
            color=cmap(i),
            label=str(lbl)
        )

    plt.title(title)
    plt.legend(markerscale=1.2, fontsize=8)
    plt.axis("off")

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# PCA Visualization
# =========================

# ~ Implementation:
# ~ 	- Fit PCA only on baseline
# ~ 	- Project distilled datasets using the same PCA
# ~ 	- Ensures geometric comparability

def visualize_pca(data_dir, method_names):
    print("Running PCA visualization")

    print("Saving PCA plots")

    X_base, Y_base, _ = load_dataset(data_dir)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z_base = pca.fit_transform(X_base)

    # Baseline
    plot_embedding_and_save(
        Z_base,
        Y_base,
        "PCA - Baseline",
        os.path.join(OUTPUT_DIR, "PCA_baseline.png")
    )

    # Distilled methods
    for method in method_names:
        X_m, Y_m, _ = load_dataset(data_dir, method)
        Z_m = pca.transform(X_m)

        plot_embedding_and_save(
            Z_m,
            Y_m,
            f"PCA - {method}",
            os.path.join(OUTPUT_DIR, f"PCA_{method}.png")
            )
		




# =========================
# UMAP Visualization
# =========================
# ~ Introduced by
# ~ McInnes, L., Healy, J. and Melville, J., 2018. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.
# Main idea: UMAP is fitted on the baseline dataset and used to embed distilled datasets into the same latent space, enabling direct geometric comparison

#needing pip install umap-learn
def visualize_umap(data_dir, method_names):
    print("Saving UMAP plots")

    X_base, Y_base, _ = load_dataset(data_dir)

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=RANDOM_STATE
    )

    Z_base = umap_model.fit_transform(X_base)

    plot_embedding_and_save(
        Z_base,
        Y_base,
        "UMAP - Baseline",
        os.path.join(OUTPUT_DIR, "UMAP_baseline.png")
    )

    for method in method_names:
        X_m, Y_m, _ = load_dataset(data_dir, method)
        Z_m = umap_model.transform(X_m)

        plot_embedding_and_save(
            Z_m,
            Y_m,
            f"UMAP - {method}",
            os.path.join(OUTPUT_DIR, f"UMAP_{method}.png")
        )




# =========================
# t-SNE Visualization
# =========================
# ~ Implementation:
		# ~ Run separately per dataset
		# ~ Same hyperparameters + random seed
		# ~ Visual comparison, not geometric alignment

def visualize_tsne(data_dir, method_names):
    print("Running t-SNE visualization")

    print("Saving t-SNE plots")

    datasets = [(None, "baseline")] + [(m, m) for m in method_names]

    for method, name in datasets:
        X, Y, _ = load_dataset(data_dir, method)

        tsne = TSNE(
            n_components=2,
            perplexity=TSNE_PERPLEXITY,
            random_state=RANDOM_STATE,
            init="pca",
            learning_rate="auto"
        )

        Z = tsne.fit_transform(X)

        plot_embedding_and_save(
            Z,
            Y,
            f"t-SNE - {name}",
            os.path.join(OUTPUT_DIR, f"TSNE_{name}.png")
        )


# =========================
# Main
# =========================

if __name__ == "__main__":
    visualize_pca(DATA_DIR, METHOD_NAMES)
    visualize_tsne(DATA_DIR, METHOD_NAMES)
    visualize_umap(DATA_DIR, METHOD_NAMES)
