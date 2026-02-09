import numpy as np
from sklearn.decomposition import PCA


# =====================================
# DIMENSION REDUCTION ENGINE
# =====================================

def reduce_dim(
    X,
    variance_threshold=0.90,   # keep 90% info
    max_components=20,
    model=None
):
    """
    Smart PCA reducer.

    Parameters:
    ----------------
    X : np.array
    variance_threshold : float
        target explained variance
    max_components : int
        hard cap to avoid over-reduction
    model : PCA object (optional)
        reuse for live trading later

    Returns:
    ----------------
    X_reduced
    model
    """

    # reuse existing PCA (IMPORTANT for live later)
    if model is not None:
        return model.transform(X), model

    # fit PCA
    pca_full = PCA()
    pca_full.fit(X)

    cumulative = np.cumsum(pca_full.explained_variance_ratio_)

    # auto choose components
    n_components = np.searchsorted(cumulative, variance_threshold) + 1

    # safety cap
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components)

    X_reduced = pca.fit_transform(X)

    print(f"[Dimensionality] Reduced {X.shape[1]} → {n_components} dimensions")
    print(f"[Dimensionality] Variance kept: {cumulative[n_components-1]:.2%}")

    return X_reduced, pca


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("DIMENSIONALITY ENGINE TEST")

    np.random.seed(42)

    # fake high-dim data
    X = np.random.normal(0, 1, (1500, 25))

    X_reduced, model = reduce_dim(X)

    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)

    print("PASSED ✅")
