import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# =====================================
# MAIN CLUSTER WRAPPER
# =====================================

def cluster(
    X,
    method="hdbscan",
    k=8,
    random_state=42
):

    if method == "hdbscan":

        if not HDBSCAN_AVAILABLE:
            print("⚠️ HDBSCAN not installed → fallback to GMM")
            return _gmm_cluster(X, k, random_state)

        labels, model = _hdbscan_cluster(X)

        unique = np.unique(labels)

        # 🔥 AUTO FAILSAFE
        if len(unique) <= 1:
            print("⚠️ HDBSCAN found no clusters → auto fallback to GMM")
            return _gmm_cluster(X, k, random_state)

        return labels, model

    elif method == "gmm":
        return _gmm_cluster(X, k, random_state)

    elif method == "kmeans":
        return _kmeans_cluster(X, k, random_state)

    else:
        raise ValueError("Unknown clustering method")


# =====================================
# HDBSCAN (FIXED)
# =====================================

def _hdbscan_cluster(X):

    n = len(X)

    # 🔥 adaptive sizing (VERY IMPORTANT)
    min_cluster_size = max(10, int(n * 0.01))   # 1% data
    min_samples = max(3, int(min_cluster_size * 0.25))

    print(f"HDBSCAN params → min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    labels = model.fit_predict(X)

    print("Clusters found:", np.unique(labels))

    return labels, model


# =====================================
# GAUSSIAN MIXTURE (SAFE)
# =====================================

def _gmm_cluster(X, k, random_state):

    model = GaussianMixture(
        n_components=min(k, max(2, len(X)//500)),
        covariance_type='full',
        random_state=random_state,
        n_init=5
    )

    labels = model.fit_predict(X)

    print("GMM clusters:", np.unique(labels))

    return labels, model


# =====================================
# KMEANS
# =====================================

def _kmeans_cluster(X, k, random_state):

    k = min(k, max(2, len(X)//500))

    model = KMeans(
        n_clusters=k,
        n_init=25,
        random_state=random_state
    )

    labels = model.fit_predict(X)

    print("KMeans clusters:", np.unique(labels))

    return labels, model
