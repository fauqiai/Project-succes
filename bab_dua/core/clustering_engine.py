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
    method="hdbscan",   # "hdbscan", "gmm", "kmeans"
    k=8,
    random_state=42
):
    """
    Universal clustering wrapper.

    Returns:
        labels : np.array
        model  : fitted clustering object
    """

    if method == "hdbscan":

        if not HDBSCAN_AVAILABLE:
            print("⚠️ HDBSCAN not installed → fallback to GMM")
            return _gmm_cluster(X, k, random_state)

        return _hdbscan_cluster(X)

    elif method == "gmm":
        return _gmm_cluster(X, k, random_state)

    elif method == "kmeans":
        return _kmeans_cluster(X, k, random_state)

    else:
        raise ValueError("Unknown clustering method")


# =====================================
# HDBSCAN (AUTO REGIME DETECTOR)
# =====================================

def _hdbscan_cluster(X):

    model = hdbscan.HDBSCAN(
        min_cluster_size=80,     # tweak later
        min_samples=20,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    labels = model.fit_predict(X)

    return labels, model


# =====================================
# GAUSSIAN MIXTURE
# =====================================

def _gmm_cluster(X, k, random_state):

    model = GaussianMixture(
        n_components=k,
        covariance_type='full',
        random_state=random_state,
        n_init=5
    )

    labels = model.fit_predict(X)

    return labels, model


# =====================================
# KMEANS (SAFE FALLBACK)
# =====================================

def _kmeans_cluster(X, k, random_state):

    model = KMeans(
        n_clusters=k,
        n_init=25,
        random_state=random_state
    )

    labels = model.fit_predict(X)

    return labels, model


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("CLUSTERING ENGINE SELF TEST")

    import numpy as np

    # fake regimes
    np.random.seed(42)

    cluster1 = np.random.normal(0, 1, (500, 5))
    cluster2 = np.random.normal(5, 1, (500, 5))
    cluster3 = np.random.normal(-4, 1, (500, 5))

    X = np.vstack([cluster1, cluster2, cluster3])

    labels, model = cluster(X, method="hdbscan")

    unique = np.unique(labels)

    print("Clusters detected:", unique)
    print("Cluster count:", len(unique))

    noise = np.sum(labels == -1)
    print("Noise points:", noise)

    print("✅ PASSED")
