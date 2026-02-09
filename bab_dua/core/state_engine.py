import pandas as pd
from sklearn.preprocessing import StandardScaler

# NEW
from .clustering_engine import cluster


# =====================================
# BUILD STATE MATRIX
# =====================================

def build_state_matrix(features, regimes, scaler=None):
    """
    Combine features + regimes
    Scale them for clustering
    """

    state = pd.concat([features, regimes], axis=1).dropna()

    # allow reuse scaler later (live trading future)
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(state)
    else:
        scaled = scaler.transform(state)

    return state, scaled, scaler


# =====================================
# CLUSTER STATES (NEW MODULAR)
# =====================================

def cluster_states(
    state,
    scaled,
    method="hdbscan",   # "hdbscan", "gmm", "kmeans"
    k=8
):
    """
    Cluster market states using clustering_engine.
    """

    labels, model = cluster(
        scaled,
        method=method,
        k=k
    )

    state["cluster"] = labels

    return state, model


# =====================================
# SELF TEST
# =====================================

if __name__ == "__main__":

    print("STATE ENGINE TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix
    from .regime_engine import build_regime_matrix

    size = 1500
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    # build features
    f = build_feature_matrix(df)
    r = build_regime_matrix(df)

    # build state
    state, scaled, scaler = build_state_matrix(f, r)

    # cluster
    state, model = cluster_states(
        state,
        scaled,
        method="hdbscan"   # change to test others
    )

    print(state.head())

    print("\nClusters:", state["cluster"].nunique())
    print("Noise points:", (state["cluster"] == -1).sum())

    print("PASSED ✅")
