import pandas as pd
from sklearn.preprocessing import StandardScaler

from .clustering_engine import cluster
from .dimensionality_engine import reduce_dim
from .feature_discovery import discover_features


# =====================================
# BUILD STATE MATRIX (FINAL FORM)
# =====================================

def build_state_matrix(
    features,
    regimes,
    scaler=None,
    dim_model=None,
    discovery_model=None
):
    """
    FINAL PIPELINE:

    handcrafted features
        + discovered features
        + regimes
        ↓
    scale
        ↓
    reduce dimension
        ↓
    READY for clustering
    """

    # 🔥 SCARY PART — auto feature discovery
    discovered, discovery_model = discover_features(
        features,
        model=discovery_model
    )

    state = pd.concat([
        features,
        discovered,
        regimes
    ], axis=1).dropna()

    # reuse scaler for live later
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(state)
    else:
        scaled = scaler.transform(state)

    # dimensionality reduction
    scaled, dim_model = reduce_dim(
        scaled,
        model=dim_model
    )

    return state, scaled, scaler, dim_model, discovery_model


# =====================================
# CLUSTER STATES
# =====================================

def cluster_states(
    state,
    scaled,
    method="hdbscan",
    k=8
):

    if method == "hmm":
        from .hmm_engine import hmm_cluster
        labels, model = hmm_cluster(scaled, n_states=k)

    else:
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

    print("FINAL STATE ENGINE TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix
    from .regime_engine import build_regime_matrix
    from .regime_validator import validate_regimes, regime_health_score

    size = 2500
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    f = build_feature_matrix(df)
    r = build_regime_matrix(df)

    state, scaled, scaler, dim_model, discovery_model = build_state_matrix(f, r)

    state, model = cluster_states(
        state,
        scaled,
        method="hdbscan"
    )

    print(state.head())

    report = validate_regimes(df, state)

    print("\nREGIME REPORT:")
    print(report)

    print("\nHEALTH:", regime_health_score(report))

    print("PASSED ✅")
