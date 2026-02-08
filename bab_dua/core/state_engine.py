from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


def build_state_matrix(features, regimes):

    state = pd.concat([features, regimes], axis=1).dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(state)

    return state, scaled, scaler


def cluster_states(state, scaled, k=8):

    model = KMeans(n_clusters=k, n_init=25, random_state=42)

    labels = model.fit_predict(scaled)

    state["cluster"] = labels

    return state, model



if __name__ == "__main__":

    print("STATE ENGINE TEST")

    import numpy as np
    from .feature_engine import build_feature_matrix
    from .regime_engine import build_regime_matrix

    size = 1200
    price = np.cumsum(np.random.randn(size)) + 100

    df = pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })

    f = build_feature_matrix(df)
    r = build_regime_matrix(df)

    state, scaled, _ = build_state_matrix(f, r)
    state, _ = cluster_states(state, scaled)

    print(state.head())

    print("PASSED ✅")
