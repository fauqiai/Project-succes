import numpy as np
import pandas as pd

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.state_engine import build_state_matrix, cluster_states
from core.interaction_engine import build_interactions
from core.transition_engine import transition_matrix, transition_expectancy
from core.expectancy_engine import state_edge


def generate_data(size=6000):

    np.random.seed(42)

    price = np.cumsum(np.random.randn(size)) + 100

    return pd.DataFrame({
        "open": price,
        "high": price + np.random.rand(size),
        "low": price - np.random.rand(size),
        "close": price
    })


def run():

    print("\n🔥 RUNNING ULTRA EDGE ENGINE\n")

    df = generate_data()

    features = build_feature_matrix(df)
    regimes = build_regime_matrix(df)

    interactions = build_interactions(features)
    features = pd.concat([features, interactions], axis=1)

    state, scaled, _ = build_state_matrix(features, regimes)
    state, _ = cluster_states(state, scaled, k=8)

    print("\nTOP STATES:")
    print(state_edge(df, state).head())

    print("\nTRANSITIONS:")
    print(transition_matrix(state))

    print("\nTRANSITION EXPECTANCY:")
    print(transition_expectancy(df, state).head())

    print("\n✅ RESEARCH COMPLETE")


if __name__ == "__main__":
    run()
