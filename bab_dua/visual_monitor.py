# visual_monitor.py

import pandas as pd
import matplotlib.pyplot as plt

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.state_engine import build_state_matrix, cluster_states
from core.interaction_engine import build_interactions
from core.expectancy_engine import state_edge
from core.regime_shift_engine import detect_regime_shift


CSV_PATH = "xauusd_m1_cleaned.csv"


def build_state(df):

    features = build_feature_matrix(df)
    regimes = build_regime_matrix(df)

    interactions = build_interactions(features)
    features = pd.concat([features, interactions], axis=1)

    state, scaled, *_ = build_state_matrix(features, regimes)
    state, _ = cluster_states(state, scaled)

    return state


def plot_clusters(df, state):

    plt.figure(figsize=(16,6))

    clusters = state["cluster"].unique()

    for c in clusters:
        mask = state["cluster"] == c
        plt.scatter(
            state.index[mask],
            df["close"][mask],
            s=5,
            label=f"Cluster {c}"
        )

    plt.plot(df["close"], linewidth=1)

    plt.title("Market States (Cluster Visualization)")
    plt.legend()

    plt.savefig("clusters.png")
    plt.close()


def plot_edge_zones(df, state):

    edge_table = state_edge(df, state)

    strong = edge_table[edge_table["edge"] > 0.05].index
    weak = edge_table[edge_table["edge"] <= 0.0].index

    plt.figure(figsize=(16,6))

    plt.plot(df["close"], linewidth=1)

    for c in strong:
        mask = state["cluster"] == c
        plt.scatter(state.index[mask], df["close"][mask], s=6)

    for c in weak:
        mask = state["cluster"] == c
        plt.scatter(state.index[mask], df["close"][mask], s=6, marker="x")

    plt.title("Edge Zones (Dots = Strong, X = Fake)")
    plt.savefig("edge_zones.png")
    plt.close()


def plot_regime_shift(df):

    returns = df.close.pct_change()

    shift = detect_regime_shift(returns)

    plt.figure(figsize=(16,4))

    plt.plot(shift.astype(int))

    plt.title("Regime Shift Detector")
    plt.savefig("regime_shift.png")
    plt.close()


def main():

    print("\n🔥 BUILDING VISUAL MONITOR...\n")

    df = pd.read_csv(CSV_PATH)

    df.columns = df.columns.str.lower().str.strip()
    df = df[["open","high","low","close"]].dropna()

    state = build_state(df)

    plot_clusters(df, state)
    plot_edge_zones(df, state)
    plot_regime_shift(df)

    print("✅ Charts saved:")
    print(" - clusters.png")
    print(" - edge_zones.png")
    print(" - regime_shift.png\n")


if __name__ == "__main__":
    main()
