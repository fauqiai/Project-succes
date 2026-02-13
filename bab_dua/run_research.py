import pandas as pd
import pickle
import matplotlib.pyplot as plt

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.interaction_engine import build_interactions
from core.state_engine import build_state_matrix, cluster_states
from core.expectancy_engine import state_edge


def main():

    print("\n🔥 STEP 1 — RUNNING RESEARCH\n")

    df = pd.read_csv("xauusd_m1_cleaned.csv")

    df.columns = df.columns.str.lower().str.strip()
    df = df[["open","high","low","close"]].dropna()

    print("Rows:", len(df))

    # =====================
    # BUILD STATE
    # =====================

    features = build_feature_matrix(df)
    regimes = build_regime_matrix(df)

    interactions = build_interactions(features)
    features = pd.concat([features, interactions], axis=1)

    state, scaled, scaler, dim_model, discovery_model = build_state_matrix(
        features,
        regimes
    )

    state, model = cluster_states(state, scaled)

    # =====================
    # EDGE
    # =====================

    edge_table = state_edge(df, state)

    print("\n🔥 TOP STATES:")
    print(edge_table.head(10))


    # =====================
    # 🔥 VISUAL CLUSTER (ADDED ONLY)
    # =====================

    top_cluster = edge_table["edge"].idxmax()

    mask = state["cluster"] == top_cluster

    plt.figure(figsize=(16,6))
    plt.plot(df["close"])
    plt.scatter(df.index[mask], df["close"][mask], s=10)

    plt.title(f"Highest Edge Cluster = {top_cluster}")
    plt.xlabel("Index")
    plt.ylabel("Price")

    plt.savefig("top_cluster.png")
    plt.close()

    print("✅ Saved chart → top_cluster.png")


    # =====================
    # SAVE (IMPORTANT)
    # =====================

    with open("research_output.pkl", "wb") as f:
        pickle.dump({
            "df": df,
            "state": state,
            "edge_table": edge_table
        }, f)

    print("\n✅ Research saved → research_output.pkl")


if __name__ == "__main__":
    main()
