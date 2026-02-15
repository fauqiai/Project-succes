import pandas as pd
import pickle

from sklearn.metrics import silhouette_score

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.interaction_engine import build_interactions
from core.state_engine import build_state_matrix, cluster_states
from core.expectancy_engine import state_edge


def main():

    print("\n🔥 STEP 1 — RUNNING RESEARCH\n")

    df = pd.read_csv("xauusd_m1_cleaned.csv")
    df.columns = df.columns.str.lower().str.strip()

    # AUTO detect kolom waktu
    time_col = None
    for col in ["time", "date", "datetime", "timestamp"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

    df = df[["open","high","low","close"]].dropna()
    df = df.sort_index()

    print("Rows:", len(df))

    # =====================
    # BUILD STATE (ASLI)
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

    # align index untuk keamanan output saja
    state = state.loc[df.index]

    # =====================
    # EDGE (ASLI)
    # =====================

    edge_table = state_edge(df, state)

    print("\n🔥 TOP STATES:")
    print(edge_table.head(10))

    # =====================
    # 🔥 VALIDASI CLUSTER (BARU)
    # =====================

    print("\n🔎 CLUSTER QUALITY CHECK")

    # 1️⃣ Silhouette score
    try:
        sil = silhouette_score(scaled, state["cluster"])
        print(f"Silhouette score: {sil:.4f}")
    except Exception as e:
        print("Silhouette score failed:", e)

    # 2️⃣ Distribusi cluster
    print("\nCluster distribution:")
    print(state["cluster"].value_counts(normalize=True).sort_index())

    # 3️⃣ Edge stability split 50:50
    mid = len(df)//2

    df_a, df_b = df.iloc[:mid], df.iloc[mid:]
    state_a, state_b = state.iloc[:mid], state.iloc[mid:]

    edge_a = state_edge(df_a, state_a)
    edge_b = state_edge(df_b, state_b)

    print("\nTop cluster FIRST half:")
    print(edge_a.sort_values("edge", ascending=False).head(3))

    print("\nTop cluster SECOND half:")
    print(edge_b.sort_values("edge", ascending=False).head(3))

    # =====================
    # 🔥 EXPORT CSV (MT5 / ANALISIS)
    # =====================

    state_export = state[["cluster"]].copy()
    state_export.index.name = "time"
    state_export.to_csv("state_per_candle.csv")
    print("\n✅ Saved → state_per_candle.csv")

    edge_table.to_csv("edge_table.csv")
    print("✅ Saved → edge_table.csv")

    # =====================
    # SAVE PKL (ASLI)
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
