import pandas as pd
import pickle

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.interaction_engine import build_interactions
from core.state_engine import build_state_matrix, cluster_states
from core.expectancy_engine import state_edge


def main():

    print("\n🔥 STEP 1 — RUNNING RESEARCH\n")

    df = pd.read_csv("xauusd_m1_cleaned.csv")
    df.columns = df.columns.str.lower().str.strip()

    # AUTO detect kolom waktu (tidak ubah logic lain)
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

    # align index untuk keamanan output saja (tidak ubah logic)
    state = state.loc[df.index]

    # =====================
    # EDGE (ASLI)
    # =====================

    edge_table = state_edge(df, state)

    print("\n🔥 TOP STATES:")
    print(edge_table.head(10))

    # =====================
    # 🔥 EXPORT CSV (BARU - VISUAL / MT5)
    # =====================

    # ---- state per candle ----
    state_export = state.copy()

    # pastikan ada kolom cluster
    if "cluster" not in state_export.columns:
        raise ValueError("Column 'cluster' not found in state output.")

    state_export = state_export[["cluster"]].copy()
    state_export.index.name = "time"
    state_export.to_csv("state_per_candle.csv")

    print("✅ Saved → state_per_candle.csv")

    # ---- edge table ----
    edge_export = edge_table.copy()
    edge_export.to_csv("edge_table.csv")

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
