import pandas as pd
import pickle
import matplotlib.pyplot as plt
import mplfinance as mpf

from core.feature_engine import build_feature_matrix
from core.regime_engine import build_regime_matrix
from core.interaction_engine import build_interactions
from core.state_engine import build_state_matrix, cluster_states
from core.expectancy_engine import state_edge


def main():

    print("\n🔥 STEP 1 — RUNNING RESEARCH\n")

    df = pd.read_csv("xauusd_m1_cleaned.csv")
    df.columns = df.columns.str.lower().str.strip()

    # Jika ada kolom waktu → jadikan index
    for col in ["time", "date", "datetime", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            break

    df = df[["open", "high", "low", "close"]].dropna()

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

    # ====================================================
    # 🔥 STABLE CANDLE VISUAL (INDEX BASED ZOOM)
    # ====================================================

    top_cluster = edge_table["edge"].idxmax()

    mask = state["cluster"] == top_cluster
    cluster_positions = mask[mask].index

    if len(cluster_positions) > 0:

        # Ambil posisi tengah cluster
        mid_pos = len(cluster_positions) // 2
        mid_index = df.index.get_loc(cluster_positions[mid_pos])

        # Zoom 200 candle kiri-kanan (aman semua timeframe)
        start = max(0, mid_index - 200)
        end   = min(len(df), mid_index + 200)

        zoom_df = df.iloc[start:end].copy()
        zoom_mask = mask.iloc[start:end]

        fig, axlist = mpf.plot(
            zoom_df,
            type='candle',
            style='charles',
            figsize=(18,8),
            returnfig=True,
            warn_too_much_data=10000
        )

        ax = axlist[0]

        # Garis tipis cluster
        for t in zoom_df.index[zoom_mask]:
            ax.axvline(t, linewidth=0.6, alpha=0.35)

        fig.savefig("top_cluster_candle.png", dpi=200)
        plt.close(fig)

        print("✅ Saved CLEAN candle chart → top_cluster_candle.png")

    else:
        print("⚠️ No candles found for top cluster.")

    # =====================
    # SAVE
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
