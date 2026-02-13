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

    # 🔥 AUTO detect kolom waktu (TIDAK ubah logic lain)
    time_col = None
    for col in ["time", "date", "datetime", "timestamp"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is not None:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)

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


    # ====================================================
    # 🔥 QUANT CANDLE VISUAL (FIXED TIME INDEX)
    # ====================================================

    top_cluster = edge_table["edge"].idxmax()

    mask = state["cluster"] == top_cluster
    cluster_indices = df.index[mask]

    if len(cluster_indices) > 0:

        center = len(cluster_indices)//2
        center_time = cluster_indices[center]

        zoom_df = df.loc[
            center_time - pd.Timedelta(hours=12):
            center_time + pd.Timedelta(hours=12)
        ].copy()

        zoom_mask = mask.loc[zoom_df.index]

        fig, axlist = mpf.plot(
            zoom_df,
            type='candle',
            style='charles',
            figsize=(24,10),
            returnfig=True
        )

        ax = axlist[0]

        # garis tipis zona cluster
        for t in zoom_df.index[zoom_mask]:
            ax.axvline(t, linewidth=0.5, alpha=0.35)

        fig.savefig("top_cluster_candle.png", dpi=300)
        plt.close(fig)

        print("✅ Saved QUANT candle chart → top_cluster_candle.png")

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
