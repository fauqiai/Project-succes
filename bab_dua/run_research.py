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
    # 🔥🔥 QUANT-GRADE CANDLE VISUAL (ADDED ONLY)
    # ====================================================

    top_cluster = edge_table["edge"].idxmax()

    mask = state["cluster"] == top_cluster
    cluster_indices = df.index[mask]

    if len(cluster_indices) > 0:

        center = cluster_indices[len(cluster_indices)//2]

        start = max(0, center - 150)
        end   = min(len(df), center + 150)

        zoom_df = df.iloc[start:end].copy()
        zoom_mask = mask.iloc[start:end]

        # mplfinance butuh datetime index
        zoom_df["date"] = pd.date_range(
            start="2024-01-01",
            periods=len(zoom_df),
            freq="T"
        )

        zoom_df.set_index("date", inplace=True)

        # cari zona cluster
        cluster_dates = zoom_df.index[zoom_mask]

        fig, axlist = mpf.plot(
            zoom_df,
            type='candle',
            style='charles',
            figsize=(24,10),
            returnfig=True
        )

        ax = axlist[0]

        # garis tipis zona cluster
        for d in cluster_dates:
            ax.axvline(d, linewidth=0.5, alpha=0.35)

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
