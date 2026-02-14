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

    # AUTO detect kolom waktu (tidak ubah logic research)
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
    # BUILD STATE  (ASLI)
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

    # pastikan align index (visual safety only)
    state = state.loc[df.index]

    # =====================
    # EDGE (ASLI)
    # =====================

    edge_table = state_edge(df, state)

    print("\n🔥 TOP STATES:")
    print(edge_table.head(10))

    # ====================================================
    # 🔥 STABLE CANDLE VISUAL (WINDOW FIXED SIZE)
    # ====================================================

    if len(edge_table) > 0:

        top_cluster = edge_table["edge"].idxmax()

        mask = state["cluster"] == top_cluster
        cluster_times = state.index[mask]

        if len(cluster_times) > 0:

            # ambil kemunculan tengah cluster terbaik
            mid_time = cluster_times[len(cluster_times)//2]

            # cari posisi index numerik
            mid_pos = df.index.get_indexer([mid_time], method="nearest")[0]

            # 🔥 WINDOW CANDLE TETAP (tidak tergantung jam trading)
            WINDOW = 600   # kiri
            WINDOW_RIGHT = 600   # kanan

            start = max(0, mid_pos - WINDOW)
            end   = min(len(df), mid_pos + WINDOW_RIGHT)

            zoom_df = df.iloc[start:end].copy()
            zoom_state = state.iloc[start:end]

            # mask cluster di window ini
            zoom_mask = zoom_state["cluster"] == top_cluster

            if len(zoom_df) > 10:

                fig, axlist = mpf.plot(
                    zoom_df,
                    type="candle",
                    style="charles",
                    figsize=(22,10),
                    returnfig=True
                )

                ax = axlist[0]

                # garis tipis transparan highlight cluster
                highlight_times = zoom_df.index[zoom_mask]

                for t in highlight_times:
                    ax.axvline(t, linewidth=0.6, alpha=0.25)

                ax.set_title(
                    f"Top Cluster = {top_cluster} | Edge = {edge_table.loc[top_cluster,'edge']:.4f}\n"
                    f"Candles shown = {len(zoom_df)}",
                    fontsize=14
                )

                fig.savefig("top_cluster_candle.png", dpi=300, bbox_inches="tight")
                plt.close(fig)

                print("✅ Saved QUANT candle chart → top_cluster_candle.png")

            else:
                print("⚠️ Too few candles to plot.")

        else:
            print("⚠️ No candles found for top cluster.")

    # =====================
    # SAVE (ASLI)
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
