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

    # AUTO detect time column
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

    # =====================
    # 🔥 VISUAL 1 HARI SAJA
    # =====================

    top_cluster = edge_table["edge"].idxmax()

    mask = state["cluster"] == top_cluster
    cluster_times = df.index[mask]

    if len(cluster_times) > 0:

        # ambil hari pertama dari cluster tertinggi
        first_time = cluster_times[0]
        day_start = first_time.normalize()
        day_end = day_start + pd.Timedelta(days=1)

        day_df = df.loc[day_start:day_end].copy()
        day_mask = mask.loc[day_df.index]

        if len(day_df) > 0:

            fig, axlist = mpf.plot(
                day_df,
                type='candle',
                style='charles',
                figsize=(14,7),
                returnfig=True,
                warn_too_much_data=10000
            )

            ax = axlist[0]

            # garis tipis untuk cluster
            for t in day_df.index[day_mask]:
                ax.axvline(t, color='blue', linewidth=0.6, alpha=0.4)

            fig.savefig("top_cluster_candle.png", dpi=200)
            plt.close(fig)

            print("✅ Saved 1-day cluster chart → top_cluster_candle.png")

        else:
            print("⚠️ No data for that day.")

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
