import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplfinance as mpf
from clustering_engine import run_clustering
from state_engine import analyze_states


def run_research():

    print("🔎 STEP 1 - RUNNING RESEARCH")

    # ===============================
    # LOAD DATA
    # ===============================
    df = pd.read_csv("xauusd_m5.csv")
    df.columns = df.columns.str.lower()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    print(f"Rows: {len(df)}")

    # ===============================
    # CLUSTERING (ASLI - TIDAK DIUBAH)
    # ===============================
    cluster_labels, features = run_clustering(df)

    df["cluster"] = cluster_labels

    # ===============================
    # STATE ANALYSIS (ASLI - TIDAK DIUBAH)
    # ===============================
    report = analyze_states(df)

    print("\n📊 TOP STATES:")
    print(report.head(10))

    # ===============================
    # SAVE RESEARCH OUTPUT
    # ===============================
    with open("research_output.pkl", "wb") as f:
        pickle.dump(report, f)

    print("\n💾 Research saved -> research_output.pkl")

    # =====================================================
    # ================= VISUAL SECTION ====================
    # =====================================================

    try:
        print("\n🎨 Generating TOP CLUSTER CANDLE VIEW...")

        # ambil cluster edge tertinggi
        top_cluster = report.sort_values("edge", ascending=False).index[0]

        cluster_df = df[df["cluster"] == top_cluster]

        if len(cluster_df) == 0:
            print("No data in top cluster.")
            return

        # ambil 1 hari pertama dari cluster tsb
        first_day = cluster_df.index.date[0]
        day_data = df[df.index.date == first_day].copy()

        if len(day_data) == 0:
            print("No daily data found.")
            return

        # tandai cluster range di hari tsb
        day_data["highlight"] = np.where(
            day_data["cluster"] == top_cluster,
            day_data["high"],
            np.nan
        )

        # buat garis highlight tipis
        apdict = mpf.make_addplot(
            day_data["highlight"],
            type='line',
            width=0.7
        )

        # plot candle
        mpf.plot(
            day_data,
            type='candle',
            style='charles',
            addplot=apdict,
            volume=False,
            figratio=(12,6),
            figscale=1.5,
            warn_too_much_data=10000,
            title=f"Top Cluster {top_cluster} - {first_day}"
        )

        print("✅ Candle chart displayed.")

    except Exception as e:
        print("Visual error:", e)


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    run_research()
