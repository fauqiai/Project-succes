import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import mplfinance as mpf

from clustering_engine import run_clustering
from state_engine import analyze_states

# ===============================
# LOAD DATA
# ===============================

DATA_PATH = "xauusd_2025_m5.csv"

df = pd.read_csv(DATA_PATH)

df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

print("STEP 1 – RUNNING RESEARCH")
print("Rows:", len(df))

# ===============================
# RUN CLUSTERING (ASLI)
# ===============================

df, model_info = run_clustering(df)

# ===============================
# ANALYZE STATES (ASLI)
# ===============================

top_states = analyze_states(df)

print("\nTOP STATES:")
print(top_states)

# ===============================
# SAVE RESEARCH OUTPUT (ASLI)
# ===============================

with open("research_output.pkl", "wb") as f:
    pickle.dump({
        "data": df,
        "top_states": top_states,
        "model_info": model_info
    }, f)

print("Research saved -> research_output.pkl")

# ===============================
# VISUALISASI CLUSTER TERBAIK
# ===============================

# Ambil cluster dengan edge tertinggi
best_cluster = top_states.iloc[0].name
print("\nBest cluster:", best_cluster)

mask = df["cluster"] == best_cluster

if mask.sum() > 0:

    # Ambil candle tengah dari cluster
    mid_index = df[mask].index[len(df[mask]) // 2]

    # ===== ZOOM 1 FULL DAY =====
    day_start = mid_index.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    zoom_df = df.loc[day_start:day_end].copy()
    zoom_mask = mask.loc[day_start:day_end]

    # Pastikan kolom sesuai mplfinance
    zoom_df = zoom_df[['open', 'high', 'low', 'close']]

    # Buat garis vertikal tipis untuk cluster
    vlines = zoom_df.index[zoom_mask]

    # Plot candle
    mpf.plot(
        zoom_df,
        type='candle',
        style='charles',
        title=f"Top Cluster {best_cluster} (1 Day Zoom)",
        vlines=dict(
            vlines=vlines,
            colors='blue',
            linewidths=0.5,
            alpha=0.4
        ),
        warn_too_much_data=10000,
        savefig="top_cluster_candle.png"
    )

    print("Saved QUANT candle chart -> top_cluster_candle.png")

else:
    print("No data found for best cluster.")
