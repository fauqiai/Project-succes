"""
chart_visualizer.py
TRUE Quant Chart Visualizer

- Behavior aware
- Strategy driven entries
- Auto regime detection
- No manual signal rules
"""

import matplotlib.pyplot as plt

from data_loader import load_and_prepare
from entry_engine import generate_entry_signals

from behavior_core.event_detection import (
    detect_impulse,
    detect_retracement,
    detect_consolidation
)

from behavior_core.regime_detection import classify_regime

from config import *


# ============================================================
# PREPARE DATA (CRITICAL FIX)
# ============================================================

def prepare_visual_data(path_to_csv, timeframe=None):

    data = load_and_prepare(path_to_csv, timeframe)

    if data is None or len(data) == 0:
        print("Failed to load data.")
        return None

    # ===== BUILD BEHAVIOR COLUMNS =====
    data["impulse"] = detect_impulse(data)
    data["retracement"] = detect_retracement(data)
    data["consolidation"] = detect_consolidation(data)
    data["regime"] = classify_regime(data)

    return data


# ============================================================
# CANDLE PLOT
# ============================================================

def plot_candles(data):

    fig, ax = plt.subplots(figsize=(18,8))

    for i in range(len(data)):

        o = data["open"].iloc[i]
        c = data["close"].iloc[i]
        h = data["high"].iloc[i]
        l = data["low"].iloc[i]

        color = "green" if c >= o else "red"

        # wick
        ax.plot([i, i], [l, h], linewidth=1)

        # body
        ax.plot([i, i], [o, c], linewidth=4, color=color)

    return fig, ax


# ============================================================
# OVERLAY ENTRIES
# ============================================================

def overlay_entries(ax, signals):

    for signal in signals:

        i = signal["index"]
        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal["tp"]

        if signal["type"] == "BUY":
            ax.scatter(i, entry, marker="^", s=140)
        else:
            ax.scatter(i, entry, marker="v", s=140)

        # Stop Loss
        ax.hlines(sl, i-2, i+2, linestyles="dashed")

        # Take Profit
        ax.hlines(tp, i-2, i+2, linestyles="solid")


# ============================================================
# FULL PIPELINE
# ============================================================

def run_chart_visualizer(path_to_csv,
                        timeframe=None,
                        max_bars=400):

    print("Loading market data...")
    data = prepare_visual_data(path_to_csv, timeframe)

    if data is None:
        return

    # 🔥 SUPER IMPORTANT (biar gak lag)
    data = data.tail(max_bars).reset_index(drop=True)

    print("Running quant engine...")
    signals = generate_entry_signals(data)

    if not signals:
        print("No signals generated.")
        return

    print("Plotting chart...")
    fig, ax = plot_candles(data)

    overlay_entries(ax, signals)

    plt.title("QUANT BEHAVIOR EXECUTION CHART")
    plt.grid(alpha=0.25)

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "xauusd_m1_cleaned.csv"

    run_chart_visualizer(
        path_to_csv=DATA_PATH,
        timeframe=DEFAULT_TIMEFRAME,
        max_bars=400
    )
