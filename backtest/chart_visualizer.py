"""
chart_visualizer.py
------------------
Quant chart visualizer with trading entries.

Features:
- Candlestick chart
- BUY / SELL arrows
- Stop Loss line
- Take Profit line
"""

import matplotlib.pyplot as plt

from data_loader import load_and_prepare
from entry_engine import generate_entry_signals
from config import *


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_visual_data(path_to_csv, timeframe=None):

    data = load_and_prepare(path_to_csv, timeframe)

    if data is None or len(data) == 0:
        print("Failed to load data.")
        return None

    data["range"] = data["high"] - data["low"]
    data["body"] = (data["close"] - data["open"]).abs()
    data["atr"] = data["range"].rolling(ATR_PERIOD, min_periods=1).mean()

    return data


# ============================================================
# SIMPLE CANDLE PLOT
# ============================================================

def plot_candles(data):

    fig, ax = plt.subplots(figsize=(16,7))

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
# DRAW TRADING SIGNALS (FIXED)
# ============================================================

def overlay_entries(ax, signals):

    for signal in signals:

        i = signal["index"]
        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal["tp"]

        if signal["type"] == "BUY":
            ax.scatter(i, entry, marker="^", s=120)
        else:
            ax.scatter(i, entry, marker="v", s=120)

        # SL
        ax.hlines(sl, i-3, i+3, linestyles="dashed")

        # TP
        ax.hlines(tp, i-3, i+3, linestyles="solid")


# ============================================================
# FULL PIPELINE (FIXED)
# ============================================================

def run_chart_visualizer(path_to_csv,
                         timeframe=None,
                         max_bars=300):

    print("Loading data...")
    data = prepare_visual_data(path_to_csv, timeframe)

    if data is None:
        return

    # 🔥 BATASI DATA DULU (SUPER IMPORTANT)
    data = data.tail(max_bars).reset_index(drop=True)

    print("Generating entries...")
    signals = generate_entry_signals(data)

    print("Plotting chart...")
    fig, ax = plot_candles(data)

    overlay_entries(ax, signals)

    plt.title("Quant Behavior Trading Chart")
    plt.grid(alpha=0.2)
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    DATA_PATH = "xauusd_m1_cleaned.csv"

    run_chart_visualizer(
        path_to_csv=DATA_PATH,
        timeframe=DEFAULT_TIMEFRAME,
        max_bars=300
    )
