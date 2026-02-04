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
from behavior_core.event_detection import *
from behavior_core.regime_detection import *
from behavior_core.microstructure import *

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

def plot_candles(data, max_bars=300):

    data = data.tail(max_bars).reset_index(drop=True)

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

    return fig, ax, data


# ============================================================
# DRAW TRADING SIGNALS
# ============================================================

def overlay_entries(ax, data, signals):

    start_index = len(data)

    for signal in signals:

        i = signal["index"]

        # hanya tampilkan yang masuk area chart
        if i < (len(signals) - start_index):
            continue

        chart_i = i - (len(signals) - start_index)

        if chart_i < 0 or chart_i >= len(data):
            continue

        entry = signal["entry"]
        sl = signal["sl"]
        tp = signal["tp"]

        if signal["type"] == "BUY":

            # BUY arrow
            ax.scatter(chart_i, entry, marker="^", s=120)

        else:

            # SELL arrow
            ax.scatter(chart_i, entry, marker="v", s=120)

        # SL line
        ax.hlines(sl,
                  chart_i-3,
                  chart_i+3,
                  linestyles="dashed")

        # TP line
        ax.hlines(tp,
                  chart_i-3,
                  chart_i+3,
                  linestyles="solid")


# ============================================================
# FULL PIPELINE
# ============================================================

def run_chart_visualizer(path_to_csv,
                         timeframe=None,
                         max_bars=300):

    print("Loading data...")
    data = prepare_visual_data(path_to_csv, timeframe)

    if data is None:
        return

    print("Generating entries...")
    signals = generate_entry_signals(data)

    print("Plotting chart...")
    fig, ax, data = plot_candles(data, max_bars)

    overlay_entries(ax, data, signals)

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
