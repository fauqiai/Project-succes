"""
chart_visualizer.py
------------------
Module untuk visualisasi hasil Quant Behavior langsung di chart.

Fungsi:
- Plot candlestick sederhana
- Overlay event
- Overlay microstructure
- Overlay regime
"""

import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_and_prepare
from behavior_core.measures import *
from behavior_core.event_detection import *
from behavior_core.regime_detection import *
from behavior_core.microstructure import *

from config import *


# ============================================================
# 1. PREPARE DATA
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
# 2. COMPUTE SIGNALS
# ============================================================

def compute_behavior_signals(data):

    signals = {}

    signals["impulse"] = detect_impulse(data, data["atr"])
    signals["retracement"] = detect_retracement(data)
    signals["consolidation"] = detect_consolidation(data)

    signals["sweep"] = detect_sweep(data)
    signals["imbalance"] = detect_imbalance(data)
    signals["fvg"] = detect_fvg(data)
    signals["displacement"] = detect_displacement(data)
    signals["sfp"] = detect_sfp(data)

    # VERY IMPORTANT -> convert ke list biar anti iloc error
    signals["regime"] = list(classify_regime(data))

    return signals


# ============================================================
# 3. SIMPLE CANDLE PLOT
# ============================================================

def plot_candles(data, max_bars=300):

    data = data.tail(max_bars).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16,7))

    for i in range(len(data)):

        open_price = data["open"].iloc[i]
        close_price = data["close"].iloc[i]
        high_price = data["high"].iloc[i]
        low_price = data["low"].iloc[i]

        color = "green" if close_price >= open_price else "red"

        # wick
        ax.plot([i, i], [low_price, high_price])

        # body
        ax.plot([i, i], [open_price, close_price], linewidth=4)

    return fig, ax, data


# ============================================================
# 4. OVERLAY EVENTS
# ============================================================

def overlay_events(ax, data, signals):

    for i in range(len(data)):

        if signals["impulse"].iloc[-len(data)+i]:
            ax.scatter(i, data["high"].iloc[i], marker="^", s=60)

        elif signals["retracement"].iloc[-len(data)+i]:
            ax.scatter(i, data["low"].iloc[i], marker="v", s=60)

        elif signals["consolidation"].iloc[-len(data)+i]:
            ax.scatter(i, data["close"].iloc[i], marker="o", s=30)


# ============================================================
# 5. OVERLAY MICROSTRUCTURE
# ============================================================

def overlay_microstructure(ax, data, signals):

    for i in range(len(data)):

        if signals["sweep"].iloc[-len(data)+i]:
            ax.scatter(i, data["high"].iloc[i], marker="x", s=80)

        if signals["displacement"].iloc[-len(data)+i]:
            ax.scatter(i, data["close"].iloc[i], marker="s", s=40)

        if signals["sfp"].iloc[-len(data)+i]:
            ax.scatter(i, data["low"].iloc[i], marker="*", s=120)


# ============================================================
# 6. OVERLAY REGIME (FIX TOTAL)
# ============================================================

def overlay_regime(ax, data, signals):

    regimes = list(signals["regime"])[-len(data):]

    for i in range(len(data)):

        regime = regimes[i]

        if regime == "trend":
            ax.axvspan(i-0.5, i+0.5, alpha=0.05)

        elif regime == "range":
            ax.axvspan(i-0.5, i+0.5, alpha=0.02)


# ============================================================
# 7. FULL PIPELINE
# ============================================================

def run_chart_visualizer(path_to_csv,
                         timeframe=None,
                         max_bars=300):

    print("Loading data for visualization...")
    data = prepare_visual_data(path_to_csv, timeframe)

    if data is None:
        return

    print("Computing behavior signals...")
    signals = compute_behavior_signals(data)

    print("Plotting chart...")
    fig, ax, data = plot_candles(data, max_bars)

    overlay_events(ax, data, signals)
    overlay_microstructure(ax, data, signals)
    overlay_regime(ax, data, signals)

    plt.title("Quant Behavior Chart")
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
