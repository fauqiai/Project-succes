"""
chart_visualizer.py
------------------
Module untuk visualisasi hasil Quant Behavior ke dalam chart.
ASCII safe.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from behavior_core.event_detection import *
from behavior_core.microstructure import *
from behavior_core.regime_detection import *

from data_loader import load_and_prepare
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
    data["atr"] = data["range"].rolling(14, min_periods=1).mean()

    return data


# ============================================================
# 2. COMPUTE SIGNALS
# ============================================================

def compute_behavior_signals(data):

    signals = {}

    # Events
    signals["impulse"] = detect_impulse(data, data["atr"])
    signals["retracement"] = detect_retracement(data)
    signals["consolidation"] = detect_consolidation(data)

    # Microstructure
    signals["sweep"] = detect_sweep(data)
    signals["imbalance"] = detect_imbalance(data)
    signals["fvg"] = detect_fvg(data)
    signals["displacement"] = detect_displacement(data)
    signals["sfp"] = detect_sfp(data)

    # Regime
    signals["regime"] = classify_regime(data)

    return signals


# ============================================================
# 3. CANDLE PLOT
# ============================================================

def plot_candles(data, max_bars=300):

    data = data.tail(max_bars)

    fig, ax = plt.subplots(figsize=(14,7))

    for i in range(len(data)):

        open_price = data["open"].iloc[i]
        close_price = data["close"].iloc[i]
        high = data["high"].iloc[i]
        low = data["low"].iloc[i]

        color = "green" if close_price >= open_price else "red"

        # wick
        ax.plot([i, i], [low, high], linewidth=1)

        # body
        ax.add_patch(
            plt.Rectangle(
                (i - 0.3, min(open_price, close_price)),
                0.6,
                abs(close_price - open_price)
            )
        ).set_color(color)

    ax.set_title("Quant Behavior Chart")
    ax.grid(alpha=0.2)

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
            ax.scatter(i, data["close"].iloc[i], marker="o", s=20)


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
# 6. OVERLAY REGIME
# ============================================================

def overlay_regime(ax, data, signals):

    regimes = signals["regime"][-len(data):]

    for i in range(len(data)):

        regime = regimes.iloc[i]

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

    plt.show()


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    print("Chart Visualizer Running...")

    DATA_PATH = "xauusd_m1_cleaned.csv"

    run_chart_visualizer(
        path_to_csv=DATA_PATH,
        timeframe=DEFAULT_TIMEFRAME,
        max_bars=300
    )

