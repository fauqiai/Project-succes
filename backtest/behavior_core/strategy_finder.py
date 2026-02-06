"""
strategy_finder.py
-----------------
Strategy Finder V3

Behavior + Regime aware.
Mencari edge nyata di market.
"""

import pandas as pd
import numpy as np

from .event_detection import detect_impulse, detect_retracement, detect_consolidation
from .regime_detection import classify_regime
from .expectancy import compute_excursions


# ============================================================
# CONFIG
# ============================================================

MIN_SAMPLES = 200
MIN_WINRATE = 0.52


# ============================================================
# BUILD CONDITIONS
# ============================================================

def build_conditions(data):

    atr = (data["high"] - data["low"]).rolling(14, min_periods=1).mean()

    impulse = detect_impulse(data, atr)
    retracement = detect_retracement(data)
    consolidation = detect_consolidation(data)

    regime = classify_regime(data)

    conditions = {

        # 🔥 TREND IMPULSE (bias continuation)
        "trend_impulse_buy":
            (impulse) & (pd.Series(regime) == "trend") & (data["close"] > data["open"]),

        "trend_impulse_sell":
            (impulse) & (pd.Series(regime) == "trend") & (data["close"] < data["open"]),

        # 🔥 RANGE RETRACE (mean reversion)
        "range_retrace_buy":
            (retracement) & (pd.Series(regime) == "range") & (data["close"] > data["open"]),

        "range_retrace_sell":
            (retracement) & (pd.Series(regime) == "range") & (data["close"] < data["open"]),

        # 🔥 CONSOLIDATION BREAK (early expansion)
        "breakout_buy":
            (consolidation) & (data["close"] > data["open"]),

        "breakout_sell":
            (consolidation) & (data["close"] < data["open"]),
    }

    return conditions


# ============================================================
# EVALUATE STRATEGY
# ============================================================

def evaluate_strategy(mask, data, forward_points):

    indices = np.where(mask)[0]

    if len(indices) < MIN_SAMPLES:
        return None

    excursions = compute_excursions(data, indices, forward_points)

    if excursions is None:
        return None

    avg_up = excursions["avg_up"]
    avg_down = excursions["avg_down"]
    winrate = excursions["winrate"]

    EV = (avg_up * winrate) - (avg_down * (1 - winrate))

    if winrate < MIN_WINRATE:
        return None

    return {
        "EV": float(EV),
        "avg_up_move": float(avg_up),
        "avg_down_move": float(avg_down),
        "winrate": float(winrate),
        "samples": int(len(indices))
    }


# ============================================================
# MAIN FINDER
# ============================================================

def find_best_strategies(data, forward_points=10):

    print("Scanning strategies (V3)...")

    conditions = build_conditions(data)

    results = []

    for name, mask in conditions.items():

        stats = evaluate_strategy(mask, data, forward_points)

        if stats:
            results.append((name, stats))

    if not results:
        print("No strategy with positive expectancy.")
        return []

    # Sort by EV
    results.sort(key=lambda x: x[1]["EV"], reverse=True)

    print(f"Found {len(results)} valid strategies.")
    print("Top strategy:", results[0][0])

    return results


# ============================================================
# SUMMARY
# ============================================================

def strategy_summary(best):

    if not best:
        return "No valid strategy found."

    strategy, stats = best[0]

    return {
        "strategy": strategy,
        "EV": stats["EV"],
        "winrate": stats["winrate"],
        "samples": stats["samples"]
    }
